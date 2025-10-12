from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from torchvision import transforms
from sklearn.metrics import roc_auc_score, accuracy_score


# -----------------------------------------------------------------------------
# Model components
# -----------------------------------------------------------------------------


class ImageEncoder(nn.Module):
    """A lightweight CNN that produces a fixed-size embedding for each image."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(256, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.net(x).flatten(1)
        proj = self.proj(features)
        return F.relu(proj)


class TabularMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, out_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(num_features=in_dim, affine=True),
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FusionClassifier(nn.Module):
    def __init__(
        self,
        num_images: int,
        num_feats: int,
        img_embed_dim: int = 256,
        tab_embed_dim: int = 128,
        fusion_hidden: int = 256,
        dropout: float = 0.2,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.num_images = num_images
        self.image_encoder = ImageEncoder(in_channels=in_channels, embed_dim=img_embed_dim)
        self.tab_mlp = TabularMLP(in_dim=num_feats, hidden=tab_embed_dim, out_dim=tab_embed_dim)
        fusion_in = img_embed_dim + tab_embed_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden // 2, 1),
        )

    def forward(self, images: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        """Return logits for a batch.

        images: (B, K, C, H, W)
        feats:   (B, F)
        """
        batch_size, k, c, h, w = images.shape
        if k != self.num_images:
            raise ValueError(f"Expected {self.num_images} images per sample, got {k}.")
        flat = images.view(batch_size * k, c, h, w)
        img_emb = self.image_encoder(flat)
        img_emb = img_emb.view(batch_size, k, -1).mean(dim=1)
        tab_emb = self.tab_mlp(feats)
        fused = torch.cat([img_emb, tab_emb], dim=1)
        logits = self.fusion(fused).squeeze(1)
        return logits


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


@dataclass
class Sample:
    csv_path: str
    match_id: str
    minute_str: str
    image_dir: str


class LolMinuteDataset(Dataset):
    """Loads per-minute CSVs together with 10 participant trajectory PNGs."""

    def __init__(
        self,
        dataset_root: str,
        image_dir_name: str = "image_dataset",
        num_players: int = 10,
        image_size: int = 224,
        augment: bool = True,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.image_root = self.dataset_root / image_dir_name
        self.num_players = num_players

        self.samples: List[Sample] = []
        for match_dir in sorted(self.dataset_root.glob("KR_*")):
            if match_dir.name == image_dir_name:
                continue
            match_id = match_dir.name
            for csv_path in sorted(match_dir.glob(f"{match_id}_min*.csv")):
                minute_match = re.search(r"_min(\d+)", csv_path.name)
                if not minute_match:
                    continue
                minute_str = f"min{int(minute_match.group(1)):02d}"
                image_dir = self.image_root / match_id / minute_str
                expected = [image_dir / f"{match_id}_{minute_str}_{i}.png" for i in range(1, num_players + 1)]
                if not all(p.exists() for p in expected):
                    continue
                self.samples.append(Sample(csv_path=str(csv_path), match_id=match_id, minute_str=minute_str, image_dir=str(image_dir)))

        if not self.samples:
            raise RuntimeError(f"No valid samples found under {self.dataset_root}.")

        t_aug = [transforms.Resize((image_size, image_size))]
        if augment:
            t_aug.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            ])
        t_aug.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.transform = transforms.Compose(t_aug)

        self.feat_dim = self._infer_feat_dim(self.samples[0].csv_path)

    def __len__(self) -> int:
        return len(self.samples)

    def _read_table_to_features(self, csv_path: str) -> Tuple[np.ndarray, int]:
        df = pd.read_csv(csv_path)

        label = None
        for col in ["win", "label", "result"]:
            if col in df.columns:
                try:
                    label = int(round(float(df[col].iloc[0])))
                except Exception:
                    label = None
                df = df.drop(columns=[col], errors="ignore")

        if df.shape[1] >= 1:
            df = df.iloc[:, 1:]
        if df.shape[1] >= 1:
            df = df.iloc[:, :-1]
        df = df.drop(columns=['champion_id'], errors='ignore')

        df_numeric = df.select_dtypes(include=[np.number]).copy()
        if df_numeric.empty:
            raise ValueError(f"No numeric features parsed from {csv_path}.")

        feats = df_numeric.to_numpy(dtype=np.float32).reshape(-1)
        if label is None:
            label = 0
        return feats, label

    def _infer_feat_dim(self, csv_path: str) -> int:
        feats, _ = self._read_table_to_features(csv_path)
        return int(feats.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        feats_np, label = self._read_table_to_features(sample.csv_path)

        images: List[torch.Tensor] = []
        for i in range(1, self.num_players + 1):
            img_path = Path(sample.image_dir) / f"{sample.match_id}_{sample.minute_str}_{i}.png"
            img = Image.open(img_path).convert("RGB")
            images.append(self.transform(img))
        image_tensor = torch.stack(images, dim=0)
        feat_tensor = torch.from_numpy(feats_np)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return image_tensor, feat_tensor, label_tensor


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def collate_batch(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images, feats, labels = zip(*batch)
    return torch.stack(images, dim=0), torch.stack(feats, dim=0), torch.stack(labels, dim=0)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> float:
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    running_loss = 0.0
    total_batches = len(loader)
    log_interval = max(1, total_batches // 20)
    step = 0
    for images, feats, labels in loader:
        step += 1
        if step % log_interval == 0:
            pct = 100.0 * step / total_batches
            print(f'  batch {step}/{total_batches} ({pct:.1f}% )')
        images = images.to(device)
        feats = feats.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(images, feats)
            loss = loss_fn(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    with torch.no_grad():
        for images, feats, labels in loader:
            images = images.to(device)
            feats = feats.to(device)
            logits = model(images, feats)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    logits_np = torch.cat(all_logits).numpy()
    labels_np = torch.cat(all_labels).numpy()
    preds = (1.0 / (1.0 + np.exp(-logits_np)) > 0.5).astype(np.int32)
    acc = accuracy_score(labels_np, preds)
    try:
        auc = roc_auc_score(labels_np, logits_np)
    except ValueError:
        auc = float("nan")
    return acc, auc


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fusion model on per-minute LOL trajectories.")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--image-dir-name", type=str, default="image_dataset")
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})" if device.type == "cuda"
      else f"Using device: {device}")

    resume_ckpt = None
    start_epoch = 0
    best_auc = -1.0
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f'Resume checkpoint not found: {resume_path}')
        print(f"Resuming from {resume_path}")
        resume_ckpt = torch.load(resume_path, map_location='cpu')
        start_epoch = resume_ckpt.get('epoch', 0)
        best_auc = resume_ckpt.get('best_auc', -1.0)

    print('Loading training dataset...')
    full_ds = LolMinuteDataset(
        dataset_root=args.dataset_root,
        image_dir_name=args.image_dir_name,
        num_players=args.num_images,
        image_size=args.image_size,
        augment=not args.no_augment,
    )
    print(f'Training samples loaded: {len(full_ds)}')
    num_feats = full_ds.feat_dim

    total = len(full_ds)
    if resume_ckpt and 'train_idx' in resume_ckpt and 'val_idx' in resume_ckpt:
        train_idx = resume_ckpt['train_idx']
        val_idx = resume_ckpt['val_idx']
        if isinstance(train_idx, np.ndarray):
            train_idx = train_idx.tolist()
        if isinstance(val_idx, np.ndarray):
            val_idx = val_idx.tolist()
        train_size = len(train_idx)
        if len(val_idx) + train_size != total:
            raise ValueError('Checkpoint indices do not match current dataset size')
    else:
        train_size = int(total * 0.9)
        permutation = np.random.permutation(total)
        train_idx = permutation[:train_size].tolist()
        val_idx = permutation[train_size:].tolist()

    print('Loading validation dataset...')
    val_dataset = LolMinuteDataset(
        dataset_root=args.dataset_root,
        image_dir_name=args.image_dir_name,
        num_players=args.num_images,
        image_size=args.image_size,
        augment=False,
    )
    print(f'Validation samples loaded: {len(val_dataset)}')

    train_subset = torch.utils.data.Subset(full_ds, train_idx)
    val_subset = torch.utils.data.Subset(val_dataset, val_idx)

    print('Creating data loaders...')
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    print('Data loaders ready')

    model = FusionClassifier(num_images=args.num_images, num_feats=num_feats).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    if resume_ckpt:
        if 'model' in resume_ckpt:
            model.load_state_dict(resume_ckpt['model'])
        if 'optimizer' in resume_ckpt and resume_ckpt['optimizer']:
            optimizer.load_state_dict(resume_ckpt['optimizer'])
        if 'scaler' in resume_ckpt and resume_ckpt['scaler']:
            try:
                scaler.load_state_dict(resume_ckpt['scaler'])
            except Exception as exc:
                print(f"Warning: could not load scaler state ({exc})")
        msg_auc = f"{best_auc:.4f}" if best_auc != -1.0 else str(best_auc)
        print(f"Resumed from checkpoint at epoch {start_epoch} with best AUC {msg_auc}")

    if resume_ckpt is None:
        best_auc = -1.0
        start_epoch = 0
    for epoch in range(start_epoch + 1, start_epoch + 1 + args.epochs):
        print(f"\nEpoch {epoch:02d}/{args.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        print(f"Train loss: {train_loss:.4f}")
        acc, auc = evaluate(model, val_loader, device)
        print(f"Val acc:   {acc:.4f}")
        print(f"Val AUC:   {auc:.4f}")
        improved = auc > best_auc
        if improved:
            best_auc = auc
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'epoch': epoch,
            'best_auc': best_auc,
            'train_idx': train_idx,
            'val_idx': val_idx,
            'dataset_root': str(args.dataset_root),
            'image_dir_name': args.image_dir_name,
        }
        torch.save(checkpoint, 'last.pt')
        if improved:
            torch.save(checkpoint, 'best.pt')
            print(f"Saved new best checkpoint with AUC={auc:.4f}")
        else:
            print(f"Checkpoint saved (AUC={auc:.4f})")


if __name__ == "__main__":
    main()