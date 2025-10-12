from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from sklearn.metrics import roc_auc_score, accuracy_score


# -----------------------------------------------------------------------------
# Model components
# -----------------------------------------------------------------------------


class TabularClassifier(nn.Module):
    def __init__(self, num_feats: int, hidden: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(num_feats),
            nn.Linear(num_feats, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats).squeeze(1)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


@dataclass
class Sample:
    csv_path: str
    match_id: str
    minute_str: str


class TabularMinuteDataset(Dataset):
    """Loads per-minute CSVs containing tabular features only."""

    def __init__(
        self,
        dataset_root: str,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.samples: List[Sample] = []
        for match_dir in sorted(self.dataset_root.glob("KR_*")):
            match_id = match_dir.name
            for csv_path in sorted(match_dir.glob(f"{match_id}_min*.csv")):
                minute_str = csv_path.stem.split("_")[-1]
                self.samples.append(Sample(csv_path=str(csv_path), match_id=match_id, minute_str=minute_str))

        if not self.samples:
            raise RuntimeError(f"No valid samples found under {self.dataset_root}.")

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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        feats_np, label = self._read_table_to_features(sample.csv_path)
        feat_tensor = torch.from_numpy(feats_np)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return feat_tensor, label_tensor


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def collate_batch(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    feats, labels = zip(*batch)
    return torch.stack(feats, dim=0), torch.stack(labels, dim=0)


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
    for feats, labels in loader:
        step += 1
        if step % log_interval == 0:
            pct = 100.0 * step / total_batches
            print(f'  batch {step}/{total_batches} ({pct:.1f}% )')
        feats = feats.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(feats)
            loss = loss_fn(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * feats.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    with torch.no_grad():
        for feats, labels in loader:
            feats = feats.to(device)
            logits = model(feats)
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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
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
    full_ds = TabularMinuteDataset(dataset_root=args.dataset_root)
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
    val_dataset = TabularMinuteDataset(dataset_root=args.dataset_root)
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

    model = TabularClassifier(num_feats=num_feats).to(device)
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
        }
        torch.save(checkpoint, 'last_tabular.pt')
        if improved:
            torch.save(checkpoint, 'best_tabular.pt')
            print(f"Saved new best checkpoint with AUC={auc:.4f}")
        else:
            print(f"Checkpoint saved (AUC={auc:.4f})")


if __name__ == "__main__":
    main()
