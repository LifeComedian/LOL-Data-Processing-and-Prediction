from __future__ import annotations
import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
import glob
import re

class TabularMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, out_dim: int = 128, dropout: float = 0.1):
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

class TabularOnlyClassifier(nn.Module):
    def __init__(self, num_feats: int, tab_embed_dim: int = 128, fusion_hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.tab_mlp = TabularMLP(in_dim=num_feats, hidden=tab_embed_dim, out_dim=tab_embed_dim)
        self.fusion = nn.Sequential(
            nn.Linear(tab_embed_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden // 2, 1)  # output logit
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        feats: (B, F)
        returns: (B,) logits
        """
        tab_emb = self.tab_mlp(feats)  # (B, T)
        logit = self.fusion(tab_emb).squeeze(1)  # (B,)
        return logit

@dataclass
class Sample:
    feats: np.ndarray  # (F,)
    label: int

class TabularOnlyDataset(Dataset):
    def __init__(self, dataset_root: str, num_feats: int, augment: bool = True):
        self.dataset_root = dataset_root
        self.num_feats = num_feats
        self.samples = []

        for match_dir in sorted(glob.glob(os.path.join(dataset_root, "KR_*"))):
            match_id = os.path.basename(match_dir)
            for csv_path in sorted(glob.glob(os.path.join(match_dir, f"{match_id}_min*.csv"))):
                m = re.search(r"_min(\d+)", os.path.basename(csv_path))
                if not m:
                    continue
                minute_str = f"min{int(m.group(1)):02d}"
                self.samples.append({
                    "csv_path": csv_path,
                    "match_id": match_id,
                    "minute_str": minute_str,
                })

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found under {dataset_root}. ")

        self.feat_dim = self._infer_feat_dim(self.samples[0]["csv_path"])

    def __len__(self):
        return len(self.samples)

    def _read_table_to_features(self, csv_path: str) -> Tuple[np.ndarray, int]:
        df = pd.read_csv(csv_path)
        label = None
        for col in ["win", "label", "result"]:
            if col in df.columns:
                v = df[col].iloc[0]
                label = int(round(float(v)))
                df = df.drop(columns=[col], errors="ignore")

        df_num = df.select_dtypes(include=[np.number]).copy()
        if df_num.size == 0:
            raise ValueError(f"No numeric features parsed from {csv_path}.")

        feats = df_num.to_numpy(dtype=np.float32).reshape(-1)
        if label is None:
            label = 0
        return feats, label

    def _infer_feat_dim(self, csv_path: str) -> int:
        feats, _ = self._read_table_to_features(csv_path)
        return int(feats.shape[0])

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        feats_np, label = self._read_table_to_features(s["csv_path"])
        feats = torch.from_numpy(feats_np)  # (F,)
        label = torch.tensor(label, dtype=torch.float32)  # scalar
        return feats, label

def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    running_loss = 0.0
    for feats, labels in loader:
        feats = feats.to(device)  # (B, F)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
            logits = model(feats)  
            loss = loss_fn(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * feats.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for feats, labels in loader:
        feats = feats.to(device)
        logits = model(feats)
        all_logits.append(logits.cpu())
        all_labels.append(labels)

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    preds = (1 / (1 + np.exp(-logits)) > 0.5).astype(np.int32)
    acc = accuracy_score(labels, preds)

    try:
        auc = roc_auc_score(labels, logits)
    except ValueError:
        auc = float('nan')  
    return acc, auc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--no-augment', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    full_ds = TabularOnlyDataset(
        dataset_root=args.dataset_root,
        num_feats=10,  
        augment=not args.no_augment,
    )
    num_feats = full_ds.feat_dim
    n_total = len(full_ds)
    n_train = int(n_total * 0.9)
    idxs = np.random.permutation(n_total)
    train_idx, val_idx = idxs[:n_train], idxs[n_train:]

    train_ds = torch.utils.data.Subset(full_ds, train_idx)
    val_ds = torch.utils.data.Subset(full_ds, val_idx)

    def collate(batch):
        feats, labels = zip(*batch)
        return torch.stack(feats, dim=0), torch.stack(labels, dim=0)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = TabularOnlyClassifier(num_feats=num_feats).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    best_auc = -1.0
    results = []  

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        acc, auc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:02d}: loss={train_loss:.4f}, val_acc={acc:.4f}, val_auc={auc:.4f}")

        results.append({'epoch': epoch, 'val_acc': acc, 'val_auc': auc, 'train_loss': train_loss})

        if auc > best_auc:
            best_auc = auc
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'auc': auc}, 'best.pt')
            print(f"Saved new best checkpoint with AUC={auc:.4f}")

    df = pd.DataFrame(results)
    df.to_csv('training_results.csv', index=False)
    print("Saved training results to 'training_results.csv'")


if __name__ == '__main__':
    main()
