from __future__ import annotations
import argparse
import os
import glob
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score

# ------------------------------
# Model components
# ------------------------------
class ImageEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, embed_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(256, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).flatten(1)
        return F.relu(self.proj(h))

class TemporalTransformer(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 256, nhead: int = 4, nlayers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Linear(in_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
                                           dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x_btd: torch.Tensor) -> torch.Tensor:

        x = self.inp(x_btd)
        x = self.enc(x)
        x = self.norm(x)
        return x.mean(dim=1)

class CrossPlayerAttention(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 4, nlayers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(nlayers):
            self.blocks.append(nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True))
            self.blocks.append(nn.LayerNorm(d_model))
            self.blocks.append(nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model*4, d_model)))
            self.blocks.append(nn.LayerNorm(d_model))
    def forward(self, e_bpd: torch.Tensor) -> torch.Tensor:
        x = e_bpd
        for i in range(0, len(self.blocks), 4):
            attn, ln1, ff, ln2 = self.blocks[i:i+4]
            x2, _ = attn(x, x, x, need_weights=False)
            x = ln1(x + x2)
            x2 = ff(x)
            x = ln2(x + x2)
        return x.mean(dim=1)

class SequenceFusionModel(nn.Module):
    """Images → per-player temporal → cross-player; Tabular per-minute → temporal → fuse."""
    def __init__(self, num_players: int = 10, img_embed_dim: int = 256, tab_in_dim: int = 0,
                 d_model: int = 256, nhead_t: int = 4, nlayers_t: int = 2,
                 nhead_p: int = 4, nlayers_p: int = 1, dropout: float = 0.1, in_channels: int = 3):
        super().__init__()
        self.num_players = num_players
        self.img_encoder = ImageEncoder(in_channels=in_channels, embed_dim=img_embed_dim)

        self.temporal_player = TemporalTransformer(in_dim=img_embed_dim, d_model=d_model, nhead=nhead_t, nlayers=nlayers_t, dropout=dropout)

        self.cross_players = CrossPlayerAttention(d_model=d_model, nhead=nhead_p, nlayers=nlayers_p, dropout=dropout)

        self.use_tab = tab_in_dim > 0
        if self.use_tab:
            self.tab_temporal = TemporalTransformer(in_dim=tab_in_dim, d_model=d_model, nhead=4, nlayers=2, dropout=dropout)
            fusion_in = d_model * 2
        else:
            fusion_in = d_model
        self.cls = nn.Sequential(
            nn.Linear(fusion_in, d_model), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(d_model, 1)
        )
    def forward(self, imgs_btkchw: torch.Tensor, tabs_btF: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        imgs: [B, T, K, C, H, W]
        tabs: [B, T, F] (flattened numeric features per minute), optional
        """
        B, T, K, C, H, W = imgs_btkchw.shape

        x = imgs_btkchw.reshape(B*T*K, C, H, W)
        z = self.img_encoder(x)
        z = z.view(B, T, K, -1)

        z_bktd = z.permute(0,2,1,3).reshape(B*K, T, -1)
        e_bkD = self.temporal_player(z_bktd) 
        e_bKD = e_bkD.view(B, K, -1)            
        g_img = self.cross_players(e_bKD)       
        if self.use_tab and tabs_btF is not None
            h = torch.cat([g_img, s_tab], dim=1)
        else:
            h = g_img
        return self.cls(h).squeeze(1)

# ------------------------------
# Dataset (multi image roots support)
# ------------------------------
@dataclass
class MatchItem:
    match_id: str
    minute_dirs: List[str]   
    csv_paths: List[str]     
    label: int

class LolSequenceDataset(Dataset):
    def __init__(self, dataset_root: str, minutes: int = 20, image_dirs: List[str] = None,
                 num_players: int = 10, image_size: int = 224, require_full: bool = True):
        self.root = dataset_root
        self.minutes = minutes
        self.img_roots = image_dirs or []  
        self.num_players = num_players
        self.require_full = require_full
        tfs = [transforms.Resize((image_size, image_size)), transforms.ToTensor(),
               transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])]
        self.tf = transforms.Compose(tfs)

        self.items: List[MatchItem] = []
        for match_dir in sorted(glob.glob(os.path.join(dataset_root, 'KR_*'))):
            mid = os.path.basename(match_dir)
            if not os.path.isdir(match_dir):
                continue
            csvs = []
            mins = []
            full_ok = True
            for t in range(1, minutes+1):
                minute_str = f"min{t:02d}"
                csv_path = os.path.join(match_dir, f"{mid}_{minute_str}.csv")
                if not os.path.isfile(csv_path):
                    full_ok = False

                for i in range(1, num_players+1):
                    rel = os.path.join(mid, minute_str, f"{mid}_{minute_str}_{i}.png")
                    found = any(os.path.isfile(os.path.join(r, rel)) for r in self.img_roots)
                    if not found:
                        full_ok = False
                        break
                csvs.append(csv_path)
                mins.append(minute_str)
            if full_ok or (not require_full and len(csvs)>0):
                label = self._infer_label_from_any(csvs)
                self.items.append(MatchItem(match_id=mid, minute_dirs=mins, csv_paths=csvs, label=label))
        if len(self.items) == 0:
            raise RuntimeError('No complete matches found. Provide --image-dirs pointing to your subset roots, or use --allow-incomplete.')

        self.tab_dim = self._infer_tab_dim(self.items[0].csv_paths[0])

    def _infer_tab_dim(self, csv_path: str) -> int:
        f, _ = self._read_table_to_features(csv_path)
        return int(f.shape[0])

    def _infer_label_from_any(self, csv_paths: List[str]) -> int:
        for p in csv_paths:
            df = pd.read_csv(p)
            for col in ['win','label','result']:
                if col in df.columns:
                    try:
                        return int(round(float(df[col].iloc[0])))
                    except Exception:
                        pass
        return 0

    def _read_table_to_features(self, csv_path: str) -> Tuple[np.ndarray, int]:
        df = pd.read_csv(csv_path)
        label = None
        for col in ['win','label','result']:
            if col in df.columns:
                try:
                    label = int(round(float(df[col].iloc[0])))
                except Exception:
                    pass
                df = df.drop(columns=[col], errors='ignore')
        if df.shape[1] >= 1:
            df = df.iloc[:,1:]
        if df.shape[1] >= 1:
            df = df.iloc[:,:-1]
        df_num = df.select_dtypes(include=[np.number]).copy()
        feats = df_num.to_numpy(dtype=np.float32).reshape(-1)
        if feats.size == 0:
            raise ValueError(f'No numeric features in {csv_path}')
        if label is None:
            label = 0
        return feats, label

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        T = self.minutes
        K = self.num_players
        imgs = []
        tabs = []
        for minute_str, csv_path in zip(it.minute_dirs, it.csv_paths):
            
            minute_imgs = []
            for i in range(1, K+1):
                rel = os.path.join(it.match_id, minute_str, f"{it.match_id}_{minute_str}_{i}.png")
                p = None
                for root in self.img_roots:
                    cand = os.path.join(root, rel)
                    if os.path.isfile(cand):
                        p = cand
                        break
                if p is None:
                    raise FileNotFoundError(f"Image not found in any subset for {rel}")
                img = Image.open(p).convert('RGB')
                minute_imgs.append(self.tf(img))
            imgs.append(torch.stack(minute_imgs, dim=0))

            feats, _ = self._read_table_to_features(csv_path)
            tabs.append(torch.from_numpy(feats))  
        imgs = torch.stack(imgs, dim=0)  
        tabs = torch.stack(tabs, dim=0)  
        y = torch.tensor(float(it.label), dtype=torch.float32)
        return imgs, tabs, y

# ------------------------------
# Train/Eval
# ------------------------------

def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    tot = 0.0
    n = 0
    for imgs, tabs, y in loader:
        imgs = imgs.to(device)
        tabs = tabs.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
            logit = model(imgs, tabs)
            loss = loss_fn(logit, y)
        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        bs = imgs.size(0)
        tot += loss.item() * bs
        n += bs
    return tot / max(n, 1)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    logits_all, labels_all = [], []
    for imgs, tabs, y in loader:
        imgs = imgs.to(device)
        tabs = tabs.to(device)
        logit = model(imgs, tabs)
        logits_all.append(logit.cpu())
        labels_all.append(y)
    logits = torch.cat(logits_all).numpy()
    labels = torch.cat(labels_all).numpy()
    prob = 1/(1+np.exp(-logits))
    pred = (prob>0.5).astype(np.int32)
    acc = accuracy_score(labels, pred)
    try:
        auc = roc_auc_score(labels, logits)
        ap = average_precision_score(labels, prob)
    except Exception:
        auc, ap = float('nan'), float('nan')
    return acc, auc, ap

# ------------------------------
# CLI
# ------------------------------

def parse_image_dirs_arg(s: str, dataset_root: str) -> List[str]:
    dirs = [p.strip() for p in s.split(',') if p.strip()]
    resolved = []
    for d in dirs:
        if os.path.isabs(d):
            resolved.append(os.path.normpath(d))
        else:
            resolved.append(os.path.normpath(os.path.join(dataset_root, d)))
    return resolved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-root', type=str, required=True, help='Numeric CSV root (contains KR_xxx dirs).')
    ap.add_argument('--image-dirs', type=str, default='image_dataset', help='Comma-separated image roots (absolute or relative to dataset-root).')
    ap.add_argument('--minutes', type=int, default=20)
    ap.add_argument('--num-players', type=int, default=10)
    ap.add_argument('--image-size', type=int, default=224)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--allow-incomplete', action='store_true', help='Not all minutes/images must exist.')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_roots = parse_image_dirs_arg(args.image_dirs, args.dataset_root)

    full_ds = LolSequenceDataset(dataset_root=args.dataset_root, minutes=args.minutes,
                                 image_dirs=img_roots, num_players=args.num_players,
                                 image_size=args.image_size, require_full=not args.allow_incomplete)


    n = len(full_ds)
    idx = np.random.permutation(n)
    n_train = max(1, int(n*0.9))
    train_idx, val_idx = idx[:n_train], idx[n_train:]
    train_ds = torch.utils.data.Subset(full_ds, train_idx)
    val_ds   = torch.utils.data.Subset(full_ds, val_idx)

    def collate(batch):
        imgs, tabs, y = zip(*batch)
        return (torch.stack(imgs,0), torch.stack(tabs,0), torch.stack(y,0))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate)

    tab_dim = full_ds.tab_dim
    model = SequenceFusionModel(num_players=args.num_players, img_embed_dim=256, tab_in_dim=tab_dim,
                                d_model=256, nhead_t=4, nlayers_t=2, nhead_p=4, nlayers_p=1, dropout=0.1).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=='cuda'))

    best_auc = -1.0
    for ep in range(1, args.epochs+1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        acc, auc, ap = evaluate(model, val_loader, device)
        print(f"Epoch {ep:02d} | loss={tr_loss:.4f} | val_acc={acc:.4f} | val_auc={auc:.4f} | val_ap={ap:.4f}")
        if auc > best_auc:
            best_auc = auc
            torch.save({'model': model.state_dict(), 'epoch': ep, 'auc': auc,
                        'cfg': {'minutes': args.minutes, 'num_players': args.num_players, 'image_size': args.image_size, 'tab_dim': tab_dim}},
                       'best_seq.pt')
            print(f"  ↳ Saved best_seq.pt with AUC={auc:.4f}")

if __name__ == '__main__':
    main()
