from __future__ import annotations
import argparse
import glob
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import torch.nn.functional as F

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
        B, T, K, C, H, W = imgs_btkchw.shape
        x = imgs_btkchw.reshape(B*T*K, C, H, W)
        z = self.img_encoder(x)
        z = z.view(B, T, K, -1)
        z_bktd = z.permute(0,2,1,3).reshape(B*K, T, -1)
        e_bkD = self.temporal_player(z_bktd)
        e_bKD = e_bkD.view(B, K, -1)
        g_img = self.cross_players(e_bKD)
        if self.use_tab and tabs_btF is not None:
            s_tab = self.tab_temporal(tabs_btF)
            h = torch.cat([g_img, s_tab], dim=1)
        else:
            h = g_img
        return self.cls(h).squeeze(1)

@dataclass
class MatchItem:
    match_id: str
    minutes: List[str]
    csvs: List[str]
    label: float

class LolEvalDataset(Dataset):
    def __init__(self, dataset_root: str, image_dirs: List[str],
                 minute_start: int, minute_end: Optional[int],  # inclusive; None=end
                 num_players: int = 10, image_size: int = 224,
                 require_full: bool = True, limit_matches: Optional[int] = None):
        self.root = dataset_root
        self.img_roots = image_dirs
        self.ms = minute_start
        self.me = minute_end
        self.K = num_players
        self.require_full = require_full
        tfs = [transforms.Resize((image_size, image_size)), transforms.ToTensor(),
               transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])]
        self.tf = transforms.Compose(tfs)
        self.items: List[MatchItem] = []

        for match_dir in sorted(glob.glob(os.path.join(dataset_root, 'KR_*'))):
            mid = os.path.basename(match_dir)
            if not os.path.isdir(match_dir):
                continue

            csv_list = sorted(glob.glob(os.path.join(match_dir, f'{mid}_min*.csv')))
            if not csv_list:
                continue
            max_min = 0
            for p in csv_list:
                base = os.path.basename(p)

                try:
                    m = int(base.split('_min')[-1].split('.')[0])
                    max_min = max(max_min, m)
                except Exception:
                    pass
            s = self.ms
            e = self.me if self.me is not None else max_min
            if s > e:
                continue
            minutes = [f'min{t:02d}' for t in range(s, e+1)]
            csvs = [os.path.join(match_dir, f'{mid}_{m}.csv') for m in minutes]

            ok = True
            for m,csvp in zip(minutes, csvs):
                if not os.path.isfile(csvp):
                    ok = False
                    if self.require_full:
                        break
            if not ok and self.require_full:
                continue
            if ok:
                for m in minutes:
                    for i in range(1, self.K+1):
                        rel = os.path.join(mid, m, f'{mid}_{m}_{i}.png')
                        if not any(os.path.isfile(os.path.join(r, rel)) for r in self.img_roots):
                            ok = False
                            break
                    if not ok and self.require_full:
                        break
            if not ok and self.require_full:
                continue

            label = self._infer_label_from_any(csvs)
            if not self.require_full:
                new_minutes, new_csvs = [], []
                for m,csvp in zip(minutes, csvs):
                    if os.path.isfile(csvp):
                        new_minutes.append(m)
                        new_csvs.append(csvp)
                minutes, csvs = new_minutes, new_csvs
                if len(minutes) == 0:
                    continue
            self.items.append(MatchItem(match_id=mid, minutes=minutes, csvs=csvs, label=label))
            if limit_matches is not None and len(self.items) >= limit_matches:
                break
        if len(self.items) == 0:
            raise RuntimeError('No matches found for this phase. Check paths or relax --allow-incomplete.')

        self.tab_dim = self._infer_tab_dim(self.items[0].csvs[0])

    def _infer_label_from_any(self, csv_paths: List[str]) -> float:
        for p in csv_paths:
            try:
                df = pd.read_csv(p)
            except Exception:
                continue
            for col in ['win','label','result']:
                if col in df.columns:
                    try:
                        return float(df[col].iloc[0])
                    except Exception:
                        pass
        return 0.0

    def _infer_tab_dim(self, csv_path: str) -> int:
        feats = self._read_table_to_features(csv_path)
        feats = np.asarray(feats, dtype=np.float32).reshape(-1)
        return int(feats.shape[0])

    def _read_table_to_features(self, csv_path: str):
        df = pd.read_csv(csv_path)

        for col in ['win','label','result']:
            if col in df.columns:
                df = df.drop(columns=[col], errors='ignore')
        if df.shape[1] >= 1:
            df = df.iloc[:,1:]
        if df.shape[1] >= 1:
            df = df.iloc[:,:-1]
        df_num = df.select_dtypes(include=[np.number]).copy()
        feats = df_num.to_numpy(dtype=np.float32).reshape(-1)
        return feats

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        K = 10
        imgs_per_min = []
        tabs_per_min = []
        for m,csvp in zip(it.minutes, it.csvs):

            minute_imgs = []
            for i in range(1, K+1):
                rel = os.path.join(it.match_id, m, f'{it.match_id}_{m}_{i}.png')
                p = None
                for root in self.img_roots:
                    cand = os.path.join(root, rel)
                    if os.path.isfile(cand):
                        p = cand
                        break
                if p is None:
                    raise FileNotFoundError(f'Image not found: {rel}')
                img = Image.open(p).convert('RGB')
                minute_imgs.append(self.tf(img))
            imgs_per_min.append(torch.stack(minute_imgs, dim=0))

            feats = self._read_table_to_features(csvp)
            tabs_per_min.append(torch.from_numpy(feats))
        imgs = torch.stack(imgs_per_min, dim=0)
        tabs = torch.stack(tabs_per_min, dim=0)
        y = torch.tensor(float(it.label), dtype=torch.float32)
        return imgs, tabs, y

@torch.no_grad()
def eval_loader(model, loader, device):
    from sklearn.metrics import roc_auc_score, average_precision_score
    model.eval()
    correct = 0
    total = 0
    all_logits = []
    all_labels = []
    for imgs, tabs, y in loader:
        imgs = imgs.to(device)
        tabs = tabs.to(device)
        y = y.to(device)
        logits = model(imgs, tabs)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == y).sum().item()
        total += y.numel()
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())
    acc = correct / max(total,1)
    if len(all_logits) == 0:
        return acc, float('nan'), float('nan'), int(correct), int(total)
    logits = torch.cat(all_logits).numpy().ravel()
    labels = torch.cat(all_labels).numpy().ravel()
    prob = 1/(1+np.exp(-logits))
    try:
        auc = roc_auc_score(labels, logits)
    except Exception:
        auc = float('nan')
    try:
        ap = average_precision_score(labels, prob)
    except Exception:
        ap = float('nan')
    return acc, auc, ap, int(correct), int(total)

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_image_dirs_arg(s: str, dataset_root: str) -> List[str]:
    if not s:
        return []
    parts = []
    for item in s.split(','):
        item = item.strip()
        if not item:
            continue
        if os.path.isabs(item):
            parts.append(os.path.normpath(item))
        else:
            parts.append(os.path.normpath(os.path.join(dataset_root, item)))
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-root', type=str, required=True)
    ap.add_argument('--image-dirs', type=str, required=True, help='Comma-separated image roots.')
    ap.add_argument('--checkpoint', type=str, required=True, help='Path to best_seq.pt')
    ap.add_argument('--num-players', type=int, default=10)
    ap.add_argument('--image-size', type=int, default=224)
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--limit', type=int, default=1000, help='Max matches to evaluate per phase')
    ap.add_argument('--allow-incomplete', action='store_true')
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--pad-mode', type=str, default='truncate', choices=['truncate','pad'],
                    help='How to batch variable-length sequences across matches: truncate to min T or pad to max T with zeros.')
    args = ap.parse_args()

    device = torch.device(args.device)


    ckpt = torch.load(args.checkpoint, map_location='cpu')

    cfg = ckpt.get('cfg', {}) if isinstance(ckpt, dict) else {}
    num_players = cfg.get('num_players', args.num_players)

    img_roots = parse_image_dirs_arg(args.image_dirs, args.dataset_root)

    phases = {
        'early_1_15': (1, 15),
        'mid_16_30': (16, 30),
        'late_31_end': (31, None),
    }

    results = {}

    for name,(s,e) in phases.items():
        ds = LolEvalDataset(
            dataset_root=args.dataset_root,
            image_dirs=img_roots,
            minute_start=s,
            minute_end=e,
            num_players=num_players,
            image_size=args.image_size,
            require_full=not args.allow_incomplete,
            limit_matches=args.limit,
        )
        
        def collate_varlen(batch):
            imgs_list, tabs_list, y_list = zip(*batch)
            Ts = [x.shape[0] for x in imgs_list]
            if args.pad_mode == 'truncate':
                T = min(Ts)
                imgs = torch.stack([x[:T] for x in imgs_list], 0)
                tabs = torch.stack([t[:T] for t in tabs_list], 0)
            elif args.pad_mode == 'pad':
                T = max(Ts)
                K, C, H, W = imgs_list[0].shape[1:]
                F = tabs_list[0].shape[1]
                imgs = torch.zeros((len(batch), T, K, C, H, W), dtype=imgs_list[0].dtype)
                tabs = torch.zeros((len(batch), T, F), dtype=tabs_list[0].dtype)
                for i,(im, tb) in enumerate(zip(imgs_list, tabs_list)):
                    t = im.shape[0]
                    imgs[i, :t] = im
                    tabs[i, :t] = tb
            else:
                raise ValueError('pad_mode must be one of {truncate, pad}')
            y = torch.stack(y_list, 0)
            return imgs, tabs, y
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_varlen)
        
        tab_dim = ds.tab_dim
        model = SequenceFusionModel(num_players=num_players, img_embed_dim=256, tab_in_dim=tab_dim,
                                    d_model=256, nhead_t=4, nlayers_t=2, nhead_p=4, nlayers_p=1, dropout=0.1).to(device)
        missing, unexpected = model.load_state_dict(ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt, strict=False)
        if missing or unexpected:
            print(f"[warn] state_dict mismatch for {name}: missing={len(missing)}, unexpected={len(unexpected)}")
        acc, auc, ap, correct, total = eval_loader(model, dl, device)
        results[name] = (acc, auc, ap, correct, total, len(ds))
        print(f"{name}: acc={acc:.4f} auc={auc:.4f} ap={ap:.4f}  ({correct}/{total}) | matches={len(ds)}")

    print("\nSummary")
    for k,(acc,auc,ap,correct,total,nm) in results.items():
        print(f"  {k:12s}  acc={acc:.4f}  auc={auc:.4f}  ap={ap:.4f}  ({correct}/{total})  matches={nm}")

if __name__ == '__main__':
    main()
