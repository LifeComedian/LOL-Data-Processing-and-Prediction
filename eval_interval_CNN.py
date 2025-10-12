import argparse
import os
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, roc_auc_score

from train_tabular import TabularMinuteDataset, TabularClassifier
from train_CNN import LolMinuteDataset, FusionClassifier, collate_batch as fusion_collate


def load_checkpoint(path: Path, model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    epoch = checkpoint.get("epoch", "?")
    auc = checkpoint.get("best_auc", checkpoint.get("auc", float("nan")))
    print(f"[Checkpoint] loaded from {path} (epoch={epoch}, best_auc={auc})")
    return model


def minute_value(sample) -> Optional[int]:
    token = getattr(sample, "minute_str", "")
    token = token.lower().replace("min", "")
    try:
        return int(token)
    except ValueError:
        return None


def select_indices(
    samples: Iterable, predicate: Callable[[object], bool], max_matches: Optional[int]
) -> List[int]:
    allowed_matches: List[str] = []
    indices: List[int] = []
    for idx, sample in enumerate(samples):
        match_id = getattr(sample, "match_id", None)
        if match_id is None:
            continue
        if match_id not in allowed_matches:
            if max_matches is not None and len(allowed_matches) >= max_matches:
                continue
            allowed_matches.append(match_id)
        if predicate(sample):
            indices.append(idx)
    return indices


def collate_tabular(batch):
    feats, labels = zip(*batch)
    feats_tensor = torch.stack(feats, dim=0).float()
    labels_tensor = torch.stack(labels, dim=0).float()
    return feats_tensor, labels_tensor


@torch.no_grad()
def evaluate(model, loader: Optional[DataLoader], device: torch.device, mode: str) -> Tuple[float, float]:
    if loader is None:
        print("[Eval] loader is empty, skipping interval.")
        return float("nan"), float("nan")

    model.eval()
    logits_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []

    for batch in loader:
        if mode == "tabular":
            feats, labels = batch
            feats = feats.to(device)
            logits = model(feats)
        else:
            images, feats, labels = batch
            images = images.to(device)
            feats = feats.to(device)
            logits = model(images, feats)
        logits_list.append(logits.detach().cpu())
        labels_list.append(labels.detach().cpu())

    if not logits_list:
        return float("nan"), float("nan")

    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    probs = torch.sigmoid(logits).numpy()
    labels_np = labels.numpy()
    preds = (probs > 0.5).astype(np.int32)

    acc = accuracy_score(labels_np, preds)
    try:
        auc = roc_auc_score(labels_np, probs)
    except ValueError:
        auc = float("nan")
    return float(acc), float(auc)


def make_loader(
    dataset,
    indices: List[int],
    batch_size: int,
    num_workers: int,
    collate_fn,
) -> Optional[DataLoader]:
    if not indices:
        print("[Loader] subset empty, skipping loader.")
        return None
    subset = Subset(dataset, indices)
    print(f"[Loader] creating loader with {len(subset)} samples (batch_size={batch_size})")
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def build_datasets_and_model(args, device: torch.device):
    if args.mode == "tabular":
        print(f"[Dataset] loading tabular dataset from {args.minute_root}")
        dataset = TabularMinuteDataset(dataset_root=args.minute_root)
        model = TabularClassifier(num_feats=dataset.feat_dim).to(device)
        collate_fn = collate_tabular
    else:
        image_dir_name = args.image_root or "image_dataset"
        if os.path.isabs(image_dir_name):
            image_dir_name = os.path.relpath(image_dir_name, args.minute_root)
        print(f"[Dataset] loading fusion dataset from {args.minute_root} with images at {image_dir_name}")
        dataset = LolMinuteDataset(
            dataset_root=args.minute_root,
            image_dir_name=image_dir_name,
            num_players=args.num_images,
            image_size=args.image_size,
            augment=False,
        )
        model = FusionClassifier(
            num_images=args.num_images,
            num_feats=dataset.feat_dim,
        ).to(device)
        collate_fn = fusion_collate
    print(f"[Dataset] total samples: {len(dataset)} (feature dim={dataset.feat_dim})")
    return dataset, model, collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate time-split accuracy on minute dataset.")
    parser.add_argument("--minute-root", type=Path, required=True, help="Path to minute_dataset root.")
    parser.add_argument(
        "--image-root",
        type=str,
        default=None,
        help="Path (relative or absolute) to trajectory image dataset. Required for fusion mode.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint file to load.")
    parser.add_argument("--mode", choices=["tabular", "fusion"], default="tabular", help="Evaluation modality.")
    parser.add_argument("--num-matches", type=int, default=None, help="Limit evaluation to the first N matches.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--num-images", type=int, default=10, help="Number of player images per sample (fusion).")
    parser.add_argument("--image-size", type=int, default=224, help="Image resize dimension (fusion).")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "fusion" and args.image_root is None:
        raise ValueError("--image-root must be provided when mode='fusion'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset, model, collate_fn = build_datasets_and_model(args, device)
    model = load_checkpoint(args.checkpoint, model, device)

    def make_predicate(lower: Optional[int], upper: Optional[int]):
        def predicate(sample) -> bool:
            minute = minute_value(sample)
            if minute is None:
                return False
            if lower is not None and minute <= lower:
                return False
            if upper is not None and minute > upper:
                return False
            return True
        return predicate

    intervals = [
        ("1-5", 0, 5),
        ("5-10", 5, 10),
        ("10-15", 10, 15),
        ("15-20", 15, 20),
        ("20-25", 20, 25),
        ("25-30", 25, 30),
        ("30+", 30, None),
    ]

    interval_indices = {
        label: select_indices(dataset.samples, make_predicate(lower, upper), args.num_matches)
        for label, lower, upper in intervals
    }

    loaders = {
        label: make_loader(dataset, indices, args.batch_size, args.num_workers, collate_fn)
        for label, indices in interval_indices.items()
    }

    print(f"[Setup] device: {device}")
    print(f"[Setup] mode: {args.mode}")
    print(f"[Setup] matches evaluated: {args.num_matches if args.num_matches else 'all'}")
    for label, indices in interval_indices.items():
        print(f"[Split] {label} min samples: {len(indices)}")

    for label, loader in loaders.items():
        print(f"[Eval] evaluating {label} minute interval...")
        acc, auc = evaluate(model, loader, device, args.mode)
        print(f"[Eval] {label} min -> accuracy: {acc:.4f}, AUC: {auc:.4f}")


if __name__ == "__main__":
    main()

