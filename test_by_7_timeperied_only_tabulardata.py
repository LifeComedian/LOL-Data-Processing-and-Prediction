import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, roc_auc_score
from newtrainwithoutpic import TabularOnlyDataset, TabularOnlyClassifier  

def load_checkpoint(checkpoint_path: str, model: torch.nn.Module):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    print(f"Checkpoint loaded from {checkpoint_path} (Epoch {checkpoint['epoch']}, AUC {checkpoint['auc']:.4f})")
    return model

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

def load_data_for_time_range(full_ds, start_min: int, end_min: int):
    idx = [i for i, s in enumerate(full_ds.samples) if int(s['minute_str'][3:5]) >= start_min and int(s['minute_str'][3:5]) < end_min]
    return Subset(full_ds, idx)

def main():
    dataset_root = '/work/zh180/test/output_root'  
    checkpoint_path = 'best.pt'  

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    full_ds = TabularOnlyDataset(
        dataset_root=dataset_root,
        num_feats=10,  
        augment=False  
    )

    min_1_5_ds = load_data_for_time_range(full_ds, 1, 5)
    min_5_10_ds = load_data_for_time_range(full_ds, 5, 10)
    min_10_15_ds = load_data_for_time_range(full_ds, 10, 15)
    min_15_20_ds = load_data_for_time_range(full_ds, 15, 20)
    min_20_25_ds = load_data_for_time_range(full_ds, 20, 25)
    min_25_30_ds = load_data_for_time_range(full_ds, 25, 30)
    min_30_plus_ds = load_data_for_time_range(full_ds, 30, 61) 

    def collate(batch):
        feats, labels = zip(*batch)
        return torch.stack(feats, dim=0), torch.stack(labels, dim=0)

    min_1_5_loader = DataLoader(min_1_5_ds, batch_size=256, shuffle=False, collate_fn=collate)
    min_5_10_loader = DataLoader(min_5_10_ds, batch_size=256, shuffle=False, collate_fn=collate)
    min_10_15_loader = DataLoader(min_10_15_ds, batch_size=256, shuffle=False, collate_fn=collate)
    min_15_20_loader = DataLoader(min_15_20_ds, batch_size=256, shuffle=False, collate_fn=collate)
    min_20_25_loader = DataLoader(min_20_25_ds, batch_size=256, shuffle=False, collate_fn=collate)
    min_25_30_loader = DataLoader(min_25_30_ds, batch_size=256, shuffle=False, collate_fn=collate)
    min_30_plus_loader = DataLoader(min_30_plus_ds, batch_size=256, shuffle=False, collate_fn=collate)

    model = TabularOnlyClassifier(num_feats=full_ds.feat_dim).to(device)

    model = load_checkpoint(checkpoint_path, model)

    print("Evaluating for 1-5 min:")
    acc_1_5, auc_1_5 = evaluate(model, min_1_5_loader, device)
    print(f"1-5 min - Accuracy: {acc_1_5:.4f}, AUC: {auc_1_5:.4f}")

    print("Evaluating for 5-10 min:")
    acc_5_10, auc_5_10 = evaluate(model, min_5_10_loader, device)
    print(f"5-10 min - Accuracy: {acc_5_10:.4f}, AUC: {auc_5_10:.4f}")

    print("Evaluating for 10-15 min:")
    acc_10_15, auc_10_15 = evaluate(model, min_10_15_loader, device)
    print(f"10-15 min - Accuracy: {acc_10_15:.4f}, AUC: {auc_10_15:.4f}")

    print("Evaluating for 15-20 min:")
    acc_15_20, auc_15_20 = evaluate(model, min_15_20_loader, device)
    print(f"15-20 min - Accuracy: {acc_15_20:.4f}, AUC: {auc_15_20:.4f}")

    print("Evaluating for 20-25 min:")
    acc_20_25, auc_20_25 = evaluate(model, min_20_25_loader, device)
    print(f"20-25 min - Accuracy: {acc_20_25:.4f}, AUC: {auc_20_25:.4f}")

    print("Evaluating for 25-30 min:")
    acc_25_30, auc_25_30 = evaluate(model, min_25_30_loader, device)
    print(f"25-30 min - Accuracy: {acc_25_30:.4f}, AUC: {auc_25_30:.4f}")

    print("Evaluating for 30+ min:")
    acc_30_plus, auc_30_plus = evaluate(model, min_30_plus_loader, device)
    print(f"30+ min - Accuracy: {acc_30_plus:.4f}, AUC: {auc_30_plus:.4f}")

if __name__ == '__main__':
    main()
