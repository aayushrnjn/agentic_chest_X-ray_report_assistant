"""Train the multi-label CNN.

python -m models.train --epochs 8 --batch-size 32
Keeps the checkpoint with the best mean validation AUROC.
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CHECKPOINT, FINDINGS, SPLITS_DIR
from models.cnn import ChestXrayNet, eval_transform, get_device, train_transform
from models.dataset import XrayDataset


def per_class_auroc(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """AUROC per finding. Skips a class if the split has only one label value for it."""
    out = {}
    for j, f in enumerate(FINDINGS):
        if len(np.unique(y_true[:, j])) == 2:
            out[f] = float(roc_auc_score(y_true[:, j], y_score[:, j]))
    return out


@torch.no_grad()
def predict_logits(model, loader, device):
    model.eval()
    logits, labels = [], []
    for x, y in tqdm(loader, desc="eval", leave=False):
        logits.append(model(x.to(device)).float().cpu())
        labels.append(y)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = get_device()
    train_ds = XrayDataset(SPLITS_DIR / "train.csv", train_transform())
    val_ds = XrayDataset(SPLITS_DIR / "val.csv", eval_transform())
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_dl = DataLoader(val_ds, args.batch_size, num_workers=args.workers)

    # Class imbalance: Hernia is ~0.2% positive, Infiltration ~18%.
    # pos_weight = negatives/positives makes each rare positive count more in the loss.
    pos = train_ds.labels.sum(0)
    neg = len(train_ds) - pos
    pos_weight = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1, 50), dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = ChestXrayNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_dl))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for x, y in tqdm(train_dl, desc=f"epoch {epoch}"):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item() * len(x)

        logits, labels = predict_logits(model, val_dl, device)
        aurocs = per_class_auroc(labels, 1 / (1 + np.exp(-logits)))
        mean_auc = float(np.mean(list(aurocs.values())))
        print(f"epoch {epoch}: train loss {running / len(train_ds):.4f}  val mean AUROC {mean_auc:.4f}  ({time.time() - t0:.0f}s)")

        if mean_auc > best:
            best = mean_auc
            torch.save({"state_dict": model.state_dict(), "findings": FINDINGS,
                        "val_auroc": aurocs, "epoch": epoch}, CHECKPOINT)
            print(f"  saved new best -> {CHECKPOINT}")

    print(f"done. best val mean AUROC = {best:.4f}")


if __name__ == "__main__":
    main()
