"""Probability calibration + per-finding decision thresholds.

python -m models.calibrate

Why: a raw network that outputs 0.9 is often right far less than 90% of the
time. We fit, per finding, calibrated_logit = logit / T + b on the VALIDATION
set by minimising log-loss (temperature scaling with a bias term, i.e. Platt
scaling). The bias is needed because training with pos_weight deliberately
pushes logits upward for rare findings; plain temperature scaling cannot
undo that shift. Both are monotonic, so ranking and AUROC are unchanged.

Thresholds: with ~1% prevalence a 0.5 cut-off almost never fires, so each
finding gets its own threshold chosen on validation (Youden's J = TPR - FPR).
"""
import json

import numpy as np
import torch
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

from config import CALIBRATION, CHECKPOINT, FINDINGS, SPLITS_DIR
from models.cnn import ChestXrayNet, eval_transform, get_device
from models.dataset import XrayDataset
from models.train import predict_logits


def fit_calibration(logits: np.ndarray, labels: np.ndarray, steps: int = 200):
    """Returns (T, b) arrays, one value per finding."""
    lg = torch.tensor(logits, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    log_t = torch.zeros(lg.shape[1], requires_grad=True)          # optimise log(T) so T stays positive
    bias = torch.zeros(lg.shape[1], requires_grad=True)
    opt = torch.optim.LBFGS([log_t, bias], lr=0.1, max_iter=steps)
    bce = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = bce(lg / log_t.exp() + bias, y)
        loss.backward()
        return loss

    opt.step(closure)
    return log_t.exp().detach().numpy(), bias.detach().numpy()


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    """Average gap between predicted probability and observed frequency, over all findings."""
    p, y = probs.ravel(), labels.ravel()
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def youden_thresholds(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    th = np.full(probs.shape[1], 0.5)
    for j in range(probs.shape[1]):
        if len(np.unique(labels[:, j])) == 2:
            fpr, tpr, cut = roc_curve(labels[:, j], probs[:, j])
            th[j] = float(np.clip(cut[np.argmax(tpr - fpr)], 0.01, 0.99))
    return th


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    device = get_device()
    model = ChestXrayNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device)["state_dict"])
    val_dl = DataLoader(XrayDataset(SPLITS_DIR / "val.csv", eval_transform()), batch_size=64, num_workers=4)
    logits, labels = predict_logits(model, val_dl, device)

    temps, biases = fit_calibration(logits, labels)
    before = expected_calibration_error(sigmoid(logits), labels)
    cal_probs = sigmoid(logits / temps + biases)
    after = expected_calibration_error(cal_probs, labels)
    thresholds = youden_thresholds(cal_probs, labels)

    CALIBRATION.write_text(json.dumps({
        "findings": FINDINGS,
        "temperature": temps.round(4).tolist(),
        "bias": biases.round(4).tolist(),
        "threshold": thresholds.round(4).tolist(),
        "val_ece_before": round(before, 4),
        "val_ece_after": round(after, 4),
    }, indent=2))
    print(f"ECE before {before:.4f} -> after {after:.4f}")
    for f, t, b, th in zip(FINDINGS, temps, biases, thresholds):
        print(f"  {f:20s} T={t:.2f}  b={b:+.2f}  threshold={th:.3f}")


if __name__ == "__main__":
    main()
