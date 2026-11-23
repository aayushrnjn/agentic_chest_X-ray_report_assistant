"""Test-set evaluation of the CNN: AUROC per finding + calibration (ECE, reliability plot).

python -m eval.evaluate_model
Writes artifacts/eval/model_metrics.json, reliability.png, auroc.png and test_preds.npz
(the last one is reused by the triage model).
"""
import json

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from config import CALIBRATION, CHECKPOINT, EVAL_DIR, FINDINGS, SPLITS_DIR, TEST_PREDS  # noqa: E402
from models.calibrate import expected_calibration_error, sigmoid  # noqa: E402
from models.cnn import ChestXrayNet, eval_transform, get_device  # noqa: E402
from models.dataset import XrayDataset  # noqa: E402
from models.train import per_class_auroc, predict_logits  # noqa: E402


def reliability_plot(raw: np.ndarray, cal: np.ndarray, y: np.ndarray, path) -> None:
    bins = np.linspace(0, 1, 11)
    fig, ax = plt.subplots(figsize=(5, 5))
    for probs, label in [(raw, "before calibration"), (cal, "after calibration")]:
        p, t = probs.ravel(), y.ravel()
        idx = np.digitize(p, bins[1:-1])
        xs = [p[idx == b].mean() for b in range(10) if (idx == b).any()]
        ys = [t[idx == b].mean() for b in range(10) if (idx == b).any()]
        ax.plot(xs, ys, marker="o", label=label)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title("Reliability diagram (all findings)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    device = get_device()
    model = ChestXrayNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device)["state_dict"])
    dl = DataLoader(XrayDataset(SPLITS_DIR / "test.csv", eval_transform()), batch_size=64, num_workers=4)
    logits, y = predict_logits(model, dl, device)

    cal = json.loads(CALIBRATION.read_text())
    raw = sigmoid(logits)
    calp = sigmoid(logits / np.array(cal["temperature"]) + np.array(cal["bias"]))
    th = np.array(cal["threshold"])
    pred = calp >= th

    auroc = per_class_auroc(y, calp)
    sens = {f: float(pred[y[:, j] == 1, j].mean()) for j, f in enumerate(FINDINGS) if (y[:, j] == 1).any()}
    spec = {f: float(1 - pred[y[:, j] == 0, j].mean()) for j, f in enumerate(FINDINGS)}
    metrics = {
        "n_test_images": int(len(y)),
        "mean_auroc": round(float(np.mean(list(auroc.values()))), 4),
        "auroc": {k: round(v, 4) for k, v in auroc.items()},
        "sensitivity_at_threshold": {k: round(v, 3) for k, v in sens.items()},
        "specificity_at_threshold": {k: round(v, 3) for k, v in spec.items()},
        "ece_before": round(expected_calibration_error(raw, y), 4),
        "ece_after": round(expected_calibration_error(calp, y), 4),
    }
    (EVAL_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(TEST_PREDS, cal_probs=calp.astype(np.float32), labels=y.astype(np.int8))
    reliability_plot(raw, calp, y, EVAL_DIR / "reliability.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    items = sorted(auroc.items(), key=lambda kv: kv[1])
    ax.barh([k for k, _ in items], [v for _, v in items])
    ax.axvline(0.5, ls="--", color="gray")
    ax.set_xlim(0.4, 1.0)
    ax.set_title(f"Test AUROC per finding (mean {metrics['mean_auroc']})")
    fig.tight_layout()
    fig.savefig(EVAL_DIR / "auroc.png", dpi=130)

    print(json.dumps({k: metrics[k] for k in ["n_test_images", "mean_auroc", "ece_before", "ece_after"]}, indent=2))
    for k, v in items:
        print(f"  {k:20s} AUROC {v:.3f}  sens {sens.get(k, float('nan')):.2f}  spec {spec[k]:.2f}")


if __name__ == "__main__":
    main()
