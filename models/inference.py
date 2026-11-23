"""Everything the running system needs from the trained model:
calibrated predictions, image embeddings and Grad-CAM heatmaps.

The imaging MCP server is a thin wrapper around this class.
"""
import hashlib
import json
from functools import lru_cache

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from config import CALIBRATION, CHECKPOINT, FINDINGS, HEATMAP_DIR, UNCERTAIN_MARGIN
from models.cnn import ChestXrayNet, eval_transform, get_device
from models.preprocess import load_xray


class XrayModel:
    def __init__(self):
        self.device = get_device()
        self.model = ChestXrayNet(pretrained=False).to(self.device).eval()
        ckpt = torch.load(CHECKPOINT, map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.tf = eval_transform()

        cal = json.loads(CALIBRATION.read_text()) if CALIBRATION.exists() else {}
        n = len(FINDINGS)
        self.temp = np.array(cal.get("temperature", [1.0] * n))
        self.bias = np.array(cal.get("bias", [0.0] * n))
        self.threshold = np.array(cal.get("threshold", [0.5] * n))
        self.calibrated = bool(cal)

    def _tensor(self, path: str) -> torch.Tensor:
        return self.tf(load_xray(path)).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, path: str) -> dict:
        logits = self.model(self._tensor(path)).float().cpu().numpy()[0]
        probs = 1 / (1 + np.exp(-(logits / self.temp + self.bias)))
        findings = []
        for j, name in enumerate(FINDINGS):
            p, th = float(probs[j]), float(self.threshold[j])
            findings.append({
                "finding": name,
                "probability": round(p, 3),
                "threshold": round(th, 3),
                "positive": p >= th,
                # close to the decision boundary either way -> the model is unsure
                "uncertain": abs(p - th) < UNCERTAIN_MARGIN,
            })
        findings.sort(key=lambda d: d["probability"] - d["threshold"], reverse=True)
        return {"calibrated": self.calibrated, "findings": findings}

    @torch.no_grad()
    def embed(self, path: str) -> np.ndarray:
        v = self.model.embed(self._tensor(path)).float().cpu().numpy()[0]
        return v / (np.linalg.norm(v) + 1e-8)          # unit length -> dot product = cosine similarity

    def gradcam(self, path: str, finding: str) -> str:
        """Grad-CAM: which regions pushed the score for `finding` up?

        1. forward pass, keep the last conv feature maps A (1024 x 7 x 7)
        2. backprop the finding's logit to get dScore/dA
        3. weight of each channel = average of its gradient
        4. heatmap = ReLU(sum_k weight_k * A_k), upsampled to image size
        Returns the path of a saved overlay PNG (small, easy to pass between tools).
        """
        j = FINDINGS.index(finding)
        x = self._tensor(path)
        with torch.enable_grad():
            fmap = self.model.feature_maps(x)
            fmap.retain_grad()
            score = self.model.classifier(self.model.pool(fmap))[0, j]
            self.model.zero_grad()
            score.backward()
        weights = fmap.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * fmap).sum(dim=1)).detach().cpu().numpy()[0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        base = np.array(load_xray(path).resize((512, 512)))
        cam = cv2.resize(cam, (512, 512))
        color = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(cv2.cvtColor(base, cv2.COLOR_RGB2BGR), 0.6, color, 0.4, 0)

        key = hashlib.md5(f"{path}:{finding}".encode()).hexdigest()[:10]
        out = HEATMAP_DIR / f"{finding}_{key}.png"
        cv2.imwrite(str(out), overlay)
        return str(out)


@lru_cache(maxsize=1)
def get_model() -> XrayModel:
    """Load once per process (model loading takes a few seconds)."""
    return XrayModel()
