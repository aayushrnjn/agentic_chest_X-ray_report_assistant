"""OpenCV preprocessing and image-quality checks (no torch needed).

The same `load_xray` is used for training AND inference, so the model always
sees images prepared the same way.
"""
import cv2
import numpy as np
from PIL import Image

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def load_gray(path: str) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def load_xray(path: str) -> Image.Image:
    """Grayscale -> CLAHE contrast enhancement -> 3-channel PIL image.

    CLAHE (adaptive histogram equalisation) evens out exposure differences
    between machines. We copy the gray channel 3 times because the
    ImageNet-pretrained DenseNet expects RGB input.
    """
    gray = load_gray(path)
    gray = _CLAHE.apply(gray)
    return Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))


def check_quality(path: str) -> dict:
    """Cheap heuristics that catch images the model should not be trusted on.

    - blur: variance of the Laplacian (low = few sharp edges = blurry)
    - contrast: pixel standard deviation (low = washed out)
    - exposure: mean brightness (too dark / too bright)
    - resolution: very small images lose fine detail
    The thresholds are heuristics; tune them on your own data.
    """
    gray = load_gray(path)
    h, w = gray.shape
    small = cv2.resize(gray, (512, 512))
    blur = float(cv2.Laplacian(small, cv2.CV_64F).var())
    contrast = float(small.std())
    brightness = float(small.mean())

    problems = []
    if min(h, w) < 256:
        problems.append(f"low resolution ({w}x{h})")
    if blur < 20:
        problems.append(f"image looks blurry (sharpness {blur:.1f})")
    if contrast < 25:
        problems.append(f"very low contrast ({contrast:.1f})")
    if brightness < 30 or brightness > 225:
        problems.append(f"poor exposure (mean brightness {brightness:.0f})")

    return {
        "ok": len(problems) == 0,
        "problems": problems,
        "metrics": {"width": w, "height": h, "sharpness": round(blur, 1),
                    "contrast": round(contrast, 1), "brightness": round(brightness, 1)},
    }
