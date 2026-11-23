"""The deep learning model: DenseNet-121 with a 14-way multi-label head."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms as T

from config import FINDINGS, IMG_SIZE

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ChestXrayNet(nn.Module):
    """DenseNet-121 backbone (ImageNet-pretrained) + one linear layer.

    Multi-label: each of the 14 outputs is an independent logit, turned into a
    probability with a sigmoid (not a softmax), because one X-ray can show
    several findings at once.
    """

    def __init__(self, num_classes: int = len(FINDINGS), pretrained: bool = True):
        super().__init__()
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.densenet121(weights=weights)
        self.features = backbone.features                   # conv layers -> (B, 1024, 7, 7)
        self.classifier = nn.Linear(backbone.classifier.in_features, num_classes)

    def feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    def pool(self, fmap: torch.Tensor) -> torch.Tensor:
        # inplace=False matters: an in-place ReLU would break Grad-CAM's gradients
        return F.adaptive_avg_pool2d(F.relu(fmap, inplace=False), 1).flatten(1)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """1024-d image embedding, used for 'find similar past cases'."""
        return self.pool(self.feature_maps(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embed(x))


def train_transform() -> T.Compose:
    # No horizontal flip: the heart sits on the left, and flipping would
    # teach the model anatomically wrong images (matters for Cardiomegaly).
    return T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        T.RandomRotation(7),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform() -> T.Compose:
    # Plain resize (no crop) so Grad-CAM maps line up with the whole image.
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
