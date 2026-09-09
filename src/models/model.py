"""Model definitions for casting defect classification."""

import torch.nn as nn
from torchvision import models


def build_model(architecture="resnet18", num_classes=2, pretrained=True):
    """Build a classifier: ImageNet-pretrained backbone + fresh linear head.

    Outputs raw logits, shape (batch, num_classes) -- pair with
    nn.CrossEntropyLoss (see src.training.train), not BCEWithLogitsLoss,
    since the head has one output unit per class rather than a single
    sigmoid unit.
    """
    if architecture == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unsupported architecture: {architecture}")
