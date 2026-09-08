"""Model definitions for casting defect classification."""

import torch.nn as nn
from torchvision import models


def build_model(architecture="resnet18", num_classes=2, pretrained=True):
    if architecture == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unsupported architecture: {architecture}")
