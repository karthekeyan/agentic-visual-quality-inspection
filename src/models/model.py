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
    if architecture == "mobilenet_v2":
        # Lighter, edge/mobile-oriented backbone -- depthwise-separable
        # convs instead of ResNet18's full convs, ~1/3 the parameters.
        # Classifier head is Sequential(Dropout, Linear); only the Linear
        # needs replacing for our binary head.
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    raise ValueError(f"Unsupported architecture: {architecture}")
