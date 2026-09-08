import torch

from src.models.model import build_model


def test_build_model_output_shape():
    model = build_model(architecture="resnet18", num_classes=2, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 2)
