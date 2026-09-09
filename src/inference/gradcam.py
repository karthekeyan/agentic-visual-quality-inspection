"""Grad-CAM explainability for the ResNet18 casting-defect classifier.

Implementation choice
----------------------
This environment has no internet access at runtime (see requirements.txt /
.venv -- `pip install pytorch-grad-cam` fails with no matching distribution),
so a third-party Grad-CAM library isn't an option. Grad-CAM itself
(Selvaraju et al., 2017) is a short, well-defined algorithm -- hook the
target layer to capture its activations and the gradient of the target
class's logit w.r.t. those activations, weight each activation channel by
its globally-averaged gradient, sum, ReLU, normalize. A direct
implementation is ~30 lines, has no extra dependencies, and keeps the
output format (raw float array + overlay image) under our control for
downstream consumers -- notably the Characterization Agent this feeds.

Target layer: `model.layer4[-1]`, the last BasicBlock of ResNet18 -- i.e.
the last feature map before global average pooling. This is the standard
Grad-CAM target for ResNet: the deepest layer that still has spatial
resolution (7x7 for a 224x224 input), so its activations carry the most
class-discriminative semantics while still being localizable back onto
the image.

No OpenCV in this environment either, so heatmap colorization and overlay
compositing use matplotlib's colormap + PIL instead of cv2.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image

from src.data.transforms import build_eval_transforms
from src.models.model import build_model
from src.utils.checkpoint import load_checkpoint

CLASS_NAMES = {0: "ok", 1: "defective"}


class GradCAM:
    """Grad-CAM for one target layer of a classification model.

    One instance can be reused across many `__call__`s (e.g. looping over
    a batch of images) -- hooks are registered once at construction.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self._activations = None
        self._gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def __call__(self, input_tensor, class_idx=None):
        """Run Grad-CAM for a single image.

        Args:
            input_tensor: (1, C, H, W) preprocessed model input.
            class_idx: class to explain. Defaults to the model's own
                prediction (argmax logit) -- what a downstream consumer
                almost always wants ("why did the model call this X").

        Returns:
            (cam, class_idx, confidence): cam is a (h, w) float32 numpy
            array in [0, 1] at the target layer's spatial resolution
            (7x7 for a 224x224 ResNet18 input) -- caller resizes it to
            whatever resolution it needs. confidence is the softmax
            probability of class_idx.
        """
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        confidence = float(probs[0, class_idx].item())

        logits[0, class_idx].backward()

        gradients = self._gradients[0]    # (C, h, w)
        activations = self._activations[0]  # (C, h, w)
        weights = gradients.mean(dim=(1, 2))  # (C,), global-average-pooled gradient per channel

        cam = torch.einsum("c,chw->hw", weights, activations)
        cam = F.relu(cam)
        cam_max = cam.max()
        cam = cam / cam_max if cam_max > 0 else cam
        return cam.cpu().numpy().astype(np.float32), class_idx, confidence


@dataclass
class GradCAMResult:
    """Grad-CAM output for one image -- the unit the Characterization Agent consumes."""

    image_path: str
    predicted_label: int
    predicted_class: str
    confidence: float
    heatmap: np.ndarray  # float32 (H, W) in [0, 1], resized to model input resolution
    overlay: Image.Image  # RGB, same H x W, heatmap alpha-blended over the input image


def load_model_for_gradcam(checkpoint_path, device):
    """Load a trained model from checkpoint and wrap it with a GradCAM on layer4.

    Returns (model, gradcam, image_size) -- image_size comes from the
    config embedded in the checkpoint (see src.utils.checkpoint), so the
    caller doesn't need to separately track which config trained it.
    """
    model = build_model(architecture="resnet18", num_classes=2, pretrained=False).to(device)
    checkpoint = load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()

    gradcam = GradCAM(model, target_layer=model.layer4[-1])
    image_size = tuple(checkpoint["config"]["data"]["image_size"])
    return model, gradcam, image_size


def _colorize(heatmap):
    """(H, W) float32 in [0, 1] -> (H, W, 3) uint8 RGB via the jet colormap."""
    jet = colormaps["jet"]
    colored = jet(heatmap)[:, :, :3]  # drop alpha
    return (colored * 255).astype(np.uint8)


def _overlay_heatmap(base_image, heatmap, alpha=0.45):
    """Alpha-blend a colorized heatmap over an RGB PIL image (same size)."""
    base = np.asarray(base_image, dtype=np.float32)
    colored = _colorize(heatmap).astype(np.float32)
    blended = (1 - alpha) * base + alpha * colored
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))


def explain_image(image_path, model, gradcam, image_size, device, class_idx=None):
    """Grad-CAM explanation for a single image on disk.

    Args:
        image_path: path to an image file.
        model, gradcam, image_size: from `load_model_for_gradcam`.
        device: torch device the model lives on.
        class_idx: explain this class instead of the model's own prediction
            (e.g. to visualize "what would make this look defective").

    Returns:
        GradCAMResult.
    """
    image_path = Path(image_path)
    pil_image = Image.open(image_path).convert("RGB")

    transform = build_eval_transforms(image_size)
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    heatmap_small, predicted_label, confidence = gradcam(input_tensor, class_idx=class_idx)

    # Resize CAM (target layer's resolution, e.g. 7x7) up to the model's
    # input resolution so it aligns pixel-for-pixel with what the model saw.
    heatmap = F.interpolate(
        torch.from_numpy(heatmap_small)[None, None],
        size=image_size,
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()

    # Background for the overlay: the resized (but not normalized) input
    # image, so colors match what a human sees, not ImageNet-normalized values.
    display_image = pil_image.resize((image_size[1], image_size[0]), Image.BILINEAR)
    overlay = _overlay_heatmap(display_image, heatmap)

    return GradCAMResult(
        image_path=str(image_path),
        predicted_label=predicted_label,
        predicted_class=CLASS_NAMES[predicted_label],
        confidence=confidence,
        heatmap=heatmap,
        overlay=overlay,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Grad-CAM explanation for a single image.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default=None, help="Path to save the overlay PNG.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, gradcam, image_size = load_model_for_gradcam(args.checkpoint, device)
    result = explain_image(args.image, model, gradcam, image_size, device)

    print(f"image:      {result.image_path}")
    print(f"prediction: {result.predicted_class} (confidence={result.confidence:.4f})")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.overlay.save(out_path)
        print(f"overlay saved to: {out_path}")


if __name__ == "__main__":
    main()
