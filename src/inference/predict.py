"""Inference wrapper -- the single entry point the agent layer calls for a prediction.

predict() is the stable contract for Phase 2: given one image (a path, an
already-loaded PIL image, or a numpy array), it returns a structured
PredictionResult with the predicted label, confidence, raw logits, and a
Grad-CAM explanation (heatmap + overlay). It reuses src.inference.gradcam
for the explainability piece so results match what generated the offline
Grad-CAM manifest (outputs/figures/gradcam/manifest.json).

Model loading is the expensive part (checkpoint read + network build), so
it's cached per (checkpoint_path, device): the first predict() call for a
given checkpoint loads it, every later call with the same arguments reuses
it. Callers that want explicit control over the loaded model -- e.g. a
long-lived API process, or an agent that will call predict() in a tight
loop -- should call load_model() once and pass the result in via
`model=`, which also skips the cache lookup.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from src.inference.gradcam import (
    CLASS_NAMES,
    GradCAM,
    explain_pil_image,
    load_model_for_gradcam,
)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = "models/checkpoints/best.pt"

# Sanity bounds for input images -- catches obviously-wrong inputs (a
# corrupt crop, a 1x1 placeholder, an accidentally-huge scan) before they
# reach the model with a confusing downstream error.
MIN_IMAGE_DIM = 10
MAX_IMAGE_DIM = 8192

ImageInput = Union[str, Path, np.ndarray, Image.Image]


@dataclass
class LoadedModel:
    """A model + Grad-CAM ready for repeated predict() calls.

    Returned by load_model(). Pass it back into predict(model=...) to
    reuse it instead of hitting the checkpoint-load cache.
    """

    model: torch.nn.Module
    gradcam: GradCAM
    image_size: Tuple[int, int]
    device: torch.device
    checkpoint_path: Path


@dataclass
class PredictionResult:
    """Structured output of predict() -- the contract the agent layer relies on."""

    label: str  # 'ok' or 'defective'
    confidence: float  # softmax probability of `label`, in [0, 1]
    raw_logits: np.ndarray  # shape (num_classes,), pre-softmax
    heatmap: np.ndarray  # float32 (H, W) in [0, 1], Grad-CAM for `label`
    overlay_image: Image.Image  # RGB heatmap alpha-blended over the input image


# checkpoint_path (resolved, absolute) + device -> LoadedModel. Module-level
# so it survives across predict() calls within a process without callers
# having to thread a LoadedModel through themselves.
_model_cache: Dict[Tuple[str, str], LoadedModel] = {}


def _resolve_device(device) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device if isinstance(device, torch.device) else torch.device(device)


def _resolve_checkpoint_path(checkpoint_path: Union[str, Path]) -> Path:
    path = Path(checkpoint_path)
    return path if path.is_absolute() else REPO / path


def load_model(checkpoint_path: Union[str, Path] = DEFAULT_CHECKPOINT, device=None) -> LoadedModel:
    """Load a checkpoint once and wrap it for repeated predict() calls.

    Callers that will call predict() many times (an API process, an agent
    loop) should call this once up front and pass the result as
    predict(..., model=loaded) so the checkpoint isn't re-read from disk
    on every prediction. Relative `checkpoint_path` resolves against the
    repo root.
    """
    device = _resolve_device(device)
    checkpoint_path = _resolve_checkpoint_path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model, gradcam, image_size = load_model_for_gradcam(checkpoint_path, device)
    return LoadedModel(
        model=model,
        gradcam=gradcam,
        image_size=tuple(image_size),
        device=device,
        checkpoint_path=checkpoint_path,
    )


def _get_cached_model(checkpoint_path: Union[str, Path], device) -> LoadedModel:
    device = _resolve_device(device)
    checkpoint_path = _resolve_checkpoint_path(checkpoint_path)
    key = (str(checkpoint_path), str(device))
    loaded = _model_cache.get(key)
    if loaded is None:
        loaded = load_model(checkpoint_path, device)
        _model_cache[key] = loaded
    return loaded


def _array_to_pil(array: np.ndarray) -> Image.Image:
    if array.size == 0:
        raise ValueError("Image array is empty.")
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3 or array.shape[2] not in (1, 3, 4):
        raise ValueError(
            f"Unsupported image array shape {array.shape}; expected (H, W), "
            "(H, W, 1), (H, W, 3), or (H, W, 4)."
        )
    if array.shape[2] == 1:
        array = array.repeat(3, axis=2)

    if np.issubdtype(array.dtype, np.floating):
        # Heuristic: [0, 1]-scaled float arrays (common from ML pipelines)
        # vs. already-0..255 float arrays.
        scale = 255.0 if array.max() <= 1.0 + 1e-6 else 1.0
        array = array * scale
    array = array.clip(0, 255).astype(np.uint8)

    return Image.fromarray(array).convert("RGB")


def _load_pil_image(image_path_or_array: ImageInput) -> Image.Image:
    """Validate and normalize any supported input into an RGB PIL image."""
    if isinstance(image_path_or_array, Image.Image):
        pil_image = image_path_or_array.convert("RGB")
    elif isinstance(image_path_or_array, np.ndarray):
        pil_image = _array_to_pil(image_path_or_array)
    elif isinstance(image_path_or_array, (str, Path)):
        path = Path(image_path_or_array)
        if not path.exists():
            raise FileNotFoundError(f"Image path does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"Image path is not a file: {path}")
        try:
            pil_image = Image.open(path).convert("RGB")
        except UnidentifiedImageError as e:
            raise ValueError(f"Could not open '{path}' as an image: {e}") from e
    else:
        raise TypeError(
            "image_path_or_array must be a file path (str/Path), a PIL.Image.Image, "
            f"or a numpy array; got {type(image_path_or_array)}."
        )

    width, height = pil_image.size
    if width < MIN_IMAGE_DIM or height < MIN_IMAGE_DIM:
        raise ValueError(f"Image is too small: {width}x{height} (minimum {MIN_IMAGE_DIM}x{MIN_IMAGE_DIM}).")
    if width > MAX_IMAGE_DIM or height > MAX_IMAGE_DIM:
        raise ValueError(f"Image is too large: {width}x{height} (maximum {MAX_IMAGE_DIM}x{MAX_IMAGE_DIM}).")

    return pil_image


def predict(
    image_path_or_array: ImageInput,
    model: Optional[LoadedModel] = None,
    checkpoint_path: Union[str, Path] = DEFAULT_CHECKPOINT,
    device=None,
) -> PredictionResult:
    """Run the trained model + Grad-CAM on one image.

    Args:
        image_path_or_array: path to an image file, an already-loaded
            PIL.Image.Image, or an (H, W) / (H, W, C) numpy array.
        model: a LoadedModel from load_model(), to reuse across calls
            without re-reading the checkpoint. If None, a model is loaded
            for `checkpoint_path`/`device` and cached for future calls
            made with the same arguments.
        checkpoint_path: checkpoint to load when `model` is not given.
            Relative paths resolve against the repo root. Ignored if
            `model` is given.
        device: torch device (or device string, e.g. "cpu") to run on
            when `model` is not given. Defaults to CUDA if available,
            else CPU. Ignored if `model` is given.

    Returns:
        PredictionResult.
    """
    if model is not None and not isinstance(model, LoadedModel):
        raise TypeError(f"model must be a LoadedModel from load_model(); got {type(model)}")

    loaded = model if model is not None else _get_cached_model(checkpoint_path, device)
    pil_image = _load_pil_image(image_path_or_array)

    predicted_label, confidence, heatmap, overlay_image = explain_pil_image(
        pil_image, loaded.model, loaded.gradcam, loaded.image_size, loaded.device
    )
    raw_logits = loaded.gradcam.last_logits[0].cpu().numpy()

    return PredictionResult(
        label=CLASS_NAMES[predicted_label],
        confidence=confidence,
        raw_logits=raw_logits,
        heatmap=heatmap,
        overlay_image=overlay_image,
    )


def main():
    parser = argparse.ArgumentParser(description="Run the casting-defect model on a single image.")
    parser.add_argument("--image", required=True, help="Path to an image file.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to a model checkpoint.")
    parser.add_argument("--device", default=None, help="'cuda' or 'cpu' (default: cuda if available).")
    parser.add_argument("--out", default=None, help="Optional path to save the Grad-CAM overlay PNG.")
    args = parser.parse_args()

    result = predict(args.image, checkpoint_path=args.checkpoint, device=args.device)

    print(f"image:      {args.image}")
    print(f"label:      {result.label}")
    print(f"confidence: {result.confidence:.4f}")
    print(f"raw_logits: {np.array2string(result.raw_logits, precision=4)}")
    print(f"heatmap:    shape={result.heatmap.shape} min={result.heatmap.min():.4f} max={result.heatmap.max():.4f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.overlay_image.save(out_path)
        print(f"overlay saved to: {out_path}")


if __name__ == "__main__":
    main()
