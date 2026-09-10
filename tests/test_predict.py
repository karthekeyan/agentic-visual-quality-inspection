"""Tests for the Phase 2 inference contract: src.inference.predict.predict().

These check that predict() behaves correctly (right output structure,
input validation, model reuse) -- not that every prediction is correct.
Includes the known false negative from the Grad-CAM manifest
(outputs/figures/gradcam/manifest.json) specifically because a wrong
prediction there is expected, and the contract must hold regardless.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.inference.predict import (
    REPO,
    LoadedModel,
    PredictionResult,
    _get_cached_model,
    _model_cache,
    load_model,
    predict,
)

TEST_IMAGES = {
    "true_positive": REPO / "data/raw/casting_data/casting_data/test/def_front/cast_def_0_1063.jpeg",
    "true_negative": REPO / "data/raw/casting_data/casting_data/test/ok_front/cast_ok_0_2927.jpeg",
    "false_negative": REPO / "data/raw/casting_data/casting_data/test/def_front/cast_def_0_2102.jpeg",
}


def _assert_well_formed(result, image_size):
    assert isinstance(result, PredictionResult)
    assert result.label in ("ok", "defective")
    assert isinstance(result.confidence, float)
    assert 0.0 <= result.confidence <= 1.0

    assert isinstance(result.raw_logits, np.ndarray)
    assert result.raw_logits.shape == (2,)

    assert isinstance(result.heatmap, np.ndarray)
    assert result.heatmap.shape == tuple(image_size)
    assert result.heatmap.min() >= 0.0
    assert result.heatmap.max() <= 1.0 + 1e-5

    assert isinstance(result.overlay_image, Image.Image)
    assert result.overlay_image.size == (image_size[1], image_size[0])


@pytest.fixture(scope="module")
def loaded_model():
    return load_model()


@pytest.mark.parametrize("name", sorted(TEST_IMAGES))
def test_predict_on_known_images_from_path(name, loaded_model):
    path = TEST_IMAGES[name]
    assert path.is_file(), f"expected test fixture image missing: {path}"

    result = predict(path, model=loaded_model)
    _assert_well_formed(result, loaded_model.image_size)


def test_predict_false_negative_structure_holds_even_when_prediction_is_wrong(loaded_model):
    """The known false negative (see manifest tag 'false_negative'): the
    model predicts 'ok' for a genuinely defective part. predict() must
    still return a well-formed result -- correctness of the label isn't
    part of this contract."""
    result = predict(TEST_IMAGES["false_negative"], model=loaded_model)
    _assert_well_formed(result, loaded_model.image_size)
    assert result.label == "ok"  # documents the known-wrong prediction


def test_predict_accepts_pil_image(loaded_model):
    pil_image = Image.open(TEST_IMAGES["true_positive"]).convert("RGB")
    result = predict(pil_image, model=loaded_model)
    _assert_well_formed(result, loaded_model.image_size)


def test_predict_accepts_numpy_array_uint8(loaded_model):
    pil_image = Image.open(TEST_IMAGES["true_negative"]).convert("RGB")
    array = np.asarray(pil_image, dtype=np.uint8)
    result = predict(array, model=loaded_model)
    _assert_well_formed(result, loaded_model.image_size)


def test_predict_accepts_numpy_array_float_0_1(loaded_model):
    pil_image = Image.open(TEST_IMAGES["true_negative"]).convert("RGB")
    array = np.asarray(pil_image, dtype=np.float32) / 255.0
    result = predict(array, model=loaded_model)
    _assert_well_formed(result, loaded_model.image_size)


def test_predict_missing_path_raises_file_not_found(loaded_model):
    with pytest.raises(FileNotFoundError):
        predict(REPO / "data/raw/does_not_exist.jpeg", model=loaded_model)


def test_predict_unopenable_file_raises_value_error(tmp_path, loaded_model):
    bad_file = tmp_path / "not_an_image.jpeg"
    bad_file.write_text("this is not image data")
    with pytest.raises(ValueError):
        predict(bad_file, model=loaded_model)


def test_predict_rejects_tiny_image(loaded_model):
    tiny = np.zeros((3, 3, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        predict(tiny, model=loaded_model)


def test_predict_rejects_bad_type(loaded_model):
    with pytest.raises(TypeError):
        predict(12345, model=loaded_model)


def test_predict_rejects_bad_model_type():
    with pytest.raises(TypeError):
        predict(TEST_IMAGES["true_negative"], model="not-a-loaded-model")


def test_load_model_missing_checkpoint_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_model(checkpoint_path="models/checkpoints/does_not_exist.pt")


def test_load_model_returns_loaded_model_with_expected_shape():
    loaded = load_model()
    assert isinstance(loaded, LoadedModel)
    assert isinstance(loaded.device, torch.device)
    assert len(loaded.image_size) == 2


def test_predict_without_explicit_model_reuses_cached_model():
    """predict() with no `model=` must not reload the checkpoint on every
    call -- the same LoadedModel instance should come back from the cache
    across repeated calls with the same checkpoint_path/device."""
    _model_cache.clear()
    predict(TEST_IMAGES["true_negative"])
    cached_after_first = _get_cached_model("models/checkpoints/best.pt", None)

    predict(TEST_IMAGES["true_positive"])
    cached_after_second = _get_cached_model("models/checkpoints/best.pt", None)

    assert cached_after_first is cached_after_second
