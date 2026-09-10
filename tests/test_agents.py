"""Tests for the Phase 3 agent graph: START -> inspection_agent -> END.

Checks that the graph runs end-to-end and structures predict()'s output
into state correctly -- not that the model's prediction is right. This
agent is intentionally 'dumb' (no reasoning, no LLM calls), so the
contract to verify is: state in -> state out, correctly shaped.
"""

import numpy as np
import pytest
from PIL import Image

from src.agents.graph import build_graph, get_graph
from src.agents.inspection_agent import AGENT_NAME
from src.inference.predict import REPO

TEST_IMAGE = REPO / "data/raw/casting_data/casting_data/test/def_front/cast_def_0_1063.jpeg"
FALSE_NEGATIVE_IMAGE = REPO / "data/raw/casting_data/casting_data/test/def_front/cast_def_0_2102.jpeg"


@pytest.fixture(scope="module")
def graph():
    return get_graph()


def test_get_graph_returns_cached_compiled_graph():
    assert get_graph() is get_graph()


def test_build_graph_is_independent_of_the_cache():
    assert build_graph() is not build_graph()


def test_graph_runs_end_to_end_and_populates_state(graph):
    final_state = graph.invoke({"image_path": str(TEST_IMAGE)})

    assert final_state["image_path"] == str(TEST_IMAGE)

    assert final_state["label"] in ("ok", "defective")
    assert isinstance(final_state["confidence"], float)
    assert 0.0 <= final_state["confidence"] <= 1.0

    assert isinstance(final_state["raw_logits"], np.ndarray)
    assert final_state["raw_logits"].shape == (2,)

    assert isinstance(final_state["heatmap"], np.ndarray)
    assert final_state["heatmap"].ndim == 2
    assert final_state["heatmap"].min() >= 0.0
    assert final_state["heatmap"].max() <= 1.0 + 1e-5

    assert isinstance(final_state["overlay_image"], Image.Image)


def test_graph_records_inspection_agent_output(graph):
    final_state = graph.invoke({"image_path": str(TEST_IMAGE)})

    assert AGENT_NAME in final_state["agent_outputs"]
    agent_output = final_state["agent_outputs"][AGENT_NAME]
    assert agent_output["label"] == final_state["label"]
    assert agent_output["confidence"] == final_state["confidence"]


def test_graph_on_known_false_negative(graph):
    """Structure must hold even for the known false negative (see
    outputs/figures/gradcam/manifest.json, tag 'false_negative') -- this
    agent doesn't judge correctness, just reports what the model said."""
    final_state = graph.invoke({"image_path": str(FALSE_NEGATIVE_IMAGE)})
    assert final_state["label"] == "ok"  # documents the known-wrong prediction


def test_graph_propagates_predict_errors(graph):
    """No error-handling logic in this agent yet -- predict()'s validation
    errors should surface unchanged through graph.invoke()."""
    with pytest.raises(FileNotFoundError):
        graph.invoke({"image_path": str(REPO / "data/raw/does_not_exist.jpeg")})
