"""Tests for the agent graph: START -> inspection_agent -> characterization_agent -> END.

Checks that the graph runs end-to-end and structures each agent's output
into state correctly -- not that the model's prediction is right, and not
that the heuristic characterization is a "correct" diagnosis (there's no
ground truth for that -- see src.agents.characterization_agent). Both
agents are intentionally simple (no reasoning/LLM calls in the Inspection
Agent, no learned classification in the Characterization Agent), so the
contract to verify is: state in -> state out, correctly shaped.
"""

import numpy as np
import pytest
from PIL import Image

from src.agents.characterization_agent import AGENT_NAME as CHARACTERIZATION_AGENT_NAME
from src.agents.characterization_agent import DefectCharacterization
from src.agents.graph import build_graph, get_graph
from src.agents.inspection_agent import AGENT_NAME as INSPECTION_AGENT_NAME
from src.inference.predict import REPO

TEST_IMAGE = REPO / "data/raw/casting_data/casting_data/test/def_front/cast_def_0_1063.jpeg"
OK_IMAGE = REPO / "data/raw/casting_data/casting_data/test/ok_front/cast_ok_0_2927.jpeg"
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

    assert INSPECTION_AGENT_NAME in final_state["agent_outputs"]
    agent_output = final_state["agent_outputs"][INSPECTION_AGENT_NAME]
    assert agent_output["label"] == final_state["label"]
    assert agent_output["confidence"] == final_state["confidence"]


def test_graph_propagates_predict_errors(graph):
    """No error-handling logic in these agents yet -- predict()'s
    validation errors should surface unchanged through graph.invoke()."""
    with pytest.raises(FileNotFoundError):
        graph.invoke({"image_path": str(REPO / "data/raw/does_not_exist.jpeg")})


# --- Characterization Agent (Phase 4) ---------------------------------


def test_characterization_runs_for_defective_prediction(graph):
    final_state = graph.invoke({"image_path": str(TEST_IMAGE)})
    assert final_state["label"] == "defective"

    characterization = final_state["defect_characterization"]
    assert isinstance(characterization, DefectCharacterization)
    assert characterization.applicable is True
    assert characterization.heuristic_approximation is True
    assert characterization.category in (
        "localized_anomaly",
        "distributed_irregularity",
        "edge_boundary_anomaly",
    )
    assert isinstance(characterization.description, str) and characterization.description
    assert characterization.region_size in ("concentrated", "diffuse")
    assert characterization.position in ("centered", "edge")
    assert characterization.confidence_tier in ("high", "moderate", "low")
    assert 0.0 <= characterization.heatmap_area_fraction <= 1.0

    agent_output = final_state["agent_outputs"][CHARACTERIZATION_AGENT_NAME]
    assert agent_output["category"] == characterization.category
    assert agent_output["heuristic_approximation"] is True


def test_characterization_skipped_for_ok_prediction(graph):
    final_state = graph.invoke({"image_path": str(OK_IMAGE)})
    assert final_state["label"] == "ok"

    characterization = final_state["defect_characterization"]
    assert isinstance(characterization, DefectCharacterization)
    assert characterization.applicable is False
    assert characterization.category == "not_applicable"
    assert characterization.heuristic_approximation is True
    # Spatial/confidence fields are meaningless when not applicable.
    assert characterization.region_size is None
    assert characterization.position is None
    assert characterization.confidence_tier is None


def test_characterization_skipped_for_false_negative(graph):
    """The known false negative (see outputs/figures/gradcam/manifest.json,
    tag 'false_negative'): the Inspection Agent calls it 'ok', so the
    Characterization Agent must skip it too -- it only runs off the
    Inspection Agent's label, not ground truth."""
    final_state = graph.invoke({"image_path": str(FALSE_NEGATIVE_IMAGE)})
    assert final_state["label"] == "ok"  # documents the known-wrong prediction

    characterization = final_state["defect_characterization"]
    assert characterization.applicable is False
    assert characterization.category == "not_applicable"
