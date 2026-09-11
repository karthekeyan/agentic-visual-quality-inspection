"""Tests for the agent graph:
START -> inspection_agent -> characterization_agent -> root_cause_agent ->
disposition_agent -> END.

Checks that the graph runs end-to-end and structures each agent's output
into state correctly -- not that the model's prediction is right, and not
that the heuristic characterization is a "correct" diagnosis (there's no
ground truth for that -- see src.agents.characterization_agent). The
Inspection, Characterization, and Disposition Agents are intentionally
simple (no reasoning/LLM calls, no learned classification/policy), so the
contract to verify is: state in -> state out, correctly shaped. The
Root-Cause Agent does call an LLM (Anthropic API) -- those tests mock the
API call so they run offline, at no cost, and deterministically; ChromaDB
retrieval itself is real (it's local, free, and part of what's being
verified).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

import src.agents.root_cause_agent as root_cause_agent
from src.agents.characterization_agent import AGENT_NAME as CHARACTERIZATION_AGENT_NAME
from src.agents.characterization_agent import DefectCharacterization
from src.agents.disposition_agent import AGENT_NAME as DISPOSITION_AGENT_NAME
from src.agents.disposition_agent import Disposition, decide_disposition
from src.agents.graph import build_graph, get_graph
from src.agents.inspection_agent import AGENT_NAME as INSPECTION_AGENT_NAME
from src.agents.root_cause_agent import AGENT_NAME as ROOT_CAUSE_AGENT_NAME
from src.agents.root_cause_agent import MODEL_ID, RootCause
from src.inference.predict import REPO

TEST_IMAGE = REPO / "data/raw/casting_data/casting_data/test/def_front/cast_def_0_1063.jpeg"
OK_IMAGE = REPO / "data/raw/casting_data/casting_data/test/ok_front/cast_ok_0_2927.jpeg"
FALSE_NEGATIVE_IMAGE = REPO / "data/raw/casting_data/casting_data/test/def_front/cast_def_0_2102.jpeg"
LOCALIZED_DEFECT_IMAGE = REPO / "data/raw/casting_data/casting_data/test/def_front/cast_def_0_108.jpeg"


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


# --- Root-Cause Agent (Phase 5) ----------------------------------------


def _mock_anthropic_client(response_text: str) -> MagicMock:
    """A MagicMock standing in for anthropic.Anthropic(), shaped like the
    real client just enough for root_cause_agent.generate_explanation()."""
    mock_client = MagicMock()
    mock_response = SimpleNamespace(content=[SimpleNamespace(type="text", text=response_text)])
    mock_client.messages.create.return_value = mock_response
    return mock_client


def test_root_cause_skipped_for_ok_prediction(graph):
    """No defect to explain -- and critically, no API call should be made."""
    with patch.object(root_cause_agent, "_get_client") as mock_get_client:
        final_state = graph.invoke({"image_path": str(OK_IMAGE)})

    assert final_state["label"] == "ok"

    root_cause = final_state["root_cause"]
    assert isinstance(root_cause, RootCause)
    assert root_cause.applicable is False
    assert root_cause.retrieved_entries == []
    mock_get_client.assert_not_called()

    agent_output = final_state["agent_outputs"][ROOT_CAUSE_AGENT_NAME]
    assert agent_output["applicable"] is False


def test_root_cause_skipped_for_false_negative(graph):
    """Mirrors test_characterization_skipped_for_false_negative: the Root-Cause
    Agent only runs off the Characterization Agent's `applicable` flag, not
    ground truth, so it must skip whenever characterization did."""
    with patch.object(root_cause_agent, "_get_client") as mock_get_client:
        final_state = graph.invoke({"image_path": str(FALSE_NEGATIVE_IMAGE)})

    assert final_state["defect_characterization"].applicable is False
    assert final_state["root_cause"].applicable is False
    mock_get_client.assert_not_called()


def test_root_cause_runs_for_defective_prediction(graph):
    mock_client = _mock_anthropic_client("Likely porosity from gas entrapment during solidification.")

    with patch.object(root_cause_agent, "_get_client", return_value=mock_client):
        final_state = graph.invoke({"image_path": str(TEST_IMAGE)})

    assert final_state["label"] == "defective"
    assert final_state["defect_characterization"].applicable is True

    root_cause = final_state["root_cause"]
    assert isinstance(root_cause, RootCause)
    assert root_cause.applicable is True
    assert root_cause.explanation == "Likely porosity from gas entrapment during solidification."
    assert root_cause.model == MODEL_ID

    # Retrieval actually happened against the real (local) ChromaDB collection.
    assert 1 <= len(root_cause.retrieved_entries) <= 3
    for entry in root_cause.retrieved_entries:
        assert set(entry.keys()) == {"defect_type", "process_causes", "distance"}
        assert isinstance(entry["defect_type"], str) and entry["defect_type"]
        assert isinstance(entry["process_causes"], str) and entry["process_causes"]

    # The API was called exactly once, with the right model and the
    # retrieved knowledge grounded in the prompt content.
    mock_client.messages.create.assert_called_once()
    _, kwargs = mock_client.messages.create.call_args
    assert kwargs["model"] == MODEL_ID
    assert isinstance(kwargs["system"], str) and kwargs["system"]

    messages = kwargs["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    for entry in root_cause.retrieved_entries:
        assert entry["defect_type"] in messages[0]["content"]
        assert entry["process_causes"] in messages[0]["content"]

    agent_output = final_state["agent_outputs"][ROOT_CAUSE_AGENT_NAME]
    assert agent_output["applicable"] is True
    assert agent_output["explanation"] == root_cause.explanation
    assert agent_output["retrieved_entries"] == root_cause.retrieved_entries


def test_root_cause_retrieval_matches_knowledge_base():
    """retrieve_knowledge() returns entries actually present in the curated
    knowledge base (data/knowledge_base/casting_defects.json), not
    hallucinated ones -- the grounding guarantee this agent depends on."""
    from src.knowledge.ingest import load_entries

    known_defect_types = {e["defect_type"] for e in load_entries()}

    characterization = DefectCharacterization(
        applicable=True,
        category="localized_anomaly",
        description=(
            "Grad-CAM activation is a small, tightly concentrated region near the "
            "bore/inner-rim area -- consistent with a single, sharply defined "
            "anomaly rather than a widespread surface issue."
        ),
        region_size="concentrated",
        position="centered",
        confidence_tier="high",
    )

    entries = root_cause_agent.retrieve_knowledge(characterization)
    assert len(entries) == root_cause_agent.TOP_K
    for entry in entries:
        assert entry["defect_type"] in known_defect_types


# --- Disposition Agent (Phase 6) ----------------------------------------


def _mocked_root_cause_client():
    """Patch target for graph runs on defective images -- the Root-Cause
    Agent would otherwise make a real (costly) API call en route to the
    Disposition Agent under test."""
    return patch.object(root_cause_agent, "_get_client", return_value=_mock_anthropic_client("mocked explanation"))


def test_disposition_accept_for_high_confidence_ok(graph):
    final_state = graph.invoke({"image_path": str(OK_IMAGE)})
    assert final_state["label"] == "ok"
    assert final_state["confidence"] >= 0.90

    disposition = final_state["disposition"]
    assert isinstance(disposition, Disposition)
    assert disposition.decision == "accept"
    assert disposition.confidence_threshold_used == pytest.approx(0.90)
    assert "accept" in disposition.reasoning.lower()

    agent_output = final_state["agent_outputs"][DISPOSITION_AGENT_NAME]
    assert agent_output["decision"] == "accept"


def test_disposition_scrap_for_diffuse_high_confidence_defective(graph):
    with _mocked_root_cause_client():
        final_state = graph.invoke({"image_path": str(TEST_IMAGE)})

    assert final_state["label"] == "defective"
    assert final_state["confidence"] >= 0.90
    assert final_state["defect_characterization"].region_size == "diffuse"

    disposition = final_state["disposition"]
    assert disposition.decision == "scrap"
    assert "diffuse" in disposition.reasoning.lower()

    agent_output = final_state["agent_outputs"][DISPOSITION_AGENT_NAME]
    assert agent_output["decision"] == "scrap"


def test_disposition_rework_for_localized_high_confidence_defective(graph):
    with _mocked_root_cause_client():
        final_state = graph.invoke({"image_path": str(LOCALIZED_DEFECT_IMAGE)})

    assert final_state["label"] == "defective"
    assert final_state["confidence"] >= 0.90
    assert final_state["defect_characterization"].region_size == "concentrated"

    disposition = final_state["disposition"]
    assert disposition.decision == "rework"
    assert "concentrated" in disposition.reasoning.lower()

    agent_output = final_state["agent_outputs"][DISPOSITION_AGENT_NAME]
    assert agent_output["decision"] == "rework"


def test_disposition_escalate_for_low_confidence_ok_false_negative(graph):
    """The known false negative (0.63 confidence, predicted 'ok'): below the
    high-confidence threshold, so it must escalate rather than auto-accept --
    this is precisely the case the escalate rule exists to catch."""
    final_state = graph.invoke({"image_path": str(FALSE_NEGATIVE_IMAGE)})
    assert final_state["label"] == "ok"
    assert final_state["confidence"] < 0.90

    disposition = final_state["disposition"]
    assert disposition.decision == "escalate"
    assert "escalate" in disposition.reasoning.lower() or "review" in disposition.reasoning.lower()


def test_disposition_escalate_for_low_confidence_defective_synthetic():
    """No naturally-occurring low-confidence defective example is on hand,
    so this exercises the rule directly against a synthetic state via
    decide_disposition() rather than through the graph."""
    characterization = DefectCharacterization(
        applicable=True,
        category="localized_anomaly",
        description="synthetic low-confidence case for testing",
        region_size="concentrated",
        position="centered",
        confidence_tier="low",
    )

    disposition = decide_disposition(
        label="defective",
        confidence=0.55,
        characterization=characterization,
    )

    assert disposition.decision == "escalate"
    assert disposition.confidence_threshold_used == pytest.approx(0.90)


def test_decide_disposition_reasoning_is_auditable():
    """Every branch must explain which rule fired, not just the decision."""
    accept = decide_disposition(label="ok", confidence=0.95)
    assert isinstance(accept.reasoning, str) and len(accept.reasoning) > 0

    scrap = decide_disposition(
        label="defective",
        confidence=0.95,
        characterization=DefectCharacterization(
            applicable=True, category="distributed_irregularity", description="", region_size="diffuse"
        ),
    )
    assert "diffuse" in scrap.reasoning.lower()

    rework = decide_disposition(
        label="defective",
        confidence=0.95,
        characterization=DefectCharacterization(
            applicable=True, category="localized_anomaly", description="", region_size="concentrated"
        ),
    )
    assert "concentrated" in rework.reasoning.lower()


def test_decide_disposition_honors_explicit_threshold_override():
    """confidence_threshold_used should reflect whatever threshold was
    actually applied -- config-driven by default, but overridable."""
    disposition = decide_disposition(label="ok", confidence=0.80, high_confidence_threshold=0.75)
    assert disposition.decision == "accept"
    assert disposition.confidence_threshold_used == 0.75


def test_disposition_config_default_threshold_matches_config_yaml():
    """The default threshold is read from config/config.yaml, not hardcoded --
    this pins that config/config.yaml: disposition.high_confidence_threshold
    is actually wired up."""
    from src.agents.disposition_agent import _high_confidence_threshold

    assert _high_confidence_threshold() == pytest.approx(0.90)
