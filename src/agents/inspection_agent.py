"""Inspection Agent -- the first node in the agent graph.

Intentionally 'dumb': it wraps src.inference.predict.predict() and
structures its output onto the shared graph state. No interpretation, no
LLM calls -- that's for later agents (e.g. a Characterization Agent) to
add. This one's whole job is getting the ML prediction + Grad-CAM
explanation onto the state so downstream agents have something to reason
about.
"""

from typing import Any, Dict

from src.agents.state import InspectionState
from src.inference.predict import DEFAULT_CHECKPOINT, LoadedModel, load_model, predict

AGENT_NAME = "inspection_agent"

_default_model: LoadedModel = None


def _get_default_model() -> LoadedModel:
    """Load the default checkpoint once per process and reuse it.

    predict() also caches internally by (checkpoint_path, device), so
    this mainly saves the cache-key resolution on every node call -- but
    it keeps the common case (one checkpoint, one process, many images)
    to exactly one checkpoint read.
    """
    global _default_model
    if _default_model is None:
        _default_model = load_model(DEFAULT_CHECKPOINT)
    return _default_model


def inspection_agent(state: InspectionState) -> Dict[str, Any]:
    """LangGraph node: run predict() on state['image_path'].

    Returns a partial state update -- LangGraph merges it into the full
    graph state -- rather than mutating `state` in place.
    """
    result = predict(state["image_path"], model=_get_default_model())

    output = {
        "label": result.label,
        "confidence": result.confidence,
        "raw_logits": result.raw_logits,
        "heatmap": result.heatmap,
        "overlay_image": result.overlay_image,
    }

    return {
        **output,
        "agent_outputs": {AGENT_NAME: output},
    }
