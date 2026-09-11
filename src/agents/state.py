"""Shared state schema threaded through the LangGraph agent pipeline.

One TypedDict describes the whole graph's state, one entry per image. Each
agent node reads whatever fields it needs and returns only the fields it
updates -- LangGraph merges each node's partial return into the full
state (see src.agents.graph), so nodes never need the rest of the state
just to pass it through.
"""

from typing import Any, Dict, Optional

import numpy as np
from PIL import Image
from typing_extensions import Annotated, TypedDict


def _merge_agent_outputs(left: Optional[Dict[str, Any]], right: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Reducer for `agent_outputs`.

    Without this, LangGraph's default behavior for a plain dict field is
    to replace it wholesale on every node update -- fine for a single
    node, but as more agents are added after this one, each would wipe
    out the previous agents' entries instead of adding to them. Merge
    keeps every agent's contribution keyed by its own name.
    """
    merged = dict(left or {})
    merged.update(right or {})
    return merged


class InspectionState(TypedDict, total=False):
    """Graph state for one image moving through the inspection pipeline.

    Fields:
        image_path: path to the image being inspected. Set by the caller
            before invoking the graph; read-only for agent nodes.
        label: predicted class, 'ok' or 'defective'. Set by the
            Inspection Agent.
        confidence: softmax probability of `label`, in [0, 1].
        raw_logits: pre-softmax model output, shape (num_classes,).
        heatmap: Grad-CAM heatmap, float32 (H, W) in [0, 1].
        overlay_image: Grad-CAM heatmap alpha-blended over the input
            image, for display.
        defect_characterization: set by the Characterization Agent -- a
            heuristic (rule-based, not learned) description of the
            Grad-CAM activation pattern, only meaningful when
            label == 'defective'. See
            src.agents.characterization_agent.DefectCharacterization;
            always carries heuristic_approximation=True.
        root_cause: set by the Root-Cause Agent -- a RAG-generated
            explanation of likely process-related root causes, grounded
            in a curated knowledge base, only meaningful when
            defect_characterization.applicable is True. See
            src.agents.root_cause_agent.RootCause.
        agent_outputs: every agent's structured output, keyed by agent
            name (e.g. "inspection_agent"), so later agents and the final
            report can see what earlier agents produced without depending
            on the top-level field names above.
    """

    image_path: str

    label: Optional[str]
    confidence: Optional[float]
    raw_logits: Optional[np.ndarray]
    heatmap: Optional[np.ndarray]
    overlay_image: Optional[Image.Image]

    defect_characterization: Optional[Any]
    root_cause: Optional[Any]

    agent_outputs: Annotated[Dict[str, Any], _merge_agent_outputs]
