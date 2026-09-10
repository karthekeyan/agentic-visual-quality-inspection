"""Characterization Agent -- heuristic, rule-based defect characterization.

IMPORTANT: this is a heuristic approximation, not a learned classification.

There is no defect-subtype ground truth in this dataset -- no labels
distinguishing porosity, shrinkage, crack, surface blemish, etc. -- so
there is nothing to train or validate a real defect-type classifier
against. This matches docs/functional-proposal.md: Section 7 lists
"defect-type characterization... dependent on additional data -- a
defect-labeled dataset" as architected but not yet a trained or validated
capability, and Section 8.1 notes "a defect-subtype-labeled dataset...
would be required to fully realize defect-type characterization. The
current dataset supports binary detection only."

Until that dataset exists, this agent falls back to simple, inspectable
rules over the Grad-CAM heatmap the Inspection Agent already produced:
region size (concentrated vs. diffuse), region position (centered on the
bore/inner-rim region vs. near the part's edge/boundary), and the model's
confidence. The bore-region rule is grounded in what the Grad-CAM review
actually showed on this dataset -- correctly-classified defective images
consistently activated the bore/inner-rim region (commit 85b5648). But
these rules describe *where the model looked*, not *what defect is
physically present* -- every result carries heuristic_approximation=True
so downstream agents/consumers never mistake this for a validated
defect-type classification.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np

from src.agents.state import InspectionState

AGENT_NAME = "characterization_agent"

# Grad-CAM heatmaps from src.inference.gradcam are float32 in [0, 1];
# treat anything at/above this as "the model looked here".
HEATMAP_ACTIVATION_THRESHOLD = 0.5

# Fraction of pixels at/above HEATMAP_ACTIVATION_THRESHOLD below which the
# activated region counts as "concentrated" rather than "diffuse".
SMALL_REGION_FRACTION = 0.20

# Normalized distance of the activated region's centroid from the image
# center (0 = dead center, ~0.71 = corner) below which it counts as
# "centered" (i.e. the bore/inner-rim region) rather than "edge".
CENTERED_DISTANCE = 0.20

# Confidence tiers, purely descriptive -- these don't change the category,
# only the tentativeness noted in the description.
HIGH_CONFIDENCE = 0.85
MODERATE_CONFIDENCE = 0.65


@dataclass
class DefectCharacterization:
    """Heuristic characterization of a defective prediction's Grad-CAM pattern.

    `heuristic_approximation` is always True -- there is no ground truth
    this could be validated against yet (see module docstring). Treat
    `category`/`description` as an inspectable, rule-based description of
    the activation pattern, not a diagnosis.
    """

    applicable: bool
    category: str
    description: str
    heuristic_approximation: bool = True
    region_size: Optional[str] = None  # 'concentrated' | 'diffuse'
    position: Optional[str] = None  # 'centered' | 'edge'
    confidence_tier: Optional[str] = None  # 'high' | 'moderate' | 'low'
    heatmap_area_fraction: Optional[float] = None
    heatmap_center_distance: Optional[float] = None


def _not_applicable() -> DefectCharacterization:
    return DefectCharacterization(
        applicable=False,
        category="not_applicable",
        description="Not applicable -- the Inspection Agent did not classify this image as defective.",
    )


def _heatmap_stats(heatmap: np.ndarray, threshold: float = HEATMAP_ACTIVATION_THRESHOLD):
    """(area_fraction, center_distance) for the activated region of a Grad-CAM heatmap."""
    mask = heatmap >= threshold
    area_fraction = float(mask.mean())

    if mask.any():
        ys, xs = np.nonzero(mask)
    else:
        # Degenerate case: nothing crosses the threshold (an unusually flat
        # CAM). Fall back to the single hottest pixel so the heuristic
        # still has a position to reason about.
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        ys, xs = np.array([y]), np.array([x])

    h, w = heatmap.shape
    centroid_y = float(ys.mean()) / h
    centroid_x = float(xs.mean()) / w
    center_distance = float(np.hypot(centroid_y - 0.5, centroid_x - 0.5))
    return area_fraction, center_distance


def _classify_region_size(area_fraction: float) -> str:
    return "concentrated" if area_fraction < SMALL_REGION_FRACTION else "diffuse"


def _classify_position(center_distance: float) -> str:
    return "centered" if center_distance < CENTERED_DISTANCE else "edge"


def _classify_confidence(confidence: float) -> str:
    if confidence >= HIGH_CONFIDENCE:
        return "high"
    if confidence >= MODERATE_CONFIDENCE:
        return "moderate"
    return "low"


def _describe(region_size: str, position: str, confidence_tier: str) -> Dict[str, str]:
    """(category, description) from the three heuristic axes.

    Position takes priority: an activation near the part's boundary is a
    qualitatively different pattern from the bore-region activation the
    Grad-CAM review associated with confirmed defects, regardless of how
    large or small that boundary region is.
    """
    if position == "edge":
        category = "edge_boundary_anomaly"
        pattern = (
            "Grad-CAM activation is concentrated near the part's outer boundary/rim "
            "rather than the bore region the model typically attends to for a "
            "confirmed defect."
        )
    elif region_size == "concentrated":
        category = "localized_anomaly"
        pattern = (
            "Grad-CAM activation is a small, tightly concentrated region near the "
            "bore/inner-rim area -- consistent with a single, sharply defined "
            "anomaly rather than a widespread surface issue."
        )
    else:
        category = "distributed_irregularity"
        pattern = (
            "Grad-CAM activation covers a broad, diffuse area around the bore "
            "region -- consistent with an irregularity spread across a larger "
            "surface area rather than one sharply localized spot."
        )

    confidence_note = {
        "high": "The model's confidence in this defective call is high.",
        "moderate": "The model's confidence in this defective call is moderate.",
        "low": "The model's confidence in this defective call is low -- treat this characterization as especially tentative.",
    }[confidence_tier]

    return category, f"{pattern} {confidence_note}"


def characterize_defect(heatmap: np.ndarray, confidence: float) -> DefectCharacterization:
    """Apply the heuristic rules to one Grad-CAM heatmap + confidence score.

    Callers should only call this for a 'defective' prediction -- for an
    'ok' prediction, use _not_applicable() (which the node function below
    does automatically).
    """
    area_fraction, center_distance = _heatmap_stats(heatmap)
    region_size = _classify_region_size(area_fraction)
    position = _classify_position(center_distance)
    confidence_tier = _classify_confidence(confidence)
    category, description = _describe(region_size, position, confidence_tier)

    return DefectCharacterization(
        applicable=True,
        category=category,
        description=description,
        region_size=region_size,
        position=position,
        confidence_tier=confidence_tier,
        heatmap_area_fraction=area_fraction,
        heatmap_center_distance=center_distance,
    )


def characterization_agent(state: InspectionState) -> Dict[str, Any]:
    """LangGraph node: heuristically characterize a defective prediction's Grad-CAM pattern.

    Only runs the characterization rules when the Inspection Agent's
    `label` is 'defective'; otherwise returns a minimal not-applicable
    result. Returns a partial state update -- LangGraph merges it into
    the full graph state -- rather than mutating `state` in place.
    """
    if state.get("label") != "defective":
        result = _not_applicable()
    else:
        result = characterize_defect(state["heatmap"], state["confidence"])

    return {
        "defect_characterization": result,
        "agent_outputs": {AGENT_NAME: asdict(result)},
    }
