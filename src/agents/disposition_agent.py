"""Disposition Agent -- rule-based accept/rework/scrap/escalate recommendation.

IMPORTANT: this is an authored decision policy, not one learned from
historical disposition outcomes.

There is no disposition-outcome ground truth in this dataset -- no record
of what actually happened to parts previously classified 'ok' or
'defective' (whether an accepted part later failed in the field, whether
a scrapped part could have been reworked, etc.) -- so there is nothing to
train or validate a learned disposition policy against. This matches the
honesty pattern in src.agents.characterization_agent: the
high_confidence_threshold (config/config.yaml: disposition section) and
the accept/rework/scrap/escalate rules below are engineering judgment,
not fitted to labeled disposition outcomes. The threshold is
config-driven specifically so it can be revised as real-world outcomes
accumulate, without a code change.

The escalate branches exist because confidence alone is an imperfect
proxy for correctness -- the known false-negative case in this dataset
(a defective part misclassified 'ok' at 0.63 confidence, see
tests/test_agents.py) is exactly the kind of case this policy is
designed to catch: below-threshold confidence, on either label, routes
to a human rather than an automated decision.
"""

import functools
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.agents.characterization_agent import DefectCharacterization
from src.agents.state import InspectionState

AGENT_NAME = "disposition_agent"

REPO = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO / "config/config.yaml"

# Fallback if config/config.yaml is missing or has no `disposition` section.
DEFAULT_HIGH_CONFIDENCE_THRESHOLD = 0.90


@functools.lru_cache(maxsize=None)
def _load_disposition_config(config_path: Path = CONFIG_PATH) -> Dict[str, Any]:
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        config = {}
    return config.get("disposition", {}) or {}


def _high_confidence_threshold() -> float:
    return _load_disposition_config().get("high_confidence_threshold", DEFAULT_HIGH_CONFIDENCE_THRESHOLD)


@dataclass
class Disposition:
    """Disposition Agent output.

    `reasoning` records which rule fired and why -- kept alongside
    `decision` so the recommendation is auditable, not just a bare label.
    `confidence_threshold_used` records the exact threshold the decision
    was made against, for traceability if config/config.yaml changes later.
    """

    decision: str  # 'accept' | 'rework' | 'scrap' | 'escalate'
    reasoning: str
    confidence_threshold_used: float


def decide_disposition(
    label: str,
    confidence: float,
    characterization: Optional[DefectCharacterization] = None,
    high_confidence_threshold: Optional[float] = None,
) -> Disposition:
    """Apply the authored disposition rules to one prediction.

    `characterization` is only consulted for a 'defective' label at or
    above the confidence threshold, to choose between 'rework' (localized/
    concentrated activation -- worth attempting repair) and 'scrap'
    (diffuse activation -- a widespread irregularity, scrapped rather than
    reworked).
    """
    threshold = high_confidence_threshold if high_confidence_threshold is not None else _high_confidence_threshold()

    if label == "ok":
        if confidence >= threshold:
            decision = "accept"
            reasoning = (
                f"label='ok' with confidence {confidence:.4f} >= high-confidence threshold "
                f"{threshold:.2f} -- accepted."
            )
        else:
            decision = "escalate"
            reasoning = (
                f"label='ok' but confidence {confidence:.4f} < high-confidence threshold "
                f"{threshold:.2f} -- an uncertain 'ok' call is escalated for human review rather "
                f"than accepted outright (a low-confidence 'ok' is exactly the profile of the "
                f"dataset's known false-negative case)."
            )
    elif label == "defective":
        if confidence >= threshold:
            region_size = characterization.region_size if characterization is not None else None
            if region_size == "diffuse":
                decision = "scrap"
                reasoning = (
                    f"label='defective' with confidence {confidence:.4f} >= threshold {threshold:.2f}, "
                    f"and characterization.region_size='diffuse' (widespread irregularity) -- scrapped "
                    f"rather than attempting repair."
                )
            else:
                decision = "rework"
                reasoning = (
                    f"label='defective' with confidence {confidence:.4f} >= threshold {threshold:.2f}, "
                    f"and characterization.region_size={region_size!r} (localized/concentrated, or "
                    f"unavailable) -- worth attempting repair rather than scrapping outright."
                )
        else:
            decision = "escalate"
            reasoning = (
                f"label='defective' but confidence {confidence:.4f} < threshold {threshold:.2f} -- "
                f"a low-confidence defective call is escalated for human review rather than an "
                f"automated rework/scrap decision."
            )
    else:
        raise ValueError(f"Unrecognized label: {label!r}")

    return Disposition(decision=decision, reasoning=reasoning, confidence_threshold_used=threshold)


def disposition_agent(state: InspectionState) -> Dict[str, Any]:
    """LangGraph node: rule-based accept/rework/scrap/escalate recommendation.

    Reads `label`, `confidence`, and `defect_characterization` from state
    (set by the Inspection and Characterization Agents) and applies the
    authored decision policy above. Unlike the Characterization and
    Root-Cause Agents, this node always produces a decision -- there is no
    not-applicable case, since every prediction (ok or defective) needs a
    disposition. Returns a partial state update -- LangGraph merges it
    into the full graph state -- rather than mutating `state` in place.
    """
    result = decide_disposition(
        label=state["label"],
        confidence=state["confidence"],
        characterization=state.get("defect_characterization"),
    )

    return {
        "disposition": result,
        "agent_outputs": {AGENT_NAME: asdict(result)},
    }
