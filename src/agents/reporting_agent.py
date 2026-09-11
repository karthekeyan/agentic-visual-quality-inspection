"""Reporting Agent -- compiles the full pipeline state into one inspection record.

The sixth and final content-producing node. Purely a formatting step: no
new reasoning, no learned or heuristic judgment, no LLM call -- everything
in the report was already decided by an earlier agent (Inspection,
Characterization, Root-Cause, Disposition, Trend). This agent's only job
is to gather those results into one structured record, render a short
human-readable summary of it, and persist both to disk as an audit trail.

Note on `generated_at` vs. Trend's `simulated_timestamp`: `generated_at`
below is real wall-clock time and means exactly what it says (when this
report was compiled) -- unlike Trend's `simulated_timestamp`, it isn't
attached to any simulated batch grouping, so it needs no such caveat.
`trend.note`/`batch_id` on the compiled report still carry the SIMULATED
warning from src.agents.trend_agent; this agent doesn't strip it.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.agents.state import InspectionState

AGENT_NAME = "reporting_agent"

REPO = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO / "outputs/reports/inspections"


@dataclass
class Report:
    """The final compiled inspection record.

    `characterization`/`root_cause` are None when not applicable (an 'ok'
    prediction) rather than a placeholder dict -- absent, not padded with
    not-applicable filler, since the report is meant to be read quickly.
    `saved_path` is filled in by save_report() after the JSON file is
    written, so the file includes its own location for traceability.
    """

    image_path: str
    generated_at: str  # real wall-clock ISO timestamp -- see module docstring
    inspection: Dict[str, Any]
    characterization: Optional[Dict[str, Any]]
    root_cause: Optional[Dict[str, Any]]
    disposition: Dict[str, Any]
    trend: Dict[str, Any]
    summary_text: str
    saved_path: Optional[str] = None


def _characterization_dict(characterization) -> Optional[Dict[str, Any]]:
    if characterization is None or not characterization.applicable:
        return None
    return {
        "category": characterization.category,
        "description": characterization.description,
        "region_size": characterization.region_size,
        "position": characterization.position,
        "confidence_tier": characterization.confidence_tier,
        "heuristic_approximation": characterization.heuristic_approximation,
    }


def _root_cause_dict(root_cause) -> Optional[Dict[str, Any]]:
    if root_cause is None or not root_cause.applicable:
        return None
    return {
        "explanation": root_cause.explanation,
        "retrieved_entries": root_cause.retrieved_entries,
        "model": root_cause.model,
    }


def _disposition_dict(disposition) -> Dict[str, Any]:
    return {
        "decision": disposition.decision,
        "reasoning": disposition.reasoning,
        "confidence_threshold_used": disposition.confidence_threshold_used,
    }


def _trend_dict(trend) -> Dict[str, Any]:
    return {
        "batch_id": trend.batch_id,
        "defect_rate": trend.defect_rate,
        "scrap_rate": trend.scrap_rate,
        "drift_flag": trend.drift_flag,
        "sample_size": trend.sample_size,
        "note": trend.note,
    }


def _build_summary_text(
    image_path: str,
    label: str,
    confidence: float,
    characterization_dict: Optional[Dict[str, Any]],
    root_cause_dict: Optional[Dict[str, Any]],
    disposition_dict: Dict[str, Any],
    trend_dict: Dict[str, Any],
) -> str:
    """A few plain-text sentences, not a data dump -- everything a quality
    engineer needs from a 10-second read, with the full detail still
    available in the structured fields alongside it."""
    image_name = Path(image_path).name
    sentences = [f"Inspection of {image_name}: predicted {label.upper()} with {confidence * 100:.1f}% confidence."]

    if characterization_dict is not None:
        sentences.append(
            f"Grad-CAM attention was {characterization_dict['region_size']} and "
            f"{characterization_dict['position']}."
        )

    if root_cause_dict is not None and root_cause_dict.get("explanation"):
        # Note: retrieved_entries is ranked by embedding-similarity to the
        # characterization, not by what the model concluded -- its top
        # entry is not necessarily the explanation's actual conclusion, so
        # the summary points to the explanation rather than asserting a
        # single defect type here.
        sentences.append("Root-cause analysis available -- see root_cause.explanation for grounded detail.")

    sentences.append(f"Disposition: {disposition_dict['decision'].upper()}.")

    drift_note = " Defect rate is trending high for this batch." if trend_dict["drift_flag"] else ""
    sentences.append(
        f"Simulated batch {trend_dict['batch_id']} trend: {trend_dict['defect_rate'] * 100:.0f}% defective "
        f"over the last {trend_dict['sample_size']} part(s).{drift_note}"
    )

    return " ".join(sentences)


def build_report(state: InspectionState) -> Report:
    """Compile a Report from the full graph state. Pure formatting -- reads
    what upstream agents already decided, adds no new judgment."""
    characterization_dict = _characterization_dict(state.get("defect_characterization"))
    root_cause_dict = _root_cause_dict(state.get("root_cause"))
    disposition_dict = _disposition_dict(state["disposition"])
    trend_dict = _trend_dict(state["trend"])

    summary_text = _build_summary_text(
        image_path=state["image_path"],
        label=state["label"],
        confidence=state["confidence"],
        characterization_dict=characterization_dict,
        root_cause_dict=root_cause_dict,
        disposition_dict=disposition_dict,
        trend_dict=trend_dict,
    )

    return Report(
        image_path=state["image_path"],
        generated_at=datetime.now(timezone.utc).isoformat(),
        inspection={"label": state["label"], "confidence": state["confidence"]},
        characterization=characterization_dict,
        root_cause=root_cause_dict,
        disposition=disposition_dict,
        trend=trend_dict,
        summary_text=summary_text,
    )


def save_report(report: Report, reports_dir: Path = REPORTS_DIR) -> Path:
    """Write the report as JSON to reports_dir/{timestamp}_{image_stem}.json.

    Mutates report.saved_path before serializing, so the file records its
    own location. Filename timestamp is derived from report.generated_at
    (not a fresh now()) so the two stay consistent.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)

    generated_dt = datetime.fromisoformat(report.generated_at)
    filename_ts = generated_dt.strftime("%Y%m%dT%H%M%S%f")[:-3] + "Z"  # milliseconds, filename-safe
    image_stem = Path(report.image_path).stem
    path = reports_dir / f"{filename_ts}_{image_stem}.json"

    report.saved_path = str(path)
    with open(path, "w") as f:
        json.dump(asdict(report), f, indent=2)

    return path


def reporting_agent(state: InspectionState) -> Dict[str, Any]:
    """LangGraph node: compile and persist the final inspection report.

    Reads label/confidence/defect_characterization/root_cause/disposition/
    trend from state (set by every upstream agent) and produces both a
    structured Report (state['report']) and a JSON file under
    outputs/reports/inspections/, one per pipeline run -- the persistent
    audit trail. Returns a partial state update -- LangGraph merges it
    into the full graph state -- rather than mutating `state` in place.
    """
    report = build_report(state)
    save_report(report, reports_dir=REPORTS_DIR)

    return {
        "report": report,
        "agent_outputs": {AGENT_NAME: asdict(report)},
    }
