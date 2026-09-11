"""Trend Agent -- monitors defect patterns across inspections over time.

*** SIMULATED DATA WARNING ***

This dataset (data/raw/casting_data) has no real batch, production-line,
or timestamp metadata -- it's a flat folder of images with no notion of
"which batch was this cast in" or "when was it inspected". To demonstrate
the trend-monitoring pattern described in docs/functional-proposal.md
("Monitors defect patterns across batches and production lines") without
that data, this agent SIMULATES it:

- `batch_id` is assigned round-robin across three fake batches
  (BATCH_IDS below) based on how many history records already exist --
  it has no relationship to how or when the underlying images were
  actually produced.
- `simulated_timestamp` is real wall-clock time (when the pipeline
  happened to run), but it's labeled "simulated" throughout because the
  *batch grouping* it's attached to is fake -- the timestamp value is
  real, the meaning ("this batch was inspected at this time") is not.

Every output field this agent produces -- and every history record it
writes to outputs/logs/inspection_history.jsonl -- carries this
simulated batch_id/simulated_timestamp. Do not treat drift_flag,
defect_rate, or scrap_rate as reflecting real production trends; this is
a demonstration of the pattern, not a validated monitoring capability.
The `note` field on every Trend result restates this for any downstream
consumer that only sees the dataclass, not this docstring.
"""

import functools
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.agents.state import InspectionState

AGENT_NAME = "trend_agent"

REPO = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO / "config/config.yaml"
HISTORY_PATH = REPO / "outputs/logs/inspection_history.jsonl"

# Fake batch identifiers -- see module docstring. Assignment is round-robin
# over this list, not tied to any real production grouping.
BATCH_IDS = ["batch_A", "batch_B", "batch_C"]

# Fallbacks if config/config.yaml is missing or has no `trend` section.
DEFAULT_DRIFT_THRESHOLD = 0.5
DEFAULT_WINDOW_SIZE = 20

SIMULATED_DATA_NOTE = (
    "Computed over SIMULATED batch/timestamp data for demonstration purposes -- "
    "this dataset has no real batch, line, or timestamp metadata. batch_id is "
    "assigned round-robin, not from real production grouping. Do not treat these "
    "stats as reflecting real production trends."
)


@functools.lru_cache(maxsize=None)
def _load_trend_config(config_path: Path = CONFIG_PATH) -> Dict[str, Any]:
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        config = {}
    return config.get("trend", {}) or {}


def _drift_threshold() -> float:
    return _load_trend_config().get("drift_threshold", DEFAULT_DRIFT_THRESHOLD)


def _window_size() -> int:
    return _load_trend_config().get("window_size", DEFAULT_WINDOW_SIZE)


@dataclass
class Trend:
    """Trend Agent output -- see module docstring for the simulated-data caveat.

    Computed over the current batch's recent history (up to `sample_size`
    records, capped at the configured window_size): `defect_rate` is the
    fraction labeled 'defective', `scrap_rate` the fraction dispositioned
    'scrap', and `drift_flag` is True when defect_rate exceeds
    drift_threshold. `sample_size` is included specifically so a consumer
    can judge how much to trust the rates -- 2/3 defective means something
    very different from 12/20.
    """

    batch_id: str
    defect_rate: float
    scrap_rate: float
    drift_flag: bool
    sample_size: int
    note: str = SIMULATED_DATA_NOTE


def read_history(path: Path = HISTORY_PATH) -> List[Dict[str, Any]]:
    """Read outputs/logs/inspection_history.jsonl (or `path`) as a list of records.

    Returns [] if the file doesn't exist yet (first run).
    """
    if not path.exists():
        return []

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_history_record(record: Dict[str, Any], path: Path = HISTORY_PATH) -> None:
    """Append one record to the history log, creating the file/dir if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def _assign_batch_id(existing_record_count: int) -> str:
    """Round-robin a fake batch_id based on how many history records already
    exist -- see module docstring. Not tied to any real production grouping."""
    return BATCH_IDS[existing_record_count % len(BATCH_IDS)]


def build_history_record(state: InspectionState, path: Path = HISTORY_PATH) -> Dict[str, Any]:
    """Build the history-log row for the current inspection from graph state.

    Reads the current history length (before appending) to assign the
    round-robin `batch_id`, so callers must append the returned record
    exactly once, before building the next one.
    """
    existing_count = len(read_history(path))
    disposition = state.get("disposition")

    return {
        "image_path": state.get("image_path"),
        "label": state.get("label"),
        "confidence": state.get("confidence"),
        "disposition_decision": disposition.decision if disposition is not None else None,
        "batch_id": _assign_batch_id(existing_count),
        "simulated_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def compute_trend(
    batch_id: str,
    path: Path = HISTORY_PATH,
    window_size: Optional[int] = None,
    drift_threshold: Optional[float] = None,
) -> Trend:
    """Compute defect_rate/scrap_rate/drift_flag over the recent history for one batch.

    "Recent" means the most recent `window_size` records whose `batch_id`
    matches (i.e. all records for this simulated batch, capped at
    window_size) -- not the last N records overall, since mixing records
    from different simulated batches would make the per-batch drift
    signal meaningless.
    """
    window_size = window_size if window_size is not None else _window_size()
    drift_threshold = drift_threshold if drift_threshold is not None else _drift_threshold()

    history = read_history(path)
    batch_records = [r for r in history if r.get("batch_id") == batch_id]
    recent = batch_records[-window_size:]

    sample_size = len(recent)
    if sample_size == 0:
        defect_rate = 0.0
        scrap_rate = 0.0
    else:
        defect_rate = sum(1 for r in recent if r.get("label") == "defective") / sample_size
        scrap_rate = sum(1 for r in recent if r.get("disposition_decision") == "scrap") / sample_size

    return Trend(
        batch_id=batch_id,
        defect_rate=defect_rate,
        scrap_rate=scrap_rate,
        drift_flag=defect_rate > drift_threshold,
        sample_size=sample_size,
    )


def trend_agent(state: InspectionState) -> Dict[str, Any]:
    """LangGraph node: log this inspection to history, then compute recent trend stats.

    Appends the current inspection (image_path, label, confidence,
    disposition.decision, SIMULATED batch_id/simulated_timestamp -- see
    module docstring) to outputs/logs/inspection_history.jsonl, then
    computes defect_rate/scrap_rate/drift_flag over the recent history for
    the batch this inspection was just assigned to. Returns a partial
    state update -- LangGraph merges it into the full graph state --
    rather than mutating `state` in place.
    """
    record = build_history_record(state, path=HISTORY_PATH)
    append_history_record(record, path=HISTORY_PATH)
    result = compute_trend(record["batch_id"], path=HISTORY_PATH)

    return {
        "trend": result,
        "agent_outputs": {AGENT_NAME: asdict(result)},
    }
