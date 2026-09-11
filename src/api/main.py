"""Phase 10 -- FastAPI backend exposing the 6-agent pipeline over HTTP.

This is a thin HTTP wrapper, not a reimplementation: every endpoint below
calls straight into the existing pipeline (src.agents.graph) or its
existing helpers (src.agents.trend_agent) rather than recomputing
anything. The pipeline's own error resilience (src.agents.orchestrator --
Phase 9) already keeps a single agent failure from crashing a request; the
one thing this layer adds is translating "the pipeline couldn't produce a
report at all" (e.g. inspection_agent failed because the upload wasn't a
valid image) into a clear 400 rather than a raw 200 with a missing report.

Run with:

    uvicorn src.api.main:app --reload

CORS is left open to any localhost/127.0.0.1 origin (any port), since the
frontend (a separate dev server, e.g. Vite on :5173) runs on a different
port than this API during development.
"""

import base64
import io
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from src.agents.graph import get_graph
from src.agents.trend_agent import compute_trend, read_history

app = FastAPI(title="Agentic Visual Quality Inspection API")

app.add_middleware(
    CORSMiddleware,
    # Any localhost/127.0.0.1 origin, any port -- the frontend dev server's
    # port isn't fixed yet, and this is a local-dev-only API.
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _encode_overlay_png_base64(overlay_image: Image.Image) -> str:
    buffer = io.BytesIO()
    overlay_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@app.get("/health")
def health() -> Dict[str, Any]:
    """Basic liveness check -- also reports whether this process sees a
    CUDA device, so a frontend/operator can confirm the backend is
    actually running the model on GPU rather than silently falling back
    to CPU (src.inference.predict defaults to CUDA when available)."""
    return {"status": "ok", "cuda_available": torch.cuda.is_available()}


@app.post("/inspect")
async def inspect(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Run an uploaded image through the full agent pipeline and return the report.

    The upload is written to a temp file (the pipeline's contract is a
    file path -- see src.agents.state.InspectionState.image_path) and run
    through the same compiled graph src.agents.run_pipeline/run_batch use;
    the temp file is removed once the graph has finished with it,
    regardless of outcome.

    Returns the same JSON shape as src.agents.reporting_agent.Report
    (dataclasses.asdict), plus `overlay_image_base64` (the Grad-CAM
    overlay as a base64-encoded PNG, so a frontend can render it without
    a separate file-serving endpoint) and `original_filename`.

    A malformed/non-image upload makes inspection_agent fail; thanks to
    Phase 9's per-node error handling that doesn't crash the graph, but
    it does mean no report was produced -- that specific case (no
    'report' key in the final state) is what gets translated into a 400
    here, with the underlying agent error(s) as the detail.
    """
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    suffix = Path(file.filename or "").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    try:
        final_state = get_graph().invoke({"image_path": str(tmp_path)})
    finally:
        tmp_path.unlink(missing_ok=True)

    if "report" not in final_state:
        errors = final_state.get("errors") or []
        detail = "; ".join(f"{e['agent']}: {e['error']}" for e in errors) or "Unknown pipeline failure."
        raise HTTPException(status_code=400, detail=f"Could not process uploaded image: {detail}")

    response = asdict(final_state["report"])
    response["original_filename"] = file.filename

    overlay_image = final_state.get("overlay_image")
    if overlay_image is not None:
        response["overlay_image_base64"] = _encode_overlay_png_base64(overlay_image)

    return response


@app.get("/trend/{batch_id}")
def get_trend(batch_id: str) -> Dict[str, Any]:
    """Current recent-history trend stats for one (SIMULATED -- see
    src.agents.trend_agent) batch_id, via trend_agent.compute_trend() --
    the exact same computation the pipeline itself uses, not a
    reimplementation. An unknown batch_id isn't an error: it just has no
    matching history yet, so compute_trend() returns a zero/empty trend
    for it rather than this endpoint raising."""
    return asdict(compute_trend(batch_id))


@app.get("/history")
def get_history(limit: int = Query(default=20, ge=1, le=500)) -> Dict[str, Any]:
    """The most recent `limit` inspection-history records, newest first,
    via trend_agent.read_history() -- for a frontend's recent-inspections
    list. Each record is whatever trend_agent.build_history_record()
    wrote (image_path, label, confidence, disposition_decision, the
    SIMULATED batch_id/simulated_timestamp)."""
    records = read_history()
    recent = list(reversed(records[-limit:]))
    return {"count": len(recent), "records": recent}
