"""Tests for the Phase 10 FastAPI backend (src.api.main).

Kept deliberately small: /inspect for a real defective image exercises
the full pipeline including a real Anthropic API call in root_cause_agent
(no mocking here -- this is meant to verify the actual HTTP contract end
to end), so this file sticks to the couple of cases that matter most
(a working request, and the 400 error path) rather than a full endpoint
matrix, to control cost. See tests/test_agents.py for the mocked,
free/offline coverage of the pipeline's own agent logic.
"""

import base64

from fastapi.testclient import TestClient

from src.api.main import app
from src.inference.predict import REPO

TEST_IMAGE = REPO / "data/raw/casting_data/casting_data/test/def_front/cast_def_0_1063.jpeg"

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert isinstance(body["cuda_available"], bool)


def test_inspect_with_real_defective_image():
    """Real end-to-end request: real model inference, real Grad-CAM, and a
    real Anthropic API call for root_cause_agent (this image is a known
    defective example -- see tests/test_agents.py)."""
    with open(TEST_IMAGE, "rb") as f:
        response = client.post("/inspect", files={"file": (TEST_IMAGE.name, f, "image/jpeg")})

    assert response.status_code == 200
    body = response.json()

    assert body["inspection"]["label"] == "defective"
    assert 0.0 <= body["inspection"]["confidence"] <= 1.0

    assert body["disposition"]["decision"] in ("accept", "rework", "scrap", "escalate")
    assert isinstance(body["human_review_required"], bool)
    assert isinstance(body["summary_text"], str) and body["summary_text"]
    assert body["errors"] == []

    assert body["original_filename"] == TEST_IMAGE.name

    # Grad-CAM overlay comes back as a base64-encoded PNG.
    overlay_bytes = base64.b64decode(body["overlay_image_base64"])
    assert overlay_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_inspect_rejects_non_image_upload():
    """A malformed upload makes inspection_agent fail; the pipeline itself
    doesn't crash (Phase 9), but no report is produced -- the API must
    surface that as a clear 400, not a 200 with a missing report or an
    uncaught 500."""
    response = client.post(
        "/inspect",
        files={"file": ("not_an_image.txt", b"this is definitely not image data", "text/plain")},
    )

    assert response.status_code == 400
    assert "inspection_agent" in response.json()["detail"]
