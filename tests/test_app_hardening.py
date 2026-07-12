"""Regression tests for Phase 3 API/security hardening.

Covers: upload filename-traversal neutralization + size/type limits, /ws/process
path containment (no existence oracle), /ws/webcam session isolation (the most
severe bug found -- concurrent users previously shared one global HeartRateMonitor),
/export_chart validation + leak-free PNG streaming, and CORS defaults.

MediaPipe face detection is stubbed throughout (reusing the patterns from
test_method_wiring.py) so these stay fast, deterministic unit tests -- no real
camera or model download required.
"""

import base64
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("mediapipe")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

# Import main AFTER stubbing so its module-level UPLOAD_DIR etc. are unaffected by
# the detector stub (only HeartRateMonitor construction is stubbed, not the app).
import src.app.main as appmod
import src.vitals.heart_rate as hr_mod
from tests.test_method_wiring import _PulseDetector, _StubDetector

BLANK_FRAME_JPEG_B64 = base64.b64encode(
    cv2.imencode(".jpg", np.zeros((480, 640, 3), dtype=np.uint8))[1]
).decode("utf-8")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(hr_mod, "MediaPipeDetector", _StubDetector)
    return TestClient(appmod.app)


# --------------------------------------------------------------------------- #
# /upload
# --------------------------------------------------------------------------- #


def test_upload_rejects_bad_content_type(client):
    r = client.post("/upload", files={"file": ("evil.txt", b"not a video", "text/plain")})
    assert r.status_code == 400


def test_upload_ignores_traversal_filename(client):
    """A crafted filename must never let the write escape UPLOAD_DIR.

    The uploaded bytes aren't a real video, so cv2 will fail to open it and the
    endpoint returns 400 -- but the key assertion is that the file was written (and
    then cleaned up) strictly inside UPLOAD_DIR the whole time, never at any path
    derived from the attacker-controlled filename.
    """
    before = set(appmod.UPLOAD_DIR.iterdir())
    r = client.post(
        "/upload",
        files={"file": ("../../../../../etc/passwd.mp4", b"not really a video", "video/mp4")},
    )
    assert r.status_code == 400
    after = set(appmod.UPLOAD_DIR.iterdir())
    assert after == before, "rejected upload must not leave files behind"
    assert not Path("/etc/passwd.mp4").exists()


def test_upload_rejects_oversized_file(client, monkeypatch):
    monkeypatch.setattr(appmod, "MAX_UPLOAD_BYTES", 10)
    before = set(appmod.UPLOAD_DIR.iterdir())
    r = client.post("/upload", files={"file": ("clip.mp4", b"x" * 1000, "video/mp4")})
    assert r.status_code == 413
    after = set(appmod.UPLOAD_DIR.iterdir())
    assert after == before, "oversized upload must not leave a partial file behind"


def test_upload_accepts_real_video_and_stores_uuid_name(client, monkeypatch):
    video_path = Path("datasets/DATASET_1/8-gt/vid.avi")
    if not video_path.exists():
        pytest.skip("real UBFC sample video not present locally")

    # This particular UBFC sample is ~1.9GB (uncompressed capture) -- well over the
    # default 200MB limit already covered by test_upload_rejects_oversized_file.
    # Raise it just for this test, which is about the happy path, not the limit.
    monkeypatch.setattr(appmod, "MAX_UPLOAD_BYTES", 3 * 1024 * 1024 * 1024)

    with open(video_path, "rb") as f:
        r = client.post("/upload", files={"file": ("my video.avi", f, "video/x-msvideo")})
    assert r.status_code == 200
    data = r.json()
    stored = Path(data["path"]).resolve()
    assert stored.is_relative_to(appmod.UPLOAD_DIR.resolve())
    assert stored.stem != "my video"  # server-generated name, not the client's
    assert data["filename"] == "my video.avi"  # original name preserved as metadata only
    stored.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# /ws/process
# --------------------------------------------------------------------------- #


def test_ws_process_identical_error_for_missing_vs_outside_upload_dir(client):
    """Regression guard: the error must NOT reveal whether a path exists outside
    UPLOAD_DIR, or a client could use it as a filesystem existence oracle."""
    with client.websocket_connect("/ws/process") as ws:
        ws.send_json({"video_path": "/definitely/does/not/exist.avi"})
        missing_response = ws.receive_json()

    outside_path = Path("datasets/DATASET_1/8-gt/gtdump.xmp")
    if not outside_path.exists():
        pytest.skip("no real out-of-UPLOAD_DIR file available locally")

    with client.websocket_connect("/ws/process") as ws:
        ws.send_json({"video_path": str(outside_path)})
        outside_response = ws.receive_json()

    assert missing_response == outside_response == {"error": "Video not found"}


# --------------------------------------------------------------------------- #
# /ws/webcam -- session isolation regression test
# --------------------------------------------------------------------------- #


def test_webcam_sessions_are_isolated(monkeypatch):
    """The most severe bug found: /process_frame + /reset_webcam used to share one
    global HeartRateMonitor across every connected client. This proves two
    concurrent /ws/webcam connections never influence each other's readings.
    """
    bpms = iter([72.0, 100.0])
    monkeypatch.setattr(hr_mod, "MediaPipeDetector", lambda *a, **k: _PulseDetector(bpm=next(bpms)))
    client = TestClient(appmod.app)

    def _drive(ws, n_frames=200):
        last = None
        for _ in range(n_frames):
            ws.send_json({"frame": BLANK_FRAME_JPEG_B64})
            last = ws.receive_json()
        return last

    # Fully drive connection 1 first (deterministically assigns it bpm=72 via the
    # `bpms` iterator above), then connection 2 (bpm=100) -- if any state leaked
    # between them, the two final readings would converge instead of staying apart.
    with client.websocket_connect("/ws/webcam") as ws1:
        result1 = _drive(ws1)
    with client.websocket_connect("/ws/webcam") as ws2:
        result2 = _drive(ws2)

    assert result1["heart_rate"] > 0 and result2["heart_rate"] > 0
    assert (
        abs(result1["heart_rate"] - result2["heart_rate"]) > 15
    ), f"sessions appear to share state: {result1['heart_rate']} vs {result2['heart_rate']}"


def test_webcam_missing_frame_data_returns_error(client):
    with client.websocket_connect("/ws/webcam") as ws:
        ws.send_json({})
        assert ws.receive_json() == {"error": "No frame data"}


# --------------------------------------------------------------------------- #
# /export_chart
# --------------------------------------------------------------------------- #


def _snapshot_temp_dir() -> set:
    return set(Path(tempfile.gettempdir()).iterdir())


def test_export_chart_returns_png_on_success(client):
    before = _snapshot_temp_dir()
    r = client.post(
        "/export_chart",
        json={
            "hr_timeline": [70, 72, 75],
            "timestamps": [0, 1, 2],
            "avg_hr": 72.3,
            "min_hr": 70,
            "max_hr": 75,
        },
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert _snapshot_temp_dir() == before, "success path must not leave a temp file behind"


def test_export_chart_mismatched_lengths_returns_400(client):
    r = client.post(
        "/export_chart",
        json={"hr_timeline": [70, 72], "timestamps": [0], "avg_hr": 70, "min_hr": 70, "max_hr": 70},
    )
    assert r.status_code == 400


def test_export_chart_empty_data_returns_400(client):
    r = client.post(
        "/export_chart",
        json={"hr_timeline": [], "timestamps": [], "avg_hr": 0, "min_hr": 0, "max_hr": 0},
    )
    assert r.status_code == 400


def test_export_chart_non_finite_values_returns_400(client):
    r = client.post(
        "/export_chart",
        content='{"hr_timeline":[70,NaN],"timestamps":[0,1],"avg_hr":70,"min_hr":70,"max_hr":70}',
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 400


# --------------------------------------------------------------------------- #
# CORS
# --------------------------------------------------------------------------- #


def test_cors_defaults_exclude_wildcard_and_credentials():
    cors_middleware = next(
        mw for mw in appmod.app.user_middleware if mw.cls.__name__ == "CORSMiddleware"
    )
    assert cors_middleware.kwargs["allow_origins"] != ["*"]
    assert cors_middleware.kwargs.get("allow_credentials", False) is False
