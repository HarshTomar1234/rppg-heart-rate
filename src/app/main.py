"""
rPPG Real-Time Dashboard - FastAPI Backend
WebSocket-based real-time heart rate monitoring
"""

import asyncio
import base64
import math
import os
import time
import uuid
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

matplotlib.use("Agg")  # Non-interactive backend
import sys
from io import BytesIO

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.logging_config import get_logger
from src.vitals import HeartRateMonitor

logger = get_logger("app.main")


app = FastAPI(
    title="rPPG - Contactless Heart Rate Monitor",
    description="Real-time heart rate detection from video",
    version="1.0.0",
)

# CORS
# No cookie/session-based auth exists anywhere in this app, so allow_credentials
# is intentionally omitted (Starlette defaults to False) -- combining a wildcard
# origin with credentials is a classic footgun that becomes exploitable the moment
# someone "fixes" allow_origins into a reflected-origin pattern later. Origins
# default to localhost only; any non-local deployment must set RPPG_CORS_ORIGINS
# explicitly rather than falling back to a wildcard.
_default_cors_origins = "http://localhost:8000,http://127.0.0.1:8000"
_cors_origins = [
    origin.strip()
    for origin in os.getenv("RPPG_CORS_ORIGINS", _default_cors_origins).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
static_path = Path(__file__).parent / "static"
static_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Uploaded videos live in their own directory, deliberately NOT under static_path --
# that tree is served publicly via the /static mount above, so anything placed
# there becomes directly downloadable by anyone.
UPLOAD_DIR = Path(os.getenv("RPPG_UPLOAD_DIR", str(Path(__file__).parent / "uploads")))
UPLOAD_DIR.mkdir(exist_ok=True)

MAX_UPLOAD_BYTES = int(os.getenv("RPPG_MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))  # 200MB default
ALLOWED_UPLOAD_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm"}
ALLOWED_UPLOAD_CONTENT_TYPES = {
    "video/mp4",
    "video/quicktime",
    "video/x-msvideo",
    "video/webm",
}
UPLOAD_MAX_AGE_SECONDS = int(
    os.getenv("RPPG_UPLOAD_MAX_AGE_SECONDS", str(24 * 60 * 60))
)  # 24h default


def _sweep_stale_uploads() -> None:
    """Opportunistic cleanup of old uploads -- runs on each /upload call rather than
    via a separate scheduler, which is unnecessary infrastructure at this app's scale."""
    now = time.time()
    for f in UPLOAD_DIR.iterdir():
        try:
            if f.is_file() and (now - f.stat().st_mtime) > UPLOAD_MAX_AGE_SECONDS:
                f.unlink()
        except OSError:
            pass  # best-effort; another request may have already removed it


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the dashboard HTML."""
    html_path = static_path / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse(
        content="""
    <html>
        <head><title>rPPG Dashboard</title></head>
        <body>
            <h1>rPPG Dashboard</h1>
            <p>Static files not found. Please create src/app/static/index.html</p>
        </body>
    </html>
    """
    )


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video for processing."""
    _sweep_stale_uploads()

    # The client-supplied filename (from the Content-Disposition header) is never
    # used for the storage path -- only its extension is inspected, and a fresh
    # server-generated name is what actually touches the filesystem. This is what
    # neutralizes filename-based path traversal, not string sanitization of the
    # original name.
    original_name = file.filename or ""
    extension = Path(original_name).suffix.lower()
    if (
        extension not in ALLOWED_UPLOAD_EXTENSIONS
        or file.content_type not in ALLOWED_UPLOAD_CONTENT_TYPES
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {sorted(ALLOWED_UPLOAD_EXTENSIONS)}",
        )

    stored_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{extension}"

    bytes_written = 0
    chunk_size = 1024 * 1024  # 1MB
    try:
        with open(stored_path, "wb") as f:
            while chunk := await file.read(chunk_size):
                bytes_written += len(chunk)
                if bytes_written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {MAX_UPLOAD_BYTES} byte upload limit",
                    )
                f.write(chunk)
    except HTTPException:
        stored_path.unlink(missing_ok=True)
        raise
    except Exception:
        # Any other failure mid-write (disk full, client disconnect, etc.) must not
        # leave a partial file behind either -- HTTPException isn't the only way
        # this loop can fail.
        stored_path.unlink(missing_ok=True)
        logger.exception("Upload write failed")
        raise HTTPException(status_code=500, detail="Upload failed") from None

    # Get video info
    cap = cv2.VideoCapture(str(stored_path))
    if not cap.isOpened():
        cap.release()
        stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Could not open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frames / fps if fps > 0 else 0
    cap.release()

    return {
        "filename": original_name,
        "path": str(stored_path),
        "fps": fps,
        "frames": frames,
        "duration": round(duration, 2),
    }


@app.websocket("/ws/process")
async def websocket_process(websocket: WebSocket):
    """WebSocket endpoint for real-time video processing."""
    await websocket.accept()

    try:
        # Wait for video path
        data = await websocket.receive_json()
        video_path = data.get("video_path")

        # Confine processing to files under UPLOAD_DIR -- resolve() collapses any
        # ".."/symlink components before the containment check, so this can't be
        # spoofed by a crafted relative path the way string-prefix matching could.
        # The error message is deliberately identical for "doesn't exist" and
        # "exists but outside UPLOAD_DIR": diverging them would let a client use the
        # response to probe which paths exist anywhere on the server filesystem.
        requested = Path(video_path).resolve() if video_path else None
        if (
            not requested
            or not requested.is_file()
            or not requested.is_relative_to(UPLOAD_DIR.resolve())
        ):
            await websocket.send_json({"error": "Video not found"})
            return

        # Initialize monitor
        cap = cv2.VideoCapture(str(requested))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            monitor = HeartRateMonitor(fps=fps, method="chrom")

            await websocket.send_json({"type": "info", "fps": fps, "total_frames": total_frames})

            # Process frames
            frame_idx = 0
            heart_rates = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                result = monitor.process_frame(frame)

                # Store HR
                hr = result["heart_rate"]
                if hr > 0:
                    heart_rates.append(hr)

                # Send update every 10 frames
                if frame_idx % 10 == 0:
                    # Encode frame as base64
                    _, buffer = cv2.imencode(
                        ".jpg", result["frame_annotated"], [cv2.IMWRITE_JPEG_QUALITY, 50]
                    )
                    frame_b64 = base64.b64encode(buffer).decode("utf-8")

                    avg_hr = np.mean(heart_rates) if heart_rates else 0

                    await websocket.send_json(
                        {
                            "type": "frame",
                            "frame_idx": frame_idx,
                            "total_frames": total_frames,
                            # Some containers (e.g. certain .webm encodes) report a
                            # zero frame count from cv2.CAP_PROP_FRAME_COUNT; guard
                            # against dividing by that rather than aborting the
                            # whole session over a cosmetic progress percentage.
                            "progress": (
                                round(frame_idx / total_frames * 100, 1) if total_frames else 0
                            ),
                            "heart_rate": round(hr, 1),
                            "avg_heart_rate": round(avg_hr, 1),
                            "confidence": round(result["confidence"], 2),
                            "status": result["status"],
                            "frame": frame_b64,
                        }
                    )

                    # Small delay for browser to process
                    await asyncio.sleep(0.01)

                frame_idx += 1

            # Send final results
            if heart_rates:
                await websocket.send_json(
                    {
                        "type": "complete",
                        "avg_heart_rate": round(np.mean(heart_rates), 1),
                        "min_heart_rate": round(np.min(heart_rates), 1),
                        "max_heart_rate": round(np.max(heart_rates), 1),
                        "std_heart_rate": round(np.std(heart_rates), 1),
                        "total_measurements": len(heart_rates),
                    }
                )
        finally:
            cap.release()

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.exception("Error during WebSocket processing")
        await websocket.send_json({"error": str(e)})


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "message": "rPPG server running"}


def _decode_webcam_frame(frame_b64: str) -> np.ndarray | None:
    """Decode a base64-encoded JPEG frame into a BGR image array, or None on failure.

    A malformed base64 payload must degrade to "skip this one frame" (matching the
    already-established behavior for a frame cv2 fails to decode), not propagate out
    and kill the whole WebSocket session -- unlike the old per-request /process_frame
    design, one bad frame here would otherwise end the entire connection.
    """
    try:
        frame_data = base64.b64decode(frame_b64, validate=True)
    except (ValueError, TypeError):
        return None
    nparr = np.frombuffer(frame_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


@app.websocket("/ws/webcam")
async def websocket_webcam(websocket: WebSocket):
    """WebSocket endpoint for real-time webcam heart rate monitoring.

    Each connection gets its own HeartRateMonitor and HR history -- mirroring the
    already-correct per-connection pattern in websocket_process (above). This
    replaces a POST-polling design (formerly /process_frame + /reset_webcam) that
    shared module-level global state across every concurrent user, corrupting one
    user's readings with another's and letting one user's "reset" wipe everyone's
    session. Connection lifetime now IS session lifetime: closing the socket is the
    reset, no separate reset endpoint needed.
    """
    await websocket.accept()
    monitor: HeartRateMonitor | None = None
    heart_rates: list[float] = []

    try:
        while True:
            data = await websocket.receive_json()
            frame_b64 = data.get("frame")
            if not frame_b64:
                await websocket.send_json({"error": "No frame data"})
                continue

            frame = _decode_webcam_frame(frame_b64)
            if frame is None:
                await websocket.send_json({"error": "Could not decode frame"})
                continue

            if monitor is None:
                monitor = HeartRateMonitor(fps=30.0, method="chrom")

            result = monitor.process_frame(frame)

            hr = result["heart_rate"]
            if hr > 0:
                heart_rates.append(hr)
                if len(heart_rates) > 300:  # Keep last 10 seconds at 30 fps
                    heart_rates.pop(0)

            avg_hr = np.mean(heart_rates) if heart_rates else 0

            await websocket.send_json(
                {
                    "heart_rate": round(hr, 1),
                    "avg_heart_rate": round(avg_hr, 1),
                    "confidence": round(result["confidence"], 2),
                    "status": result["status"],
                }
            )
    except WebSocketDisconnect:
        logger.info("Webcam client disconnected")
    except Exception:
        logger.exception("Error during webcam WebSocket processing")


def _validate_chart_payload(data: dict) -> tuple[list[float], list[float], float, float, float]:
    """Validate /export_chart's JSON body. Raises HTTPException(400) on any problem.

    Guards against malformed input reaching matplotlib mid-render, which previously
    surfaced as a 200-status JSON error body that the frontend's `response.ok` check
    would treat as success and try to download as a broken PNG.
    """
    hr_timeline = data.get("hr_timeline")
    timestamps = data.get("timestamps")
    avg_hr = data.get("avg_hr", 0)
    min_hr = data.get("min_hr", 0)
    max_hr = data.get("max_hr", 0)

    if not isinstance(hr_timeline, list) or not isinstance(timestamps, list):
        raise HTTPException(status_code=400, detail="hr_timeline and timestamps must be arrays")
    if not hr_timeline or not timestamps:
        raise HTTPException(status_code=400, detail="No data to export")
    if len(hr_timeline) != len(timestamps):
        raise HTTPException(
            status_code=400, detail="hr_timeline and timestamps must be the same length"
        )

    def _finite_floats(values, name: str) -> list[float]:
        try:
            floats = [float(v) for v in values]
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400, detail=f"{name} must contain only numbers"
            ) from exc
        if not all(math.isfinite(v) for v in floats):
            raise HTTPException(status_code=400, detail=f"{name} must contain only finite numbers")
        return floats

    hr_timeline = _finite_floats(hr_timeline, "hr_timeline")
    timestamps = _finite_floats(timestamps, "timestamps")

    for name, value in (("avg_hr", avg_hr), ("min_hr", min_hr), ("max_hr", max_hr)):
        if not isinstance(value, int | float) or not math.isfinite(value):
            raise HTTPException(status_code=400, detail=f"{name} must be a finite number")

    return hr_timeline, timestamps, float(avg_hr), float(min_hr), float(max_hr)


@app.post("/export_chart")
async def export_chart(request: Request):
    """
    Export heart rate chart as PNG image.
    Generates a report similar to physnet_results.png
    """
    data = await request.json()
    hr_timeline, timestamps, avg_hr, min_hr, max_hr = _validate_chart_payload(data)

    try:
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle("rPPG Heart Rate Detection Report", fontsize=16, fontweight="bold")

        # Plot 1: Heart Rate Timeline
        ax1.plot(timestamps, hr_timeline, "b-", linewidth=2, label="Heart Rate")
        ax1.axhline(
            y=avg_hr, color="r", linestyle="--", linewidth=1.5, label=f"Avg: {avg_hr:.1f} BPM"
        )
        ax1.fill_between(timestamps, min_hr, max_hr, alpha=0.2, color="blue")
        ax1.set_xlabel("Time (seconds)", fontsize=12)
        ax1.set_ylabel("Heart Rate (BPM)", fontsize=12)
        ax1.set_title("Heart Rate Over Time", fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right")
        ax1.set_ylim(40, 180)

        # Plot 2: Signal Quality Info
        ax2.text(
            0.5,
            0.7,
            "Session Summary",
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            transform=ax2.transAxes,
        )
        ax2.text(
            0.5,
            0.5,
            f"Average HR: {avg_hr:.1f} BPM",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax2.transAxes,
        )
        ax2.text(
            0.5,
            0.35,
            f"Range: {min_hr:.1f} - {max_hr:.1f} BPM",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax2.transAxes,
        )
        ax2.text(
            0.5,
            0.2,
            f"Samples: {len(hr_timeline)}",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax2.transAxes,
        )
        ax2.text(
            0.5,
            0.05,
            "Research demonstration only. Not medically validated.",
            ha="center",
            va="center",
            fontsize=10,
            style="italic",
            transform=ax2.transAxes,
            color="gray",
        )
        ax2.axis("off")

        plt.tight_layout()

        # Save to bytes
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)

        # Stream the already-in-memory PNG bytes directly -- no reason for this to
        # touch disk at all, which sidesteps the temp-file-cleanup problem entirely
        # rather than needing a BackgroundTask to remember to clean one up.
        return Response(
            content=buf.getvalue(),
            media_type="image/png",
            headers={
                "Content-Disposition": f'attachment; filename="rppg_report_{int(time.time())}.png"'
            },
        )

    except HTTPException:
        raise
    except Exception:
        logger.exception("Chart export failed")
        raise HTTPException(status_code=500, detail="Chart export failed") from None


if __name__ == "__main__":

    import uvicorn

    # Loopback-only by default -- binding to all interfaces (0.0.0.0) is an
    # explicit opt-in for containerized/production deployment, not a hardcoded
    # default, since this dev server has no auth layer of its own.
    host = os.getenv("RPPG_HOST", "127.0.0.1")
    port = int(os.getenv("RPPG_PORT", "8000"))

    print("Starting rPPG Dashboard Server...")
    print(f"Open http://{host}:{port} in your browser")
    uvicorn.run(app, host=host, port=port)
