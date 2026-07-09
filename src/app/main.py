"""
rPPG Real-Time Dashboard - FastAPI Backend
WebSocket-based real-time heart rate monitoring
"""

import asyncio
import base64
import tempfile
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from fastapi import FastAPI, File, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
static_path = Path(__file__).parent / "static"
static_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


# Global heart rate monitor
monitor: HeartRateMonitor | None = None


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
    # Save to temp file
    temp_path = Path(tempfile.gettempdir()) / f"rppg_{file.filename}"

    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Get video info
    cap = cv2.VideoCapture(str(temp_path))
    if not cap.isOpened():
        return {"error": "Could not open video"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frames / fps if fps > 0 else 0
    cap.release()

    return {
        "filename": file.filename,
        "path": str(temp_path),
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

        if not video_path or not Path(video_path).exists():
            await websocket.send_json({"error": "Video not found"})
            return

        # Initialize monitor
        cap = cv2.VideoCapture(video_path)
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
                        "progress": round(frame_idx / total_frames * 100, 1),
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

        cap.release()

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

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.exception("Error during WebSocket processing")
        await websocket.send_json({"error": str(e)})


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "message": "rPPG server running"}


# Global monitor for webcam mode
webcam_monitor: HeartRateMonitor | None = None
webcam_heart_rates = []


@app.post("/process_frame")
async def process_frame(request: Request):
    """Process a single frame from webcam."""
    global webcam_monitor, webcam_heart_rates

    data = await request.json()
    frame_b64 = data.get("frame")

    if not frame_b64:
        return {"error": "No frame data"}

    # Decode frame
    try:
        frame_data = base64.b64decode(frame_b64)
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"error": "Could not decode frame"}

        # Initialize monitor if needed
        if webcam_monitor is None:
            webcam_monitor = HeartRateMonitor(fps=30.0, method="chrom")
            webcam_heart_rates = []

        # Process frame
        result = webcam_monitor.process_frame(frame)

        # Store HR
        hr = result["heart_rate"]
        if hr > 0:
            webcam_heart_rates.append(hr)
            if len(webcam_heart_rates) > 300:  # Keep last 10 seconds at 30 fps
                webcam_heart_rates.pop(0)

        avg_hr = np.mean(webcam_heart_rates) if webcam_heart_rates else 0

        return {
            "heart_rate": round(hr, 1),
            "avg_heart_rate": round(avg_hr, 1),
            "confidence": round(result["confidence"], 2),
            "status": result["status"],
        }

    except Exception as e:
        return {"error": str(e)}


@app.post("/reset_webcam")
async def reset_webcam():
    """Reset webcam monitor state."""
    global webcam_monitor, webcam_heart_rates
    webcam_monitor = None
    webcam_heart_rates = []
    return {"status": "reset"}


@app.post("/export_chart")
async def export_chart(request: Request):
    """
    Export heart rate chart as PNG image.
    Generates a report similar to physnet_results.png
    """
    try:
        data = await request.json()
        hr_timeline = data.get("hr_timeline", [])
        timestamps = data.get("timestamps", [])
        avg_hr = data.get("avg_hr", 0)
        min_hr = data.get("min_hr", 0)
        max_hr = data.get("max_hr", 0)

        if not hr_timeline:
            return {"error": "No data to export"}

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

        # Save to temp file and return
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        temp_file.write(buf.getvalue())
        temp_file.close()

        return FileResponse(
            temp_file.name, media_type="image/png", filename=f"rppg_report_{int(time.time())}.png"
        )

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":

    import uvicorn

    print("Starting rPPG Dashboard Server...")
    print("Open http://localhost:8000 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8000)
