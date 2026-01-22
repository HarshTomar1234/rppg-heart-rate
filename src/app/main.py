"""
rPPG Real-Time Dashboard - FastAPI Backend
WebSocket-based real-time heart rate monitoring
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import asyncio
import json
import base64
from pathlib import Path
import tempfile
import time
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.vitals import HeartRateMonitor

app = FastAPI(
    title="rPPG - Contactless Heart Rate Monitor",
    description="Real-time heart rate detection from video",
    version="1.0.0"
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
monitor: Optional[HeartRateMonitor] = None


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the dashboard HTML."""
    html_path = static_path / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse(content="""
    <html>
        <head><title>rPPG Dashboard</title></head>
        <body>
            <h1>rPPG Dashboard</h1>
            <p>Static files not found. Please create src/app/static/index.html</p>
        </body>
    </html>
    """)


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
        "duration": round(duration, 2)
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
        
        monitor = HeartRateMonitor(fps=fps, method='chrom')
        
        await websocket.send_json({
            "type": "info",
            "fps": fps,
            "total_frames": total_frames
        })
        
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
            hr = result['heart_rate']
            if hr > 0:
                heart_rates.append(hr)
            
            # Send update every 10 frames
            if frame_idx % 10 == 0:
                # Encode frame as base64
                _, buffer = cv2.imencode('.jpg', result['frame_annotated'], 
                                         [cv2.IMWRITE_JPEG_QUALITY, 50])
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                
                avg_hr = np.mean(heart_rates) if heart_rates else 0
                
                await websocket.send_json({
                    "type": "frame",
                    "frame_idx": frame_idx,
                    "total_frames": total_frames,
                    "progress": round(frame_idx / total_frames * 100, 1),
                    "heart_rate": round(hr, 1),
                    "avg_heart_rate": round(avg_hr, 1),
                    "confidence": round(result['confidence'], 2),
                    "status": result['status'],
                    "frame": frame_b64
                })
                
                # Small delay for browser to process
                await asyncio.sleep(0.01)
            
            frame_idx += 1
        
        cap.release()
        
        # Send final results
        if heart_rates:
            await websocket.send_json({
                "type": "complete",
                "avg_heart_rate": round(np.mean(heart_rates), 1),
                "min_heart_rate": round(np.min(heart_rates), 1),
                "max_heart_rate": round(np.max(heart_rates), 1),
                "std_heart_rate": round(np.std(heart_rates), 1),
                "total_measurements": len(heart_rates)
            })
        
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        await websocket.send_json({"error": str(e)})


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "message": "rPPG server running"}


if __name__ == "__main__":
    import uvicorn
    print("Starting rPPG Dashboard Server...")
    print("Open http://localhost:8000 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8000)
