"""
Test trained PhysNet on video
"""

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection import FaceDetector
from src.models.physnet import PhysNetLite
from src.processing import BandpassFilter, FFTAnalyzer


def test_physnet(video_path: str, model_path: str = "models/physnet_trained.pth"):
    """Test trained PhysNet on a video."""

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = PhysNetLite()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded model from: {model_path}")

    # Face detector
    face_detector = FaceDetector()

    # FFT analyzer
    fft = FFTAnalyzer(fps=30.0)
    bandpass = BandpassFilter(fps=30.0)

    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {video_path}")
    print(f"Frames: {total_frames}, FPS: {fps}")

    # Collect frames
    frames = []
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect face
        detection = face_detector.detect(frame)
        if detection is not None:
            x, y, w, h = detection["bbox"]
            face = frame[y : y + h, x : x + w]
        else:
            face = frame

        # Resize
        face = cv2.resize(face, (64, 64))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype(np.float32) / 255.0
        face = (face - mean) / std

        frames.append(face)

    cap.release()
    print(f"Collected {len(frames)} frames")

    # Process in clips
    clip_length = 64
    stride = 32
    all_signals = []
    heart_rates = []

    for start in range(0, len(frames) - clip_length, stride):
        clip_frames = frames[start : start + clip_length]
        clip = np.array(clip_frames).transpose(3, 0, 1, 2).astype(np.float32)
        clip = clip[np.newaxis, ...]

        with torch.no_grad():
            clip_tensor = torch.from_numpy(clip).to(device)
            signal = model(clip_tensor)
            signal = signal.cpu().numpy()[0]

        # Apply bandpass
        signal = bandpass.filter(signal)
        all_signals.append(signal)

        # Get heart rate
        hr, conf = fft.get_heart_rate(signal)
        heart_rates.append(hr)

    # Average signal
    avg_hr = np.mean(heart_rates)
    std_hr = np.std(heart_rates)

    print("\n=== PhysNet Results ===")
    print(f"Average Heart Rate: {avg_hr:.1f} ± {std_hr:.1f} BPM")
    print(f"Range: {np.min(heart_rates):.1f} - {np.max(heart_rates):.1f} BPM")

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

    # Heart rates over time
    times = np.arange(len(heart_rates)) * stride / fps
    ax1.plot(times, heart_rates, "b-", linewidth=1.5)
    ax1.axhline(avg_hr, color="r", linestyle="--", label=f"Avg: {avg_hr:.1f} BPM")
    ax1.set_ylabel("Heart Rate (BPM)")
    ax1.set_title("PhysNet Heart Rate Detection")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Sample signal
    ax2.plot(all_signals[len(all_signals) // 2], "g-")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("rPPG Signal")
    ax2.set_title("Extracted rPPG Signal (middle clip)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("physnet_results.png", dpi=150)
    print("\nSaved plot to: physnet_results.png")

    return avg_hr, heart_rates


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Path to video")
    parser.add_argument("--model", default="models/physnet_trained.pth", help="Model path")
    args = parser.parse_args()

    test_physnet(args.video, args.model)
