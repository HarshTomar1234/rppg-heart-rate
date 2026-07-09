"""
rPPG Video Demo - Production Grade
Test the heart rate detection on a video file

Usage:
    python demo_video.py path/to/video.mp4
    python demo_video.py path/to/video.mp4 --output output.mp4
    python demo_video.py path/to/video.mp4 --no-preview  # Headless mode
    python demo_video.py path/to/video.mp4 --method pos  # Use POS method
"""

import argparse
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")  # Use non-interactive backend
# Allow running the script directly from a checkout without installing the package.
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from src.vitals import HeartRateMonitor


def process_video(
    video_path: str, output_path: str = None, show_preview: bool = True, method: str = "chrom"
):
    """
    Process a video file and extract heart rate.

    Args:
        video_path: Path to input video
        output_path: Optional path to save output video
        show_preview: Whether to show preview window
        method: Signal extraction method ('chrom', 'pos', 'auto')
    """
    print("=" * 50)
    print("rPPG - Contactless Heart Rate Detection")
    print(f"Method: {method.upper()}")
    print("=" * 50)

    # Open video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nVideo: {video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {total_frames/fps:.2f} seconds")
    print("-" * 50)

    # Initialize heart rate monitor with selected method
    monitor = HeartRateMonitor(fps=fps, method=method)

    # Setup video writer if output specified
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Results storage
    heart_rates = []
    confidences = []
    timestamps = []

    print("\nProcessing video...")
    if show_preview:
        print("Press 'q' to quit preview, 'r' to reset\n")

    frame_idx = 0

    # Check if display is available
    display_available = show_preview
    if show_preview:
        try:
            cv2.namedWindow("rPPG Demo", cv2.WINDOW_NORMAL)
        except cv2.error:
            print("Warning: Display not available, running in headless mode")
            display_available = False

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Process frame
        result = monitor.process_frame(frame)

        # Get annotated frame
        display_frame = result["frame_annotated"]

        # Add status text
        status = result["status"]
        cv2.putText(
            display_frame,
            status,
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # Store results
        if result["heart_rate"] > 0:
            heart_rates.append(result["heart_rate"])
            confidences.append(result["confidence"])
            timestamps.append(frame_idx / fps)

        # Write to output
        if writer:
            writer.write(display_frame)

        # Show preview if display is available
        if display_available:
            try:
                cv2.imshow("rPPG Demo", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    monitor.reset()
                    heart_rates.clear()
                    confidences.clear()
                    timestamps.clear()
                    print("Reset!")
            except cv2.error:
                display_available = False

        # Progress
        frame_idx += 1
        if frame_idx % 30 == 0:
            progress = frame_idx / total_frames * 100
            current_hr = result["heart_rate"]
            current_conf = result["confidence"]
            print(
                f"Progress: {progress:.1f}% | Frame: {frame_idx}/{total_frames} | HR: {current_hr:.1f} BPM | Conf: {current_conf:.2f}"
            )

    # Cleanup
    cap.release()
    if writer:
        writer.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass

    # Print summary
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)

    if len(heart_rates) > 0:
        avg_hr = np.mean(heart_rates)
        std_hr = np.std(heart_rates)
        min_hr = np.min(heart_rates)
        max_hr = np.max(heart_rates)
        avg_conf = np.mean(confidences)

        print(f"Average Heart Rate: {avg_hr:.1f} ± {std_hr:.1f} BPM")
        print(f"Range: {min_hr:.1f} - {max_hr:.1f} BPM")
        print(f"Average Confidence: {avg_conf:.2f}")
        print(f"Total measurements: {len(heart_rates)}")

        # Plot results
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        # Heart rate plot
        ax1.plot(timestamps, heart_rates, "b-", linewidth=1.5, label="Heart Rate")
        ax1.axhline(y=avg_hr, color="r", linestyle="--", label=f"Average: {avg_hr:.1f} BPM")
        ax1.fill_between(timestamps, avg_hr - std_hr, avg_hr + std_hr, alpha=0.2, color="red")
        ax1.set_ylabel("Heart Rate (BPM)")
        ax1.set_title(f"rPPG Heart Rate Detection ({method.upper()} method)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([max(40, min_hr - 10), min(180, max_hr + 10)])

        # Confidence plot
        ax2.plot(timestamps, confidences, "g-", linewidth=1.5, label="Confidence")
        ax2.axhline(y=0.5, color="orange", linestyle=":", label="Threshold")
        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel("Confidence")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])

        plt.tight_layout()

        # Save plot
        plot_path = Path(video_path).stem + "_heart_rate.png"
        plt.savefig(plot_path, dpi=150)
        print(f"\nPlot saved to: {plot_path}")

        # Try to show plot
        try:
            plt.show()
        except Exception:
            print("(Plot display not available, saved to file)")
    else:
        print("No heart rate measurements recorded.")
        print("Make sure the video contains a clearly visible face.")


def main():
    parser = argparse.ArgumentParser(description="rPPG Heart Rate Detection from Video")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--output", "-o", help="Path to save output video")
    parser.add_argument("--no-preview", action="store_true", help="Disable preview window")
    parser.add_argument(
        "--method",
        "-m",
        choices=["chrom", "pos", "auto"],
        default="chrom",
        help="Signal extraction method",
    )

    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        return

    process_video(
        video_path=args.video,
        output_path=args.output,
        show_preview=not args.no_preview,
        method=args.method,
    )


if __name__ == "__main__":
    main()
