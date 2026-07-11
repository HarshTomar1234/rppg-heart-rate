"""
UBFC-rPPG Dataset Loader
Loads and processes UBFC-rPPG dataset for training PhysNet
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..logging_config import get_logger

logger = get_logger("data.ubfc_loader")


@dataclass
class UBFCSubject:
    """Represents one subject's data from UBFC-rPPG dataset."""

    subject_id: str
    video_path: str
    gt_ppg: np.ndarray | None  # Ground truth PPG signal
    gt_hr: np.ndarray | None  # Ground truth heart rate
    gt_time: np.ndarray | None  # Time stamps
    fps: float
    total_frames: int
    has_ground_truth: bool
    dataset_version: str  # "DATASET_1" or "DATASET_2"


class UBFCDatasetLoader:
    """
    Loader for UBFC-rPPG dataset.
    Handles both DATASET_1 (gtdump.xmp) and DATASET_2 (ground_truth.txt) formats.
    """

    def __init__(self, dataset_root: str):
        """
        Initialize loader.

        Args:
            dataset_root: Path to datasets folder containing DATASET_1, DATASET_2
        """
        self.dataset_root = Path(dataset_root)
        self.subjects: list[UBFCSubject] = []
        self._scan_datasets()

    def _scan_datasets(self):
        """Scan for all available subjects."""
        logger.info("Scanning dataset at: %s", self.dataset_root)

        # Scan DATASET_1
        dataset1_path = self.dataset_root / "DATASET_1"
        if dataset1_path.exists():
            self._scan_dataset1(dataset1_path)

        # Scan DATASET_2
        dataset2_path = self.dataset_root / "DATASET_2"
        if dataset2_path.exists():
            self._scan_dataset2(dataset2_path)

        with_gt = sum(1 for s in self.subjects if s.has_ground_truth)
        logger.info(
            "Total subjects found: %d (with ground truth: %d, without: %d)",
            len(self.subjects),
            with_gt,
            len(self.subjects) - with_gt,
        )

    def _scan_dataset1(self, path: Path):
        """Scan DATASET_1 format."""
        for folder in sorted(path.iterdir()):
            if not folder.is_dir():
                continue

            video_path = folder / "vid.avi"
            gt_path = folder / "gtdump.xmp"

            # Check if video exists
            has_video = video_path.exists()
            has_gt = gt_path.exists()

            if not has_video and not has_gt:
                continue

            # Load ground truth if available
            gt_ppg, gt_hr, gt_time = None, None, None
            if has_gt:
                try:
                    gt_data = pd.read_csv(gt_path, header=None).values
                    gt_time = gt_data[:, 0] / 1000  # ms to seconds
                    gt_hr = gt_data[:, 1]
                    gt_ppg = gt_data[:, 3]
                except Exception as e:
                    logger.warning("Could not read %s: %s", gt_path, e)
                    has_gt = False

            # Get video info
            fps, total_frames = 30.0, 0
            if has_video:
                cap = cv2.VideoCapture(str(video_path))
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

            subject = UBFCSubject(
                subject_id=f"D1_{folder.name}",
                video_path=str(video_path) if has_video else "",
                gt_ppg=gt_ppg,
                gt_hr=gt_hr,
                gt_time=gt_time,
                fps=fps,
                total_frames=total_frames,
                has_ground_truth=has_gt and gt_ppg is not None,
                dataset_version="DATASET_1",
            )

            if has_video or has_gt:
                self.subjects.append(subject)
                status = "[OK]" if has_video and has_gt else "[WARN]"
                logger.info("%s %s: video=%s, gt=%s", status, folder.name, has_video, has_gt)

    def _scan_dataset2(self, path: Path):
        """Scan DATASET_2 format."""
        for folder in sorted(path.iterdir()):
            if not folder.is_dir():
                continue

            video_path = folder / "vid.avi"
            gt_path = folder / "ground_truth.txt"

            has_video = video_path.exists()
            has_gt = gt_path.exists()

            if not has_video and not has_gt:
                continue

            # Load ground truth if available
            gt_ppg, gt_hr, gt_time = None, None, None
            if has_gt:
                try:
                    gt_data = np.loadtxt(gt_path)
                    gt_ppg = gt_data[0, :]
                    gt_hr = gt_data[1, :]
                    gt_time = gt_data[2, :]
                except Exception as e:
                    logger.warning("Could not read %s: %s", gt_path, e)
                    has_gt = False

            # Get video info
            fps, total_frames = 30.0, 0
            if has_video:
                cap = cv2.VideoCapture(str(video_path))
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

            subject = UBFCSubject(
                subject_id=f"D2_{folder.name}",
                video_path=str(video_path) if has_video else "",
                gt_ppg=gt_ppg,
                gt_hr=gt_hr,
                gt_time=gt_time,
                fps=fps,
                total_frames=total_frames,
                has_ground_truth=has_gt and gt_ppg is not None,
                dataset_version="DATASET_2",
            )

            if has_video or has_gt:
                self.subjects.append(subject)
                status = "[OK]" if has_video and has_gt else "[WARN]"
                logger.info("%s %s: video=%s, gt=%s", status, folder.name, has_video, has_gt)

    def get_trainable_subjects(self) -> list[UBFCSubject]:
        """Get subjects that have both video and ground truth."""
        return [s for s in self.subjects if s.video_path and s.has_ground_truth]

    def get_video_only_subjects(self) -> list[UBFCSubject]:
        """Get subjects with video but no ground truth (for pseudo-labeling)."""
        return [s for s in self.subjects if s.video_path and not s.has_ground_truth]

    def summary(self) -> dict:
        """Get dataset summary."""
        trainable = self.get_trainable_subjects()
        video_only = self.get_video_only_subjects()

        return {
            "total_subjects": len(self.subjects),
            "trainable": len(trainable),
            "video_only": len(video_only),
            "trainable_ids": [s.subject_id for s in trainable],
            "video_only_ids": [s.subject_id for s in video_only],
        }


def resample_gt_to_frames(
    gt_signal: np.ndarray, gt_time: np.ndarray, num_frames: int, fps: float
) -> np.ndarray:
    """
    Resample ground truth signal to match video frame rate.

    Args:
        gt_signal: Ground truth PPG signal
        gt_time: Ground truth timestamps
        num_frames: Number of video frames
        fps: Video frame rate

    Returns:
        Resampled signal matching video frames
    """
    # Create frame timestamps
    frame_times = np.arange(num_frames) / fps

    # Interpolate ground truth to frame times
    resampled = np.interp(frame_times, gt_time, gt_signal)

    # Normalize
    resampled = resampled - np.mean(resampled)
    std = np.std(resampled)
    if std > 0:
        resampled = resampled / std

    return resampled


def compute_windowed_gt_hr(
    gt_signal: np.ndarray,
    gt_time: np.ndarray,
    num_frames: int,
    fps: float,
    window_frames: int = 180,
    stride_frames: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ground-truth BPM per sliding window, using the SAME windowed-FFT
    procedure the production pipeline applies to its own extracted pulse signal.

    This keeps the comparison symmetric: rather than trusting the pulse oximeter's
    built-in HR channel (which has its own, unknown, internal smoothing), the raw PPG
    waveform is resampled to the video's frame grid and then windowed/bandpassed/FFT'd
    exactly like ``SignalProcessor.get_pulse_signal()`` + ``FFTAnalyzer`` does for the
    camera-derived signal. Any remaining discrepancy is then attributable to the rPPG
    extraction itself, not to a windowing or smoothing mismatch between the two sides.

    Only full windows are emitted (the first output is anchored at frame index
    ``window_frames - 1``), matching a strict 6-second buffer rather than the live
    monitor's shorter early-warmup estimates — this keeps evaluation windows uniform.

    Args:
        gt_signal: Raw ground-truth PPG waveform.
        gt_time: Ground-truth sample timestamps, in seconds.
        num_frames: Number of video frames.
        fps: Video frame rate.
        window_frames: Window size in frames (default 180 = 6s at 30fps, matching
            ``HeartRateMonitor``'s default buffer).
        stride_frames: Step between windows in frames (default 30 = 1s at 30fps).

    Returns:
        Tuple of (frame_indices, gt_bpm) — frame_indices are the last frame index of
        each window (aligned to when a live estimate at that frame would be available).
    """
    # Local imports: avoids a module-level src.data -> src.processing import cycle risk
    # and keeps this dataset-analysis helper's dependency footprint explicit.
    from ..processing.fft_analyzer import FFTAnalyzer
    from ..processing.filters import BandpassFilter, detrend_signal

    frame_aligned = resample_gt_to_frames(gt_signal, gt_time, num_frames, fps)
    bandpass = BandpassFilter(fps=fps)
    fft_analyzer = FFTAnalyzer(fps=fps)

    frame_indices = []
    gt_bpm = []
    for end in range(window_frames - 1, num_frames, stride_frames):
        start = end - window_frames + 1
        window = frame_aligned[start : end + 1]

        pulse = detrend_signal(window)
        pulse = bandpass.filter(pulse)
        bpm, _confidence = fft_analyzer.get_heart_rate(pulse)

        frame_indices.append(end)
        gt_bpm.append(bpm)

    return np.array(frame_indices), np.array(gt_bpm)


if __name__ == "__main__":
    # Test the loader
    import sys

    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "datasets"

    print("=" * 60)
    print("UBFC-rPPG Dataset Loader")
    print("=" * 60)

    loader = UBFCDatasetLoader(dataset_path)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    summary = loader.summary()
    print(f"Total subjects: {summary['total_subjects']}")
    print(f"Trainable (video + GT): {summary['trainable']}")
    print(f"  IDs: {summary['trainable_ids']}")
    print(f"Video only (for pseudo-labels): {summary['video_only']}")
    print(f"  IDs: {summary['video_only_ids']}")
