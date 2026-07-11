"""Validate the ground-truth windowing helper used for honest classical-pipeline
evaluation against real UBFC-rPPG data (Phase 1).

Uses only numpy/scipy — no pandas/cv2/mediapipe required — so this stays a fast,
CI-safe unit test even though ubfc_loader.py itself needs pandas/cv2 to load real
UBFC files.
"""

import numpy as np
import pytest

pytest.importorskip("pandas")
pytest.importorskip("cv2")

from src.data.ubfc_loader import compute_windowed_gt_hr, resample_gt_to_frames

FPS = 30.0


def _synthetic_ppg(bpm: float, duration_s: float = 12.0, sample_rate_hz: float = 60.0):
    """Synthesize a raw PPG waveform + irregular-ish timestamps, like a pulse oximeter."""
    n = int(duration_s * sample_rate_hz)
    gt_time = np.arange(n) / sample_rate_hz
    gt_signal = np.sin(2 * np.pi * (bpm / 60.0) * gt_time)
    return gt_signal, gt_time


@pytest.mark.parametrize("bpm", [60.0, 75.0, 100.0])
def test_compute_windowed_gt_hr_recovers_bpm(bpm):
    gt_signal, gt_time = _synthetic_ppg(bpm)
    num_frames = int(12.0 * FPS)

    frame_indices, gt_bpm = compute_windowed_gt_hr(
        gt_signal, gt_time, num_frames, FPS, window_frames=180, stride_frames=30
    )

    assert len(frame_indices) > 0
    assert frame_indices[0] == 179  # first full 6s/180-frame window
    assert np.all(np.diff(frame_indices) == 30)  # 1s stride
    assert np.all(np.abs(gt_bpm - bpm) <= 3.0)


def test_compute_windowed_gt_hr_no_windows_if_too_short():
    gt_signal, gt_time = _synthetic_ppg(72.0, duration_s=3.0)  # only 90 frames worth
    frame_indices, gt_bpm = compute_windowed_gt_hr(
        gt_signal, gt_time, num_frames=90, fps=FPS, window_frames=180, stride_frames=30
    )
    assert len(frame_indices) == 0
    assert len(gt_bpm) == 0


def test_resample_gt_to_frames_matches_frame_count():
    gt_signal, gt_time = _synthetic_ppg(72.0)
    resampled = resample_gt_to_frames(gt_signal, gt_time, num_frames=360, fps=FPS)
    assert len(resampled) == 360
    assert abs(float(np.mean(resampled))) < 1e-6
