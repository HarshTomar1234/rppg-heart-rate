"""Validate the Kalman heart-rate filter's smoothing and spike rejection."""

import numpy as np

from src.processing.kalman_filter import KalmanHRFilter


def test_spike_is_rejected():
    kf = KalmanHRFilter(initial_hr=75.0)
    measurements = [75, 76, 74, 78, 150, 73, 77, 75, 74, 76]  # 150 is a spike
    estimates = [kf.update(float(m), confidence=0.7) for m in measurements]
    # The estimate right after the spike must not jump anywhere near it.
    assert estimates[4] < 100.0
    # Output is far smoother than the raw input.
    assert np.std(estimates) < np.std(measurements)


def test_tracks_steady_rate():
    kf = KalmanHRFilter(initial_hr=70.0)
    for _ in range(60):
        kf.update(80.0, confidence=0.9)
    assert abs(kf.x - 80.0) <= 3.0


def test_output_clamped_to_physiological_range():
    kf = KalmanHRFilter(initial_hr=70.0)
    for _ in range(30):
        kf.update(500.0, confidence=1.0)  # absurd measurements
    assert 45.0 <= kf.x <= 170.0
