"""Validate the FFT heart-rate analyzer on clean synthetic signals."""

import numpy as np
import pytest

from src.processing.fft_analyzer import FFTAnalyzer, WelchAnalyzer

FPS = 30.0


def _sine(bpm: float, fps: float = FPS, seconds: float = 12.0) -> np.ndarray:
    t = np.arange(int(fps * seconds)) / fps
    return np.sin(2 * np.pi * (bpm / 60.0) * t)


@pytest.mark.parametrize("bpm", [50.0, 72.0, 90.0, 150.0])
def test_fft_recovers_bpm(bpm):
    hr, conf = FFTAnalyzer(fps=FPS).get_heart_rate(_sine(bpm))
    assert abs(hr - bpm) <= 2.0
    assert 0.0 <= conf <= 1.0


def test_parabolic_interpolation_beats_raw_bin():
    """Sub-bin interpolation should land closer than the raw FFT bin spacing.

    A true 77 BPM tone is off-grid for a 360-sample FFT; interpolation should still
    resolve it to within ~2 BPM.
    """
    hr, _ = FFTAnalyzer(fps=FPS, use_interpolation=True).get_heart_rate(_sine(77.0))
    assert abs(hr - 77.0) <= 2.0


def test_out_of_band_returns_clamped():
    hr, conf = FFTAnalyzer(fps=FPS).get_heart_rate(np.zeros(360))
    assert hr == 0.0 or 42.0 <= hr <= 180.0


def test_short_signal_is_rejected():
    hr, conf = FFTAnalyzer(fps=FPS).get_heart_rate(np.ones(10))
    assert hr == 0.0 and conf == 0.0


@pytest.mark.parametrize("bpm", [66.0, 108.0])
def test_welch_analyzer(bpm):
    hr, conf = WelchAnalyzer(fps=FPS).get_heart_rate(_sine(bpm))
    assert abs(hr - bpm) <= 4.0
