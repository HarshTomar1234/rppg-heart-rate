"""Validate CHROM and POS pulse extraction end-to-end via the SignalProcessor.

At 30 fps a 12 s trace gives ~360 samples; combined with the FFT analyzer's
zero-padding + parabolic interpolation, a clean synthetic pulse is recovered to well
under 1 BPM, so a ±3 BPM tolerance is a safe correctness bound.
"""

import numpy as np
import pytest

from src.processing import CHROMMethod, POSMethod, SignalProcessor
from src.processing.fft_analyzer import FFTAnalyzer

FPS = 30.0
TOL_BPM = 3.0


def _recover_bpm(method: str, r, g, b) -> float:
    proc = SignalProcessor(buffer_size=len(g), fps=FPS, method=method)
    for ri, gi, bi in zip(r, g, b, strict=False):
        proc.add_sample((ri, gi, bi))
    signal = proc.get_pulse_signal()
    assert signal.size > 0, f"{method}: empty pulse signal"
    bpm, _ = FFTAnalyzer(fps=FPS).get_heart_rate(signal)
    return bpm


@pytest.mark.parametrize("bpm", [60.0, 72.0, 96.0, 120.0])
def test_chrom_recovers_bpm(rgb_pulse, bpm):
    r, g, b = rgb_pulse(bpm)
    assert abs(_recover_bpm("chrom", r, g, b) - bpm) <= TOL_BPM


@pytest.mark.parametrize("bpm", [60.0, 72.0, 96.0, 120.0])
def test_pos_recovers_bpm(rgb_pulse, bpm):
    """POS must actually run and recover the rate (guards the POS no-op bug fix)."""
    r, g, b = rgb_pulse(bpm)
    assert abs(_recover_bpm("pos", r, g, b) - bpm) <= TOL_BPM


def test_signal_processor_uses_requested_method(rgb_pulse):
    proc = SignalProcessor(buffer_size=180, fps=FPS, method="pos")
    assert proc.pulse_extractor is POSMethod
    proc_chrom = SignalProcessor(buffer_size=180, fps=FPS, method="chrom")
    assert proc_chrom.pulse_extractor is CHROMMethod


def test_invalid_method_raises():
    with pytest.raises(ValueError):
        SignalProcessor(buffer_size=180, fps=FPS, method="not-a-method")


def test_extractors_return_zero_mean(rgb_pulse):
    r, g, b = rgb_pulse(72.0)
    for extractor in (CHROMMethod, POSMethod):
        pulse = extractor.extract_pulse(r, g, b)
        assert pulse.size == len(g)
        assert abs(float(np.mean(pulse))) < 1e-6
