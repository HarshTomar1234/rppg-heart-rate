"""Shared pytest fixtures and synthetic-signal helpers for the rPPG test suite.

All tests are deterministic and self-contained: they synthesize rPPG signals with a
known ground-truth BPM rather than relying on a camera, a face, or network access.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Allow `import src...` even when the package is not pip-installed.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _make_rgb_pulse(
    bpm: float,
    fps: float = 30.0,
    seconds: float = 12.0,
    noise: float = 0.005,
    seed: int = 0,
):
    """Synthesize a realistic rPPG RGB trace with a known heart rate.

    Physiological modelling choices:
    - Pulse waveform = fundamental + a weak 2nd harmonic (real PPG has a sharp
      systolic peak, not a pure sinusoid).
    - The pulsatile (AC) component is strongest in green, weaker in red, weakest in
      blue — matching hemoglobin absorption and rPPG literature.
    - AC/DC ratio ~2% (physiologically plausible), plus small Gaussian sensor noise.

    Args:
        bpm: Ground-truth heart rate in beats per minute.
        fps: Sampling rate (frames per second).
        seconds: Trace length; 12 s at 30 fps = 360 samples for good FFT resolution.
        noise: Relative Gaussian noise standard deviation (fraction of DC).
        seed: RNG seed for reproducibility.

    Returns:
        (r, g, b) arrays of per-frame mean channel values.
    """
    rng = np.random.default_rng(seed)
    n = int(fps * seconds)
    t = np.arange(n) / fps
    f = bpm / 60.0

    pulse = np.sin(2 * np.pi * f * t) + 0.25 * np.sin(2 * np.pi * 2 * f * t)
    pulse /= np.max(np.abs(pulse))

    dc = np.array([180.0, 120.0, 110.0])  # R, G, B skin baseline
    ac = np.array([0.35, 1.0, 0.2]) * 0.02  # green strongest; ~2% AC/DC

    r = dc[0] * (1 + ac[0] * pulse) + noise * dc[0] * rng.standard_normal(n)
    g = dc[1] * (1 + ac[1] * pulse) + noise * dc[1] * rng.standard_normal(n)
    b = dc[2] * (1 + ac[2] * pulse) + noise * dc[2] * rng.standard_normal(n)
    return r, g, b


@pytest.fixture
def rgb_pulse():
    """Return the synthetic RGB-pulse generator function."""
    return _make_rgb_pulse
