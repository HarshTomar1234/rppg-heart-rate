"""Validate HeartRateMonitor method wiring without needing a camera or MediaPipe model.

The MediaPipe detector is stubbed so these stay fast unit tests; they still exercise the
real __init__ wiring that the POS no-op bug lived in.
"""

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("mediapipe")

import src.vitals.heart_rate as hr_mod
from src.vitals.heart_rate import HeartRateMonitor


class _StubDetector:
    """Minimal stand-in for MediaPipeDetector (no model download, no inference)."""

    def __init__(self, *args, **kwargs):
        pass

    def detect(self, frame):
        return None

    def draw_detection(self, frame, detection, **kwargs):
        return frame


@pytest.fixture
def stub_detector(monkeypatch):
    monkeypatch.setattr(hr_mod, "MediaPipeDetector", _StubDetector)


def test_pos_method_is_wired(stub_detector):
    # Regression guard: previously the processor was hard-wired to CHROM.
    assert HeartRateMonitor(method="pos").signal_processor.method == "pos"


def test_chrom_is_default(stub_detector):
    assert HeartRateMonitor().signal_processor.method == "chrom"


def test_auto_uses_chrom_as_base(stub_detector):
    assert HeartRateMonitor(method="auto").signal_processor.method == "chrom"


def test_green_method_is_wired(stub_detector):
    # green is fully supported by SignalProcessor and used as an evaluation
    # baseline (Phase 1); must be reachable through the real production monitor.
    assert HeartRateMonitor(method="green").signal_processor.method == "green"


def test_invalid_method_raises():
    with pytest.raises(ValueError):
        HeartRateMonitor(method="banana")


def test_blank_frame_reports_no_face(stub_detector):
    monitor = HeartRateMonitor(method="pos", enable_low_light=False)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    result = monitor.process_frame(frame)
    assert result["status"] == "No face detected"
    assert result["heart_rate"] == 0.0
    assert result["face_detected"] is False
    assert "raw_heart_rate" not in result


class _PulseDetector:
    """Stand-in detector that feeds a synthetic RGB pulse instead of decoding frames.

    Mirrors the physiological model in tests/conftest.py: pulse strongest in green,
    ~2% AC/DC ratio, small noise.
    """

    def __init__(self, *args, bpm: float = 72.0, fps: float = 30.0, **kwargs):
        rng = np.random.default_rng(0)
        n = 300
        t = np.arange(n) / fps
        pulse = np.sin(2 * np.pi * (bpm / 60.0) * t)
        dc = np.array([180.0, 120.0, 110.0])
        ac = np.array([0.35, 1.0, 0.2]) * 0.02
        noise = 0.005
        self._rgb = np.stack(
            [
                dc[i] * (1 + ac[i] * pulse) + noise * dc[i] * rng.standard_normal(n)
                for i in range(3)
            ],
            axis=1,
        )
        self._i = 0

    def detect(self, frame):
        return {"landmarks": [], "forehead_polygon": [], "bbox": (0, 0, 1, 1)}

    def get_multi_roi_colors(self, frame, detection):
        rgb = tuple(self._rgb[min(self._i, len(self._rgb) - 1)])
        self._i += 1
        return rgb

    def draw_detection(self, frame, detection, **kwargs):
        return frame


def test_raw_heart_rate_exposed_once_ready(monkeypatch):
    monkeypatch.setattr(hr_mod, "MediaPipeDetector", lambda *a, **k: _PulseDetector(bpm=72.0))
    monitor = HeartRateMonitor(method="chrom", enable_low_light=False)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    results = [monitor.process_frame(frame) for _ in range(200)]

    not_ready = [r for r in results if "Collecting data" in r["status"]]
    assert not_ready and all("raw_heart_rate" not in r for r in not_ready)

    ready = [r for r in results if "raw_heart_rate" in r]
    assert ready, "expected at least one measurement once the buffer filled"
    assert all(40.0 <= r["raw_heart_rate"] <= 200.0 for r in ready)
