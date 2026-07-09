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
