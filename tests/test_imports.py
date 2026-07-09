"""Smoke test: every src module imports cleanly.

Skipped automatically when heavy runtime deps are absent (e.g. a lightweight local
run); the CI job installs the full dependency set and runs these for real.
"""

import importlib

import pytest

pytest.importorskip("cv2")
pytest.importorskip("mediapipe")
pytest.importorskip("torch")
pytest.importorskip("fastapi")

MODULES = [
    "src",
    "src.logging_config",
    "src.processing",
    "src.processing.filters",
    "src.processing.fft_analyzer",
    "src.processing.kalman_filter",
    "src.processing.low_light",
    "src.detection",
    "src.detection.face_detector",
    "src.detection.roi_extractor",
    "src.detection.mediapipe_detector",
    "src.vitals",
    "src.vitals.heart_rate",
    "src.models",
    "src.models.physnet",
    "src.data",
    "src.data.ubfc_loader",
    "src.app.main",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module):
    importlib.import_module(module)
