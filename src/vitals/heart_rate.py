"""
Heart Rate Monitor - Production Grade
Uses MediaPipe Face Mesh for accurate forehead ROI detection
With Kalman filtering, spike rejection, and low-light compensation
"""

from collections import deque
from typing import Any

import cv2
import numpy as np

from ..detection.mediapipe_detector import MediaPipeDetector
from ..processing import FFTAnalyzer, SignalProcessor
from ..processing.kalman_filter import KalmanHRFilter
from ..processing.low_light import LowLightEnhancer


class HeartRateMonitor:
    """
    Production-grade heart rate monitoring from video.

    Uses MediaPipe Face Mesh for accurate landmark-based ROI detection.
    Includes Kalman filtering for smooth, stable readings.
    Low-light enhancement for webcam in poor conditions.
    """

    def __init__(
        self,
        fps: float = 30.0,
        buffer_seconds: float = 6.0,
        roi_region: str = "forehead",
        smoothing_window: int = 12,
        method: str = "chrom",
        enable_low_light: bool = True,
    ):
        """
        Initialize heart rate monitor.

        Args:
            fps: Video frames per second
            buffer_seconds: Seconds of data to buffer
            roi_region: Face region to use (forehead, left_cheek, right_cheek)
            smoothing_window: Number of readings to smooth over
            method: Signal extraction method ('chrom', 'pos', 'green', 'auto')
            enable_low_light: Enable low-light frame enhancement
        """
        valid_methods = ("chrom", "pos", "green", "auto")
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got {method!r}")

        self.fps = fps
        self.buffer_size = int(fps * buffer_seconds)
        self.method = method
        self.roi_region = roi_region
        self.enable_low_light = enable_low_light

        # MediaPipe detector
        self.detector = MediaPipeDetector(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Signal processing.
        # For 'auto' the base method is a fallback only (get_best_signal tries all);
        # for 'chrom'/'pos'/'green' the processor must actually use the requested method.
        self.signal_processor = SignalProcessor(
            buffer_size=self.buffer_size,
            fps=fps,
            method="chrom" if method == "auto" else method,
        )
        self.fft_analyzer = FFTAnalyzer(fps=fps)

        # Kalman filter with aggressive spike rejection
        self.kalman_filter = KalmanHRFilter(
            initial_hr=70.0,
            process_noise=0.5,  # Low for stability
            measurement_noise=15.0,  # High for noise rejection
            spike_threshold=15.0,  # Tight threshold
            hard_clamp_range=25.0,  # Never deviate more than ±25 BPM
        )

        # Low-light enhancer
        if enable_low_light:
            self.low_light = LowLightEnhancer(
                clip_limit=2.5,
                brightness_threshold=70.0,  # Enhance if darker than this
                target_brightness=110.0,
            )
        else:
            self.low_light = None

        # Smoothing buffers
        self.hr_buffer = deque(maxlen=smoothing_window)
        self.confidence_buffer = deque(maxlen=smoothing_window)

        # State
        self.frame_count = 0
        self.last_detection = None
        self.last_valid_hr = 0.0

        # Quality thresholds - relaxed for webcam
        self.min_confidence = 0.10  # Lowered for poor lighting
        self.hr_change_threshold = 15  # Tighter control
        self.confidence_gate = 0.50  # Lowered from 0.6 for webcam

    def process_frame(self, frame: np.ndarray) -> dict[str, Any]:
        """
        Process a single video frame using MediaPipe landmarks.

        Args:
            frame: BGR image from video

        Returns:
            Dictionary with results
        """
        self.frame_count += 1

        # Apply low-light enhancement if enabled
        if self.low_light is not None:
            brightness, needs_enhancement = self.low_light.analyze_brightness(frame)
            if needs_enhancement:
                frame = self.low_light.enhance(frame)

        # Detect face with MediaPipe
        detection = self.detector.detect(frame)

        if detection is None:
            return self._create_result(frame, None, 0.0, 0.0, False, "No face detected")

        self.last_detection = detection

        # Get mean RGB from multiple face regions (forehead + cheeks)
        # Multi-ROI fusion provides more robust signal with redundancy
        mean_rgb = self.detector.get_multi_roi_colors(frame, detection)

        if mean_rgb == (0.0, 0.0, 0.0):
            return self._create_result(frame, detection, 0.0, 0.0, True, "ROI extraction failed")

        # Add to signal processor
        self.signal_processor.add_sample(mean_rgb)

        # Check if we have enough data
        if not self.signal_processor.is_ready():
            buffer_pct = len(self.signal_processor.g_buffer) / (self.buffer_size // 2) * 100
            return self._create_result(
                frame, detection, 0.0, 0.0, True, f"Collecting data... {buffer_pct:.0f}%"
            )

        # Get pulse signal using auto-selection if enabled
        if self.method == "auto":
            signal, method_used, sig_quality = self.signal_processor.get_best_signal()
        else:
            signal = self.signal_processor.get_pulse_signal()
            method_used = self.method
            sig_quality = self.signal_processor.get_signal_quality()

        if len(signal) == 0:
            return self._create_result(frame, detection, 0.0, 0.0, True, "Signal extraction failed")

        # Extract heart rate
        heart_rate, fft_confidence = self.fft_analyzer.get_heart_rate(signal)

        # Combine confidences
        confidence = (sig_quality + fft_confidence) / 2

        # Apply outlier rejection and smoothing
        smoothed_hr = self._smooth_heart_rate(heart_rate, confidence)

        # Create result
        status = f"Measuring ({method_used.upper()})"

        return self._create_result(
            frame,
            detection,
            smoothed_hr,
            confidence,
            True,
            status,
            raw_signal=signal,
            method=method_used,
            raw_heart_rate=heart_rate,
        )

    def _smooth_heart_rate(self, hr: float, confidence: float) -> float:
        """
        Apply Kalman filtering with outlier rejection.

        The Kalman filter provides:
        - Smooth, stable readings
        - Automatic spike rejection
        - Confidence-adaptive noise model
        """

        # Reject very low confidence readings entirely
        if confidence < self.min_confidence:
            return self.last_valid_hr if self.last_valid_hr > 0 else 0.0

        # Apply Kalman filter (handles spike rejection internally)
        kalman_hr = self.kalman_filter.update(hr, confidence)

        # Add to buffer for weighted averaging
        self.hr_buffer.append(kalman_hr)
        self.confidence_buffer.append(confidence)

        if len(self.hr_buffer) == 0:
            return 0.0

        # Weighted average based on confidence for extra stability
        hrs = np.array(self.hr_buffer)
        confs = np.array(self.confidence_buffer)

        # Remove obvious outliers (median-based)
        if len(hrs) >= 3:
            median = np.median(hrs)
            mad = np.median(np.abs(hrs - median))
            valid_mask = np.abs(hrs - median) < 3 * (mad + 5)
            hrs = hrs[valid_mask]
            confs = confs[valid_mask]

        if len(hrs) == 0:
            return self.last_valid_hr if self.last_valid_hr > 0 else 0.0

        # Weighted average
        smoothed_hr = np.average(hrs, weights=confs)

        self.last_valid_hr = smoothed_hr

        return smoothed_hr

    def _create_result(
        self,
        frame: np.ndarray,
        detection: dict | None,
        heart_rate: float,
        confidence: float,
        face_detected: bool,
        status: str,
        raw_signal: np.ndarray = None,
        method: str = None,
        raw_heart_rate: float = None,
    ) -> dict[str, Any]:
        """Create result dictionary with annotated frame."""

        annotated = frame.copy()

        # Draw face detection with MediaPipe
        if detection is not None:
            annotated = self.detector.draw_detection(annotated, detection, show_forehead=True)

        # Add HR text
        if heart_rate > 0:
            hr_text = f"HR: {heart_rate:.0f} BPM"
            conf_text = f"Conf: {confidence:.2f}"

            # Color based on confidence
            if confidence >= 0.7:
                color = (0, 255, 0)  # Green
            elif confidence >= 0.4:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 165, 255)  # Orange

            cv2.putText(annotated, hr_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(annotated, conf_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        result = {
            "heart_rate": heart_rate,
            "confidence": confidence,
            "face_detected": face_detected,
            "status": status,
            "frame_annotated": annotated,
        }

        if raw_signal is not None:
            result["raw_signal"] = raw_signal
        if method is not None:
            result["method"] = method
        if raw_heart_rate is not None:
            # Pre-Kalman FFT-derived BPM, exposed for evaluation/diagnostics —
            # lets callers compare raw signal-processing accuracy against the
            # smoothed value actually shown to users.
            result["raw_heart_rate"] = raw_heart_rate

        return result

    def reset(self):
        """Reset the monitor."""
        self.signal_processor.clear()
        self.hr_buffer.clear()
        self.confidence_buffer.clear()
        self.frame_count = 0
        self.last_valid_hr = 0.0

    def get_signal_plot_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get data for plotting the signal."""
        r, g, b = self.signal_processor.get_raw_signal()
        processed = self.signal_processor.get_pulse_signal()

        time_raw = np.arange(len(g)) / self.fps

        return (time_raw, g, processed)


# Quick test
if __name__ == "__main__":
    print("Heart Rate Monitor - MediaPipe Edition")
    print("=" * 50)
    print("Features:")
    print("  - MediaPipe Face Mesh for precise detection")
    print("  - Landmark-based forehead ROI extraction")
    print("  - CHROM/POS signal extraction methods")
    print("  - Outlier rejection and robust smoothing")
