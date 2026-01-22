"""
Heart Rate Monitor - Production Grade
Complete pipeline from video frame to heart rate with improved accuracy
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
from collections import deque

from ..detection import FaceDetector, ROIExtractor
from ..processing import SignalProcessor, FFTAnalyzer


class HeartRateMonitor:
    """
    Production-grade heart rate monitoring from video.
    
    Key improvements over basic version:
    - Uses CHROM/POS methods for robust signal extraction
    - Adaptive method selection based on signal quality
    - Better smoothing with outlier rejection
    - Confidence-based filtering
    """
    
    def __init__(
        self,
        fps: float = 30.0,
        buffer_seconds: float = 6.0,  # Increased for better FFT resolution
        roi_region: str = 'forehead',
        smoothing_window: int = 7,
        method: str = 'chrom'  # 'chrom', 'pos', or 'auto'
    ):
        """
        Initialize heart rate monitor.
        
        Args:
            fps: Video frames per second
            buffer_seconds: Seconds of data to buffer
            roi_region: Face region to use
            smoothing_window: Number of readings to smooth over
            method: Signal extraction method ('chrom', 'pos', 'auto')
        """
        self.fps = fps
        self.buffer_size = int(fps * buffer_seconds)
        self.method = method
        
        # Initialize components
        self.face_detector = FaceDetector()
        self.roi_extractor = ROIExtractor(region=roi_region)
        self.signal_processor = SignalProcessor(
            buffer_size=self.buffer_size,
            fps=fps,
            method='chrom' if method != 'auto' else 'chrom'
        )
        self.fft_analyzer = FFTAnalyzer(fps=fps)
        
        # For smoothing with outlier rejection
        self.hr_buffer = deque(maxlen=smoothing_window)
        self.confidence_buffer = deque(maxlen=smoothing_window)
        
        # State
        self.frame_count = 0
        self.last_detection = None
        self.last_valid_hr = 0.0
        
        # Quality thresholds (lowered for real-world robustness)
        self.min_confidence = 0.05  # Accept readings with low confidence but apply smoothing
        self.hr_change_threshold = 30  # Max BPM change per reading
        
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single video frame.
        
        Args:
            frame: BGR image from video
            
        Returns:
            Dictionary with results
        """
        self.frame_count += 1
        
        # Detect face
        detection = self.face_detector.detect(frame)
        
        if detection is None:
            return self._create_result(
                frame, None, 0.0, 0.0, False, 'No face detected'
            )
            
        self.last_detection = detection
        
        # Extract ROI
        roi = self.roi_extractor.extract(frame, detection['bbox'])
        
        if roi is None or roi.size == 0:
            return self._create_result(
                frame, detection, 0.0, 0.0, True, 'ROI extraction failed'
            )
            
        # Get mean RGB
        mean_rgb = self.roi_extractor.get_mean_color(roi)
        
        # Add to signal processor
        self.signal_processor.add_sample(mean_rgb)
        
        # Check if we have enough data
        if not self.signal_processor.is_ready():
            buffer_pct = len(self.signal_processor.g_buffer) / (self.buffer_size // 2) * 100
            return self._create_result(
                frame, detection, 0.0, 0.0, True,
                f'Collecting data... {buffer_pct:.0f}%'
            )
        
        # Get pulse signal using auto-selection if enabled
        if self.method == 'auto':
            signal, method_used, sig_quality = self.signal_processor.get_best_signal()
        else:
            signal = self.signal_processor.get_pulse_signal()
            method_used = self.method
            sig_quality = self.signal_processor.get_signal_quality()
        
        if len(signal) == 0:
            return self._create_result(
                frame, detection, 0.0, 0.0, True, 'Signal extraction failed'
            )
        
        # Extract heart rate
        heart_rate, fft_confidence = self.fft_analyzer.get_heart_rate(signal)
        
        # Combine confidences
        confidence = (sig_quality + fft_confidence) / 2
        
        # Apply outlier rejection and smoothing
        smoothed_hr = self._smooth_heart_rate(heart_rate, confidence)
        
        # Create result
        status = f'Measuring ({method_used.upper()})'
        
        return self._create_result(
            frame, detection, smoothed_hr, confidence, True, status,
            raw_signal=signal,
            method=method_used
        )
    
    def _smooth_heart_rate(self, hr: float, confidence: float) -> float:
        """Apply smoothing with outlier rejection."""
        
        # Reject low confidence readings
        if confidence < self.min_confidence:
            return self.last_valid_hr if self.last_valid_hr > 0 else 0.0
        
        # Reject sudden large changes (likely noise)
        if self.last_valid_hr > 0:
            if abs(hr - self.last_valid_hr) > self.hr_change_threshold:
                # Likely an outlier, use weighted average
                hr = 0.7 * self.last_valid_hr + 0.3 * hr
        
        # Add to buffer
        self.hr_buffer.append(hr)
        self.confidence_buffer.append(confidence)
        
        if len(self.hr_buffer) == 0:
            return 0.0
        
        # Weighted average based on confidence
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
        detection: Optional[dict],
        heart_rate: float,
        confidence: float,
        face_detected: bool,
        status: str,
        raw_signal: np.ndarray = None,
        method: str = None
    ) -> Dict[str, Any]:
        """Create result dictionary with annotated frame."""
        
        annotated = frame.copy()
        
        # Draw face detection
        if detection is not None:
            annotated = self.face_detector.draw_detection(annotated, detection)
            annotated = self.roi_extractor.draw_roi(annotated, detection['bbox'])
        
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
            
            cv2.putText(annotated, hr_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(annotated, conf_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        result = {
            'heart_rate': heart_rate,
            'confidence': confidence,
            'face_detected': face_detected,
            'status': status,
            'frame_annotated': annotated
        }
        
        if raw_signal is not None:
            result['raw_signal'] = raw_signal
        if method is not None:
            result['method'] = method
            
        return result
    
    def reset(self):
        """Reset the monitor."""
        self.signal_processor.clear()
        self.hr_buffer.clear()
        self.confidence_buffer.clear()
        self.frame_count = 0
        self.last_valid_hr = 0.0
        
    def get_signal_plot_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get data for plotting the signal."""
        r, g, b = self.signal_processor.get_raw_signal()
        processed = self.signal_processor.get_pulse_signal()
        
        time_raw = np.arange(len(g)) / self.fps
        time_proc = np.arange(len(processed)) / self.fps if len(processed) > 0 else np.array([])
            
        return (time_raw, g, processed)


# Quick test
if __name__ == "__main__":
    print("Heart Rate Monitor - Production Grade")
    print("=" * 50)
    print("Improvements:")
    print("  - CHROM/POS signal extraction methods")
    print("  - Auto method selection based on quality")
    print("  - Outlier rejection and robust smoothing")
    print("  - Confidence-based filtering")
