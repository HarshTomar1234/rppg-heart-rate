"""
Kalman Filter for Heart Rate Smoothing
Simple 1D Kalman filter to reduce noise and reject outliers
"""

import numpy as np


class KalmanHRFilter:
    """
    1D Kalman Filter optimized for heart rate tracking.

    Provides smooth, stable heart rate estimates by:
    - Filtering measurement noise
    - Rejecting outliers (spikes)
    - Adapting to signal quality
    - Hard clamping to rolling average
    """

    def __init__(
        self,
        initial_hr: float = 70.0,
        process_noise: float = 0.5,  # Reduced for more stability
        measurement_noise: float = 15.0,  # Increased for more filtering
        spike_threshold: float = 15.0,  # Reduced from 25 for tighter rejection
        hard_clamp_range: float = 25.0,  # Hard clamp: never deviate more than this
    ):
        """
        Args:
            initial_hr: Initial heart rate estimate (BPM)
            process_noise: Q - how much HR can change per frame
            measurement_noise: R - measurement uncertainty
            spike_threshold: Reject measurements > this from estimate
            hard_clamp_range: Never deviate more than this from rolling average
        """
        self.x = initial_hr  # State estimate (HR)
        self.P = 100.0  # Error covariance (high initial uncertainty)
        self.Q = process_noise  # Process noise
        self.R = measurement_noise  # Measurement noise
        self.spike_threshold = spike_threshold
        self.hard_clamp_range = hard_clamp_range

        # Rolling average for hard clamping
        self.rolling_window = []
        self.rolling_size = 20

        # Track history for analysis
        self.measurements = []
        self.estimates = []

    def update(self, measurement: float, confidence: float = 1.0) -> float:
        """
        Update filter with new measurement.

        Args:
            measurement: Raw HR measurement (BPM)
            confidence: Measurement confidence (0-1), affects noise

        Returns:
            Filtered HR estimate (BPM)
        """
        # Store measurement
        self.measurements.append(measurement)

        # Calculate rolling average for hard clamping
        if len(self.rolling_window) > 0:
            rolling_avg = np.mean(self.rolling_window)
        else:
            rolling_avg = self.x

        # HARD CLAMP: Never deviate more than hard_clamp_range from rolling average
        if len(self.rolling_window) >= 5:  # Only after we have enough data
            measurement = max(
                rolling_avg - self.hard_clamp_range,
                min(rolling_avg + self.hard_clamp_range, measurement),
            )

        # Adaptive measurement noise based on confidence
        # Low confidence = high noise = trust prediction more
        adaptive_R = self.R / (confidence + 0.1)

        # Spike rejection: if measurement is too far from estimate, heavily discount
        deviation = abs(measurement - self.x)
        if deviation > self.spike_threshold:
            # Strong spike detected - massively discount this measurement
            adaptive_R *= 20  # Increased from 10 for stronger rejection
        elif deviation > self.spike_threshold * 0.7:
            # Medium spike - moderate discount
            adaptive_R *= 5

        # Extra penalty for low confidence readings
        if confidence < 0.5:
            adaptive_R *= 3

        # ==== PREDICT ====
        x_pred = self.x  # HR doesn't change much between frames
        P_pred = self.P + self.Q

        # ==== UPDATE ====
        # Kalman gain
        K = P_pred / (P_pred + adaptive_R)

        # Update estimate
        self.x = x_pred + K * (measurement - x_pred)

        # Update covariance
        self.P = (1 - K) * P_pred

        # Clamp to valid range
        self.x = max(45.0, min(170.0, self.x))

        # Update rolling window
        self.rolling_window.append(self.x)
        if len(self.rolling_window) > self.rolling_size:
            self.rolling_window.pop(0)

        # Store estimate
        self.estimates.append(self.x)

        return self.x

    def reset(self, initial_hr: float = 70.0):
        """Reset filter state."""
        self.x = initial_hr
        self.P = 100.0
        self.measurements = []
        self.estimates = []

    def get_stats(self) -> dict:
        """Get filter statistics."""
        if not self.estimates:
            return {"mean": 0, "std": 0, "samples": 0}
        return {
            "mean": np.mean(self.estimates),
            "std": np.std(self.estimates),
            "samples": len(self.estimates),
            "raw_std": np.std(self.measurements) if self.measurements else 0,
        }


class AdaptiveKalmanHRFilter(KalmanHRFilter):
    """
    Adaptive Kalman filter that adjusts parameters based on signal quality.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.confidence_history = []

    def update(self, measurement: float, confidence: float = 1.0) -> float:
        """Update with adaptive noise based on running confidence."""
        self.confidence_history.append(confidence)

        # Keep last 30 confidence values
        if len(self.confidence_history) > 30:
            self.confidence_history.pop(0)

        # Adjust process noise based on confidence trend
        avg_confidence = np.mean(self.confidence_history)

        # If confidence is consistently low, trust model more (reduce Q)
        if avg_confidence < 0.5:
            self.Q = 0.5  # More stable predictions
        else:
            self.Q = 1.5  # Allow more variation

        return super().update(measurement, confidence)


if __name__ == "__main__":
    # Quick test
    kf = KalmanHRFilter(initial_hr=75)

    # Simulate noisy measurements with a spike
    measurements = [75, 76, 74, 78, 150, 73, 77, 75, 74, 76]  # 150 is a spike

    print("Kalman Filter Test:")
    print("-" * 40)
    for m in measurements:
        filtered = kf.update(m, confidence=0.7)
        print(f"Raw: {m:3d} BPM -> Filtered: {filtered:.1f} BPM")

    print("-" * 40)
    print(f"Raw StdDev: {np.std(measurements):.1f}")
    print(f"Filtered StdDev: {np.std(kf.estimates):.1f}")
