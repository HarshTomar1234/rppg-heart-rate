"""
Low-Light Compensation for rPPG
Enhances video frames for better signal extraction in poor lighting conditions
"""

import cv2
import numpy as np
from typing import Tuple


class LowLightEnhancer:
    """
    Enhances dark/low-contrast frames for better rPPG signal extraction.
    
    Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) and
    adaptive brightness normalization to improve face visibility.
    """
    
    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
        brightness_threshold: float = 80.0,
        target_brightness: float = 120.0
    ):
        """
        Args:
            clip_limit: CLAHE clip limit (higher = more contrast)
            tile_grid_size: CLAHE tile size for local adaptation
            brightness_threshold: Apply enhancement if mean brightness below this
            target_brightness: Target mean brightness after enhancement
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.brightness_threshold = brightness_threshold
        self.target_brightness = target_brightness
        
        # Create CLAHE object
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size
        )
        
    def analyze_brightness(self, frame: np.ndarray) -> Tuple[float, bool]:
        """
        Analyze if frame needs enhancement.
        
        Returns:
            (mean_brightness, needs_enhancement)
        """
        # Convert to grayscale for brightness analysis
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        mean_brightness = np.mean(gray)
        needs_enhancement = mean_brightness < self.brightness_threshold
        
        return mean_brightness, needs_enhancement
    
    def enhance(self, frame: np.ndarray, force: bool = False) -> np.ndarray:
        """
        Enhance frame for better rPPG extraction.
        
        Args:
            frame: BGR input frame
            force: Apply enhancement regardless of brightness
            
        Returns:
            Enhanced BGR frame
        """
        mean_brightness, needs_enhancement = self.analyze_brightness(frame)
        
        if not needs_enhancement and not force:
            return frame
            
        # Convert to LAB color space for better enhancement
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        l_enhanced = self.clahe.apply(l)
        
        # Adaptive brightness boost
        if mean_brightness < self.brightness_threshold:
            # Calculate boost factor
            boost_factor = min(2.0, self.target_brightness / (mean_brightness + 1))
            l_enhanced = np.clip(l_enhanced * boost_factor, 0, 255).astype(np.uint8)
        
        # Merge channels back
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def enhance_roi(self, frame: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
        """
        Enhance only the ROI region (face area).
        
        This is more efficient and avoids enhancing background.
        """
        # Create copy
        result = frame.copy()
        
        # Enhance full frame
        enhanced = self.enhance(frame, force=True)
        
        # Apply only to masked region
        if roi_mask is not None and np.any(roi_mask):
            result[roi_mask > 0] = enhanced[roi_mask > 0]
            
        return result


class AdaptiveColorNormalizer:
    """
    Normalizes color channels for consistent rPPG signal regardless of lighting.
    """
    
    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: Smoothing factor for running average
        """
        self.alpha = alpha
        self.running_mean = None
        
    def normalize(self, rgb: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Normalize RGB values to reduce lighting variation.
        
        Uses running mean subtraction for temporal stability.
        """
        r, g, b = rgb
        
        # Update running mean
        current_mean = (r + g + b) / 3
        
        if self.running_mean is None:
            self.running_mean = current_mean
        else:
            self.running_mean = (1 - self.alpha) * self.running_mean + self.alpha * current_mean
        
        # Normalize by ratio to running mean
        if self.running_mean > 0:
            ratio = 128.0 / self.running_mean  # Normalize to ~128 mean
            r *= ratio
            g *= ratio
            b *= ratio
            
        return (r, g, b)
    
    def reset(self):
        """Reset normalizer state."""
        self.running_mean = None


if __name__ == "__main__":
    print("Low-Light Enhancer Test")
    print("=" * 40)
    
    enhancer = LowLightEnhancer()
    
    # Create a dark test image
    dark_frame = np.ones((480, 640, 3), dtype=np.uint8) * 50  # Very dark
    
    brightness, needs = enhancer.analyze_brightness(dark_frame)
    print(f"Dark frame brightness: {brightness:.1f}")
    print(f"Needs enhancement: {needs}")
    
    enhanced = enhancer.enhance(dark_frame)
    new_brightness, _ = enhancer.analyze_brightness(enhanced)
    print(f"After enhancement: {new_brightness:.1f}")
