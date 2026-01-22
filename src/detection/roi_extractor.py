"""
ROI (Region of Interest) Extractor
Extracts face regions best suited for rPPG signal extraction
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class ROIExtractor:
    """
    Extract Region of Interest from detected face.
    
    For rPPG, we need skin regions with good blood flow visibility:
    - Forehead: Large area, less movement, good signal
    - Cheeks: Good blood flow, less occlusion
    
    The color changes from blood flow are most visible in these regions.
    """
    
    # ROI regions as percentages of face bounding box
    # Format: (x_start%, y_start%, width%, height%)
    # Note: Haar Cascade returns face box that often includes forehead
    # Forehead should be in the TOP portion of the detected face
    ROI_REGIONS = {
        'forehead': (0.25, 0.02, 0.50, 0.15),  # Very top of face box
        'left_cheek': (0.08, 0.55, 0.28, 0.20),  # Left cheek
        'right_cheek': (0.64, 0.55, 0.28, 0.20),  # Right cheek
        'full_face': (0.15, 0.20, 0.70, 0.60),  # Central face
    }
    
    def __init__(self, region: str = 'forehead'):
        """
        Initialize ROI extractor.
        
        Args:
            region: Which region to extract ('forehead', 'left_cheek', 
                    'right_cheek', 'full_face')
        """
        if region not in self.ROI_REGIONS:
            raise ValueError(f"Invalid region. Choose from: {list(self.ROI_REGIONS.keys())}")
        
        self.region = region
        self.roi_params = self.ROI_REGIONS[region]
        
    def extract(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Extract ROI from frame given face bounding box.
        
        Args:
            frame: Original BGR frame
            face_bbox: Face bounding box (x, y, width, height)
            
        Returns:
            Cropped ROI region or None if invalid
        """
        if face_bbox is None:
            return None
            
        x, y, w, h = face_bbox
        
        # Calculate ROI coordinates based on percentages
        rx_start, ry_start, rw, rh = self.roi_params
        
        roi_x = int(x + w * rx_start)
        roi_y = int(y + h * ry_start)
        roi_w = int(w * rw)
        roi_h = int(h * rh)
        
        # Ensure ROI is within frame bounds
        frame_h, frame_w = frame.shape[:2]
        roi_x = max(0, min(roi_x, frame_w - 1))
        roi_y = max(0, min(roi_y, frame_h - 1))
        roi_w = min(roi_w, frame_w - roi_x)
        roi_h = min(roi_h, frame_h - roi_y)
        
        # Check for valid ROI
        if roi_w <= 0 or roi_h <= 0:
            return None
            
        # Extract ROI
        roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        
        return roi
    
    def get_roi_coords(self, face_bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Get ROI coordinates without extracting.
        
        Useful for drawing ROI on frame.
        
        Args:
            face_bbox: Face bounding box (x, y, width, height)
            
        Returns:
            ROI coordinates (x, y, width, height)
        """
        x, y, w, h = face_bbox
        rx_start, ry_start, rw, rh = self.roi_params
        
        roi_x = int(x + w * rx_start)
        roi_y = int(y + h * ry_start)
        roi_w = int(w * rw)
        roi_h = int(h * rh)
        
        return (roi_x, roi_y, roi_w, roi_h)
    
    def get_mean_color(self, roi: np.ndarray) -> Tuple[float, float, float]:
        """
        Get mean RGB color of ROI.
        
        This is the core of rPPG - tracking how the mean color
        changes over time reveals the pulse signal!
        
        Args:
            roi: ROI image (BGR format)
            
        Returns:
            Mean (R, G, B) values
        """
        if roi is None or roi.size == 0:
            return (0.0, 0.0, 0.0)
            
        # Calculate mean of each channel
        # OpenCV uses BGR, we return RGB
        mean_b = np.mean(roi[:, :, 0])
        mean_g = np.mean(roi[:, :, 1])
        mean_r = np.mean(roi[:, :, 2])
        
        return (mean_r, mean_g, mean_b)
    
    def draw_roi(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int],
                 color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
        """
        Draw ROI rectangle on frame.
        
        Args:
            frame: Original frame
            face_bbox: Face bounding box
            color: Rectangle color (BGR)
            
        Returns:
            Frame with ROI drawn
        """
        frame_copy = frame.copy()
        
        if face_bbox is None:
            return frame_copy
            
        x, y, w, h = self.get_roi_coords(face_bbox)
        
        # Draw ROI rectangle
        cv2.rectangle(frame_copy, (x, y), (x + w, y + h), color, 2)
        
        # Add label
        cv2.putText(frame_copy, self.region.upper(), (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame_copy


# Quick test
if __name__ == "__main__":
    print("ROI Extractor Module")
    print("=" * 40)
    print("Available regions:", list(ROIExtractor.ROI_REGIONS.keys()))
    print("\nUsage:")
    print("  extractor = ROIExtractor('forehead')")
    print("  roi = extractor.extract(frame, face_bbox)")
    print("  mean_rgb = extractor.get_mean_color(roi)")
