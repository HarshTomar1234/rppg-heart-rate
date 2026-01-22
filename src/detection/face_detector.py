"""
Face Detector using OpenCV's DNN module
Uses a pre-trained face detection model for rPPG processing

Note: We're using OpenCV's DNN instead of MediaPipe because
MediaPipe's newer versions (0.10.31+) changed their API significantly.
OpenCV's face detector is reliable and works across all versions.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from pathlib import Path
import urllib.request
import os


class FaceDetector:
    """
    Face detection using OpenCV's Haar Cascade or DNN.
    
    This is a reliable, cross-platform face detector that:
    - Works on all OpenCV versions
    - No external dependencies needed
    - Fast enough for real-time
    - Returns bounding box for ROI extraction
    """
    
    def __init__(self, min_detection_confidence: float = 0.5):
        """
        Initialize the face detector.
        
        Args:
            min_detection_confidence: Minimum confidence threshold (0.0 to 1.0)
        """
        self.min_confidence = min_detection_confidence
        
        # Use OpenCV's built-in Haar Cascade (always available)
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Track face across frames for stability
        self.last_bbox = None
        self.frames_since_detection = 0
        
    def detect(self, frame: np.ndarray) -> Optional[dict]:
        """
        Detect face in a frame.
        
        Args:
            frame: BGR image (OpenCV format)
            
        Returns:
            Dictionary with face info or None if no face detected
            {
                'bbox': (x, y, width, height),
                'confidence': float,
                'keypoints': dict of facial keypoints (empty for Haar)
            }
        """
        # Convert to grayscale for Haar cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) == 0:
            # Use last known position for a few frames (smoothing)
            self.frames_since_detection += 1
            if self.last_bbox is not None and self.frames_since_detection < 10:
                return {
                    'bbox': self.last_bbox,
                    'confidence': 0.5,
                    'keypoints': {}
                }
            return None
        
        # Get the largest face (most likely the main subject)
        largest_idx = np.argmax([w * h for (x, y, w, h) in faces])
        x, y, w, h = faces[largest_idx]
        
        # Apply some padding
        padding = 0.1
        pad_x = int(w * padding)
        pad_y = int(h * padding)
        
        frame_h, frame_w = frame.shape[:2]
        x = max(0, x - pad_x)
        y = max(0, y - pad_y)
        w = min(w + 2 * pad_x, frame_w - x)
        h = min(h + 2 * pad_y, frame_h - y)
        
        self.last_bbox = (x, y, w, h)
        self.frames_since_detection = 0
        
        return {
            'bbox': (x, y, w, h),
            'confidence': 0.9,  # Haar doesn't give confidence, assume high
            'keypoints': {}
        }
    
    def draw_detection(self, frame: np.ndarray, detection: dict) -> np.ndarray:
        """
        Draw face detection on frame (for visualization).
        
        Args:
            frame: Original frame
            detection: Detection dictionary from detect()
            
        Returns:
            Frame with detection drawn
        """
        frame_copy = frame.copy()
        
        if detection is None:
            # Draw "No face detected" message
            cv2.putText(frame_copy, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame_copy
            
        x, y, w, h = detection['bbox']
        
        # Draw bounding box
        cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw label
        label = f"Face ({detection['confidence']:.0%})"
        cv2.putText(frame_copy, label, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        return frame_copy


# Quick test
if __name__ == "__main__":
    print("Face Detector Module")
    print("=" * 40)
    print("This module detects faces using OpenCV Haar Cascade.")
    print("Usage:")
    print("  detector = FaceDetector()")
    print("  detection = detector.detect(frame)")
    
    # Quick test with camera if available
    print("\nTesting with sample image...")
    
    # Create a test image
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    detector = FaceDetector()
    result = detector.detect(test_img)
    print(f"Result on blank image: {result}")
