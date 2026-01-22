"""
Face Detection Module
Handles face detection and ROI (Region of Interest) extraction
"""

from .face_detector import FaceDetector
from .roi_extractor import ROIExtractor

__all__ = ["FaceDetector", "ROIExtractor"]
