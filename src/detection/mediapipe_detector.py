"""
MediaPipe Face Mesh Detector (v0.10.31+ compatible)
Uses FaceLandmarker from Tasks API for precise forehead ROI detection
"""

import os
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ..logging_config import get_logger

logger = get_logger("detection.mediapipe")


class MediaPipeDetector:
    """
    Face detection using MediaPipe FaceLandmarker (Tasks API).

    Provides 478 facial landmarks for precise ROI extraction.
    Much more stable than Haar Cascade for rPPG applications.
    """

    # Forehead polygon landmarks (for 478 landmark model)
    FOREHEAD_POLYGON = [
        # Left side going up
        109,
        67,
        103,
        54,
        21,
        162,
        127,
        # Top arc
        10,
        # Right side going down
        356,
        389,
        251,
        284,
        332,
        297,
        338,
    ]

    # Left cheek landmarks
    LEFT_CHEEK_LANDMARKS = [116, 117, 118, 119, 100, 126, 209, 49, 203, 205, 206]

    # Right cheek landmarks
    RIGHT_CHEEK_LANDMARKS = [345, 346, 347, 348, 329, 355, 429, 279, 423, 425, 426]

    MODEL_PATH = "models/face_landmarker.task"
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe FaceLandmarker.

        Args:
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for landmark tracking
        """
        # Download model if not exists
        if not os.path.exists(self.MODEL_PATH):
            os.makedirs(os.path.dirname(self.MODEL_PATH), exist_ok=True)
            logger.info("Downloading FaceLandmarker model...")
            urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)
            logger.info("Model saved to %s", self.MODEL_PATH)

        # Create FaceLandmarker options
        base_options = python.BaseOptions(model_asset_path=self.MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame: np.ndarray) -> dict | None:
        """
        Detect face and extract landmarks.

        Args:
            frame: BGR image (OpenCV format)

        Returns:
            Dictionary with face info or None if no face detected
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Process with FaceLandmarker
        result = self.landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        # Get first face landmarks
        face_landmarks = result.face_landmarks[0]

        # Convert normalized landmarks to pixel coordinates
        landmarks = []
        for lm in face_landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            landmarks.append((x, y))

        # Get forehead polygon points
        forehead_points = [landmarks[i] for i in self.FOREHEAD_POLYGON if i < len(landmarks)]

        # Calculate bounding box from all landmarks
        all_x = [p[0] for p in landmarks]
        all_y = [p[1] for p in landmarks]
        bbox = (min(all_x), min(all_y), max(all_x) - min(all_x), max(all_y) - min(all_y))

        return {
            "landmarks": landmarks,
            "forehead_polygon": forehead_points,
            "confidence": 0.9,
            "bbox": bbox,
        }

    def get_mean_color(
        self, frame: np.ndarray, detection: dict, region: str = "forehead"
    ) -> tuple[float, float, float]:
        """
        Get mean RGB color from a specific face region.

        Args:
            frame: Original BGR frame
            detection: Detection dict from detect()
            region: 'forehead', 'left_cheek', or 'right_cheek'

        Returns:
            Mean (R, G, B) values
        """
        if detection is None:
            return (0.0, 0.0, 0.0)

        landmarks = detection["landmarks"]

        if region == "forehead":
            indices = self.FOREHEAD_POLYGON
        elif region == "left_cheek":
            indices = self.LEFT_CHEEK_LANDMARKS
        elif region == "right_cheek":
            indices = self.RIGHT_CHEEK_LANDMARKS
        else:
            indices = self.FOREHEAD_POLYGON

        # Get polygon points
        points = [landmarks[i] for i in indices if i < len(landmarks)]

        if len(points) < 3:
            return (0.0, 0.0, 0.0)

        # Create mask and extract mean
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        pts = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

        # Calculate mean color in masked region
        mean_bgr = cv2.mean(frame, mask=mask)[:3]

        # Return as RGB
        return (mean_bgr[2], mean_bgr[1], mean_bgr[0])

    def get_multi_roi_colors(
        self, frame: np.ndarray, detection: dict
    ) -> tuple[float, float, float]:
        """
        Get fused RGB signal from multiple face regions.

        Uses weighted average of forehead (primary) and both cheeks (secondary)
        for more robust signal extraction.

        Args:
            frame: Original BGR frame
            detection: Detection dict from detect()

        Returns:
            Fused mean (R, G, B) values
        """
        if detection is None:
            return (0.0, 0.0, 0.0)

        # Get colors from each region
        forehead = self.get_mean_color(frame, detection, "forehead")
        left_cheek = self.get_mean_color(frame, detection, "left_cheek")
        right_cheek = self.get_mean_color(frame, detection, "right_cheek")

        # Weights: forehead has strongest signal, cheeks provide redundancy
        # Based on rPPG literature, forehead has best SNR
        weights = [0.5, 0.25, 0.25]  # forehead, left, right

        # Check for invalid regions (might be occluded)
        regions = [forehead, left_cheek, right_cheek]
        valid_regions = []
        valid_weights = []

        for i, (r, g, b) in enumerate(regions):
            if r > 0 or g > 0 or b > 0:  # Valid if any color detected
                valid_regions.append((r, g, b))
                valid_weights.append(weights[i])

        if not valid_regions:
            return (0.0, 0.0, 0.0)

        # Normalize weights
        total_weight = sum(valid_weights)
        valid_weights = [w / total_weight for w in valid_weights]

        # Weighted average
        fused_r = sum(r * w for (r, g, b), w in zip(valid_regions, valid_weights, strict=False))
        fused_g = sum(g * w for (r, g, b), w in zip(valid_regions, valid_weights, strict=False))
        fused_b = sum(b * w for (r, g, b), w in zip(valid_regions, valid_weights, strict=False))

        return (fused_r, fused_g, fused_b)

    def draw_detection(
        self,
        frame: np.ndarray,
        detection: dict,
        show_mesh: bool = False,
        show_forehead: bool = True,
    ) -> np.ndarray:
        """
        Draw detection visualization on frame.

        Args:
            frame: Original frame
            detection: Detection dict from detect()
            show_mesh: Whether to draw full face mesh
            show_forehead: Whether to highlight forehead region

        Returns:
            Annotated frame
        """
        frame_copy = frame.copy()

        if detection is None:
            cv2.putText(
                frame_copy,
                "No face detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            return frame_copy

        landmarks = detection["landmarks"]

        # Draw forehead polygon
        if show_forehead and "forehead_polygon" in detection:
            pts = np.array(detection["forehead_polygon"], dtype=np.int32)
            cv2.polylines(frame_copy, [pts], True, (0, 255, 255), 2)

            # Label
            if len(pts) > 0:
                cv2.putText(
                    frame_copy,
                    "FOREHEAD",
                    (pts[0][0], pts[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                )

        # Draw face mesh (optional)
        if show_mesh:
            for x, y in landmarks:
                cv2.circle(frame_copy, (x, y), 1, (0, 255, 0), -1)

        # Draw bounding box
        x, y, w, h = detection["bbox"]
        cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return frame_copy

    def close(self):
        """Release resources."""
        self.landmarker.close()


# Quick test
if __name__ == "__main__":
    print("MediaPipe Face Landmarker Test (Tasks API)")
    print("=" * 45)

    detector = MediaPipeDetector()

    # Test with demo video
    if os.path.exists("data/demo.mp4"):
        cap = cv2.VideoCapture("data/demo.mp4")
        ret, frame = cap.read()
        if ret:
            result = detector.detect(frame)
            if result:
                print(f"✓ Detected {len(result['landmarks'])} landmarks")
                print(f"✓ Forehead polygon: {len(result['forehead_polygon'])} points")
                print(f"✓ BBox: {result['bbox']}")

                # Test mean color extraction
                rgb = detector.get_mean_color(frame, result, "forehead")
                print(f"✓ Forehead RGB: R={rgb[0]:.1f}, G={rgb[1]:.1f}, B={rgb[2]:.1f}")
            else:
                print("✗ No face detected")
        cap.release()
    else:
        print("No demo video available for test")

    detector.close()
    print("\nTest complete!")
