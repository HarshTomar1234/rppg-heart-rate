"""
PhysNet - 3D CNN for Remote Photoplethysmography
Based on: "Remote Heart Rate Measurement from Highly Compressed Facial Videos"
by Yu et al. (2019)

This model extracts rPPG signals from facial video using 3D convolutions
to capture spatio-temporal patterns of blood flow.
"""

import numpy as np
from typing import Tuple, Optional
from pathlib import Path

from ..logging_config import get_logger

logger = get_logger("models.physnet")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed. PhysNet will not be available.")


if TORCH_AVAILABLE:

    class PhysNet3D(nn.Module):
        """
        PhysNet: 3D-CNN for rPPG signal extraction.

        Architecture:
        - Input: (B, 3, T, H, W) - batch of RGB video clips
        - 3D convolutions for spatio-temporal features
        - Output: (B, T) - rPPG signal for each frame

        The model learns to find the subtle color changes in skin
        that correspond to blood flow (pulse signal).
        """

        def __init__(self, in_channels: int = 3, out_channels: int = 1, frame_depth: int = 64):
            super().__init__()

            # Encoder - extracts spatio-temporal features
            self.encoder1 = self._make_block(in_channels, 32, (1, 5, 5), (1, 2, 2))
            self.encoder2 = self._make_block(32, 64, (3, 3, 3), (1, 2, 2))
            self.encoder3 = self._make_block(64, 64, (3, 3, 3), (1, 2, 2))

            # Temporal processing - maintains temporal resolution
            self.temporal1 = self._make_temporal_block(64, 64)
            self.temporal2 = self._make_temporal_block(64, 64)

            # Decoder - upsamples back to rPPG signal
            self.decoder1 = self._make_temporal_block(64, 32)
            self.decoder2 = self._make_temporal_block(32, 16)

            # Output layer - produces single channel rPPG signal
            self.output = nn.Conv3d(16, out_channels, kernel_size=1)

            # Global average pooling over spatial dimensions
            self.gap = nn.AdaptiveAvgPool3d((None, 1, 1))

        def _make_block(
            self, in_ch: int, out_ch: int, kernel: Tuple[int, int, int], pool: Tuple[int, int, int]
        ) -> nn.Sequential:
            """Create encoder block with conv, bn, relu, pool."""
            return nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel, padding="same"),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(pool),
            )

        def _make_temporal_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
            """Create temporal processing block."""
            return nn.Sequential(
                nn.Conv3d(in_ch, out_ch, (3, 1, 1), padding="same"),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.

            Args:
                x: Input tensor (B, C, T, H, W) - video clip

            Returns:
                rPPG signal (B, T)
            """
            # Encoder
            x = self.encoder1(x)
            x = self.encoder2(x)
            x = self.encoder3(x)

            # Temporal processing
            x = self.temporal1(x)
            x = self.temporal2(x)

            # Decoder
            x = self.decoder1(x)
            x = self.decoder2(x)

            # Output
            x = self.output(x)

            # Global spatial pooling
            x = self.gap(x)  # (B, 1, T, 1, 1)

            # Remove channel and spatial dimensions
            x = x.squeeze(-1).squeeze(-1).squeeze(1)  # (B, T)

            return x

    class PhysNetLite(nn.Module):
        """
        Lightweight PhysNet for faster inference.
        ONNX-compatible version with explicit padding.
        """

        def __init__(self, frames: int = 64):
            super().__init__()
            self.frames = frames

            # Encoder with explicit padding (ONNX compatible)
            self.conv1 = nn.Conv3d(3, 16, (1, 5, 5), padding=(0, 2, 2))
            self.bn1 = nn.BatchNorm3d(16)
            self.pool1 = nn.MaxPool3d((1, 2, 2))

            self.conv2 = nn.Conv3d(16, 32, (3, 3, 3), padding=(1, 1, 1))
            self.bn2 = nn.BatchNorm3d(32)
            self.pool2 = nn.MaxPool3d((1, 2, 2))

            self.conv3 = nn.Conv3d(32, 32, (3, 3, 3), padding=(1, 1, 1))
            self.bn3 = nn.BatchNorm3d(32)
            self.pool3 = nn.MaxPool3d((1, 2, 2))

            # Temporal conv
            self.temporal = nn.Conv3d(32, 16, (3, 1, 1), padding=(1, 0, 0))
            self.bn_t = nn.BatchNorm3d(16)

            # Output
            self.output = nn.Conv3d(16, 1, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, 3, T, 64, 64)
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.pool1(x)  # (B, 16, T, 32, 32)

            x = F.relu(self.bn2(self.conv2(x)))
            x = self.pool2(x)  # (B, 32, T, 16, 16)

            x = F.relu(self.bn3(self.conv3(x)))
            x = self.pool3(x)  # (B, 32, T, 8, 8)

            x = F.relu(self.bn_t(self.temporal(x)))  # (B, 16, T, 8, 8)

            x = self.output(x)  # (B, 1, T, 8, 8)

            # Global average over spatial dimensions
            x = x.mean(dim=[-2, -1])  # (B, 1, T)
            x = x.squeeze(1)  # (B, T)

            return x


class PhysNetPreprocessor:
    """
    Preprocesses video frames for PhysNet input.

    Handles:
    - Face detection and cropping
    - Resizing to model input size
    - Normalization
    - Batching into clips
    """

    def __init__(
        self, input_size: Tuple[int, int] = (64, 64), clip_length: int = 64, normalize: bool = True
    ):
        self.input_size = input_size
        self.clip_length = clip_length
        self.normalize = normalize

        # Normalization parameters (ImageNet-style)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        # Frame buffer
        self.frames = []

    def add_frame(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int] = None):
        """
        Add a frame to the buffer.

        Args:
            frame: BGR frame from video
            face_bbox: (x, y, w, h) face bounding box, or None to use full frame
        """
        import cv2

        # Crop face if bbox provided
        if face_bbox is not None:
            x, y, w, h = face_bbox
            frame = frame[y : y + h, x : x + w]

        # Resize
        frame = cv2.resize(frame, self.input_size)

        # BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # To float [0, 1]
        frame = frame.astype(np.float32) / 255.0

        # Normalize
        if self.normalize:
            frame = (frame - self.mean) / self.std

        self.frames.append(frame)

    def get_clip(self) -> Optional[np.ndarray]:
        """
        Get a clip ready for model input.

        Returns:
            Clip tensor (1, 3, T, H, W) or None if not enough frames
        """
        if len(self.frames) < self.clip_length:
            return None

        # Get last clip_length frames
        clip = np.array(self.frames[-self.clip_length :])

        # (T, H, W, C) -> (C, T, H, W)
        clip = clip.transpose(3, 0, 1, 2)

        # Add batch dimension
        clip = clip[np.newaxis, ...]

        return clip.astype(np.float32)

    def clear(self):
        """Clear frame buffer."""
        self.frames = []


class PhysNetInference:
    """
    High-level interface for PhysNet inference.

    Combines model, preprocessing, and postprocessing for
    easy heart rate estimation from video frames.
    """

    def __init__(self, model_path: str = None, device: str = "auto", use_lite: bool = True):
        """
        Initialize PhysNet inference.

        Args:
            model_path: Path to trained model weights (optional)
            device: 'cuda', 'cpu', or 'auto'
            use_lite: Use lightweight model for faster inference
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed. Run: pip install torch")

        # Select device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info("PhysNet using device: %s", self.device)

        # Initialize model
        if use_lite:
            self.model = PhysNetLite()
        else:
            self.model = PhysNet3D()

        self.model.to(self.device)
        self.model.eval()

        # Load weights if provided
        if model_path and Path(model_path).exists():
            self.load_weights(model_path)

        # Preprocessor
        self.preprocessor = PhysNetPreprocessor()

        # Signal buffer for post-processing
        self.signal_buffer = []

    def load_weights(self, path: str):
        """Load pre-trained weights."""
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        logger.info("Loaded weights from: %s", path)

    def save_weights(self, path: str):
        """Save model weights."""
        torch.save(self.model.state_dict(), path)
        logger.info("Saved weights to: %s", path)

    def export_onnx(self, path: str, clip_length: int = 64):
        """
        Export model to ONNX format.

        Args:
            path: Output ONNX file path
            clip_length: Temporal dimension for exported model
        """
        dummy_input = torch.randn(1, 3, clip_length, 64, 64).to(self.device)

        torch.onnx.export(
            self.model,
            dummy_input,
            path,
            input_names=["video_clip"],
            output_names=["rppg_signal"],
            dynamic_axes={
                "video_clip": {0: "batch", 2: "time"},
                "rppg_signal": {0: "batch", 1: "time"},
            },
            opset_version=14,
        )
        logger.info("Exported ONNX model to: %s", path)

    def process_frame(
        self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int] = None
    ) -> Optional[np.ndarray]:
        """
        Process a single frame and return rPPG signal if ready.

        Args:
            frame: BGR frame from video
            face_bbox: Face bounding box (x, y, w, h)

        Returns:
            rPPG signal array or None if not enough frames yet
        """
        # Add frame
        self.preprocessor.add_frame(frame, face_bbox)

        # Get clip
        clip = self.preprocessor.get_clip()

        if clip is None:
            return None

        # Inference
        with torch.no_grad():
            clip_tensor = torch.from_numpy(clip).to(self.device)
            signal = self.model(clip_tensor)
            signal = signal.cpu().numpy()[0]  # Remove batch dim

        return signal

    def reset(self):
        """Reset buffers."""
        self.preprocessor.clear()
        self.signal_buffer = []


# Quick test
if __name__ == "__main__":
    print("PhysNet Module")
    print("=" * 50)
    print(f"PyTorch available: {TORCH_AVAILABLE}")

    if TORCH_AVAILABLE:
        print(f"CUDA available: {torch.cuda.is_available()}")

        # Test model
        model = PhysNetLite()
        print(f"\nPhysNetLite parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Test forward pass
        dummy = torch.randn(1, 3, 64, 64, 64)
        output = model(dummy)
        print(f"Input shape: {dummy.shape}")
        print(f"Output shape: {output.shape}")
