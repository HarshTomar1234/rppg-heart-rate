"""
PhysNet Training Script
Train PhysNet model using pseudo-labels from signal processing methods (CHROM/POS)

This is a self-training approach:
1. Use CHROM method to extract rPPG signal from videos (teacher)
2. Train PhysNet to predict the same signal (student)
3. PhysNet learns to extract signal directly from pixels

This can work because:
- CHROM is based on solid optical/physiological principles
- PhysNet learns to find better features than hand-crafted ones
- StudentPhysNet can potentially beat teacher CHROM with enough data
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.physnet import TORCH_AVAILABLE, PhysNetLite
from src.processing import BandpassFilter, SignalProcessor


class VideoDataset(Dataset):
    """
    Dataset for training PhysNet.

    Loads video clips and generates pseudo-labels using CHROM method.
    """

    def __init__(
        self,
        video_paths: list[str],
        clip_length: int = 64,
        frame_size: tuple[int, int] = (64, 64),
        stride: int = 32,  # Overlap between clips
        fps: float = 30.0,
    ):
        self.video_paths = video_paths
        self.clip_length = clip_length
        self.frame_size = frame_size
        self.stride = stride
        self.fps = fps

        # Build index of all clips
        self.clips = []  # List of (video_path, start_frame)
        self._build_index()

        # Signal processor for generating labels
        self.signal_processor = SignalProcessor(buffer_size=clip_length, fps=fps)
        self.bandpass = BandpassFilter(fps=fps)

        # Normalization
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def _build_index(self):
        """Build index of all clips from all videos."""
        for video_path in self.video_paths:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            # Generate clip indices
            for start in range(0, total_frames - self.clip_length, self.stride):
                self.clips.append((video_path, start))

        print(f"Built dataset with {len(self.clips)} clips from {len(self.video_paths)} videos")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        video_path, start_frame = self.clips[idx]

        # Load video clip
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        rgb_values = []

        for _ in range(self.clip_length):
            ret, frame = cap.read()
            if not ret:
                break

            # Resize and convert
            frame_resized = cv2.resize(frame, self.frame_size)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            # Normalize for model input
            frame_norm = frame_rgb.astype(np.float32) / 255.0
            frame_norm = (frame_norm - self.mean) / self.std
            frames.append(frame_norm)

            # Get mean RGB for CHROM label
            mean_rgb = frame_rgb.mean(axis=(0, 1))
            rgb_values.append(mean_rgb)

        cap.release()

        # Pad if needed
        while len(frames) < self.clip_length:
            frames.append(frames[-1])
            rgb_values.append(rgb_values[-1])

        # Stack frames: (T, H, W, C) -> (C, T, H, W)
        clip = np.array(frames).transpose(3, 0, 1, 2).astype(np.float32)

        # Generate pseudo-label using CHROM
        rgb_values = np.array(rgb_values)
        label = self._generate_label(rgb_values)

        return torch.from_numpy(clip), torch.from_numpy(label)

    def _generate_label(self, rgb_values: np.ndarray) -> np.ndarray:
        """Generate pseudo-label using CHROM method."""
        r, g, b = rgb_values[:, 0], rgb_values[:, 1], rgb_values[:, 2]

        # Normalize by mean
        r_n = r / (np.mean(r) + 1e-8)
        g_n = g / (np.mean(g) + 1e-8)
        b_n = b / (np.mean(b) + 1e-8)

        # CHROM
        Xs = 3 * r_n - 2 * g_n
        Ys = 1.5 * r_n + g_n - 1.5 * b_n

        std_x = np.std(Xs)
        std_y = np.std(Ys)

        if std_y < 1e-8:
            pulse = Xs - np.mean(Xs)
        else:
            alpha = std_x / std_y
            pulse = Xs - alpha * Ys

        # Normalize pulse
        pulse = pulse - np.mean(pulse)
        pulse = pulse / (np.std(pulse) + 1e-8)

        # Bandpass filter
        pulse = self.bandpass.filter(pulse)

        return pulse.astype(np.float32)


class NegPearsonLoss(nn.Module):
    """
    Negative Pearson Correlation Loss.

    Used in PhysNet paper - ensures predicted waveform
    has same shape as ground truth, regardless of amplitude.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: (B, T)
        pred_mean = pred - pred.mean(dim=1, keepdim=True)
        target_mean = target - target.mean(dim=1, keepdim=True)

        numerator = (pred_mean * target_mean).sum(dim=1)
        denominator = torch.sqrt((pred_mean**2).sum(dim=1) * (target_mean**2).sum(dim=1))

        pearson = numerator / (denominator + 1e-8)

        return 1 - pearson.mean()  # 1 - correlation = loss


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for batch_idx, (clips, labels) in enumerate(dataloader):
        clips = clips.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(clips)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)


def validate(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
) -> float:
    """Validate model."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for clips, labels in dataloader:
            clips = clips.to(device)
            labels = labels.to(device)
            outputs = model(clips)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def train_physnet(
    video_paths: list[str],
    output_dir: str = "models",
    epochs: int = 50,
    batch_size: int = 4,
    lr: float = 1e-4,
    device: str = "auto",
):
    """
    Train PhysNet on videos using CHROM pseudo-labels.

    Args:
        video_paths: List of paths to training videos
        output_dir: Directory to save model
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: 'cuda', 'cpu', or 'auto'
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not installed")

    # Device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Training on: {device}")

    # Dataset
    print("\nBuilding dataset...")
    dataset = VideoDataset(video_paths)

    if len(dataset) == 0:
        raise ValueError("No clips found in videos!")

    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Train clips: {len(train_dataset)}, Val clips: {len(val_dataset)}")

    # Model
    model = PhysNetLite().to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = NegPearsonLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.5)

    # Training loop
    best_val_loss = float("inf")
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"\nStarting training for {epochs} epochs...")

    for epoch in range(epochs):
        start = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        elapsed = time.time() - start
        print(
            f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Time: {elapsed:.1f}s"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_path / "physnet_trained.pth")
            print(f"  -> Saved best model (val_loss: {val_loss:.4f})")

    # Export to ONNX
    model.eval()
    dummy = torch.randn(1, 3, 64, 64, 64).to(device)
    torch.onnx.export(
        model,
        dummy,
        str(output_path / "physnet_trained.onnx"),
        input_names=["video_clip"],
        output_names=["rppg_signal"],
        opset_version=14,
    )
    print(f"\nExported ONNX model to: {output_path / 'physnet_trained.onnx'}")

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train PhysNet for rPPG")
    parser.add_argument("videos", nargs="+", help="Paths to training videos")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output", default="models", help="Output directory")

    args = parser.parse_args()

    train_physnet(
        video_paths=args.videos,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
