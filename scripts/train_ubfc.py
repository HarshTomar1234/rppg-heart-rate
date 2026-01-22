"""
Train PhysNet on UBFC-rPPG Dataset with Real Ground Truth
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List
import time

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.physnet import PhysNetLite, TORCH_AVAILABLE
from src.data.ubfc_loader import UBFCDatasetLoader, resample_gt_to_frames
from src.processing import SignalProcessor, BandpassFilter


class UBFCVideoDataset(Dataset):
    """
    Dataset that loads UBFC-rPPG videos with ground truth.
    Uses real PPG signal as label when available, falls back to CHROM pseudo-labels.
    """
    
    def __init__(
        self,
        loader: UBFCDatasetLoader,
        clip_length: int = 64,
        frame_size: Tuple[int, int] = (64, 64),
        stride: int = 32,
        use_ground_truth: bool = True,
        use_pseudo_labels: bool = True
    ):
        self.loader = loader
        self.clip_length = clip_length
        self.frame_size = frame_size
        self.stride = stride
        self.use_ground_truth = use_ground_truth
        self.use_pseudo_labels = use_pseudo_labels
        
        # Normalization
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        # Bandpass for pseudo-labels
        self.bandpass = BandpassFilter(fps=30.0)
        
        # Build clip index
        self.clips = []
        self._build_index()
        
    def _build_index(self):
        """Build index of all clips."""
        # First add GT subjects
        if self.use_ground_truth:
            for subject in self.loader.get_trainable_subjects():
                self._add_subject_clips(subject, has_gt=True)
                
        # Then add pseudo-label subjects
        if self.use_pseudo_labels:
            for subject in self.loader.get_video_only_subjects():
                self._add_subject_clips(subject, has_gt=False)
                
        print(f"Built dataset with {len(self.clips)} clips")
        
    def _add_subject_clips(self, subject, has_gt: bool):
        """Add clips from one subject."""
        if not subject.video_path or not Path(subject.video_path).exists():
            return
            
        total_frames = subject.total_frames
        for start in range(0, total_frames - self.clip_length, self.stride):
            self.clips.append({
                'subject': subject,
                'start_frame': start,
                'has_gt': has_gt
            })
            
    def __len__(self):
        return len(self.clips)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        clip_info = self.clips[idx]
        subject = clip_info['subject']
        start_frame = clip_info['start_frame']
        has_gt = clip_info['has_gt']
        
        # Load video frames
        cap = cv2.VideoCapture(subject.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        rgb_values = []
        
        for _ in range(self.clip_length):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize and normalize
            frame = cv2.resize(frame, self.frame_size)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # For pseudo-labels
            rgb_values.append(frame_rgb.mean(axis=(0, 1)))
            
            # Normalize for model
            frame_norm = frame_rgb.astype(np.float32) / 255.0
            frame_norm = (frame_norm - self.mean) / self.std
            frames.append(frame_norm)
            
        cap.release()
        
        # Pad if needed
        while len(frames) < self.clip_length:
            frames.append(frames[-1])
            rgb_values.append(rgb_values[-1])
            
        # Stack: (T, H, W, C) -> (C, T, H, W)
        clip = np.array(frames).transpose(3, 0, 1, 2).astype(np.float32)
        
        # Get label
        if has_gt and subject.gt_ppg is not None:
            # Use real ground truth - resample to video frames
            gt_resampled = resample_gt_to_frames(
                subject.gt_ppg, subject.gt_time, 
                subject.total_frames, subject.fps
            )
            label = gt_resampled[start_frame:start_frame + self.clip_length]
            
            # Pad if needed
            if len(label) < self.clip_length:
                label = np.pad(label, (0, self.clip_length - len(label)), mode='edge')
        else:
            # Generate pseudo-label using CHROM
            rgb_values = np.array(rgb_values)
            label = self._generate_chrom_label(rgb_values)
            
        # Normalize label
        label = label - np.mean(label)
        std = np.std(label)
        if std > 0:
            label = label / std
            
        return torch.from_numpy(clip), torch.from_numpy(label.astype(np.float32))
    
    def _generate_chrom_label(self, rgb: np.ndarray) -> np.ndarray:
        """Generate CHROM pseudo-label."""
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        
        r_n = r / (np.mean(r) + 1e-8)
        g_n = g / (np.mean(g) + 1e-8)
        b_n = b / (np.mean(b) + 1e-8)
        
        Xs = 3 * r_n - 2 * g_n
        Ys = 1.5 * r_n + g_n - 1.5 * b_n
        
        std_y = np.std(Ys)
        if std_y > 1e-8:
            alpha = np.std(Xs) / std_y
            pulse = Xs - alpha * Ys
        else:
            pulse = Xs
            
        return self.bandpass.filter(pulse - np.mean(pulse))


class NegPearsonLoss(nn.Module):
    """Negative Pearson correlation loss."""
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_m = pred - pred.mean(dim=1, keepdim=True)
        target_m = target - target.mean(dim=1, keepdim=True)
        
        num = (pred_m * target_m).sum(dim=1)
        denom = torch.sqrt((pred_m**2).sum(dim=1) * (target_m**2).sum(dim=1))
        
        return 1 - (num / (denom + 1e-8)).mean()


def train_on_ubfc(
    dataset_path: str = "datasets",
    epochs: int = 30,
    batch_size: int = 4,
    lr: float = 1e-4,
    output_dir: str = "models"
):
    """Train PhysNet on UBFC-rPPG dataset."""
    
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not found!")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    # Load dataset
    print("\n[1/4] Loading UBFC-rPPG dataset...")
    loader = UBFCDatasetLoader(dataset_path)
    summary = loader.summary()
    
    print(f"  Trainable (with GT): {summary['trainable']}")
    print(f"  Video-only (pseudo): {summary['video_only']}")
    
    if summary['trainable'] + summary['video_only'] == 0:
        raise ValueError("No usable data found!")
    
    # Create dataset
    print("\n[2/4] Building training dataset...")
    dataset = UBFCVideoDataset(loader, use_ground_truth=True, use_pseudo_labels=True)
    
    if len(dataset) == 0:
        raise ValueError("No clips found!")
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"  Train: {len(train_ds)} clips, Val: {len(val_ds)} clips")
    
    # Model
    print("\n[3/4] Initializing model...")
    model = PhysNetLite().to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = NegPearsonLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    # Train
    print(f"\n[4/4] Training for {epochs} epochs...")
    best_val_loss = float('inf')
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for clips, labels in train_loader:
            clips, labels = clips.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(clips)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for clips, labels in val_loader:
                clips, labels = clips.to(device), labels.to(device)
                outputs = model(clips)
                val_loss += criterion(outputs, labels).item()
        val_loss /= len(val_loader)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_path / "physnet_ubfc.pth")
            print(f"  -> Saved best model!")
            
    # Export ONNX
    model.eval()
    dummy = torch.randn(1, 3, 64, 64, 64).to(device)
    torch.onnx.export(model, dummy, str(output_path / "physnet_ubfc.onnx"), opset_version=14)
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets", help="Dataset path")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    train_on_ubfc(args.dataset, args.epochs, args.batch_size, args.lr)
