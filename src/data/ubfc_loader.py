"""
UBFC-rPPG Dataset Loader
Loads and processes UBFC-rPPG dataset for training PhysNet
"""

import os
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass


@dataclass
class UBFCSubject:
    """Represents one subject's data from UBFC-rPPG dataset."""
    subject_id: str
    video_path: str
    gt_ppg: Optional[np.ndarray]  # Ground truth PPG signal
    gt_hr: Optional[np.ndarray]   # Ground truth heart rate
    gt_time: Optional[np.ndarray] # Time stamps
    fps: float
    total_frames: int
    has_ground_truth: bool
    dataset_version: str  # "DATASET_1" or "DATASET_2"


class UBFCDatasetLoader:
    """
    Loader for UBFC-rPPG dataset.
    Handles both DATASET_1 (gtdump.xmp) and DATASET_2 (ground_truth.txt) formats.
    """
    
    def __init__(self, dataset_root: str):
        """
        Initialize loader.
        
        Args:
            dataset_root: Path to datasets folder containing DATASET_1, DATASET_2
        """
        self.dataset_root = Path(dataset_root)
        self.subjects: List[UBFCSubject] = []
        self._scan_datasets()
        
    def _scan_datasets(self):
        """Scan for all available subjects."""
        print(f"Scanning dataset at: {self.dataset_root}")
        
        # Scan DATASET_1
        dataset1_path = self.dataset_root / "DATASET_1"
        if dataset1_path.exists():
            self._scan_dataset1(dataset1_path)
            
        # Scan DATASET_2
        dataset2_path = self.dataset_root / "DATASET_2"
        if dataset2_path.exists():
            self._scan_dataset2(dataset2_path)
            
        print(f"\nTotal subjects found: {len(self.subjects)}")
        print(f"  With ground truth: {sum(1 for s in self.subjects if s.has_ground_truth)}")
        print(f"  Without ground truth: {sum(1 for s in self.subjects if not s.has_ground_truth)}")
        
    def _scan_dataset1(self, path: Path):
        """Scan DATASET_1 format."""
        for folder in sorted(path.iterdir()):
            if not folder.is_dir():
                continue
                
            video_path = folder / "vid.avi"
            gt_path = folder / "gtdump.xmp"
            
            # Check if video exists
            has_video = video_path.exists()
            has_gt = gt_path.exists()
            
            if not has_video and not has_gt:
                continue
                
            # Load ground truth if available
            gt_ppg, gt_hr, gt_time = None, None, None
            if has_gt:
                try:
                    gt_data = pd.read_csv(gt_path, header=None).values
                    gt_time = gt_data[:, 0] / 1000  # ms to seconds
                    gt_hr = gt_data[:, 1]
                    gt_ppg = gt_data[:, 3]
                except Exception as e:
                    print(f"  Warning: Could not read {gt_path}: {e}")
                    has_gt = False
            
            # Get video info
            fps, total_frames = 30.0, 0
            if has_video:
                cap = cv2.VideoCapture(str(video_path))
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
            
            subject = UBFCSubject(
                subject_id=f"D1_{folder.name}",
                video_path=str(video_path) if has_video else "",
                gt_ppg=gt_ppg,
                gt_hr=gt_hr,
                gt_time=gt_time,
                fps=fps,
                total_frames=total_frames,
                has_ground_truth=has_gt and gt_ppg is not None,
                dataset_version="DATASET_1"
            )
            
            if has_video or has_gt:
                self.subjects.append(subject)
                status = "✅" if has_video and has_gt else "⚠️"
                print(f"  {status} {folder.name}: video={has_video}, gt={has_gt}")
    
    def _scan_dataset2(self, path: Path):
        """Scan DATASET_2 format."""
        for folder in sorted(path.iterdir()):
            if not folder.is_dir():
                continue
                
            video_path = folder / "vid.avi"
            gt_path = folder / "ground_truth.txt"
            
            has_video = video_path.exists()
            has_gt = gt_path.exists()
            
            if not has_video and not has_gt:
                continue
            
            # Load ground truth if available
            gt_ppg, gt_hr, gt_time = None, None, None
            if has_gt:
                try:
                    gt_data = np.loadtxt(gt_path)
                    gt_ppg = gt_data[0, :]
                    gt_hr = gt_data[1, :]
                    gt_time = gt_data[2, :]
                except Exception as e:
                    print(f"  Warning: Could not read {gt_path}: {e}")
                    has_gt = False
            
            # Get video info
            fps, total_frames = 30.0, 0
            if has_video:
                cap = cv2.VideoCapture(str(video_path))
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
            
            subject = UBFCSubject(
                subject_id=f"D2_{folder.name}",
                video_path=str(video_path) if has_video else "",
                gt_ppg=gt_ppg,
                gt_hr=gt_hr,
                gt_time=gt_time,
                fps=fps,
                total_frames=total_frames,
                has_ground_truth=has_gt and gt_ppg is not None,
                dataset_version="DATASET_2"
            )
            
            if has_video or has_gt:
                self.subjects.append(subject)
                status = "✅" if has_video and has_gt else "⚠️"
                print(f"  {status} {folder.name}: video={has_video}, gt={has_gt}")
    
    def get_trainable_subjects(self) -> List[UBFCSubject]:
        """Get subjects that have both video and ground truth."""
        return [s for s in self.subjects if s.video_path and s.has_ground_truth]
    
    def get_video_only_subjects(self) -> List[UBFCSubject]:
        """Get subjects with video but no ground truth (for pseudo-labeling)."""
        return [s for s in self.subjects if s.video_path and not s.has_ground_truth]
    
    def summary(self) -> Dict:
        """Get dataset summary."""
        trainable = self.get_trainable_subjects()
        video_only = self.get_video_only_subjects()
        
        return {
            "total_subjects": len(self.subjects),
            "trainable": len(trainable),
            "video_only": len(video_only),
            "trainable_ids": [s.subject_id for s in trainable],
            "video_only_ids": [s.subject_id for s in video_only]
        }


def resample_gt_to_frames(gt_signal: np.ndarray, gt_time: np.ndarray, 
                          num_frames: int, fps: float) -> np.ndarray:
    """
    Resample ground truth signal to match video frame rate.
    
    Args:
        gt_signal: Ground truth PPG signal
        gt_time: Ground truth timestamps
        num_frames: Number of video frames
        fps: Video frame rate
        
    Returns:
        Resampled signal matching video frames
    """
    # Create frame timestamps
    frame_times = np.arange(num_frames) / fps
    
    # Interpolate ground truth to frame times
    resampled = np.interp(frame_times, gt_time, gt_signal)
    
    # Normalize
    resampled = resampled - np.mean(resampled)
    std = np.std(resampled)
    if std > 0:
        resampled = resampled / std
    
    return resampled


if __name__ == "__main__":
    # Test the loader
    import sys
    
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "datasets"
    
    print("=" * 60)
    print("UBFC-rPPG Dataset Loader")
    print("=" * 60)
    
    loader = UBFCDatasetLoader(dataset_path)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    summary = loader.summary()
    print(f"Total subjects: {summary['total_subjects']}")
    print(f"Trainable (video + GT): {summary['trainable']}")
    print(f"  IDs: {summary['trainable_ids']}")
    print(f"Video only (for pseudo-labels): {summary['video_only']}")
    print(f"  IDs: {summary['video_only_ids']}")
