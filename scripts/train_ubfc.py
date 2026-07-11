"""Train PhysNet on UBFC-rPPG with Leave-One-Subject-Out cross-validation.

Phase 1 rewrite: the previous version used `torch.utils.data.random_split` on
individual (overlapping) clips rather than on subjects. Since clips overlap 50%
(clip_length=64, stride=32), near-duplicate clips from the same subject could land
on both sides of the train/val split -- and even non-overlapping clips from one
subject share skin tone / lighting / camera confounds a small 3D-CNN can latch onto
instead of the pulse signal. That inflates validation accuracy without the model
actually generalizing. This version fixes it with subject-level Leave-One-Subject-
Out (LOSO) cross-validation: every subject is held out exactly once, trained on all
others, and never appears in its own training set.

With only a small number of real-ground-truth subjects, this is explicitly framed as
a pipeline proof-of-concept, not a production model -- see specs/phase-1.md section 5
for what these results do NOT show.

Usage:
    python scripts/train_ubfc.py --dataset datasets --output results/phase1/physnet_loso
    python scripts/train_ubfc.py --smoke-test   # 1 fold, 2 epochs, fast sanity check
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats as scipy_stats
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.ubfc_loader import (
    UBFCDatasetLoader,
    UBFCSubject,
    compute_windowed_gt_hr,
    resample_gt_to_frames,
)
from src.logging_config import get_logger
from src.models.physnet import TORCH_AVAILABLE, PhysNetLite
from src.processing import BandpassFilter, FFTAnalyzer

logger = get_logger("scripts.train_ubfc")

CLIP_LENGTH = 64
FRAME_SIZE = (64, 64)
TRAIN_STRIDE = 32  # dense overlap for training clip coverage
INFERENCE_STRIDE = 8  # denser overlap at inference for smoother stitched reconstruction
WINDOW_FRAMES = 180  # matches the classical-pipeline evaluation's 6s window
STRIDE_FRAMES = 30  # matches the classical-pipeline evaluation's 1s stride


def load_subject_cache_entry(
    subject: UBFCSubject, frame_size: tuple[int, int] = FRAME_SIZE
) -> dict[str, np.ndarray]:
    """Decode a subject's video ONCE (single sequential pass, no seeking) and
    precompute its frame-aligned ground truth.

    Motivation: `cv2.VideoCapture.set(CAP_PROP_POS_FRAMES, ...)` seeking measured at
    ~0.4s/clip on these UBFC videos, and the original per-clip-open-and-seek pattern
    made a full LOSO sweep take an estimated ~13 hours. Since each resized 64x64 frame
    is tiny (~1500 frames * 64*64*3 bytes =~ 18MB/subject), caching every subject's
    fully-decoded, already-resized frames in memory turns "seek per clip" into "decode
    once, then slice a numpy array" -- the frames are read here in one linear pass.
    """
    cap = cv2.VideoCapture(subject.video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, frame_size)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    gt_resampled = resample_gt_to_frames(
        subject.gt_ppg, subject.gt_time, subject.total_frames, subject.fps
    )
    return {"frames": np.array(frames, dtype=np.uint8), "gt_resampled": gt_resampled}


class UBFCVideoDataset(Dataset):
    """Clips from a fixed, caller-provided subset of subjects (subject-level, not
    clip-level, filtering -- this is what makes LOSO leakage-free).

    `subject_cache` is a dict shared across ALL folds' datasets in a LOSO sweep
    (populated lazily here, reused by every fold a subject appears in) so each
    subject's video is decoded once across the entire sweep, not once per fold.
    """

    def __init__(
        self,
        subjects: list[UBFCSubject],
        subject_cache: dict[str, dict[str, np.ndarray]],
        clip_length: int = CLIP_LENGTH,
        frame_size: tuple[int, int] = FRAME_SIZE,
        stride: int = TRAIN_STRIDE,
        augment: bool = False,
    ):
        self.subjects = subjects
        self.subject_cache = subject_cache
        self.clip_length = clip_length
        self.frame_size = frame_size
        self.stride = stride
        self.augment = augment

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.bandpass = BandpassFilter(fps=30.0)

        self.clips: list[dict[str, Any]] = []
        for subject in self.subjects:
            self._add_subject_clips(subject)
        logger.info("Built dataset with %d clips from %d subjects", len(self.clips), len(subjects))

    def _add_subject_clips(self, subject: UBFCSubject):
        if not subject.video_path or not Path(subject.video_path).exists():
            return
        if subject.gt_ppg is None:
            # This dataset only ever trains against real ground truth (see run_loso's
            # deliberate exclusion of pseudo-labeled subjects); fail loudly rather
            # than silently produce garbage labels if that invariant is ever broken.
            raise ValueError(
                f"{subject.subject_id} has no ground-truth PPG; cannot build labeled clips"
            )
        if subject.subject_id not in self.subject_cache:
            self.subject_cache[subject.subject_id] = load_subject_cache_entry(
                subject, self.frame_size
            )
        n_frames = len(self.subject_cache[subject.subject_id]["frames"])
        for start in range(0, n_frames - self.clip_length, self.stride):
            self.clips.append({"subject": subject, "start_frame": start})

    def __len__(self):
        return len(self.clips)

    def _augment_frame(self, frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Spatial-only augmentation. Deliberately NO color-jitter / brightness
        perturbation: the pulse signal IS a subtle inter-channel color relationship,
        so augmenting channel ratios would destroy exactly what the model must learn.
        """
        h, w = frame.shape[:2]
        if rng.random() < 0.5:
            frame = frame[:, ::-1, :]  # horizontal flip
        crop_frac = rng.uniform(0.9, 1.0)
        ch, cw = int(h * crop_frac), int(w * crop_frac)
        y0 = rng.integers(0, h - ch + 1)
        x0 = rng.integers(0, w - cw + 1)
        frame = cv2.resize(frame[y0 : y0 + ch, x0 : x0 + cw], (w, h))
        noise = rng.normal(0, 2.0, frame.shape).astype(np.float32)
        return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        clip_info = self.clips[idx]
        subject = clip_info["subject"]
        start_frame = clip_info["start_frame"]
        rng = np.random.default_rng(idx)  # deterministic per-item augmentation

        cached = self.subject_cache[subject.subject_id]
        clip_frames = cached["frames"][start_frame : start_frame + self.clip_length]

        frames = []
        for frame_rgb in clip_frames:
            if self.augment:
                frame_rgb = self._augment_frame(frame_rgb, rng)
            frame_norm = frame_rgb.astype(np.float32) / 255.0
            frame_norm = (frame_norm - self.mean) / self.std
            frames.append(frame_norm)

        while len(frames) < self.clip_length:
            frames.append(frames[-1])

        clip = np.array(frames).transpose(3, 0, 1, 2).astype(np.float32)

        label = cached["gt_resampled"][start_frame : start_frame + self.clip_length]
        if len(label) < self.clip_length:
            label = np.pad(label, (0, self.clip_length - len(label)), mode="edge")

        label = label - np.mean(label)
        std = np.std(label)
        if std > 0:
            label = label / std

        return torch.from_numpy(clip), torch.from_numpy(label.astype(np.float32))


class NegPearsonLoss(nn.Module):
    """Negative Pearson correlation loss (waveform-shape matching, not magnitude)."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_m = pred - pred.mean(dim=1, keepdim=True)
        target_m = target - target.mean(dim=1, keepdim=True)
        num = (pred_m * target_m).sum(dim=1)
        denom = torch.sqrt((pred_m**2).sum(dim=1) * (target_m**2).sum(dim=1))
        return 1 - (num / (denom + 1e-8)).mean()


def train_one_fold(
    train_subjects: list[UBFCSubject],
    val_subject: UBFCSubject,
    subject_cache: dict[str, dict[str, np.ndarray]],
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    grad_accum_steps: int,
    use_amp: bool,
    weight_decay: float,
    early_stop_patience: int,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Train one LOSO fold: `val_subject` never appears in `train_subjects`."""
    assert val_subject.subject_id not in {
        s.subject_id for s in train_subjects
    }, "leakage: val subject in train set"

    train_ds = UBFCVideoDataset(train_subjects, subject_cache, augment=True)
    val_ds = UBFCVideoDataset([val_subject], subject_cache, augment=False)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(f"Empty dataset for fold held-out={val_subject.subject_id}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = PhysNetLite().to(device)
    criterion = NegPearsonLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5)
    # AMP on this GPU (Pascal, no tensor cores) is a memory lever, not a speed one --
    # fp16 activation storage still roughly halves footprint on a 4GB card.
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        for step, (clips, labels) in enumerate(train_loader):
            clips, labels = clips.to(device), labels.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(clips)
                loss = criterion(outputs, labels) / grad_accum_steps
            scaler.scale(loss).backward()
            if (step + 1) % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            train_loss += loss.item() * grad_accum_steps
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for clips, labels in val_loader:
                clips, labels = clips.to(device), labels.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    outputs = model(clips)
                    val_loss += criterion(outputs, labels).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        logger.info(
            "  fold[held_out=%s] epoch %d/%d | train=%.4f val=%.4f",
            val_subject.subject_id,
            epoch + 1,
            epochs,
            train_loss,
            val_loss,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience:
                logger.info("  early stopping (no improvement for %d epochs)", early_stop_patience)
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    return {"best_val_loss": best_val_loss, "history": history, "model": model}


def extract_physnet_hr_trace(
    model: nn.Module,
    subject: UBFCSubject,
    subject_cache: dict[str, dict[str, np.ndarray]],
    device: torch.device,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Stitch overlapping predicted-waveform clips into one continuous per-frame
    trace, then apply the SAME 6s-window FFT convention as the classical pipeline
    (a single 64-frame/2.1s clip's FFT would only have ~28 BPM resolution -- far too
    coarse -- so overlapping clips are averaged into a longer trace first).
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    if subject.subject_id not in subject_cache:
        subject_cache[subject.subject_id] = load_subject_cache_entry(subject, FRAME_SIZE)
    all_frames = [
        (f.astype(np.float32) / 255.0 - mean) / std
        for f in subject_cache[subject.subject_id]["frames"]
    ]

    total_frames = len(all_frames)
    signal_sum = np.zeros(total_frames, dtype=np.float64)
    signal_count = np.zeros(total_frames, dtype=np.float64)

    model.eval()
    with torch.no_grad():
        for start in range(0, max(1, total_frames - CLIP_LENGTH), INFERENCE_STRIDE):
            clip_frames = all_frames[start : start + CLIP_LENGTH]
            if len(clip_frames) < CLIP_LENGTH:
                break
            clip = np.array(clip_frames).transpose(3, 0, 1, 2).astype(np.float32)[np.newaxis, ...]
            clip_tensor = torch.from_numpy(clip).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                pred = model(clip_tensor).float().cpu().numpy()[0]
            signal_sum[start : start + CLIP_LENGTH] += pred
            signal_count[start : start + CLIP_LENGTH] += 1

    valid = signal_count > 0
    stitched = np.zeros(total_frames)
    stitched[valid] = signal_sum[valid] / signal_count[valid]

    bandpass = BandpassFilter(fps=subject.fps or 30.0)
    fft_analyzer = FFTAnalyzer(fps=subject.fps or 30.0)

    frame_indices, pred_bpm = [], []
    for end in range(WINDOW_FRAMES - 1, total_frames, STRIDE_FRAMES):
        start = end - WINDOW_FRAMES + 1
        window = stitched[start : end + 1]
        if not np.all(signal_count[start : end + 1] > 0):
            continue
        pulse = bandpass.filter(window - np.mean(window))
        bpm, _confidence = fft_analyzer.get_heart_rate(pulse)
        frame_indices.append(end)
        pred_bpm.append(bpm)

    return np.array(frame_indices), np.array(pred_bpm)


def run_loso(
    dataset_path: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    grad_accum_steps: int,
    weight_decay: float,
    early_stop_patience: int,
    held_out_ids: list[str] | None = None,
) -> dict[str, Any]:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not found!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    logger.info("Training on: %s (AMP=%s)", device, use_amp)

    loader = UBFCDatasetLoader(dataset_path)
    # Headline run uses real-ground-truth subjects ONLY. The video-only (pseudo-
    # labeled) subjects are excluded deliberately: their labels come from running
    # CHROM on the same RGB the model predicts from, so training against them would
    # partly teach PhysNet to imitate CHROM's own blind spots rather than learn
    # independently -- see specs/phase-1.md section 3.2.
    trainable = loader.get_trainable_subjects()
    if not trainable:
        raise ValueError("No trainable (video + ground-truth) subjects found!")

    # `trainable` stays the FULL pool -- every fold still trains on all other real
    # subjects. `held_out_ids` only restricts which subjects get their own fold run
    # (useful for --smoke-test or resuming a partial LOSO sweep).
    fold_subjects = (
        [s for s in trainable if s.subject_id in held_out_ids] if held_out_ids else trainable
    )
    if held_out_ids and not fold_subjects:
        raise ValueError(f"None of --held-out {held_out_ids} matched a trainable subject_id")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_path / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    summary_path = output_path / "loso_summary.json"

    # Resumable: a full LOSO sweep can run long enough to hit an external time limit
    # (tool timeout, machine sleep, etc.), so completed folds are persisted to disk
    # after EVERY fold (not just at the end) and skipped on a subsequent run with the
    # same --output directory.
    fold_results: list[dict[str, Any]] = []
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            fold_results = json.load(f).get("fold_results", [])
        logger.info("Resuming: %d fold(s) already completed in %s", len(fold_results), summary_path)
    done_subject_ids = {f["subject_id"] for f in fold_results}

    # Shared across every fold: each of the (at most ~15) trainable subjects' videos
    # is decoded exactly once for the whole LOSO sweep, no matter how many folds it
    # appears in (see load_subject_cache_entry's docstring for why this matters).
    subject_cache: dict[str, dict[str, np.ndarray]] = {}

    for i, val_subject in enumerate(fold_subjects, 1):
        if val_subject.subject_id in done_subject_ids:
            logger.info(
                "[Fold %d/%d] held out=%s: already done, skipping",
                i,
                len(fold_subjects),
                val_subject.subject_id,
            )
            continue

        train_subjects = [s for s in trainable if s.subject_id != val_subject.subject_id]
        logger.info(
            "[Fold %d/%d] held out=%s, train on %d subjects",
            i,
            len(fold_subjects),
            val_subject.subject_id,
            len(train_subjects),
        )
        t0 = time.time()
        checkpoint_path = checkpoints_dir / f"physnet_loso_{val_subject.subject_id}.pth"

        fold = train_one_fold(
            train_subjects,
            val_subject,
            subject_cache,
            device,
            epochs,
            batch_size,
            lr,
            grad_accum_steps,
            use_amp,
            weight_decay,
            early_stop_patience,
            checkpoint_path,
        )

        gt_frame_idx, gt_bpm = compute_windowed_gt_hr(
            val_subject.gt_ppg,
            val_subject.gt_time,
            val_subject.total_frames,
            val_subject.fps or 30.0,
            window_frames=WINDOW_FRAMES,
            stride_frames=STRIDE_FRAMES,
        )
        pred_frame_idx, pred_bpm = extract_physnet_hr_trace(
            fold["model"], val_subject, subject_cache, device, use_amp
        )

        gt_lookup = dict(zip(gt_frame_idx.tolist(), gt_bpm.tolist(), strict=False))
        matched_pred, matched_gt = [], []
        for fi, pb in zip(pred_frame_idx.tolist(), pred_bpm.tolist(), strict=False):
            if fi in gt_lookup and gt_lookup[fi] > 0:
                matched_pred.append(pb)
                matched_gt.append(gt_lookup[fi])
        matched_pred, matched_gt = np.array(matched_pred), np.array(matched_gt)

        if len(matched_pred) >= 2:
            mae = float(np.mean(np.abs(matched_pred - matched_gt)))
            rmse = float(np.sqrt(np.mean((matched_pred - matched_gt) ** 2)))
            r, _p = scipy_stats.pearsonr(matched_pred, matched_gt)
        else:
            mae = rmse = r = float("nan")

        elapsed = time.time() - t0
        logger.info(
            "  [Fold %d/%d] %s: MAE=%.2f RMSE=%.2f r=%.2f (%.1fs, %d matched windows)",
            i,
            len(fold_subjects),
            val_subject.subject_id,
            mae,
            rmse,
            r,
            elapsed,
            len(matched_pred),
        )

        fold_results.append(
            {
                "subject_id": val_subject.subject_id,
                "best_val_loss": fold["best_val_loss"],
                "n_epochs_run": len(fold["history"]),
                "n_matched_windows": len(matched_pred),
                "mae": mae,
                "rmse": rmse,
                "pearson_r": float(r) if r == r else None,  # NaN-safe for JSON
                "elapsed_s": elapsed,
            }
        )
        _write_summary(summary_path, fold_results)
        logger.info(
            "  Wrote %s (%d/%d folds done)", summary_path, len(fold_results), len(fold_subjects)
        )

    return _write_summary(summary_path, fold_results)


def _write_summary(summary_path: Path, fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate + persist fold results so far (called after every fold, not just at
    the end, so an interrupted run never loses already-completed folds)."""
    maes = np.array([f["mae"] for f in fold_results if f["mae"] == f["mae"]])
    summary = {
        "n_folds": len(fold_results),
        "fold_results": fold_results,
        "aggregate": {
            "mean_mae": float(np.mean(maes)) if len(maes) else None,
            "sd_mae": float(np.std(maes)) if len(maes) else None,
            "median_mae": float(np.median(maes)) if len(maes) else None,
            "min_mae": float(np.min(maes)) if len(maes) else None,
            "max_mae": float(np.max(maes)) if len(maes) else None,
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="datasets", help="Path to datasets/ folder")
    parser.add_argument("--output", default="results/phase1/physnet_loso", help="Output directory")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument(
        "--held-out",
        nargs="+",
        default=None,
        help="Subset of subject_ids to run as LOSO folds (default: all)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="1 fold, 2 epochs, fast sanity check before a full run",
    )
    args = parser.parse_args()

    epochs = args.epochs
    held_out = args.held_out
    if args.smoke_test:
        loader = UBFCDatasetLoader(args.dataset)
        held_out = [loader.get_trainable_subjects()[0].subject_id]
        epochs = 2
        logger.info("SMOKE TEST: 1 fold (held_out=%s), epochs=2", held_out[0])

    run_loso(
        args.dataset,
        args.output,
        epochs,
        args.batch_size,
        args.lr,
        args.grad_accum_steps,
        args.weight_decay,
        args.early_stop_patience,
        held_out_ids=held_out,
    )


if __name__ == "__main__":
    main()
