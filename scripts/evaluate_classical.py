"""Honest, multi-subject evaluation of the classical (no-training) rPPG pipeline.

Runs the EXACT production HeartRateMonitor (CHROM, POS, green, auto) against every
trainable UBFC-rPPG subject, comparing to ground truth computed with the SAME
6-second windowed-FFT procedure applied to the raw PPG waveform (not the pulse
oximeter's own precomputed HR channel, which has unknown internal smoothing).

Reports two tiers of metrics, never blended:
  - Per-subject (N = number of trainable subjects): MAE / RMSE / Pearson r per
    subject, plus a subject-level bootstrap 95% CI on mean MAE. This is the only
    tier that supports an inferential claim.
  - Per-window (all subjects pooled): Bland-Altman agreement, %-within-+/-5 BPM,
    scatter data. Explicitly descriptive only -- windows within a subject overlap
    5 of every 6 seconds and share subject-level confounds, so they are NOT
    independent samples.

Also reports: auto-mode method-selection frequency (flags a method silently never
selected -- same failure signature as the historical POS no-op bug), a confidence-
calibration check (does higher confidence actually mean lower error), a Kalman-clamp
diagnostic (does smoothing suppress genuine variance rather than truly track), and a
paired Wilcoxon signed-rank test (CHROM vs POS, paired by subject).

See specs/phase-1.md for the full methodology and what these numbers do NOT show.

Usage:
    python scripts/evaluate_classical.py
    python scripts/evaluate_classical.py --dataset datasets --output results/phase1
    python scripts/evaluate_classical.py --smoke-test   # 1 subject, 1 method, fast
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.ubfc_loader import UBFCDatasetLoader, UBFCSubject, compute_windowed_gt_hr
from src.detection.mediapipe_detector import MediaPipeDetector
from src.logging_config import get_logger
from src.processing.low_light import LowLightEnhancer
from src.vitals.heart_rate import HeartRateMonitor

logger = get_logger("scripts.evaluate_classical")

METHODS = ["chrom", "pos", "green", "auto"]
WINDOW_FRAMES = 180  # 6s @ 30fps -- matches HeartRateMonitor's default buffer
STRIDE_FRAMES = 30  # 1s @ 30fps
TOL_BPM = 5.0  # for the "%% within tolerance" descriptive metric
N_BOOTSTRAP = 10_000
RNG_SEED = 0


# --------------------------------------------------------------------------- #
# Stage 1: extract the RGB trace ONCE per subject (the expensive MediaPipe pass)
# --------------------------------------------------------------------------- #


def extract_rgb_trace(
    video_path: str, enable_low_light: bool = True
) -> tuple[list[tuple[float, float, float] | None], float, int]:
    """Run face detection + multi-ROI RGB extraction once for the whole video.

    Returns (per-frame RGB or None, fps, total_frames). Reused across all 4 method
    evaluations for a subject so MediaPipe inference isn't repeated 4x.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detector = MediaPipeDetector(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    low_light = (
        LowLightEnhancer(clip_limit=2.5, brightness_threshold=70.0, target_brightness=110.0)
        if enable_low_light
        else None
    )

    trace: list[tuple[float, float, float] | None] = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if low_light is not None:
            _brightness, needs_enhancement = low_light.analyze_brightness(frame)
            if needs_enhancement:
                frame = low_light.enhance(frame)

        detection = detector.detect(frame)
        if detection is None:
            trace.append(None)
        else:
            rgb = detector.get_multi_roi_colors(frame, detection)
            trace.append(None if rgb == (0.0, 0.0, 0.0) else rgb)

        frame_idx += 1
        if frame_idx % 300 == 0:
            logger.info("  ...%d/%d frames detected", frame_idx, total_frames)

    cap.release()
    detector.close()
    return trace, fps, total_frames


class _ReplayDetector:
    """Stand-in MediaPipeDetector that replays a precomputed RGB trace.

    Lets us drive the REAL HeartRateMonitor.process_frame() (including its real
    confidence gating, buffering, and Kalman filtering) once per method, without
    re-running expensive face detection for every method.
    """

    def __init__(self, trace: list[tuple[float, float, float] | None]):
        self._trace = trace
        self._i = 0

    def detect(self, frame):
        rgb = self._trace[self._i] if self._i < len(self._trace) else None
        return (
            None if rgb is None else {"landmarks": [], "forehead_polygon": [], "bbox": (0, 0, 1, 1)}
        )

    def get_multi_roi_colors(self, frame, detection):
        rgb = self._trace[self._i] if self._i < len(self._trace) else (0.0, 0.0, 0.0)
        self._i += 1
        return rgb if rgb is not None else (0.0, 0.0, 0.0)

    def draw_detection(self, frame, detection, **kwargs):
        return frame


# --------------------------------------------------------------------------- #
# Stage 2: replay the trace through each method's real production pipeline
# --------------------------------------------------------------------------- #


@dataclass
class WindowRecord:
    subject_id: str
    method: str
    frame_idx: int
    raw_bpm: float
    smoothed_bpm: float
    confidence: float
    selected_method: str
    gt_bpm: float


def evaluate_subject_method(
    subject_id: str,
    trace: list[tuple[float, float, float] | None],
    fps: float,
    total_frames: int,
    method: str,
    gt_frame_idx: np.ndarray,
    gt_bpm: np.ndarray,
) -> list[WindowRecord]:
    """Replay one subject's RGB trace through one method's real HeartRateMonitor."""
    monitor = HeartRateMonitor(fps=fps, method=method, enable_low_light=False)
    monitor.detector = _ReplayDetector(trace)

    gt_lookup = dict(zip(gt_frame_idx.tolist(), gt_bpm.tolist(), strict=False))
    records: list[WindowRecord] = []
    dummy_frame = np.zeros((2, 2, 3), dtype=np.uint8)  # never inspected by the stub

    for frame_idx in range(total_frames):
        result = monitor.process_frame(dummy_frame)
        if frame_idx not in gt_lookup:
            continue
        if "raw_heart_rate" not in result:
            continue  # not enough buffered data yet, or no face on this frame
        gt = gt_lookup[frame_idx]
        if gt <= 0:
            continue
        records.append(
            WindowRecord(
                subject_id=subject_id,
                method=method,
                frame_idx=frame_idx,
                raw_bpm=result["raw_heart_rate"],
                smoothed_bpm=result["heart_rate"],
                confidence=result["confidence"],
                selected_method=result.get("method", method),
                gt_bpm=gt,
            )
        )
    return records


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def _mae(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - gt)))


def _rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def per_subject_metrics(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    """One row per subject: MAE, RMSE, Pearson r (N = number of subjects)."""
    rows = []
    for subject_id, g in df.groupby("subject_id"):
        pred, gt = g[pred_col].to_numpy(), g["gt_bpm"].to_numpy()
        r, p = (
            scipy_stats.pearsonr(pred, gt)
            if len(pred) >= 2 and np.std(pred) > 0
            else (np.nan, np.nan)
        )
        rows.append(
            {
                "subject_id": subject_id,
                "n_windows": len(g),
                "mae": _mae(pred, gt),
                "rmse": _rmse(pred, gt),
                "pearson_r": r,
                "pearson_p": p,
            }
        )
    return pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)


def bootstrap_mean_mae_ci(
    subject_maes: np.ndarray, n_resamples: int = N_BOOTSTRAP, seed: int = RNG_SEED
) -> dict[str, float]:
    """Percentile bootstrap 95% CI on the mean of per-subject MAE values.

    Deliberately NOT a normal-theory CI -- N is too small (num. trainable subjects)
    to trust a Gaussian assumption on the sampling distribution of the mean.
    """
    if len(subject_maes) < 2:
        # e.g. --smoke-test with a single subject: a CI is meaningless at N=1.
        mean = float(np.mean(subject_maes)) if len(subject_maes) else float("nan")
        return {"mean": mean, "ci_low": float("nan"), "ci_high": float("nan")}
    res = scipy_stats.bootstrap(
        (subject_maes,),
        np.mean,
        n_resamples=n_resamples,
        method="percentile",
        confidence_level=0.95,
        random_state=np.random.default_rng(seed),
    )
    return {
        "mean": float(np.mean(subject_maes)),
        "ci_low": float(res.confidence_interval.low),
        "ci_high": float(res.confidence_interval.high),
    }


def bland_altman_stats(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Descriptive only -- see module docstring on pooled-window independence."""
    diff = pred - gt
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    return {
        "bias": bias,
        "sd": sd,
        "loa_low": bias - 1.96 * sd,
        "loa_high": bias + 1.96 * sd,
        "pct_within_tol": float(np.mean(np.abs(diff) <= TOL_BPM) * 100.0),
        "n_windows": len(diff),
    }


def kalman_clamp_diagnostic(df: pd.DataFrame) -> pd.DataFrame:
    """Per-subject SD of raw vs smoothed vs GT trace, plus smoothed-vs-GT lag.

    If smoothed SD collapses far below GT SD while MAE still looks good, that's the
    Kalman hard-clamp regressing toward a rolling mean rather than truly tracking --
    not a data leak, but flags the metric as potentially flattered.
    """
    rows = []
    for subject_id, g in df.groupby("subject_id"):
        g = g.sort_values("frame_idx")
        raw_sd = float(np.std(g["raw_bpm"]))
        smoothed_sd = float(np.std(g["smoothed_bpm"]))
        gt_sd = float(np.std(g["gt_bpm"]))

        smoothed = g["smoothed_bpm"].to_numpy() - np.mean(g["smoothed_bpm"])
        gt = g["gt_bpm"].to_numpy() - np.mean(g["gt_bpm"])
        if len(smoothed) > 1 and np.std(smoothed) > 0 and np.std(gt) > 0:
            xcorr = np.correlate(smoothed, gt, mode="full")
            lags = np.arange(-len(gt) + 1, len(smoothed))
            lag_windows = int(lags[np.argmax(xcorr)])  # in units of 1s windows
        else:
            lag_windows = 0

        rows.append(
            {
                "subject_id": subject_id,
                "raw_sd": raw_sd,
                "smoothed_sd": smoothed_sd,
                "gt_sd": gt_sd,
                "smoothed_sd_ratio_to_gt": smoothed_sd / gt_sd if gt_sd > 0 else np.nan,
                "smoothed_vs_gt_lag_windows": lag_windows,
            }
        )
    return pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)


def confidence_calibration(df: pd.DataFrame, pred_col: str = "smoothed_bpm") -> pd.DataFrame:
    """Bin pooled windows (all methods) by confidence decile; MAE per decile."""
    d = df.copy()
    d["abs_error"] = np.abs(d[pred_col] - d["gt_bpm"])
    d["decile"] = pd.qcut(d["confidence"], 10, labels=False, duplicates="drop")
    return (
        d.groupby("decile")
        .agg(
            mean_confidence=("confidence", "mean"),
            mean_abs_error=("abs_error", "mean"),
            n=("abs_error", "size"),
        )
        .reset_index()
    )


def method_selection_frequency(df: pd.DataFrame) -> dict[str, float]:
    """For auto-mode rows: how often was each underlying method actually chosen."""
    auto_rows = df[df["method"] == "auto"]
    if len(auto_rows) == 0:
        return {}
    counts = auto_rows["selected_method"].value_counts(normalize=True) * 100.0
    return counts.to_dict()


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #


def plot_bland_altman(pred: np.ndarray, gt: np.ndarray, stats: dict, title: str, out_path: Path):
    mean_hr = (pred + gt) / 2
    diff = pred - gt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(mean_hr, diff, alpha=0.4, s=15)
    ax.axhline(stats["bias"], color="red", linestyle="--", label=f"Bias: {stats['bias']:.2f} BPM")
    ax.axhline(stats["loa_high"], color="gray", linestyle=":", label="+/-1.96 SD")
    ax.axhline(stats["loa_low"], color="gray", linestyle=":")
    ax.set_xlabel("Mean of predicted & ground-truth HR (BPM)")
    ax.set_ylabel("Predicted - Ground truth (BPM)")
    ax.set_title(f"Bland-Altman: {title}\n(descriptive only -- pooled windows are not independent)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scatter(pred: np.ndarray, gt: np.ndarray, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(gt, pred, alpha=0.4, s=15)
    lims = [min(gt.min(), pred.min()) - 5, max(gt.max(), pred.max()) + 5]
    ax.plot(lims, lims, "r--", label="Perfect agreement")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Ground truth HR (BPM)")
    ax.set_ylabel("Predicted HR (BPM)")
    ax.set_title(f"Predicted vs Ground Truth: {title}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confidence_calibration(calib_df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(calib_df["decile"], calib_df["mean_abs_error"])
    ax.set_xlabel("Confidence decile (0 = lowest confidence)")
    ax.set_ylabel("Mean absolute error (BPM)")
    ax.set_title("Confidence calibration: does higher confidence mean lower error?")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def evaluate_all(
    subjects: list[UBFCSubject], methods: list[str], output_dir: Path
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_records: list[WindowRecord] = []

    for i, subject in enumerate(subjects, 1):
        logger.info(
            "[%d/%d] Subject %s: extracting RGB trace...", i, len(subjects), subject.subject_id
        )
        t0 = time.time()
        trace, fps, total_frames = extract_rgb_trace(subject.video_path)
        logger.info("  done in %.1fs (%d frames, %.1f fps)", time.time() - t0, total_frames, fps)

        gt_frame_idx, gt_bpm = compute_windowed_gt_hr(
            subject.gt_ppg,
            subject.gt_time,
            total_frames,
            fps,
            window_frames=WINDOW_FRAMES,
            stride_frames=STRIDE_FRAMES,
        )

        for method in methods:
            records = evaluate_subject_method(
                subject.subject_id, trace, fps, total_frames, method, gt_frame_idx, gt_bpm
            )
            logger.info("  %s: %d matched windows", method, len(records))
            all_records.extend(records)

    df = pd.DataFrame([r.__dict__ for r in all_records])
    if df.empty:
        raise RuntimeError(
            "No windows were evaluated -- check dataset paths and video decodability."
        )

    df.to_csv(output_dir / "per_window_raw.csv", index=False)
    logger.info("Wrote %d rows to per_window_raw.csv", len(df))

    summary: dict[str, Any] = {"methods": {}}

    for method in methods:
        mdf = df[df["method"] == method]
        if mdf.empty:
            continue

        per_subj_raw = per_subject_metrics(mdf, "raw_bpm")
        per_subj_smoothed = per_subject_metrics(mdf, "smoothed_bpm")
        per_subj_raw.to_csv(output_dir / f"per_subject_{method}_raw.csv", index=False)
        per_subj_smoothed.to_csv(output_dir / f"per_subject_{method}_smoothed.csv", index=False)

        ba_smoothed = bland_altman_stats(mdf["smoothed_bpm"].to_numpy(), mdf["gt_bpm"].to_numpy())
        ba_raw = bland_altman_stats(mdf["raw_bpm"].to_numpy(), mdf["gt_bpm"].to_numpy())

        plot_bland_altman(
            mdf["smoothed_bpm"].to_numpy(),
            mdf["gt_bpm"].to_numpy(),
            ba_smoothed,
            f"{method} (Kalman-smoothed)",
            output_dir / f"bland_altman_{method}_smoothed.png",
        )
        plot_scatter(
            mdf["smoothed_bpm"].to_numpy(),
            mdf["gt_bpm"].to_numpy(),
            f"{method} (Kalman-smoothed)",
            output_dir / f"scatter_{method}_smoothed.png",
        )

        kalman_diag = kalman_clamp_diagnostic(mdf)
        kalman_diag.to_csv(output_dir / f"kalman_diagnostic_{method}.csv", index=False)

        summary["methods"][method] = {
            "n_subjects": int(mdf["subject_id"].nunique()),
            "n_windows": int(len(mdf)),
            "per_subject_raw": {
                "mean_mae": float(per_subj_raw["mae"].mean()),
                "sd_mae": float(per_subj_raw["mae"].std()),
                "median_mae": float(per_subj_raw["mae"].median()),
                "min_mae": float(per_subj_raw["mae"].min()),
                "max_mae": float(per_subj_raw["mae"].max()),
                "bootstrap_mean_mae_ci95": bootstrap_mean_mae_ci(per_subj_raw["mae"].to_numpy()),
            },
            "per_subject_smoothed": {
                "mean_mae": float(per_subj_smoothed["mae"].mean()),
                "sd_mae": float(per_subj_smoothed["mae"].std()),
                "median_mae": float(per_subj_smoothed["mae"].median()),
                "min_mae": float(per_subj_smoothed["mae"].min()),
                "max_mae": float(per_subj_smoothed["mae"].max()),
                "bootstrap_mean_mae_ci95": bootstrap_mean_mae_ci(
                    per_subj_smoothed["mae"].to_numpy()
                ),
            },
            "per_window_descriptive_smoothed": ba_smoothed,
            "per_window_descriptive_raw": ba_raw,
            "kalman_variance_ratio_mean": float(kalman_diag["smoothed_sd_ratio_to_gt"].mean()),
        }

    # Method-selection frequency for auto mode (only meaningful if 'auto' was evaluated)
    auto_pct = method_selection_frequency(df)
    summary["auto_method_selection_pct"] = auto_pct
    zero_selected = (
        [m for m in ["chrom", "pos", "green"] if auto_pct.get(m, 0) == 0] if auto_pct else []
    )
    if zero_selected:
        logger.warning(
            "Method(s) %s were NEVER selected by auto mode -- possible silent failure, same "
            "signature as the historical POS no-op bug. Investigate before trusting auto results.",
            zero_selected,
        )
    summary["auto_method_never_selected"] = zero_selected

    # Confidence calibration (pooled across all methods)
    calib = confidence_calibration(df)
    calib.to_csv(output_dir / "confidence_calibration.csv", index=False)
    plot_confidence_calibration(calib, output_dir / "confidence_calibration.png")
    summary["confidence_calibration_monotonic_ish"] = bool(
        calib["mean_abs_error"].iloc[0] >= calib["mean_abs_error"].iloc[-1]
    )

    # Paired Wilcoxon: CHROM vs POS, paired by subject, on smoothed MAE
    if "chrom" in methods and "pos" in methods:
        chrom_mae = per_subject_metrics(df[df["method"] == "chrom"], "smoothed_bpm").set_index(
            "subject_id"
        )["mae"]
        pos_mae = per_subject_metrics(df[df["method"] == "pos"], "smoothed_bpm").set_index(
            "subject_id"
        )["mae"]
        common = chrom_mae.index.intersection(pos_mae.index)
        if len(common) >= 2:
            w_stat, w_p = scipy_stats.wilcoxon(chrom_mae.loc[common], pos_mae.loc[common])
            summary["chrom_vs_pos_wilcoxon"] = {
                "n_subjects": len(common),
                "statistic": float(w_stat),
                "p_value": float(w_p),
                "interpretation": (
                    "not distinguishable at this sample size"
                    if w_p >= 0.05
                    else "statistically distinguishable (p < 0.05)"
                ),
            }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote summary.json")

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="datasets", help="Path to datasets/ folder")
    parser.add_argument("--output", default="results/phase1", help="Output directory")
    parser.add_argument(
        "--smoke-test", action="store_true", help="Run 1 subject x 1 method as a fast sanity check"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=None,
        help=f"Subset of methods to run (default: all of {METHODS})",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Subset of subject_ids to run (default: all trainable)",
    )
    args = parser.parse_args()

    loader = UBFCDatasetLoader(args.dataset)
    subjects = loader.get_trainable_subjects()
    if not subjects:
        raise ValueError(f"No trainable subjects found under {args.dataset}")

    if args.subjects:
        subjects = [s for s in subjects if s.subject_id in args.subjects]

    methods = args.methods or METHODS
    if args.smoke_test:
        subjects = subjects[:1]
        methods = methods[:1] if not args.methods else methods
        logger.info("SMOKE TEST: 1 subject (%s), method(s)=%s", subjects[0].subject_id, methods)

    logger.info("Evaluating %d subjects x %d methods...", len(subjects), len(methods))
    summary = evaluate_all(subjects, methods, Path(args.output))

    print("\n" + "=" * 70)
    print("CLASSICAL PIPELINE EVALUATION SUMMARY")
    print("=" * 70)
    for method, m in summary["methods"].items():
        s = m["per_subject_smoothed"]
        ci = s["bootstrap_mean_mae_ci95"]
        print(
            f"\n{method.upper()} (N={m['n_subjects']} subjects, {m['n_windows']} windows)\n"
            f"  Smoothed MAE: {s['mean_mae']:.2f} +/- {s['sd_mae']:.2f} BPM "
            f"(95% CI: [{ci['ci_low']:.2f}, {ci['ci_high']:.2f}])"
        )
    if summary.get("auto_method_never_selected"):
        print(f"\n[WARNING] Never selected by auto: {summary['auto_method_never_selected']}")
    if "chrom_vs_pos_wilcoxon" in summary:
        w = summary["chrom_vs_pos_wilcoxon"]
        print(
            f"\nCHROM vs POS (paired Wilcoxon, N={w['n_subjects']}): {w['interpretation']} (p={w['p_value']:.4f})"
        )
    print(f"\nFull results written to: {args.output}")


if __name__ == "__main__":
    main()
