# Known Limitations and Future Work

This document outlines the current limitations of the rPPG Heart Rate Monitor and areas for future improvement.

## Current Limitations

### 1. Lighting Sensitivity

| Condition | Performance |
|-----------|-------------|
| Good lighting (well-lit room) | Excellent (90%+ confidence) |
| Average office lighting | Good (60-80% confidence) |
| Low light / dim room | Poor (needs enhancement) |
| Backlighting (light behind) | Unreliable |
| Direct sunlight on face | Variable |

**Reason**: rPPG relies on detecting subtle color changes in skin. Poor lighting reduces signal-to-noise ratio.

### 2. Motion Sensitivity

- **Head movement** causes signal artifacts
- **Facial expressions** can temporarily affect readings
- **Talking** introduces noise in the signal
- **Best performance** when subject is stationary

**Recommendation**: Stay still for 10-15 seconds for accurate readings.

### 3. Training/Validation Data Limitations

The classical pipeline (CHROM/POS/green/auto) and PhysNet have each been validated
against **15 real UBFC-rPPG subjects** (all 8 currently-downloaded Dataset-1 subjects
+ 7 Dataset-2 subjects) using honest, uncertainty-bounded, per-subject statistics —
see the README's [Results](README.md#results) section for the full numbers and
methodology.

**Not validated on**:
- Diverse skin tones (UBFC-rPPG is a single lab setup with a narrow demographic)
- Age groups outside dataset demographics
- People with facial hair, glasses, or makeup
- Real-world webcam conditions at scale
- Any camera/webcam other than UBFC's original recording setup

### 4. Accuracy Variance

| Metric | UBFC Dataset (measured, N=15) | Real-World Webcam |
|--------|--------------------------------|--------------------|
| Classical pipeline (POS/auto) MAE | 1.6 ± 1.0-1.1 BPM | 5-10 BPM (estimated, not yet measured) |
| PhysNet (LOSO) MAE | 31.4 ± 16.6 BPM (proof-of-concept only, not production) | Not evaluated |
| Confidence | Calibration verified monotonic (higher confidence -> lower error) | Not yet measured |

### 5. Hardware Requirements

- **Camera quality**: Minimum 720p recommended
- **Frame rate**: 30 FPS optimal (lower FPS reduces accuracy)
- **CPU usage**: MediaPipe + processing can be intensive
- **Browser**: Modern browser with WebSocket support required

### 6. Environmental Factors

- **Flickering lights** (fluorescent) can introduce noise at 50/60 Hz
- **Screen glare** on glasses blocks face detection
- **Background movement** may affect processing
- **Camera auto-exposure** changes can cause spikes

### 7. Algorithm Limitations

- FFT-based frequency detection has limited resolution
- Kalman filter introduces slight lag in readings
- Multi-ROI fusion assumes all regions are visible
- Hard clamp (±25 BPM) may hide genuine rapid heart rate changes

---

## NOT Suitable For

> **IMPORTANT**: This system is for research and demonstration purposes only.

- Medical diagnosis or monitoring
- Clinical heart rate measurement
- Fitness tracking requiring high precision
- Detecting arrhythmias or heart conditions
- Emergency or critical care situations

---

## Future Improvements

> **Note on numbering**: this is a wishlist of future ideas, not a sequenced
> execution plan — the numbered "Phase 0-5" production-readiness roadmap (foundation,
> real validation, test coverage, API hardening, PhysNet/ONNX runtime integration,
> deploy) is tracked separately and is what's actually being executed against.

### Scale PhysNet Training Beyond the Current 15 Subjects

LOSO (Leave-One-Subject-Out) training and evaluation is now done on all 15 currently-
downloaded real-ground-truth UBFC-rPPG subjects (8 from Dataset-1 + 7 of Dataset-2's
42) — see the README's [Results](README.md#results) for the real numbers (mean MAE
31.4 ± 16.6 BPM). This confirmed the training pipeline works correctly end-to-end but,
as expected at this scale, does not yet approach the classical pipeline's accuracy.

Remaining work:
- Incrementally download more of UBFC-rPPG Dataset-2's remaining 35 subjects and
  re-run the (already resumable) LOSO sweep as more real ground truth becomes available
- Export a trained checkpoint to ONNX and wire it into the live runtime (currently
  the app only ever uses CHROM/POS/green/auto, never PhysNet) — tracked as a later
  production-readiness phase, not blocked on more data
- Benchmark inference speed once wired in

### Signal Processing Improvements

| Task | Description | Expected Impact |
|------|-------------|-----------------|
| Motion Compensation | Detect head movement and filter artifacts | 30% better stability |
| Adaptive FFT Window | Dynamic window size based on signal quality | More accurate min/max |
| ICA Decomposition | Separate pulse signal from noise | Cleaner signal extraction |

(CHROM, POS, green-channel, and confidence-based auto-selection are already
implemented and evaluated — see [Results](README.md#results).)

### Webcam Optimization

- [ ] Optimize frame preprocessing pipeline
- [ ] Add automatic lighting quality detection
- [ ] Implement camera auto-exposure stabilization
- [ ] Create "calibration" mode for initial face detection
- [ ] Add audio/visual feedback for poor signal quality

### Validation on Diverse Datasets

| Dataset | Purpose | Status |
|---------|---------|--------|
| UBFC-rPPG Dataset-2 (Full, 42 subjects) | Primary training/validation | 7/42 downloaded |
| PURE | Skin tone diversity | Planned |
| MMSE-HR | Expression variation | Planned |
| VIPL-HR | Large-scale validation | Research |

### Production Features

- [ ] Add session recording and playback
- [ ] Implement heart rate variability (HRV) analysis
- [ ] Create mobile-responsive dashboard
- [ ] Add multi-language support
- [ ] Implement data export to health platforms (Apple Health, Google Fit)

### Long-term Research Goals

1. **Arrhythmia Detection** - Detect irregular heartbeat patterns from PPG waveform
2. **Respiratory Rate** - Extract breathing rate from face video
3. **Blood Oxygen Estimation** - Research into SpO2 estimation without contact
4. **Stress Detection** - Combine HRV with facial expressions for stress assessment
5. **Multi-Person Detection** - Track heart rate of multiple people simultaneously

### ONNX and TensorRT Optimization (Learning Priority)

#### ONNX (Open Neural Network Exchange)
- [ ] Learn ONNX export from PyTorch models
- [ ] Export trained PhysNet to ONNX format
- [ ] Integrate ONNX Runtime for CPU inference
- [ ] Benchmark ONNX vs PyTorch inference speed
- [ ] Add ONNX model loading in HeartRateMonitor

#### TensorRT (GPU Acceleration)
- [ ] Learn TensorRT basics and optimization techniques
- [ ] Convert ONNX model to TensorRT engine
- [ ] Implement FP16/INT8 quantization for faster inference
- [ ] Benchmark TensorRT vs ONNX Runtime performance
- [ ] Add GPU inference option in dashboard

#### Expected Performance Gains
| Backend | Expected FPS | Hardware |
|---------|-------------|----------|
| PyTorch (CPU) | 5-8 FPS | Laptop CPU |
| ONNX Runtime (CPU) | 12-15 FPS | Laptop CPU |
| TensorRT (GPU) | 30-50 FPS | NVIDIA GPU |

#### Learning Resources
- [ONNX Official Docs](https://onnx.ai/)
- [PyTorch ONNX Export Guide](https://pytorch.org/docs/stable/onnx.html)
- [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/)
- [ONNX Runtime Python API](https://onnxruntime.ai/docs/)



---

## Validation Status

| Dataset | Subjects Tested | Classical Pipeline (POS/auto) | PhysNet (LOSO) | Status |
|---------|-----------------|-------------------------------|-----------------|--------|
| UBFC-rPPG Dataset-1 | 8 (all downloaded) | Included in the 15-subject evaluation below | Included in the 15-subject LOSO below | Validated |
| UBFC-rPPG Dataset-2 | 7 (of 42; more can be added incrementally) | 1.57 ± 1.02-1.07 BPM MAE (N=15 combined) | 31.4 ± 16.6 BPM MAE (N=15 folds, proof-of-concept) | Validated |
| Real-world Webcam | 0 | Not yet measured | Not applicable | Not Tested |
| Diverse Populations | 0 | Not yet measured | Not applicable | Not Tested |

Full methodology (two-tier per-subject/per-window statistics, bootstrap confidence
intervals, Bland-Altman agreement, confidence calibration, Kalman-smoothing
diagnostic) is in the README's [Results](README.md#results) section.

---

## How to Report Issues

If you encounter accuracy issues, please provide:
1. Screenshot of the dashboard
2. Lighting conditions description
3. Camera specifications
4. Any relevant environmental factors

---

## References

1. UBFC-rPPG Dataset - [Bobbia et al., 2019]
2. CHROM Method - [De Haan & Jeanne, 2013]
3. PhysNet Architecture - [Yu et al., 2019]
4. MediaPipe Face Mesh - [Google, 2020]
