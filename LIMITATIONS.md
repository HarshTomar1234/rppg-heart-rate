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

### 3. Training Data Limitations

The current model has been validated on:
- UBFC-rPPG Dataset-1 (limited subjects)
- UBFC-rPPG Dataset-2 (limited subjects)

**Not validated on**:
- Diverse skin tones (limited representation in training data)
- Age groups outside dataset demographics
- People with facial hair, glasses, or makeup
- Real-world webcam conditions at scale

### 4. Accuracy Variance

| Metric | UBFC Dataset | Real-World Webcam |
|--------|--------------|-------------------|
| Mean HR Error | ~1-2 BPM | 5-10 BPM (estimated) |
| Min/Max Accuracy | ±10-15 BPM | Higher variance |
| Confidence | 70-90% | 40-70% typically |

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

### Phase 1: PhysNet Training on Full UBFC Dataset (Next Priority)

Once all subjects from UBFC-rPPG Dataset-2 are downloaded:

1. **Data Preparation**
   - Load all 42 subjects from UBFC-rPPG Dataset-2
   - Split into train (80%) and validation (20%) sets
   - Extract synchronized video frames and ground truth PPG signals

2. **Model Training**
   - Train PhysNet 3D-CNN with ground truth PPG supervision
   - Use negative Pearson correlation loss
   - Target: < 1 BPM mean absolute error

3. **Export and Integration**
   - Export trained model to ONNX format
   - Replace FFT-based HR extraction with neural network inference
   - Benchmark inference speed and accuracy

### Phase 2: Signal Processing Improvements

| Task | Description | Expected Impact |
|------|-------------|-----------------|
| Motion Compensation | Detect head movement and filter artifacts | 30% better stability |
| Adaptive FFT Window | Dynamic window size based on signal quality | More accurate min/max |
| POS Method Integration | Add Plane-Orthogonal-to-Skin algorithm | Better low-light performance |
| ICA Decomposition | Separate pulse signal from noise | Cleaner signal extraction |

### Phase 3: Webcam Optimization

- [ ] Optimize frame preprocessing pipeline
- [ ] Add automatic lighting quality detection
- [ ] Implement camera auto-exposure stabilization
- [ ] Create "calibration" mode for initial face detection
- [ ] Add audio/visual feedback for poor signal quality

### Phase 4: Validation on Diverse Datasets

| Dataset | Purpose | Status |
|---------|---------|--------|
| UBFC-rPPG Dataset-2 (Full) | Primary training | Pending download |
| PURE | Skin tone diversity | Planned |
| MMSE-HR | Expression variation | Planned |
| VIPL-HR | Large-scale validation | Research |

### Phase 5: Production Features

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


---

## Validation Status

| Dataset | Subjects Tested | Mean Accuracy | Status |
|---------|-----------------|---------------|--------|
| UBFC-rPPG Dataset-1 | 1 | ~1 BPM error | Validated |
| UBFC-rPPG Dataset-2 | 1 | ~1-2 BPM error | Validated |
| Real-world Webcam | Limited | Variable | In Progress |
| Diverse Populations | 0 | N/A | Not Tested |

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
