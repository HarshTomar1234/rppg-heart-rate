<p align="center">
  <img src="result_imgs/rppg_logo.png" alt="rPPG Logo" width="200"/>
</p>

# rPPG: Remote Photoplethysmography for Contactless Heart Rate Detection

<p align="center">
  <em>Contactless heart rate detection using computer vision and deep learning</em>
</p>

---

Remote Photoplethysmography (rPPG) is a technique for measuring blood volume changes in the microvascular bed of tissue using a standard camera. This repository implements a production-grade rPPG system combining classical signal processing methods with deep learning for accurate, contactless heart rate estimation.

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Applications](#applications)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Algorithms](#algorithms)
- [Training](#training)
- [Results](#results)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [References](#references)
- [License](#license)

## Overview

This project provides a complete pipeline for extracting physiological signals from facial video recordings. The system detects subtle color variations in skin caused by blood flow during the cardiac cycle, processes these signals to isolate the pulse waveform, and estimates heart rate in beats per minute (BPM).

### Key Capabilities

- Real-time heart rate estimation from webcam or video files
- Multiple signal extraction methods (CHROM, POS, PhysNet)
- Web-based dashboard with live visualization
- ONNX model export for optimized deployment
- Confidence scoring for signal quality assessment

## Motivation

Traditional heart rate monitoring requires physical contact with sensors, wearables, or medical equipment. This creates barriers in several scenarios:

- **Accessibility**: Not everyone has access to medical-grade pulse oximeters
- **Comfort**: Continuous wearable monitoring can be uncomfortable
- **Hygiene**: Contact-based sensors in hospitals require frequent sanitization
- **Cost**: Medical monitoring equipment is expensive

This project explores how computer vision and deep learning can democratize health monitoring by using just a camera—available on every smartphone, laptop, and webcam.

## Applications

### Healthcare and Telemedicine

| Application | Description |
|-------------|-------------|
| **Remote Patient Monitoring** | Monitor vital signs during video consultations without specialized equipment |
| **Neonatal ICU** | Non-invasive monitoring of premature infants where contact sensors are problematic |
| **Elderly Care** | Ambient monitoring in assisted living facilities without wearables |
| **Mental Health** | Track physiological stress indicators during therapy sessions |

### Fitness and Wellness

| Application | Description |
|-------------|-------------|
| **Smart Mirrors** | Display heart rate during workouts without wearing sensors |
| **Meditation Apps** | Track heart rate variability for stress and relaxation feedback |
| **Sleep Studies** | Non-contact monitoring during sleep without uncomfortable sensors |

### Human-Computer Interaction

| Application | Description |
|-------------|-------------|
| **Driver Monitoring** | Detect drowsiness and stress levels while driving |
| **Gaming** | Adaptive gameplay based on player's physiological state |
| **Video Conferencing** | Real-time wellness indicators during meetings |
| **Lie Detection** | Research applications in detecting physiological stress |

### Research and Education

| Application | Description |
|-------------|-------------|
| **Affective Computing** | Study emotions and their physiological manifestations |
| **Sports Science** | Non-invasive performance monitoring during training |
| **Psychology Studies** | Measure arousal without disrupting natural behavior |
| **Teaching Tool** | Demonstrate signal processing and computer vision concepts |

### Security and Surveillance

| Application | Description |
|-------------|-------------|
| **Authentication** | Liveness detection using pulse presence as anti-spoofing |
| **Crowd Monitoring** | Mass screening for fever or stress in public spaces |

## Features

| Feature | Description |
|---------|-------------|
| **PhysNet 3D-CNN** | End-to-end deep learning model for rPPG signal extraction |
| **CHROM Method** | Chrominance-based signal processing (De Haan & Jeanne, 2013) |
| **POS Method** | Plane-orthogonal-to-skin projection (Wang et al., 2017) |
| **Real-time Dashboard** | FastAPI backend with WebSocket streaming |
| **ONNX Support** | Model export for cross-platform deployment |
| **Signal Quality** | SNR-based confidence estimation |

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA 11.8+ (optional, for GPU acceleration)
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/HarshTomar1234/rppg-heart-rate.git
cd rppg-heart-rate

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate      # Windows
source venv/bin/activate     # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### GPU Support (Optional)

For CUDA-accelerated training and inference:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Web Dashboard

Launch the real-time monitoring dashboard:

```bash
python -m uvicorn src.app.main:app --host 127.0.0.1 --port 8000
```

Navigate to `http://localhost:8000` in your browser, then upload a video file for analysis.

### Command-Line Demo

Process a video file directly:

```bash
python demo_video.py path/to/video.mp4 --method chrom --no-preview
```

Options:
- `--method`: Signal extraction method (`chrom`, `pos`, `auto`)
- `--no-preview`: Disable real-time display (headless mode)

### Python API

```python
from src.vitals import HeartRateMonitor
import cv2

# Initialize monitor
monitor = HeartRateMonitor(fps=30.0, method='chrom')

# Process video
cap = cv2.VideoCapture('video.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    result = monitor.process_frame(frame)
    print(f"Heart Rate: {result['heart_rate']:.1f} BPM")
```

## Architecture

```
Input Video --> Face Detection --> ROI Extraction --> Signal Processing --> FFT --> Heart Rate
     |              |                   |                    |               |          |
  Frames      Haar Cascade        Forehead ROI         CHROM/POS/PhysNet   Peak      BPM
```

### Processing Pipeline

1. **Face Detection**: Haar Cascade classifier localizes facial region
2. **ROI Extraction**: Forehead region selected for optimal pulse visibility
3. **Signal Extraction**: RGB temporal traces converted to pulse signal
4. **Bandpass Filtering**: 0.7-3.0 Hz filter isolates cardiac frequencies (42-180 BPM)
5. **Spectral Analysis**: FFT identifies dominant frequency component
6. **Heart Rate Estimation**: Peak frequency converted to BPM with confidence scoring

## Algorithms

### CHROM (Chrominance-based)

The CHROM method builds two orthogonal chrominance signals from normalized RGB values:

```
Xs = 3*Rn - 2*Gn
Ys = 1.5*Rn + Gn - 1.5*Bn
Pulse = Xs - (std(Xs)/std(Ys)) * Ys
```

Reference: De Haan, G., & Jeanne, V. (2013). Robust pulse rate from chrominance-based rPPG. IEEE Transactions on Biomedical Engineering.

### POS (Plane-Orthogonal-to-Skin)

The POS method projects normalized RGB signals onto a plane orthogonal to the skin-tone direction:

```
Xs = Gn - Bn
Ys = -2*Rn + Gn + Bn
Pulse = Xs + (std(Xs)/std(Ys)) * Ys
```

Reference: Wang, W., et al. (2017). Algorithmic principles of remote PPG. IEEE Transactions on Biomedical Engineering.

### PhysNet

PhysNet is a 3D Convolutional Neural Network that learns spatio-temporal features directly from video frames:

- **Input**: Video clip (T × H × W × 3)
- **Architecture**: 3D convolution blocks with batch normalization
- **Output**: Reconstructed rPPG signal (T samples)
- **Loss Function**: Negative Pearson correlation

Reference: Yu, Z., et al. (2019). Remote heart rate measurement from highly compressed facial videos.

## Training

### Train on UBFC-rPPG Dataset

```bash
python scripts/train_ubfc.py --dataset datasets --epochs 30 --batch-size 4
```

Options:
- `--dataset`: Path to UBFC-rPPG dataset directory
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size for training
- `--lr`: Learning rate (default: 1e-4)

### Training Output

- `models/physnet_ubfc.pth`: PyTorch model weights
- `models/physnet_ubfc.onnx`: ONNX exported model

## Results

### Demo Screenshots

#### Application Interface
![Application Interface](result_imgs/app_interface.png)

#### Heart Rate Detection in Action
![User Detection](result_imgs/user.png)

#### Session Report Chart
![Heart Rate Chart](result_imgs/user_chart.png)

### Accuracy Metrics

| Dataset | Ground Truth Mean | System Mean | Error | Confidence |
|---------|------------------|-------------|-------|------------|
| UBFC Dataset-1 Subject 11 | 77.7 BPM | 78.0 BPM | **0.3 BPM** | 54% |
| UBFC Dataset-2 Subject 1 | 106.7 BPM | 105.3 BPM | **1.4 BPM** | 91% |

### Training Performance

| Metric | Value |
|--------|-------|
| Training Subjects | 15 |
| Training Clips | 953 |
| Final Training Loss | 0.41 |
| Final Validation Loss | 0.58 |
| Loss Improvement | 42% |

### Performance Characteristics
- Mean HR accuracy: **< 2 BPM error** on controlled datasets
- Real-time processing capable (~12 FPS with ONNX)
- Confidence-based quality assessment
- Best results with good lighting conditions

> **Note**: See [LIMITATIONS.md](LIMITATIONS.md) for detailed accuracy limitations and future improvements.


## Project Structure

```
rppg-heart-rate/
├── src/
│   ├── detection/          # Face detection and ROI extraction
│   │   ├── face_detector.py
│   │   └── roi_extractor.py
│   ├── processing/         # Signal processing algorithms
│   │   ├── filters.py      # CHROM, POS, bandpass filtering
│   │   └── fft_analyzer.py # Spectral analysis
│   ├── vitals/             # Heart rate estimation pipeline
│   │   └── heart_rate.py
│   ├── models/             # Deep learning architectures
│   │   └── physnet.py      # PhysNet 3D-CNN
│   ├── data/               # Dataset loaders
│   │   └── ubfc_loader.py
│   └── app/                # Web application
│       ├── main.py         # FastAPI backend
│       └── static/         # Frontend assets
├── scripts/
│   ├── train_ubfc.py       # Training script
│   └── test_physnet.py     # Testing and evaluation
├── models/                 # Trained model weights
├── requirements.txt
├── demo_video.py
├── LICENSE
└── README.md
```

## Dataset

This project uses the UBFC-rPPG dataset for training and validation.

### Citation

If you use the UBFC-rPPG dataset, please cite:

```bibtex
@article{bobbia2019unsupervised,
  title={Unsupervised skin tissue segmentation for remote photoplethysmography},
  author={Bobbia, S. and Macwan, R. and Benezeth, Y. and Mansouri, A. and Dubois, J.},
  journal={Pattern Recognition Letters},
  volume={124},
  pages={82--90},
  year={2019},
  publisher={Elsevier}
}
```

Dataset available at: https://sites.google.com/view/ybenezeth/ubfcrppg

### Dataset Structure

```
datasets/
├── DATASET_1/
│   ├── subject1/
│   │   ├── vid.avi           # Video recording
│   │   └── gtdump.xmp        # Ground truth (timestamp, HR, SpO2, PPG)
│   └── ...
└── DATASET_2/
    ├── subject1/
    │   ├── vid.avi
    │   └── ground_truth.txt  # Ground truth (PPG, HR, timestamp)
    └── ...
```

## References

1. De Haan, G., & Jeanne, V. (2013). Robust pulse rate from chrominance-based rPPG. *IEEE Transactions on Biomedical Engineering*, 60(10), 2878-2886.

2. Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017). Algorithmic principles of remote PPG. *IEEE Transactions on Biomedical Engineering*, 64(7), 1479-1491.

3. Yu, Z., Li, X., & Zhao, G. (2019). Remote photoplethysmograph signal measurement from facial videos using spatio-temporal networks. *BMVC*.

4. Bobbia, S., Macwan, R., Benezeth, Y., Mansouri, A., & Dubois, J. (2019). Unsupervised skin tissue segmentation for remote photoplethysmography. *Pattern Recognition Letters*, 124, 82-90.

## Disclaimer

This software is provided for research and educational purposes only. Heart rate measurements from this system are **not medically validated** and should **not be used for clinical diagnosis or medical decision-making**. For accurate health monitoring, use FDA-approved medical devices.

## License

Copyright 2026 Harsh Tomar

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## Author

**Harsh Tomar**  
AI/ML Developer  
https://kernel-crush.netlify.app/
