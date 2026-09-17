# Standalone Face Detection & Recognition System

A high-performance, real-time face detection, tracking, quality assessment, and recognition engine powered by DeepFace, OpenCV, and TensorFlow.

> *Note: This AI vision engine was originally developed as part of an IoT healthcare prototype and has been completely decoupled and extracted into a standalone, hardware-independent computer vision system.*

---

## Overview

This project provides a complete, production-ready face biometric pipeline that operates in real time from a live webcam or video stream. Key features include:

- **Asynchronous Architecture**: Face detection and feature extraction execute in dedicated background worker threads, preventing UI lockups and maintaining a fluid 30+ FPS camera feed.
- **Precomputed Embedding Search**: Instead of executing slow directory scans or end-to-end classification models on each frame, faces are matched using fast L2-normalized cosine dot-products against precomputed embeddings.
- **Adaptive Image Enhancement & Quality Filtering**: Rejects blurry, dark, low-contrast, or undersized face crops, and applies conservative CLAHE (Contrast Limited Adaptive Histogram Equalization) and non-linear gamma curves only to dimly lit captures.
- **Temporal Voting & Tracking**: Lightweight IoU-based multi-face tracking with temporal smoothing (voting across multiple frames) to eliminate flickering and transient misclassifications.
- **Multi-Level Spoof Prevention**:
  - Motion challenge requiring subtle head turns or scale variations.
  - Optional deep learning anti-spoofing verification (MiniFASNet via DeepFace).
  - PIN fallback for manual administrative override.
- **Natural Audio Prompts**: Optional rate-limited, asynchronous spoken guidance via Edge-TTS and Pygame.

---

## AI Architecture & Pipeline

```text
Webcam Frame (1280x720)
       │
       ▼
Downsampled Analysis Frame (480x270 / 640x360)
       │
       ▼
Face Detection (OpenCV YuNet / DeepFace RetinaFace)
       │
       ▼
IoU Multi-Object Tracking & Face Cropping (18% contextual margin)
       │
       ▼
Face Quality Scoring (Laplacian sharpness, contrast, illumination)
       │
       ▼
Adaptive Enhancement (Y-channel CLAHE + Gamma LUT for dim crops)
       │
       ▼
Feature Extraction (FaceNet512 / ArcFace 512-D Embedding Vector)
       │
       ▼
L2 Normalization (Unit float32 vector)
       │
       ▼
Cosine Matrix Search (Top-3 Median per Enrolled Identity)
       │
       ▼
Temporal Smoothing (2 to 4 votes over 6-frame window)
       │
       ▼
Security Verification (Motion Challenge / DeepFace Anti-Spoof)
       │
       ▼
Output: OpenCV Real-Time HUD + Edge-TTS Natural Audio Voice
```

### Models Used

1. **Face Detection Backends**:
   - **YuNet** (Default for `low` and `balanced` profiles): Lightweight, fast convolutional face detector integrated into OpenCV DNN.
   - **RetinaFace** (Default for `quality` profile): High-accuracy ResNet-based face detector capable of handling extreme angles and uneven illumination.
2. **Face Feature Extractors**:
   - **FaceNet512** (Default): Inception-ResNet-v1 architecture mapping facial crops to a 512-dimensional Euclidean space.
   - **ArcFace**: Additive Angular Margin loss architecture generating highly discriminative 512-D embeddings.

---

## Directory Structure

```text
AI-IoT-Smart-Medicine-Box/
├── model/
│   ├── facenet512.npz         # 512-D FaceNet512 normalized embeddings database
│   ├── facenet512.json        # FaceNet512 database metadata & identity tallies
│   ├── arcface.npz            # 512-D ArcFace normalized embeddings database
│   └── arcface.json           # ArcFace database metadata
├── src/
│   ├── recognize.py           # Real-time webcam inference and visualization engine
│   ├── build_embeddings.py    # Offline embedding generator & database compiler
│   ├── capture_enrollment.py  # Interactive webcam enrollment assistant
│   └── face_quality.py        # Image quality metrics (sharpness, lighting, CLAHE)
├── requirements.txt           # Python package dependencies
├── AI_EXTRACTION_REPORT.md    # Full extraction and file audit report
├── LICENSE                    # MIT License
├── .gitignore                 # Python and environment ignores
└── README.md                  # Project documentation
```

---

## Installation

### Prerequisites
- Python 3.10, 3.11, or 3.12 (64-bit recommended)
- A connected USB or integrated webcam
- (Optional) NVIDIA GPU with CUDA support for accelerated inference (TensorFlow automatically falls back to CPU if no GPU is found).

### Setup Virtual Environment

```bash
# Clone or navigate to the repository
cd AI-IoT-Smart-Medicine-Box

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## How to Run

### 1. Real-Time Face Recognition (Inference)

Run real-time webcam recognition with the default CPU-optimized `low` profile and FaceNet512 model:

```bash
python src/recognize.py
```

#### Running with High-Accuracy Profile (RetinaFace):
```bash
python src/recognize.py --profile quality
```

#### Running with Optional Security Features:
```bash
# Require head movement challenge and anti-spoofing check
python src/recognize.py --require-motion --deep-liveness
```

#### Running with Caregiver / Admin PIN Fallback:
```powershell
# Set PIN in the current session
$env:FACE_AI_PIN = "1234"
python src/recognize.py
```
*Press `P` in the camera window to enter PIN mode, type your PIN, and press `Enter`.*

#### Additional CLI Options:
| Flag | Description | Default |
|---|---|---|
| `--database <path>` | Path to precomputed `.npz` database | `model/facenet512.npz` |
| `--camera <int>` | OpenCV camera device index | `0` |
| `--profile <str>` | Performance profile (`low`, `balanced`, `quality`) | `low` |
| `--detector <str>` | Force detector (`yunet`, `retinaface`, `mtcnn`) | Profile default |
| `--no-voice` | Disable spoken audio guidance | `False` |
| `--show-fps` | Overlay live frame rate counter on the feed | `False` |
| `--require-motion` | Require head rotation before authorization | `False` |
| `--deep-liveness` | Run DeepFace anti-spoof classifier on match | `False` |

---

### 2. Enrolling New Users

#### Step A: Capture Face Samples via Webcam
Run the interactive enrollment utility to capture 12–20 clear, varied face crops:

```bash
python src/capture_enrollment.py --name "Alex" --target 15
```
- Stand in front of the camera.
- Position your face inside the bounding box.
- Press **Space** to save clear samples with slight head turns, different expressions, or with/without glasses.
- Press **q** to finish. Images are saved to `faces/<Name>/`.

#### Step B: Compile Embeddings Database
Rebuild the embeddings database from the captured photos:

```bash
python src/build_embeddings.py --faces-dir faces --output model/facenet512.npz --detector retinaface
```

To build with the **ArcFace** model instead:
```bash
python src/build_embeddings.py --faces-dir faces --output model/arcface.npz --model ArcFace
```

---

## Technical Details & Thresholds

### Cosine Matching Policy
FaceNet512 produces unit-length vectors ($\|v\|_2 = 1$). The cosine similarity is computed directly via vector dot-product:

$$\text{similarity} = \mathbf{e}_{\text{live}} \cdot \mathbf{e}_{\text{enrolled}}$$

Scores are aggregated using the **median of the top 3 matches** for each enrolled person.

| Similarity Range | Outcome | Action |
|---|---|---|
| **$\ge 0.70$** | `AUTHORIZED` | Identity confirmed and displayed in green. |
| **$0.60 - 0.70$** | `LOW CONFIDENCE` | Amber warning banner; prompts user to hold still. |
| **$< 0.60$** | `UNKNOWN` | Red alert; authorization rejected. |

*Note: The authorization threshold dynamically scales up if the captured face crop exhibits low sharpness, contrast, or lighting.*

### Performance Profiles
- **`low`**: 1280x720 capture, 480x270 YuNet detection every 8 frames, IoU tracking, CPU thread limit. Ideal for older quad-core / low-voltage CPUs.
- **`balanced`**: 1280x720 capture, 640x360 YuNet detection every 6 frames. Balanced responsiveness and accuracy.
- **`quality`**: 1280x720 capture, 960x540 RetinaFace detection every 5 frames. Recommended for systems with an NVIDIA GPU.

---

## Troubleshooting

1. **Webcam Does Not Open**:
   - Ensure no other application (Zoom, Teams, Camera app) is using the webcam.
   - On Windows, the system defaults to `cv2.CAP_DSHOW`. Pass `--camera 1` if you have multiple video devices.
2. **Audio Prompts Fail or Produce Errors**:
   - Voice guidance requires an active internet connection for Edge-TTS synthesis and `pygame` for playback. Pass `--no-voice` to disable speech entirely.
3. **Low Recognition Accuracy / False Rejections**:
   - Check the lighting status on the screen banner. Avoid backlit setups (e.g. bright window behind you).
   - Capture 15–20 high-quality photos per person covering normal head positions and glasses if worn.
   - Re-run `build_embeddings.py` using `--detector retinaface`.

---

## License
This project is open-source under the terms of the [MIT License](LICENSE).
