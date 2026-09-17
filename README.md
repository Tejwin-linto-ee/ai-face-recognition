# Multimodal Biometric Smart Hub

A high-performance, real-time edge facial recognition and access control terminal powered by **SCRFD / YOLOv8-Face**, **ByteTrack**, **FaceNet512 / ArcFace**, and **MiniFASNet Anti-Spoofing**.

---

## Key Highlights

- **Domain-Generalized Architecture**: Designed for physical access control, secure terminal unlocking, and user identity verification with structured security audit logging (`system_audit_log`) and hardware triggers (`unlock_secure_state`).
- **SCRFD / YOLOv8-Face Detection**: State-of-the-art face detector running via `onnxruntime` with dynamic `CUDAExecutionProvider` on NVIDIA GPUs and optimized multi-threaded `CPUExecutionProvider` for laptops.
- **ByteTrack Multi-Face Association**: Pure Python + NumPy implementation of ByteTrack (Zhang et al., 2022) with Kalman filter motion modeling. Tracks identities smoothly through partial occlusions, motion blur, and extreme head rotations.
- **True Multiprocessing Concurrency**: Heavy AI detection and embedding inference execute inside an isolated `multiprocessing.Process` sampling every 3rd frame through an IPC queue, ensuring the main camera loop runs at a fluid 30+ FPS without frame drops.
- **Multi-Level Spoof Prevention**:
  - Motion challenge requiring center shift or face scale change.
  - Deep learning anti-spoofing verification (MiniFASNet).
  - High-security HMAC-verified PIN override (`SMART_HUB_PIN`).
- **Natural Voice Guidance**: Synthesizes asynchronous spoken guidance prompts via Edge-TTS and Pygame.

---

## Architecture Pipeline

```text
Webcam Stream (1280x720 @ 30 FPS)
       │
       ├────────────────────────────────────────┐
       │ (Every 3rd frame)                      │ (Every frame)
       ▼                                        ▼
[multiprocessing.Queue]                  ByteTrack Multi-Object
       │                                 Kalman Filter Prediction
       ▼                                        │
BiometricInferenceProcess (Child Process)        ▼
  ├─ SCRFD ONNX (CUDA / CPU)             Update Active Tracks
  ├─ Quality Gate (Sharpness & Light)           │
  ├─ Adaptive CLAHE/Gamma Enhancement           ▼
  ├─ FaceNet512 / ArcFace (512-D)         OpenCV Real-Time HUD
  └─ MiniFASNet Anti-Spoofing                   │
       │                                        ▼
       ▼                                 Access Decision Gate
[multiprocessing.Queue Response]          ├─ unlock_secure_state()
       │                                  └─ system_audit_log()
       └────────────────────────────────────────┘
```

---

## Directory Structure

```text
AI-IoT-Smart-Medicine-Box/
├── model/
│   ├── facenet512.npz          # 101 FaceNet512 precomputed embeddings (512-D float32)
│   ├── facenet512.json         # FaceNet512 metadata
│   ├── arcface.npz             # 101 ArcFace precomputed embeddings (512-D float32)
│   └── arcface.json            # ArcFace metadata
├── src/
│   ├── recognize.py            # Main access terminal, ByteTracker, HUD, & loop
│   ├── worker.py               # Multiprocessing inference worker (SCRFD + FaceNet512)
│   ├── detector.py             # SCRFD & YOLOv8-Face ONNXRuntime engine
│   ├── bytetrack.py            # ByteTrack multi-face association algorithm
│   ├── build_embeddings.py     # Offline database generator
│   ├── capture_enrollment.py   # Multi-angle face capture tool
│   └── face_quality.py         # Image quality assessment & CLAHE enhancement
├── requirements.txt            # Python dependencies
├── AI_EXTRACTION_REPORT.md     # Full file audit and dependency report
├── LICENSE                     # MIT License
└── README.md                   # Documentation
```

---

## Installation

```bash
# 1. Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Running the Terminal

### 1. Start Access Terminal
```bash
python src/recognize.py
```

### 2. High-Security Mode (Motion Challenge + Deep Anti-Spoofing)
```bash
python src/recognize.py --require-motion --deep-liveness --show-fps
```

### 3. Using Custom SCRFD ONNX Weights
```bash
python src/recognize.py --detector-onnx model/scrfd_2.5g.onnx
```

### 4. Setting Security Override PIN
```powershell
$env:SMART_HUB_PIN = "9876"
python src/recognize.py
```
*Press `P` in the camera view to activate PIN mode, enter PIN, and press `Enter`.*

---

## Enrolling Authorized Users

```bash
# Step 1: Capture 15 varied face samples
python src/capture_enrollment.py --name "Alice" --target 15

# Step 2: Compile embeddings database
python src/build_embeddings.py --faces-dir faces --output model/facenet512.npz
```

---

## License
MIT License.
