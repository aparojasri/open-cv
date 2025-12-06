# Companion Vision OS (CV-OS) — Social Perception Engine

### Real-time Face Mesh • Head Pose • Drowsiness (EAR) • Hand Gesture Recognition • Kalman Tracking • Profiling HUD

This project is a computer-vision perception engine built for real-time human-robot interaction.

It performs live facial analysis, attention estimation, gesture recognition, and state profiling using **OpenCV**, **MediaPipe**, and **Streamlit WebRTC**. It serves as a debugger and prototype platform for social robots (e.g., Miko), ADAS-inspired perception systems, and interaction research.

---

## 🚀 Live Demo Features

### ✔ Real-time Detection
* **Face Mesh (468-point):** High-fidelity facial landmarking.
* **Head Pose Estimation:** Real-time Pitch, Yaw, and Roll calculation.
* **Drowsiness Detection:** Eye Aspect Ratio (EAR) monitoring.
* **Hand Analysis:** Landmarks + Gesture Classification (static & motion).
* **Tracking:** Face Center Tracking with **Kalman Smoothing**.
* **Performance:** Per-frame Profiling (FPS, Latency, P99, Jitter).

### ✔ Debugger HUD
* **State Indicators:** "Focused", "Distracted", "Drowsy", etc.
* **Telemetry HUD:** Live FPS, Latency, and Jitter metrics.
* **Visual Overlays:** Gesture labels, face center markers, and smoothed trajectory paths.

---

## 📁 Project Structure

```bash
.
│── app.py                     # Streamlit WebRTC Debug App
│── requirements.txt           # Light dependency set
│── README.md                  # This file
│
├── src/
│   ├── perception.py          # Face, hand, pose, EAR processing pipeline
│   ├── gestures.py            # Gesture Engine (static & motion)
│   ├── geometry.py            # Head pose estimation & math utilities
│   ├── tracking.py            # Kalman tracker + State smoothing
│   ├── profiling.py           # Real-time FPS/Latency profiler
│   ├── analytics.py           # Async CSV/JSONL logger
│   └── ui_controls.py         # Streamlit sidebar controls
│
├── models/                    # Place ONNX / trained models here
├── logs/                      # Runtime logs (auto-generated)
└── demo/
    ├── screenshot.png         # Add your UI screenshot
    └── demo.mp4               # Add a short demo video
🛠 Setup & Installation
1. Create a virtual environment
Bash

python -m venv .venv
Windows: .venv\Scripts\activate

macOS/Linux: source .venv/bin/activate

2. Install dependencies
Bash

pip install -r requirements.txt
3. Run the app
Bash

streamlit run app.py
Then open the browser window that appears (usually http://localhost:8501).

🎥 How to Use
Click Select Device → choose your webcam.

Click Start.

Observe the overlays:

Face mesh + head pose angles.

Eye aspect ratio (EAR) values.

Gesture labels.

Smoothed tracking point.

View real-time FPS, P99, jitter in the sidebar.

Events get logged automatically in the logs/ folder.

🤖 Tech Stack
Language: Python 3.10+

Core Vision: OpenCV (Geometry, Image Processing), MediaPipe (Landmarks)

Frontend: Streamlit + WebRTC (Low-latency streaming)

Algorithms: Kalman Filter (Trajectory Smoothing)

System: Async Logger (Telemetry), Custom Profiler (Performance Metrics)

📌 Future Extensions (Planned)
The architecture is designed to support:

[ ] Emotion classifier (custom CNN/ViT)

[ ] Object detection (YOLOv8 / SSD)

[ ] Face ID embeddings (MobileFaceNet)

[ ] TensorRT optimization for Jetson deployment

[ ] Multi-person tracking with DeepSORT

📜 License
MIT License
