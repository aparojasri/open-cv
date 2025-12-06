# Companion Vision OS (CV-OS) — Social Perception Engine

![Status](https://img.shields.io/badge/Status-Prototype-orange) ![Python](https://img.shields.io/badge/Python-3.10-blue) ![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)

**Companion Vision OS** is a modular perception framework designed for social robotics. It aims to bridge the gap between raw visual data and high-level social cues (like "boredom" or "attention") using lightweight computer vision techniques.

> **Note:** This project is currently in the **Active Prototyping Phase**. The core architecture is established, and modules are being iteratively optimized for latency and edge-deployment.

---

## 🧠 The Vision
Social robots (like Miko) require more than just object detection; they need **Social Perception**. This engine is designed to answer:
* *"Is the user looking at me?"* (Head Pose/Gaze)
* *"Is the user engaged or sleepy?"* (EAR/Drowsiness)
* *"Is the user gesturing to me?"* (Hand Tracking)

---

## 🏗 Architecture & Modules

The system is designed with a modular pipeline approach to ensure independent testing of vision tasks.

### 1. Facial Analysis Module (Core)
* **Face Mesh:** Utilizes 468-point landmarks for high-fidelity surface geometry.
* **Head Pose Estimation:** Solves PnP (Perspective-n-Point) problems to determine Pitch, Yaw, and Roll.
* **Attention Metric:** correlates face orientation with camera vectors to estimate "Focus."

### 2. State & Profiling (In Progress)
* **Kalman Filtering:** Implementing linear quadratic estimation to smooth jittery landmark detection in low-light conditions.
* **Telemetry HUD:** A planned overlay system to visualize FPS, inference latency (ms), and P99 metrics for performance debugging.

### 3. Gesture Recognition
* **Pipeline:** Hand landmark extraction → Vector normalization → Heuristic classification.
* **Goals:** Real-time recognition of "Stop," "Wave," and "Pointer" gestures for robot control.
