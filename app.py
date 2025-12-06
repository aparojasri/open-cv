import traceback

import av
import cv2
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

from src.perception import VisionSystem
from src.profiling import EdgeProfiler
from src.analytics import AsyncSessionLogger

# -----------------------------------------------------------------------------
# Streamlit Page Config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Robot Vision Debugger",
    layout="wide",
    page_icon="🤖",
)

st.title("🤖 Robot Vision Debugger")
st.markdown(
    """
Monitor **facial expression**, **attention**, **gestures**, **identity** and
**performance** in real-time.

**How to use**

1. Click **SELECT DEVICE** and choose your webcam.  
2. Click **START** to begin the vision pipeline.  
3. Look at the camera and raise your hand to see overlays and gesture labels.  
"""
)

# -----------------------------------------------------------------------------
# Video Processor
# -----------------------------------------------------------------------------
class RobotVisionProcessor(VideoProcessorBase):
    def __init__(self):
        try:
            self.vision = VisionSystem()
            self.profiler = EdgeProfiler()
            self.logger = AsyncSessionLogger()

            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_hands = mp.solutions.hands
            self.mp_face = mp.solutions.face_mesh

            # Start async logger thread
            self.logger.start()

            self.init_success = True
            self.init_error = ""
        except Exception as e:
            print(f"[Init Error] {e}")
            traceback.print_exc()
            self.init_success = False
            self.init_error = str(e)

    def recv(self, frame):
        # Always start with a valid frame buffer
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        try:
            # Init failure: just overlay error and return
            if not self.init_success:
                cv2.putText(
                    img,
                    "INIT FAILED",
                    (50, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    img,
                    self.init_error[:60],
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
                return av.VideoFrame.from_ndarray(img, format="bgr24")

            # 1) Profiler start
            self.profiler.tick()

            # 2) Main perception
            packet = self.vision.process(img)

            # Defaults used if no face
            current_state = "UNKNOWN"
            pitch = yaw = ear = 0.0

            # ------------------------------------------------------------------
            # Hands
            # ------------------------------------------------------------------
            if "hands" in packet:
                for hand in packet["hands"]:
                    landmarks = hand["landmarks"]

                    # Skeleton
                    self.mp_drawing.draw_landmarks(
                        img,
                        landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                    )

                    gesture_text = hand.get("gesture", "UNKNOWN")
                    hand_id = hand.get("id", "?")

                    # Green for motion, yellow for static
                    color = (
                        (0, 255, 0)
                        if ("SWIPE" in gesture_text or "WAVING" in gesture_text)
                        else (0, 255, 255)
                    )

                    h, w, _ = img.shape
                    wrist = landmarks.landmark[0]
                    cx, cy = int(wrist.x * w), int(wrist.y * h)

                    cv2.putText(
                        img,
                        f"{hand_id}: {gesture_text}",
                        (cx - 40, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

                    if gesture_text and gesture_text != "UNKNOWN":
                        self.logger.log_event(
                            "GESTURE", {"id": hand_id, "name": gesture_text}
                        )

            # ------------------------------------------------------------------
            # Face
            # ------------------------------------------------------------------
            if packet.get("face_found", False):
                fd = packet["face_data"]

                current_state = fd.get("emotion_state", "Unknown")
                pitch = float(fd.get("pitch", 0.0))
                yaw = float(fd.get("yaw", 0.0))
                ear = float(fd.get("ear", 0.0))
                center = fd.get("center", (0, 0))

                # Face mesh
                self.mp_drawing.draw_landmarks(
                    img,
                    fd["landmarks"],
                    self.mp_face.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles.
                    get_default_face_mesh_tesselation_style(),
                )

                # Kalman target (yellow)
                cv2.circle(img, center, 5, (0, 255, 255), -1)

                # Parse emotion vs attention from current_state
                try:
                    expr, attn = current_state.split("/", maxsplit=1)
                    expr = expr.strip()
                    attn = attn.strip()
                except ValueError:
                    expr = current_state
                    attn = ""

                # Color: based on attention part so AsyncSessionLogger
                # still sees "Drowsy" or "Distracted" in the string.
                color = (0, 255, 0)
                if "Distracted" in current_state:
                    color = (0, 165, 255)
                if "Drowsy" in current_state:
                    color = (0, 0, 255)

                # Emotion & attention HUD
                cv2.putText(
                    img,
                    f"Emotion: {expr}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )
                if attn:
                    cv2.putText(
                        img,
                        f"Attention: {attn}",
                        (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (220, 220, 220),
                        1,
                    )
                cv2.putText(
                    img,
                    f"P: {pitch:.1f}  Y: {yaw:.1f}  EAR: {ear:.2f}",
                    (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (220, 220, 220),
                    1,
                )

            # ------------------------------------------------------------------
            # Profiling + logging
            # ------------------------------------------------------------------
            metrics = self.profiler.tock()
            fps = metrics.get("fps", 0)
            p99 = metrics.get("latency_p99", 0)
            jitter = metrics.get("jitter", 0.0)

            self.logger.log_telemetry(pitch, yaw, ear, fps, current_state)

            hud_text = f"FPS: {fps} | P99: {p99} ms | Jitter: {jitter} ms"
            cv2.putText(
                img,
                hud_text,
                (20, img.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        except Exception as e:
            err = str(e)
            print(f"[Runtime Error] {err}")
            traceback.print_exc()
            cv2.putText(
                img,
                "CRASHED!",
                (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                3,
            )
            cv2.putText(
                img,
                err[:60],
                (50, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def __del__(self):
        if hasattr(self, "logger"):
            self.logger.stop()


# -----------------------------------------------------------------------------
# WebRTC configuration (SYNC, no async processing to avoid flicker)
# -----------------------------------------------------------------------------
rtc_config = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="robot-vision-debugger",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_config,
    media_stream_constraints={
        "video": {"width": 640, "height": 480},
        "audio": False,
    },
    video_processor_factory=RobotVisionProcessor,
    async_processing=False,
)

st.markdown(
    """
---

### ℹ️ What this demo is doing

- **Face pipeline**: face mesh, head pose (pitch/yaw), EAR, drowsiness & distraction.  
- **Emotion model**: FER2013 CNN predicting one of 7 emotions in real time.  
- **Hand pipeline**: palm skeleton + static & motion gestures (wave, swipe, etc).  
- **Tracking**: Kalman-filtered face center (yellow dot) and smoothed head pose.  
- **Performance**: real-time FPS, P99 latency and jitter.  
- **Logging**: all frames + gesture events in the `logs/` folder.
"""
)
