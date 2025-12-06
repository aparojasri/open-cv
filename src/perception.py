import os
import cv2
import mediapipe as mp
import numpy as np

from .geometry import GeometryEngine
from .gestures import GestureEngine
from .tracking import KalmanTracker, StateSmoother


class DeepEmotionModel:
    """
    Real emotion model wrapper.

    - Loads a Keras or ONNX model from disk (prefers Keras).
    - Expects a cropped face ROI (BGR image).
    - Returns one of: Angry / Disgust / Fear / Happy / Sad / Surprise / Neutral.
    """

    def __init__(self,
                 model_path: str = "models/emotion_fer.tflite",
                 input_size=(48, 48)):
        self.model_path = model_path
        self.input_size = input_size  # (width, height)
        self.model_loaded = False
        self.backend = None  # "keras" or "onnx"

        # fixed class names list (7 classes typical for many FER datasets)
        self.class_names = [
            "Angry",
            "Disgust",
            "Fear",
            "Happy",
            "Sad",
            "Surprise",
            "Neutral",
        ]

        # If model file doesn't exist -> heuristic-only (safe)
        if not os.path.exists(self.model_path):
            print(f"[DeepEmotionModel] Model not found at {self.model_path}. "
                  f"Falling back to heuristic-only mode.")
            return

        # Try Keras first if .keras or .h5
        if self.model_path.endswith(".keras") or self.model_path.endswith(".h5"):
            try:
                import tensorflow as tf  # lazy import
                self._keras_model = tf.keras.models.load_model(self.model_path)
                self.backend = "keras"
                self.model_loaded = True
                print(f"[DeepEmotionModel] Loaded Keras model from {self.model_path}")

                # Optional sanity-check: model output size
                try:
                    out_shape = self._keras_model.output_shape
                    # output_shape could be (None, N) or nested; handle common case
                    if isinstance(out_shape, tuple) and len(out_shape) >= 2:
                        out_classes = int(out_shape[-1])
                        if out_classes != len(self.class_names):
                            print(f"[DeepEmotionModel] Warning: Keras model outputs {out_classes} classes "
                                  f"but class_names has {len(self.class_names)} entries.")
                except Exception:
                    pass

                return
            except Exception as e:
                print(f"[DeepEmotionModel] Failed to load Keras model: {e}")

        # Fallback: ONNX via OpenCV DNN
        if self.model_path.endswith(".onnx"):
            try:
                self._onnx_net = cv2.dnn.readNetFromONNX(self.model_path)
                self.backend = "onnx"
                self.model_loaded = True
                print(f"[DeepEmotionModel] Loaded ONNX model from {self.model_path}")
                # Can't easily check output classes for ONNX via cv2.dnn without a forward pass,
                # but you can run a dummy forward later if needed.
                return
            except Exception as e:
                print(f"[DeepEmotionModel] Failed to load ONNX model: {e}")

        print("[DeepEmotionModel] No valid model backend loaded. "
              "Using heuristic-only mode.")

    def _preprocess_face(self, face_roi):
        """
        Convert BGR ROI -> model input tensor for Keras (grayscale 48x48).
        Default assumption: FER-like 48x48 grayscale, scaled to [0,1].
        """
        if face_roi is None or face_roi.size == 0:
            return None

        # Convert to grayscale
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            print(f"[DeepEmotionModel] _preprocess_face cvtColor error: {e}")
            return None

        gray = cv2.resize(gray, self.input_size)

        # Normalize
        img = gray.astype("float32") / 255.0

        # Shape for Keras: (1, H, W, 1)
        img = np.expand_dims(img, axis=-1)  # (H, W, 1)
        img = np.expand_dims(img, axis=0)   # (1, H, W, 1)
        return img

    def predict(self, face_roi):
        """
        Predict emotion label from face ROI.
        Returns a string like "Happy" / "Sad" / "Neutral".
        """
        if not self.model_loaded or self.backend is None:
            return "Neutral"  # safe default

        if face_roi is None:
            return "Neutral"

        if self.backend == "keras":
            x = self._preprocess_face(face_roi)
            if x is None:
                return "Neutral"

            try:
                preds = self._keras_model.predict(x, verbose=0)[0]
                cls_idx = int(np.argmax(preds))
                if 0 <= cls_idx < len(self.class_names):
                    return self.class_names[cls_idx]
                else:
                    print(f"[DeepEmotionModel] Keras returned invalid class idx {cls_idx}")
                return "Neutral"
            except Exception as e:
                print(f"[DeepEmotionModel] Keras inference error: {e}")
                return "Neutral"

        elif self.backend == "onnx":
            # ONNX models often expect RGB and possibly different size / channels.
            # We use blobFromImage to be safe and allow swapping channels, scaling, cropping.
            try:
                # Resize to expected input size
                # blobFromImage expects (width, height)
                blob = cv2.dnn.blobFromImage(
                    image=face_roi,
                    scalefactor=1.0/255.0,
                    size=self.input_size,
                    mean=(0, 0, 0),
                    swapRB=True,  # convert BGR->RGB if model expects RGB
                    crop=False,
                )
                self._onnx_net.setInput(blob)
                preds = self._onnx_net.forward()
                # preds might be shape (1, N) or (N,) depending on model
                if preds is None:
                    print("[DeepEmotionModel] ONNX returned no predictions.")
                    return "Neutral"

                preds = np.array(preds).squeeze()
                # If preds is scalar, handle it
                if preds.ndim == 0:
                    cls_idx = int(preds)
                else:
                    cls_idx = int(np.argmax(preds))

                if 0 <= cls_idx < len(self.class_names):
                    return self.class_names[cls_idx]
                else:
                    print(f"[DeepEmotionModel] ONNX returned invalid class idx {cls_idx}")
                return "Neutral"
            except Exception as e:
                print(f"[DeepEmotionModel] ONNX inference error: {e}")
                return "Neutral"

        return "Neutral"


class VisionSystem:
    """
    High-level perception system:

    - Face mesh & head pose (pitch, yaw)
    - Eye aspect ratio (EAR) -> drowsiness
    - Kalman-tracked face center
    - Smoothed pose & EAR
    - Hand gestures via GestureEngine
    - Deep emotion model (FER) + heuristic attention
    """

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        # If running as script or older mediapipe may not have refine_landmarks param
        try:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.7,
            )
        except TypeError:
            # Older versions or different builds may not accept refine_landmarks
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.7,
            )

        self.gesture_engine = GestureEngine()
        self.geo = GeometryEngine()
        self.smoother = StateSmoother()
        self.tracker = KalmanTracker()

        # 👉 Update this path if your model file has a different name
        self.emotion_net = DeepEmotionModel(
            model_path="models/emotion_fer.keras",
            input_size=(48, 48),
        )

        # Anti-flicker for face: hold last face data a few frames
        self._last_face_data = None
        self._face_missing_frames = 0
        self._face_max_hold_frames = 5

    # ------------------------------------------------------------------ #
    # Main processing entry
    # ------------------------------------------------------------------ #
    def process(self, image):
        img_h, img_w, _ = image.shape
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        packet = {
            "face_found": False,
            "hands": [],
            "face_data": {},
            "perf_metrics": {},
        }

        # -------------------- Hands --------------------
        try:
            packet["hands"] = self.gesture_engine.process(img_rgb)
        except Exception as e:
            print(f"[VisionSystem] GestureEngine error: {e}")
            packet["hands"] = []

        # -------------------- Face ---------------------
        results = self.face_mesh.process(img_rgb)

        if results.multi_face_landmarks:
            self._face_missing_frames = 0
            face_landmarks = results.multi_face_landmarks[0]
            lm = face_landmarks.landmark

            # Head pose
            pitch, yaw = self.geo.solve_pnp_head_pose(lm, img_w, img_h)

            # EAR (eyes)
            r_idxs = [33, 160, 158, 133, 153, 144]
            l_idxs = [362, 385, 387, 263, 373, 380]
            ear_l = self.geo.calculate_ear(lm, l_idxs, img_w, img_h)
            ear_r = self.geo.calculate_ear(lm, r_idxs, img_w, img_h)
            avg_ear = (ear_l + ear_r) / 2.0

            # Smooth signals
            self.smoother.update(pitch, yaw, avg_ear)
            s_pitch, s_yaw, s_ear = self.smoother.get_smoothed_values()

            # Kalman track nose
            nose_x = int(lm[1].x * img_w)
            nose_y = int(lm[1].y * img_h)
            kx, ky = self.tracker.update(nose_x, nose_y)

            # Crop face ROI for emotion model
            face_roi = None
            try:
                face_roi = self.geo.crop_face(image, lm)
            except Exception as e:
                print(f"[VisionSystem] crop_face error: {e}")
                face_roi = None

            # Deep emotion model
            base_emotion = self.emotion_net.predict(face_roi)

            # --------------- Attention / Drowsiness fusion ---------------
            # Start with model's class, then adjust by pose & EAR
            emotion_state = base_emotion

            # Eyes closed / very low EAR
            if s_ear < 0.18:
                emotion_state = "Drowsy / Eyes Closed"

            # Looking far away from camera → distracted
            elif abs(s_pitch) > 20 or abs(s_yaw) > 25:
                emotion_state = f"{base_emotion} (Distracted)"

            face_data = {
                "pitch": float(s_pitch),
                "yaw": float(s_yaw),
                "ear": float(s_ear),
                "center": (int(kx), int(ky)),
                "emotion_state": emotion_state,
                "raw_emotion": base_emotion,
                "landmarks": face_landmarks,
            }

            self._last_face_data = face_data
            packet["face_found"] = True
            packet["face_data"] = face_data
            return packet

        # ------------------------------------------------------------------
        # No face this frame -> use last face for a few frames to avoid flicker
        # ------------------------------------------------------------------
        self._face_missing_frames += 1

        if (
            self._last_face_data is not None
            and self._face_missing_frames <= self._face_max_hold_frames
        ):
            packet["face_found"] = True
            packet["face_data"] = self._last_face_data
            return packet

        # Too long without face: fall back to pure prediction (no mesh)
        px, py = self.tracker.predict()
        packet["face_found"] = False
        packet["face_data"] = {
            "center": (int(px), int(py)),
        }
        return packet
