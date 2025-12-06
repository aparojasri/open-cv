import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class FaceEncoder:
    """
    Lightweight Face Embedding Extractor (MobileFaceNet / ArcFace style).

    If ONNX model or onnxruntime is not available, falls back to a
    random but consistent-sized embedding so the rest of the pipeline
    keeps working for demo purposes.
    """

    def __init__(self, model_path: str = "models/mobile_face.onnx"):
        self.available = False
        self.session = None
        self.input_name = None

        if ort is None:
            print("[FaceEncoder] onnxruntime not installed. Using fallback embeddings.")
            return

        try:
            self.session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name
            self.available = True
            print("[FaceEncoder] Loaded ONNX face model.")
        except Exception as e:
            print(f"[FaceEncoder WARNING] Could not load model: {e}")
            print(" → Using fallback random embeddings.")
            self.available = False

    def _preprocess(self, face_roi: np.ndarray) -> np.ndarray:
        """Convert face crop into normalized NCHW tensor for ONNX model."""
        img = cv2.resize(face_roi, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        return img[np.newaxis, ...]

    def get_embedding(self, face_roi: np.ndarray) -> np.ndarray | None:
        """
        Returns a normalized feature vector (e.g., 128D/512D) or None.
        """
        if face_roi is None or face_roi.size == 0:
            return None

        # Fallback: random vector
        if not self.available or self.session is None:
            vec = np.random.rand(128).astype(np.float32)
            return vec / np.linalg.norm(vec)

        blob = self._preprocess(face_roi)
        pred = self.session.run(None, {self.input_name: blob})[0]
        emb = pred.flatten().astype(np.float32)

        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        return emb
