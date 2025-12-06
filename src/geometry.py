import math
from typing import List, Sequence

import cv2
import numpy as np


class GeometryEngine:
    """
    Face geometry utilities:
    - Head pose via solvePnP
    - Eye Aspect Ratio (EAR)
    - Mouth Aspect Ratio (MAR)
    - Face cropping from landmarks
    """

    def __init__(self) -> None:
        # 3D model reference points (approximate)
        self.model_points_3d = np.array(
            [
                (0.0, 0.0, 0.0),        # Nose tip
                (0.0, -63.6, -12.5),    # Chin
                (-43.3, 32.7, -26.0),   # Left eye left corner
                (43.3, 32.7, -26.0),    # Right eye right corner
                (-28.9, -28.9, -24.1),  # Left mouth corner
                (28.9, -28.9, -24.1),   # Right mouth corner
            ],
            dtype=np.float32,
        )

        # Indices in MediaPipe FaceMesh for above points
        self.landmark_indices = [1, 152, 33, 263, 61, 291]

        self.focal_length_scale = 1.0

    # ---------------- Head Pose ----------------
    def solve_pnp_head_pose(
        self,
        landmarks: Sequence,
        img_w: int,
        img_h: int,
    ) -> tuple[float, float]:
        """
        Compute head pose (pitch, yaw) using PnP with a 3D model.
        Returns (pitch_deg, yaw_deg).
        """
        image_points = []
        for idx in self.landmark_indices:
            lm = landmarks[idx]
            x = lm.x * img_w
            y = lm.y * img_h
            image_points.append((x, y))

        image_points = np.array(image_points, dtype=np.float32)

        focal_length = img_w * self.focal_length_scale
        center = (img_w / 2.0, img_h / 2.0)
        camera_matrix = np.array(
            [
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        dist_coeffs = np.zeros((4, 1), dtype=np.float32)

        success, rvec, _ = cv2.solvePnP(
            self.model_points_3d,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return 0.0, 0.0

        R, _ = cv2.Rodrigues(rvec)
        sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

        singular = sy < 1e-6
        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])  # pitch
            y = math.atan2(-R[2, 0], sy)      # yaw
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)

        pitch_deg = math.degrees(x)
        yaw_deg = math.degrees(y)
        return pitch_deg, yaw_deg

    # ---------------- EAR ----------------
    def calculate_ear(
        self,
        landmarks: Sequence,
        indices: List[int],
        img_w: int,
        img_h: int,
    ) -> float:
        """
        Compute Eye Aspect Ratio given 6 landmark indices.
        indices = [p1, p2, p3, p4, p5, p6]
        """
        pts = []
        for idx in indices:
            lm = landmarks[idx]
            pts.append((lm.x * img_w, lm.y * img_h))

        p1, p2, p3, p4, p5, p6 = pts

        def dist(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        A = dist(p2, p6)
        B = dist(p3, p5)
        C = dist(p1, p4)

        ear = (A + B) / (2.0 * C + 1e-6)
        return float(ear)

    # ---------------- MAR ----------------
    def calculate_mar(
        self,
        landmarks: Sequence,
        indices: List[int],
        img_w: int,
        img_h: int,
    ) -> float:
        """
        Mouth Aspect Ratio (MAR).
        Typical indices example:
            [78, 308, 13, 14, 82, 312]
        (horizontal corners + upper/lower mid + side midpoints)
        """
        pts = []
        for idx in indices:
            lm = landmarks[idx]
            pts.append((lm.x * img_w, lm.y * img_h))

        # p1, p2: corners (left, right)
        # p3, p4: vertical inner top/bottom
        # p5, p6: side verticals (for robustness)
        p1, p2, p3, p4, p5, p6 = pts

        def dist(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        horiz = dist(p1, p2) + 1e-6  # avoid /0
        v1 = dist(p3, p4)
        v2 = dist(p5, p6)

        mar = (v1 + v2) / (2.0 * horiz)
        return float(mar)

    # ---------------- Face Crop ----------------
    def crop_face(self, img, landmarks: Sequence):
        """
        Crop face region using min/max of all landmarks.
        """
        h, w = img.shape[:2]
        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]

        x1 = int(max(0, min(xs) - 10))
        x2 = int(min(w, max(xs) + 10))
        y1 = int(max(0, min(ys) - 10))
        y2 = int(min(h, max(ys) + 10))

        if x2 <= x1 or y2 <= y1:
            return None

        face = img[y1:y2, x1:x2]
        return face if face.size > 0 else None
