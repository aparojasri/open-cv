import math
from collections import deque
from typing import Dict, List, Optional, Tuple

import mediapipe as mp
import numpy as np


class GestureEngine:
    """
    Hand geometry + gesture engine built on top of MediaPipe Hands
    for Companion Vision OS.

    Outputs per detected hand:
        {
            "id": str,                   # e.g. "Left_0"
            "label": "Left" / "Right",
            "score": float,              # MP handedness score
            "landmarks": NormalizedLandmarkList,
            "fingers_open": [Thumb, Index, Middle, Ring, Pinky],
            "gesture": str               # e.g. "OPEN_PALM", "WAVING", ...
        }
    """

    def __init__(self, history_len: int = 20):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.7,
            model_complexity=1,
        )

        # Temporal history for motion gestures
        self.history_len = history_len
        self.motion_buffer: Dict[str, deque] = {}

        # Anti-flicker: keep last output for a FEW frames
        # If set too high, overlays appear "stuck". At low FPS, 1–2 is ideal.
        self._last_output: List[Dict] = []
        self._missing_frames: int = 0
        self._max_hold_frames: int = 2  # frames to hold gestures when hands vanish

    # ------------------------------------------------------------------ #
    # Main entry
    # ------------------------------------------------------------------ #

    def process(self, image_rgb: np.ndarray) -> List[Dict]:
        """
        Run MediaPipe Hands and classify static + motion gestures.
        """
        h, w, _ = image_rgb.shape
        results = self.hands.process(image_rgb)

        hands_output: List[Dict] = []
        active_ids = set()

        if results.multi_hand_landmarks and results.multi_handedness:
            # We see hands this frame → reset missing counter
            self._missing_frames = 0

            for idx, (landmarks, handedness) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)
            ):
                label = handedness.classification[0].label  # "Left" / "Right"
                hand_id = f"{label}_{idx}"
                active_ids.add(hand_id)

                lm_list = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]

                # 1) Per-frame 3D finger states
                finger_states = self._get_finger_states_3d(lm_list)

                # 2) Static gesture
                static_gesture = self._classify_static(finger_states, lm_list, label)

                # 3) Motion gesture (swipes, waving)
                dynamic_gesture = self._analyze_motion(hand_id, lm_list, w, h)

                final_gesture = dynamic_gesture or static_gesture

                hands_output.append(
                    {
                        "id": hand_id,
                        "label": label,
                        "score": handedness.classification[0].score,
                        "landmarks": landmarks,
                        "fingers_open": finger_states,
                        "gesture": final_gesture,
                    }
                )

            # Update motion history and last stable output
            self._prune_history(active_ids)
            self._last_output = hands_output
            return hands_output

        # ------------------------------------------------------------------ #
        # No hands detected this frame
        # → hold last output briefly to reduce flicker at boundaries.
        # ------------------------------------------------------------------ #
        self._missing_frames += 1

        if self._last_output and self._missing_frames <= self._max_hold_frames:
            # DO NOT prune history here; we want a small grace period
            return self._last_output

        # Too long without hands: fully clear state
        self._last_output = []
        self.motion_buffer.clear()
        return []

    # ------------------------------------------------------------------ #
    # Finger state / static gestures
    # ------------------------------------------------------------------ #

    def _get_finger_states_3d(
        self, lm: List[Tuple[float, float, float]]
    ) -> List[int]:
        """
        Returns [Thumb, Index, Middle, Ring, Pinky] as 0/1 states.
        Uses 3D angles at joints, making it robust to rotation.
        """
        states: List[int] = []

        # Thumb: angle at joint (0 - wrist, 2 - thumb CMC, 4 - thumb tip)
        angle_thumb = self._vector_angle(lm[0], lm[2], lm[4])
        states.append(1 if angle_thumb > 150 else 0)

        # Fingers: angle at PIP joint (MCP -> PIP -> TIP)
        finger_indices = [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]
        for (mcp, pip, tip) in finger_indices:
            angle = self._vector_angle(lm[mcp], lm[pip], lm[tip])
            states.append(1 if angle > 160 else 0)

        return states

    def _vector_angle(self, a, b, c) -> float:
        """
        Angle ABC in degrees for 3D points a,b,c.
        """
        v1 = np.array([a[0] - b[0], a[1] - b[1], a[2] - b[2]])
        v2 = np.array([c[0] - b[0], c[1] - b[1], c[2] - b[2]])

        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0.0

        cos_angle = float(np.dot(v1, v2) / (n1 * n2))
        cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
        return math.degrees(math.acos(cos_angle))

    def _classify_static(
        self, states: List[int], lm: List[Tuple], label: str
    ) -> str:
        """
        Map finger open/closed states to coarse static gestures.
        """
        count = sum(states)

        if count == 5:
            return "OPEN_PALM"
        if count == 0:
            return "FIST"

        # Victory: index + middle
        if states[1] and states[2] and not states[3] and not states[4]:
            return "VICTORY"

        # Pointing: index only
        if states[1] and count == 1:
            return "POINTING"

        # Rock on: index + pinky
        if states[1] and states[4] and not states[2] and not states[3]:
            return "ROCK_ON"

        # Call me: thumb + pinky
        if states[0] and states[4] and not states[1] and not states[2] and not states[3]:
            return "CALL_ME"

        # Thumbs up / down: only thumb
        if (
            states[0]
            and not states[1]
            and not states[2]
            and not states[3]
            and not states[4]
        ):
            thumb_tip_y = lm[4][1]
            thumb_ip_y = lm[3][1]
            return "THUMBS_UP" if thumb_tip_y < thumb_ip_y else "THUMBS_DOWN"

        # OK sign: thumb tip close to index tip, others open
        dist_ok = math.hypot(lm[4][0] - lm[8][0], lm[4][1] - lm[8][1])
        if dist_ok < 0.05 and states[2] and states[3] and states[4]:
            return "OK_SIGN"

        return "UNKNOWN"

    # ------------------------------------------------------------------ #
    # Motion gestures (swipe, waving)
    # ------------------------------------------------------------------ #

    def _analyze_motion(
        self, hand_id: str, lm_list: List[Tuple], w: int, h: int
    ) -> Optional[str]:
        """
        Analyze wrist trajectory over recent frames for coarse dynamic gestures.
        """
        wrist = lm_list[0]
        curr_pos = (wrist[0] * w, wrist[1] * h)

        if hand_id not in self.motion_buffer:
            self.motion_buffer[hand_id] = deque(maxlen=self.history_len)

        self.motion_buffer[hand_id].append(curr_pos)
        buffer = self.motion_buffer[hand_id]

        # Require enough history for reliable motion pattern
        if len(buffer) < self.history_len:
            return None

        # 1) Net displacement
        dx_net = buffer[-1][0] - buffer[0][0]
        dy_net = buffer[-1][1] - buffer[0][1]
        dist_net = math.hypot(dx_net, dy_net)

        # Movement threshold (15% of frame width)
        min_move = w * 0.15
        if dist_net > min_move:
            if abs(dx_net) > abs(dy_net):
                return "SWIPE_RIGHT" if dx_net > 0 else "SWIPE_LEFT"
            else:
                return "SWIPE_DOWN" if dy_net > 0 else "SWIPE_UP"

        # 2) Oscillation in X → Waving
        x_vals = [p[0] for p in buffer]
        inflections = 0
        direction = 0  # 0 = unknown, 1 = positive, -1 = negative

        for i in range(1, len(x_vals)):
            diff = x_vals[i] - x_vals[i - 1]
            if abs(diff) < 2.0:
                # small noise
                continue

            new_dir = 1 if diff > 0 else -1
            if direction != 0 and new_dir != direction:
                inflections += 1
            direction = new_dir

        path_len = sum(
            math.hypot(buffer[i][0] - buffer[i - 1][0], buffer[i][1] - buffer[i - 1][1])
            for i in range(1, len(buffer))
        )

        if inflections > 2 and path_len > w * 0.2:
            return "WAVING"

        return None

    # ------------------------------------------------------------------ #
    # Housekeeping
    # ------------------------------------------------------------------ #

    def _prune_history(self, active_ids: set) -> None:
        """
        Remove motion history for hands that are no longer in view.
        """
        for key in list(self.motion_buffer.keys()):
            if key not in active_ids:
                del self.motion_buffer[key]
