import math
from typing import Dict, Optional


class EmotionEngine:
    """
    Lightweight emotion/attention classifier.

    Uses:
      - EAR (eye aspect ratio)
      - MAR (mouth aspect ratio)
      - Head pose (pitch, yaw)

    Output is a high-level state string, e.g.:
      - "Focused"
      - "Drowsy / Eyes Closed"
      - "Looking Away"
      - "Smiling"
      - "Talking"
    """

    def __init__(
        self,
        ear_drowsy_thresh: float = 0.22,
        yaw_attention_thresh: float = 18.0,
        pitch_attention_thresh: float = 18.0,
        mar_talk_thresh: float = 0.35,
        mar_smile_thresh: float = 0.30,
    ) -> None:
        self.ear_drowsy_thresh = ear_drowsy_thresh
        self.yaw_attention_thresh = yaw_attention_thresh
        self.pitch_attention_thresh = pitch_attention_thresh
        self.mar_talk_thresh = mar_talk_thresh
        self.mar_smile_thresh = mar_smile_thresh

    def infer(
        self,
        ear: float,
        mar: float,
        pitch: float,
        yaw: float,
    ) -> str:
        """
        Simple heuristic classifier.

        Priority:
          1) Drowsy (eyes closed)
          2) Looking Away (head turned)
          3) Talking (mouth wide)
          4) Smiling (medium mouth open, probably grin)
          5) Focused (default)
        """
        # 1. Eyes closed / drowsy
        if ear < self.ear_drowsy_thresh:
            return "Drowsy / Eyes Closed"

        # 2. Looking away (head pose)
        if abs(yaw) > self.yaw_attention_thresh or abs(pitch) > self.pitch_attention_thresh:
            return "Looking Away"

        # 3. Talking or laughing (mouth very open)
        if mar > self.mar_talk_thresh:
            return "Talking / Laughing"

        # 4. Smiling (mouth mildly open but eyes open and looking forward)
        if mar > self.mar_smile_thresh:
            return "Smiling"

        # 5. Fallback
        return "Focused"
