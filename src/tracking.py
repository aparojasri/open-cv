import time
from collections import deque
from typing import Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
#  Simple Kalman Tracker for 2D Face Center
# ---------------------------------------------------------------------------
class KalmanTracker:
    """
    Lightweight 2D position + velocity Kalman Filter for Companion Vision OS.
    State       = [x, y, vx, vy]
    Measurement = [x, y]
    """

    def __init__(self) -> None:
        self.kf = cv2.KalmanFilter(4, 2)

        # State transition matrix (F) – we update dt each frame
        self.kf.transitionMatrix = np.eye(4, dtype=np.float32)

        # We measure x, y directly
        self.kf.measurementMatrix = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=np.float32,
        )

        # Process noise (model uncertainty)
        self.base_q = 0.01
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * self.base_q

        # Measurement noise (sensor uncertainty)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1

        # Initial covariance
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0

        self.initialized = False
        self.last_time = time.time()

    def _update_transition(self, dt: float) -> None:
        """Update F and Q based on elapsed time."""
        self.kf.transitionMatrix[0, 2] = dt
        self.kf.transitionMatrix[1, 3] = dt

        # Increase process noise slightly with dt
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * (
            self.base_q * (1.0 + dt)
        )

    def update(self, x: int, y: int) -> Tuple[int, int]:
        """
        Update the filter with a new measurement.
        Returns smoothed (x, y).
        """
        now = time.time()
        dt = now - self.last_time
        self.last_time = now

        meas = np.array([[np.float32(x)], [np.float32(y)]])
        self._update_transition(max(dt, 1e-3))

        if not self.initialized:
            self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
            self.initialized = True
            return x, y

        # Standard Kalman cycle
        self.kf.predict()
        self.kf.correct(meas)

        state = self.kf.statePost
        fx, fy = int(state[0][0]), int(state[1][0])
        return fx, fy

    def predict(self) -> Tuple[int, int]:
        """
        Predict-only mode when detection is missing.
        Returns predicted (x, y).
        """
        now = time.time()
        dt = now - self.last_time
        self.last_time = now

        self._update_transition(max(dt, 1e-3))
        predicted = self.kf.predict()
        return int(predicted[0][0]), int(predicted[1][0])


# ---------------------------------------------------------------------------
#  Statistical Smoother for scalar signals
# ---------------------------------------------------------------------------
class StatisticalSmoother:
    """
    Double Exponential Smoothing (Holt's method) for 1D signals.
    Used for pitch, yaw, EAR.
    """

    def __init__(self, alpha: float = 0.6, beta: float = 0.3) -> None:
        self.alpha = alpha  # level smoothing
        self.beta = beta    # trend smoothing

        self.level = None
        self.trend = None
        self.buffer = deque(maxlen=5)

    def update(self, value: float) -> float:
        """
        Feed one sample and return smoothed signal.
        """
        self.buffer.append(value)

        # Initialize level/trend using first few samples
        if self.level is None:
            if len(self.buffer) > 1:
                self.level = self.buffer[-1]
                self.trend = self.buffer[-1] - self.buffer[-2]
            return float(value)

        last_level = self.level

        # Level update
        self.level = self.alpha * value + (1.0 - self.alpha) * (self.level + self.trend)

        # Trend update
        self.trend = self.beta * (self.level - last_level) + (1.0 - self.beta) * self.trend

        return float(self.level)


# ---------------------------------------------------------------------------
#  StateSmoother: manages pitch / yaw / EAR together
# ---------------------------------------------------------------------------
class StateSmoother:
    """
    Wrapper around three StatisticalSmoother instances:
        - pitch
        - yaw
        - EAR

    Usage:
        smoother = StateSmoother()
        smoother.update(pitch, yaw, ear)
        p_s, y_s, e_s = smoother.get_smoothed_values()
    """

    def __init__(self) -> None:
        self._pitch_smoother = StatisticalSmoother(alpha=0.5, beta=0.1)
        self._yaw_smoother = StatisticalSmoother(alpha=0.5, beta=0.1)
        # EAR needs to react quicker (blinks)
        self._ear_smoother = StatisticalSmoother(alpha=0.7, beta=0.2)

        self._pitch = 0.0
        self._yaw = 0.0
        self._ear = 0.0

    def update(self, pitch: float, yaw: float, ear: float) -> None:
        """
        Feed new raw values into the smoother.
        """
        self._pitch = self._pitch_smoother.update(pitch)
        self._yaw = self._yaw_smoother.update(yaw)
        self._ear = self._ear_smoother.update(ear)

    def get_smoothed_values(self) -> Tuple[float, float, float]:
        """
        Returns (smoothed_pitch, smoothed_yaw, smoothed_ear).
        """
        return self._pitch, self._yaw, self._ear
