import time
import numpy as np
from collections import deque
from typing import Dict, Union


class EdgeProfiler:
    """
    Lightweight profiler for edge/real-time inference loops
    in Companion Vision OS.

    Tracks:
      - FPS (smoothed over a buffer)
      - Average latency (ms)
      - Min / Max / P99 latency (ms)
      - Jitter (std dev of latency in ms)
    """

    def __init__(self, buffer_size: int = 100, update_interval: int = 5):
        """
        :param buffer_size: Number of recent frames to keep in the rolling window.
        :param update_interval: Recalculate metrics every N frames to save CPU.
        """
        self.buffer_size = buffer_size
        self.update_interval = max(1, update_interval)

        # Time buffers
        self.frame_start_times = deque(maxlen=buffer_size)
        self.inference_durations = deque(maxlen=buffer_size)  # in ms
        self.fps_buffer = deque(maxlen=buffer_size)

        # Per-frame timing helpers
        self.start_time: float = 0.0

        # Cached metrics (always has all keys)
        self._cached_metrics: Dict[str, Union[int, float]] = {
            "fps": 0,
            "latency_avg": 0,
            "latency_min": 0,
            "latency_max": 0,
            "latency_p99": 0,
            "jitter": 0.0,
        }
        self._update_counter: int = 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def tick(self) -> None:
        """Call at the start of a frame/inference."""
        now = time.perf_counter()
        self.start_time = now
        self.frame_start_times.append(now)

    def tock(self) -> Dict[str, Union[int, float]]:
        """
        Call at the end of a frame/inference.
        Returns the latest cached metrics dict.
        """
        end_time = time.perf_counter()

        # Latency in milliseconds for this frame
        inference_time_ms = (end_time - self.start_time) * 1000.0
        self.inference_durations.append(inference_time_ms)

        # Instant FPS based on frame start times (if we have at least 2 frames)
        if len(self.frame_start_times) >= 2:
            t_curr = self.frame_start_times[-1]
            t_prev = self.frame_start_times[-2]
            frame_delta = t_curr - t_prev
            if frame_delta > 0:
                instant_fps = 1.0 / frame_delta
                self.fps_buffer.append(instant_fps)

        self._update_counter += 1

        # Recalculate metrics every N frames
        if self._update_counter % self.update_interval == 0:
            self._recalculate_metrics()

        return self._cached_metrics

    def get_detailed_stats(self) -> Dict[str, Union[int, float]]:
        """Return the latest computed metrics."""
        return dict(self._cached_metrics)

    def reset(self) -> None:
        """Clear all history and metrics."""
        self.frame_start_times.clear()
        self.inference_durations.clear()
        self.fps_buffer.clear()
        self._update_counter = 0
        self._cached_metrics = {
            "fps": 0,
            "latency_avg": 0,
            "latency_min": 0,
            "latency_max": 0,
            "latency_p99": 0,
            "jitter": 0.0,
        }

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _recalculate_metrics(self) -> None:
        """Recompute FPS, latency stats, and jitter from buffers."""
        if not self.inference_durations:
            return

        lat_array = np.array(self.inference_durations, dtype=float)

        if self.fps_buffer:
            fps_array = np.array(self.fps_buffer, dtype=float)
            avg_fps = float(np.mean(fps_array))
        else:
            avg_fps = 0.0

        avg_latency = float(np.mean(lat_array))
        p99_latency = float(np.percentile(lat_array, 99))
        min_latency = float(np.min(lat_array))
        max_latency = float(np.max(lat_array))
        jitter = float(np.std(lat_array)) if len(lat_array) > 1 else 0.0

        self._cached_metrics = {
            "fps": int(round(avg_fps)),
            "latency_avg": int(round(avg_latency)),
            "latency_min": int(round(min_latency)),
            "latency_max": int(round(max_latency)),
            "latency_p99": int(round(p99_latency)),
            "jitter": round(jitter, 2),
        }
