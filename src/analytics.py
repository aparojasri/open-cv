import os
import time
import csv
import json
import queue
import threading
import uuid
import platform
from collections import deque
from datetime import datetime
from typing import Dict, Any, Optional


class AsyncSessionLogger:
    """
    Asynchronous session logger.

    - Telemetry -> CSV
    - Events    -> JSONL
    - Metadata  -> JSON
    - Final summary report -> JSON
    - Realtime stats (sliding window) for UI
    """

    def __init__(self, base_dir: str = "logs") -> None:
        self.base_dir = base_dir
        self.session_id = str(uuid.uuid4())[:8]
        self.start_time = datetime.now()

        self.log_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.is_running = False
        self.writer_thread: Optional[threading.Thread] = None

        self.telemetry_path = ""
        self.events_path = ""
        self.meta_path = ""

        self._setup_directories()

        # Aggregate metrics (whole session)
        self.session_metrics: Dict[str, Any] = {
            "total_frames": 0,
            "drowsy_frames": 0,
            "distracted_frames": 0,
            "gestures_detected": {},  # name -> count
            "start_timestamp": time.time(),
        }

        # Sliding window of recent states (for "real-time" stats)
        # stores strings like: "FOCUSED", "DROWSY", "DISTRACTED", "UNKNOWN"
        self._recent_states = deque(maxlen=300)  # ~10s at 30 FPS

    # ------------------------------------------------------------------ #
    # Setup / lifecycle
    # ------------------------------------------------------------------ #

    def _setup_directories(self) -> None:
        os.makedirs(self.base_dir, exist_ok=True)
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")

        self.telemetry_path = os.path.join(
            self.base_dir, f"telemetry_{timestamp}_{self.session_id}.csv"
        )
        self.events_path = os.path.join(
            self.base_dir, f"events_{timestamp}_{self.session_id}.jsonl"
        )
        self.meta_path = os.path.join(
            self.base_dir, f"meta_{timestamp}_{self.session_id}.json"
        )

    def start(self) -> None:
        if self.is_running:
            return

        self.is_running = True
        self._write_metadata()
        self._init_csv()

        self.writer_thread = threading.Thread(
            target=self._writer_loop, daemon=True
        )
        self.writer_thread.start()

    def stop(self) -> None:
        if not self.is_running:
            return

        self.is_running = False
        self.log_queue.join()

        if self.writer_thread:
            self.writer_thread.join(timeout=2.0)

        self._write_final_report()

    # ------------------------------------------------------------------ #
    # Public logging API
    # ------------------------------------------------------------------ #

    def log_telemetry(
        self,
        pitch: float,
        yaw: float,
        ear: float,
        fps: int,
        state: str,
    ) -> None:
        """
        Log one telemetry sample.
        `state` can be any label, but "DROWSY" / "DISTRACTED" are
        treated specially for stats.
        """
        ts = time.time()
        self.session_metrics["total_frames"] += 1

        state_norm = (state or "UNKNOWN").upper()

        if "DROWSY" in state_norm:
            self.session_metrics["drowsy_frames"] += 1
        if "DISTRACTED" in state_norm:
            self.session_metrics["distracted_frames"] += 1

        # Push into sliding window
        if "DROWSY" in state_norm:
            self._recent_states.append("DROWSY")
        elif "DISTRACTED" in state_norm:
            self._recent_states.append("DISTRACTED")
        elif "FOCUS" in state_norm or "FOCUSED" in state_norm:
            self._recent_states.append("FOCUSED")
        else:
            self._recent_states.append("UNKNOWN")

        record = {
            "type": "telemetry",
            "timestamp": ts,
            "data": [
                ts,
                round(pitch, 2),
                round(yaw, 2),
                round(ear, 3),
                int(fps),
                state,
            ],
        }
        self.log_queue.put(record)

    def log_event(self, event_type: str, details: Dict[str, Any]) -> None:
        if event_type == "GESTURE":
            g_name = details.get("name", "Unknown")
            gestures = self.session_metrics["gestures_detected"]
            gestures[g_name] = gestures.get(g_name, 0) + 1

        record = {
            "type": "event",
            "timestamp": time.time(),
            "data": {
                "timestamp": datetime.now().isoformat(),
                "type": event_type,
                "details": details,
            },
        }
        self.log_queue.put(record)

    def get_realtime_stats(self) -> Dict[str, Any]:
        """
        Return a lightweight summary suitable for displaying on UI.
        Uses a sliding window for focus / drowsy percentages.
        """
        duration = time.time() - self.session_metrics["start_timestamp"]
        duration = max(0.0, duration)

        # --- window-based stats (recent few seconds) ---
        window = list(self._recent_states)
        window_total = max(1, len(window))

        win_drowsy = sum(1 for s in window if s == "DROWSY")
        win_distracted = sum(1 for s in window if s == "DISTRACTED")

        # Weight distracted a bit higher than drowsy in focus score
        bad_score = (win_distracted * 1.0 + win_drowsy * 0.7) / window_total
        focus_score = max(0.0, 1.0 - bad_score) * 100.0

        drowsy_pct = (win_drowsy / window_total) * 100.0

        return {
            "duration_sec": int(duration),
            "focus_score": round(focus_score, 1),
            "drowsy_alert_count": self.session_metrics["drowsy_frames"],
            "drowsy_percentage": round(drowsy_pct, 1),
            "gestures_total": int(
                sum(self.session_metrics["gestures_detected"].values())
            ),
        }

    # ------------------------------------------------------------------ #
    # Writer thread
    # ------------------------------------------------------------------ #

    def _writer_loop(self) -> None:
        csv_file = open(self.telemetry_path, "a", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        event_file = open(self.events_path, "a", encoding="utf-8")

        try:
            while self.is_running or not self.log_queue.empty():
                try:
                    record = self.log_queue.get(timeout=0.1)

                    if record["type"] == "telemetry":
                        csv_writer.writerow(record["data"])
                    elif record["type"] == "event":
                        event_file.write(
                            json.dumps(record["data"], ensure_ascii=False) + "\n"
                        )

                    csv_file.flush()
                    event_file.flush()
                    self.log_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"[AsyncSessionLogger] Logging error: {e}")
                    self.log_queue.task_done()
        finally:
            csv_file.close()
            event_file.close()

    # ------------------------------------------------------------------ #
    # File init / summaries
    # ------------------------------------------------------------------ #

    def _init_csv(self) -> None:
        with open(self.telemetry_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["timestamp_unix", "pitch", "yaw", "ear", "fps", "state"]
            )

    def _write_metadata(self) -> None:
        meta = {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "system": {
                "node": platform.node(),
                "os": platform.system(),
                "release": platform.release(),
                "processor": platform.processor(),
            },
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4)

    def _write_final_report(self) -> None:
        report_path = os.path.join(
            self.base_dir, f"report_{self.session_id}.json"
        )
        duration = time.time() - self.session_metrics["start_timestamp"]
        duration = max(0.0, duration)

        total_frames = max(1, self.session_metrics["total_frames"])
        drowsy_pct_total = (
            self.session_metrics["drowsy_frames"] / total_frames
        ) * 100.0
        distracted_pct_total = (
            self.session_metrics["distracted_frames"] / total_frames
        ) * 100.0

        report = {
            "session_summary": {
                "duration_seconds": round(duration, 2),
                "total_frames_processed": self.session_metrics["total_frames"],
                "performance": {
                    "drowsiness_percentage": round(drowsy_pct_total, 2),
                    "distraction_percentage": round(distracted_pct_total, 2),
                },
                "interaction_log": self.session_metrics["gestures_detected"],
            }
        }

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
