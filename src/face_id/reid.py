import json
import os
import uuid
from typing import Tuple, Dict, Any

import numpy as np


class FaceReIdentifier:
    """
    Simple cosine-similarity based face re-identification.
    Stores embeddings in a JSON database on disk.
    """

    def __init__(self, db_path: str = "src/face_id/store.json", threshold: float = 0.55):
        self.db_path = db_path
        self.threshold = threshold
        self.db: Dict[str, Dict[str, Any]] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r") as f:
                    self.db = json.load(f)
            except Exception:
                self.db = {}
        else:
            # ensure directory exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            self.db = {}
            self._save()

    def _save(self) -> None:
        with open(self.db_path, "w") as f:
            json.dump(self.db, f, indent=4)

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    def identify(self, embedding: np.ndarray | None) -> Tuple[str, float]:
        """
        Returns (name, similarity_score).
        If no good match, returns ("Unknown", best_similarity).
        """
        if embedding is None or not self.db:
            return "Unknown", 0.0

        best_uid = None
        best_sim = -1.0

        for uid, entry in self.db.items():
            ref_emb = np.array(entry["embedding"], dtype=np.float32)
            sim = float(np.dot(embedding, ref_emb))  # cosine if both normalized

            if sim > best_sim:
                best_sim = sim
                best_uid = uid

        if best_uid is None or best_sim < self.threshold:
            return "Unknown", best_sim

        return self.db[best_uid]["name"], best_sim

    def register(self, name: str, embedding: np.ndarray | None) -> str | None:
        """
        Register a new user with given name and embedding.
        Returns the new user ID or None if failed.
        """
        if embedding is None:
            return None

        uid = str(uuid.uuid4())[:8]
        self.db[uid] = {
            "name": name,
            "embedding": embedding.tolist(),
        }
        self._save()
        print(f"[FaceReIdentifier] Registered new user: {name} ({uid})")
        return uid
