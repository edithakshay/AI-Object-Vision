"""
Evidence Manager — DualVision AI Phase 3
Captures screenshots + metadata for every significant detection,
writes to mission folder, and maintains an in-memory evidence list.
"""

import csv
import json
import os
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2

from mission.mission_state import MissionState, get_priority, priority_label


class Evidence:
    __slots__ = (
        "evidence_id", "timestamp", "class_name", "confidence",
        "track_id", "camera", "frame_id", "priority", "image_path",
        "video_timestamp", "verified", "notes",
    )

    def __init__(self, class_name: str, confidence: float,
                 track_id: int, camera: str, frame_id: int,
                 image_path: str = ""):
        self.evidence_id     = str(uuid.uuid4())[:12].upper()
        self.timestamp       = datetime.now()
        self.class_name      = class_name
        self.confidence      = confidence
        self.track_id        = track_id
        self.camera          = camera
        self.frame_id        = frame_id
        self.priority        = get_priority(class_name)
        self.image_path      = image_path
        self.video_timestamp = self.timestamp.strftime("%H:%M:%S")
        self.verified        = False
        self.notes           = ""

    def to_dict(self) -> dict:
        return {
            "evidence_id":     self.evidence_id,
            "timestamp":       self.timestamp.isoformat(),
            "class":           self.class_name,
            "confidence":      round(self.confidence, 3),
            "track_id":        self.track_id,
            "camera":          self.camera,
            "frame_id":        self.frame_id,
            "priority":        self.priority,
            "image_path":      self.image_path,
            "video_timestamp": self.video_timestamp,
            "verified":        self.verified,
            "notes":           self.notes,
        }


class EvidenceManager:
    """
    Sits between the main detection loop and the mission folder.
    Call `capture(frame, class_name, ...)` for each detection you want
    to record as evidence.
    """

    def __init__(self, mission_state: MissionState):
        self._ms     = mission_state
        self._lock   = threading.Lock()
        self._items: List[Evidence] = []
        self._frame_counter = 0

        # Settings (wired from MissionDialog)
        self.screenshot_every_new_track = True
        self.screenshot_high_priority_only = False
        self.save_interval_seconds   = 0       # 0 = disabled
        self.min_confidence          = 0.0

        # Seen track IDs for "new track" logic
        self._seen_tracks: set = set()

        # Callback — called on the calling thread when new evidence added
        self._on_evidence: Optional[Callable] = None

    def set_callback(self, fn: Callable):
        self._on_evidence = fn

    def reset(self):
        with self._lock:
            self._items.clear()
            self._seen_tracks.clear()
            self._frame_counter = 0

    def capture(self, frame, class_name: str, confidence: float,
                track_id: int, camera: str):
        """
        Decide whether to capture evidence for this detection.
        Call from the main detection loop (worker thread).
        """
        if not self._ms.is_active:
            return
        if self._ms.mission_dir is None:
            return
        if confidence < self.min_confidence:
            return

        priority = get_priority(class_name)

        # Filter by priority if setting enabled
        if self.screenshot_high_priority_only and priority != "high":
            return

        is_new_track = track_id not in self._seen_tracks
        if self.screenshot_every_new_track and not is_new_track:
            return

        with self._lock:
            self._seen_tracks.add(track_id)
            self._frame_counter += 1
            fid = self._frame_counter

        img_path = ""
        if frame is not None:
            img_path = self._save_image(frame, class_name, fid)

        ev = Evidence(class_name, confidence, track_id, camera, fid, img_path)

        with self._lock:
            self._items.append(ev)

        # Update mission stats
        self._ms.record_detection(class_name, confidence)
        if img_path:
            self._ms.increment_screenshots()
            self._ms.log_event(
                f"{class_name.title()} detected — {priority_label(priority)} — "
                f"conf={confidence:.2f}  [Track#{track_id}]",
                level="detection" if priority == "high" else "info")

        if self._on_evidence:
            try:
                self._on_evidence(ev)
            except Exception:
                pass

        self._append_csv(ev)
        self._update_json()

    def _save_image(self, frame, class_name: str, fid: int) -> str:
        try:
            ev_dir = self._ms.mission_dir / "evidence"
            ev_dir.mkdir(parents=True, exist_ok=True)
            ts  = datetime.now().strftime("%H%M%S")
            fn  = f"{ts}_{class_name}_{fid:04d}.jpg"
            out = str(ev_dir / fn)
            cv2.imwrite(out, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return out
        except Exception:
            return ""

    def get_all(self) -> List[Evidence]:
        with self._lock:
            return list(self._items)

    def delete(self, evidence_id: str):
        with self._lock:
            self._items = [e for e in self._items if e.evidence_id != evidence_id]

    def verify(self, evidence_id: str, verified: bool, notes: str = ""):
        with self._lock:
            for e in self._items:
                if e.evidence_id == evidence_id:
                    e.verified = verified
                    e.notes    = notes
                    break
        self._update_json()

    def _append_csv(self, ev: Evidence):
        if self._ms.mission_dir is None:
            return
        try:
            path = self._ms.mission_dir / "detections.csv"
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    ev.evidence_id, ev.timestamp.isoformat(),
                    ev.class_name, f"{ev.confidence:.3f}",
                    ev.track_id, ev.priority, ev.camera,
                    ev.frame_id, ev.verified, ev.notes,
                ])
        except Exception:
            pass

    def _update_json(self):
        if self._ms.mission_dir is None:
            return
        try:
            with self._lock:
                data = [e.to_dict() for e in self._items]
            path = self._ms.mission_dir / "detections.json"
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass
