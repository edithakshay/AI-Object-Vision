"""
Evidence Manager — DualVision AI Phase 3

Key isolation guarantee
───────────────────────
Every capture() call reads self._ms.mission_dir ONCE at the start and
holds that reference for the entire operation.  If the mission finishes
and mission_dir is cleared to None between two detections, the in-flight
capture still writes to the correct (previous) folder and the next capture
correctly drops because mission_dir is None.

This prevents the "Mission 2 writes into Mission 1 folder" bug even under
race conditions.
"""

import csv
import json
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

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
    Bridges the detection loop and the mission folder.
    Call capture() for every detection — it decides what to save.
    Call reset() when a new mission starts to clear in-memory state.
    """

    def __init__(self, mission_state: MissionState):
        self._ms     = mission_state
        self._lock   = threading.Lock()
        self._items: List[Evidence] = []
        self._frame_counter = 0

        # Per-mission in-memory tracking of seen track IDs
        # Cleared by reset() before each new mission
        self._seen_tracks: set = set()

        # Settings — wired from MissionDialog
        self.screenshot_every_new_track    = True
        self.screenshot_high_priority_only = False
        self.min_confidence                = 0.0

        # Callback — called (on worker thread) when new evidence added
        self._on_evidence: Optional[Callable] = None

        # ── Pinned folder for the current mission ─────────────────────────────
        # Set by pin_folder() at mission start.  Used by all writes.
        # This prevents writes crossing into a new mission's folder.
        self._active_folder: Optional[Path] = None

    def set_callback(self, fn: Callable):
        self._on_evidence = fn

    def pin_folder(self, folder: Path):
        """
        Call this immediately after MissionState.start() returns.
        Pins the mission folder so all subsequent writes go to this
        exact folder even if mission_state.mission_dir changes later.
        """
        with self._lock:
            self._active_folder = folder

    def reset(self):
        """
        Must be called before every new mission (before MissionState.start()).
        Clears all in-memory state so Mission N never sees Mission N-1 data.
        """
        with self._lock:
            self._items.clear()
            self._seen_tracks.clear()
            self._frame_counter = 0
            self._active_folder = None   # cleared — pin_folder() must be called again

    def capture(self, frame, class_name: str, confidence: float,
                track_id: int, camera: str):
        """
        Decide whether to capture evidence for this detection.
        Safe to call from any thread.
        """
        # ── Guard 1: only when mission is active ──────────────────────────────
        if not self._ms.is_active:
            return

        # ── Guard 2: read the pinned folder ONCE — atomic snapshot ───────────
        with self._lock:
            folder = self._active_folder
        if folder is None:
            return

        # ── Guard 3: confidence threshold ────────────────────────────────────
        if confidence < self.min_confidence:
            return

        priority = get_priority(class_name)

        # ── Guard 4: high-priority filter ────────────────────────────────────
        if self.screenshot_high_priority_only and priority != "high":
            return

        # ── Guard 5: new-track filter ─────────────────────────────────────────
        with self._lock:
            is_new = track_id not in self._seen_tracks
            if self.screenshot_every_new_track and not is_new:
                return
            self._seen_tracks.add(track_id)
            self._frame_counter += 1
            fid = self._frame_counter

        # ── Save image into the PINNED folder (not mission_dir) ───────────────
        img_path = ""
        if frame is not None:
            img_path = self._save_image(frame, class_name, fid, folder)

        ev = Evidence(class_name, confidence, track_id, camera, fid, img_path)

        with self._lock:
            self._items.append(ev)

        # ── Update mission stats ──────────────────────────────────────────────
        self._ms.record_detection(class_name, confidence)
        if img_path:
            self._ms.increment_screenshots()
        self._ms.log_event(
            f"{class_name.title()} detected — {priority_label(priority)} — "
            f"conf={confidence:.2f}  [Track#{track_id}]",
            level="detection" if priority == "high" else "info")

        # ── Write to disk (pinned folder) ─────────────────────────────────────
        self._append_csv(ev, folder)
        self._flush_json(folder)

        if self._on_evidence:
            try:
                self._on_evidence(ev)
            except Exception:
                pass

    # ── Image save ────────────────────────────────────────────────────────────
    def _save_image(self, frame, class_name: str, fid: int,
                    folder: Path) -> str:
        try:
            ev_dir = folder / "evidence"
            ev_dir.mkdir(parents=True, exist_ok=True)
            ts  = datetime.now().strftime("%H%M%S")
            fn  = f"{ts}_{class_name}_{fid:04d}.jpg"
            out = str(ev_dir / fn)
            cv2.imwrite(out, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return out
        except Exception:
            return ""

    # ── CSV (append mode — never overwrites) ──────────────────────────────────
    def _append_csv(self, ev: Evidence, folder: Path):
        try:
            path = folder / "detections.csv"
            with open(path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    ev.evidence_id, ev.timestamp.isoformat(),
                    ev.class_name, f"{ev.confidence:.3f}",
                    ev.track_id, ev.priority, ev.camera,
                    ev.frame_id, ev.verified, ev.notes,
                ])
        except Exception:
            pass

    # ── JSON (full rewrite from in-memory list) ───────────────────────────────
    def _flush_json(self, folder: Path):
        try:
            with self._lock:
                data = [e.to_dict() for e in self._items]
            (folder / "detections.json").write_text(
                json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def flush_final(self):
        """
        Called by MissionDialog.finish() to ensure the last JSON state
        is written to the pinned folder before it is cleared.
        """
        with self._lock:
            folder = self._active_folder
            data   = [e.to_dict() for e in self._items]
        if folder is None:
            return
        try:
            (folder / "detections.json").write_text(
                json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    # ── In-memory accessors ───────────────────────────────────────────────────
    def get_all(self) -> List[Evidence]:
        with self._lock:
            return list(self._items)

    def delete(self, evidence_id: str):
        with self._lock:
            self._items = [e for e in self._items
                           if e.evidence_id != evidence_id]
        with self._lock:
            folder = self._active_folder
        if folder:
            self._flush_json(folder)

    def verify(self, evidence_id: str, verified: bool, notes: str = ""):
        with self._lock:
            for e in self._items:
                if e.evidence_id == evidence_id:
                    e.verified = verified
                    e.notes    = notes
                    break
            folder = self._active_folder
        if folder:
            self._flush_json(folder)
