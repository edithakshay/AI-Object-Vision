"""
Mission State — DualVision AI Phase 3
Manages mission lifecycle, folder structure, timeline, statistics,
and JSON persistence.
"""

import json
import os
import threading
import time
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class MissionType(str, Enum):
    SEARCH_RESCUE    = "Search & Rescue"
    DISASTER         = "Disaster Assessment"
    FIRE             = "Fire Monitoring"
    VEHICLE          = "Vehicle Search"
    WILDLIFE         = "Wildlife Monitoring"


class MissionStatus(str, Enum):
    IDLE    = "Idle"
    ACTIVE  = "Active"
    PAUSED  = "Paused"
    FINISHED = "Finished"


# ── Detection priority map ────────────────────────────────────────────────────
HIGH_PRIORITY   = {"person", "fire", "smoke", "boat"}
MEDIUM_PRIORITY = {"car", "truck", "bus", "motorcycle", "bicycle",
                   "vehicle", "animal", "cat", "dog", "horse", "backpack"}
LOW_PRIORITY    = {"chair", "bottle", "tv", "monitor", "laptop",
                   "cup", "vase", "book", "clock"}


def get_priority(class_name: str) -> str:
    name = class_name.lower()
    if name in HIGH_PRIORITY:
        return "high"
    if name in MEDIUM_PRIORITY:
        return "medium"
    return "low"


def priority_label(p: str) -> str:
    return {"high": "🔴 High", "medium": "🟡 Medium", "low": "🟢 Low"}.get(p, "🟢 Low")


class TimelineEntry:
    def __init__(self, message: str, level: str = "info"):
        self.timestamp = datetime.now()
        self.message   = message
        self.level     = level   # info | warning | detection | alert

    def to_dict(self) -> dict:
        return {
            "time":    self.timestamp.strftime("%H:%M:%S"),
            "message": self.message,
            "level":   self.level,
        }


class MissionState:
    """
    Single mission session.
    Thread-safe via self._lock.
    """

    # ── Mission database root ─────────────────────────────────────────────────
    DB_ROOT = Path("missions")

    def __init__(self):
        self._lock   = threading.Lock()
        self._status = MissionStatus.IDLE

        # Form fields
        self.mission_name   = ""
        self.mission_id     = ""
        self.operator_name  = ""
        self.drone_name     = ""
        self.search_area    = ""
        self.mission_type   = MissionType.SEARCH_RESCUE
        self.start_time:  Optional[datetime] = None
        self.end_time:    Optional[datetime]  = None
        self.mission_dir: Optional[Path]      = None

        # Timeline
        self._timeline: List[TimelineEntry] = []

        # Stats
        self._stats: Dict[str, Any] = {
            "total_detections": 0,
            "persons":          0,
            "vehicles":         0,
            "animals":          0,
            "fire_smoke":       0,
            "screenshots":      0,
            "avg_confidence":   0.0,
            "conf_sum":         0.0,
            "frames_processed": 0,
            "detection_rate":   0.0,
        }

        # Callbacks
        self._on_timeline_update: Optional[Callable] = None
        self._on_stats_update:    Optional[Callable] = None

    # ── Properties ────────────────────────────────────────────────────────────
    @property
    def status(self) -> MissionStatus:
        with self._lock:
            return self._status

    @property
    def is_active(self) -> bool:
        return self._status == MissionStatus.ACTIVE

    @property
    def elapsed_seconds(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()

    @property
    def elapsed_str(self) -> str:
        secs = int(self.elapsed_seconds)
        h, rem = divmod(secs, 3600)
        m, s   = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def set_callbacks(self, on_timeline=None, on_stats=None):
        self._on_timeline_update = on_timeline
        self._on_stats_update    = on_stats

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def start(self, name: str, operator: str, drone: str,
              area: str, mtype: MissionType) -> Path:
        with self._lock:
            self._status      = MissionStatus.ACTIVE
            self.mission_name = name
            self.mission_id   = self._gen_id()
            self.operator_name = operator
            self.drone_name   = drone
            self.search_area  = area
            self.mission_type = mtype
            self.start_time   = datetime.now()
            self.end_time     = None
            self._timeline.clear()
            self._reset_stats()
            self.mission_dir  = self._create_folder()
        self._log("Mission Started", "info")
        self._save_mission_json()
        return self.mission_dir

    def pause(self):
        with self._lock:
            if self._status == MissionStatus.ACTIVE:
                self._status = MissionStatus.PAUSED
        self._log("Mission Paused", "warning")

    def resume(self):
        with self._lock:
            if self._status == MissionStatus.PAUSED:
                self._status = MissionStatus.ACTIVE
        self._log("Mission Resumed", "info")

    def finish(self):
        with self._lock:
            if self._status in (MissionStatus.ACTIVE, MissionStatus.PAUSED):
                self._status  = MissionStatus.FINISHED
                self.end_time = datetime.now()
        self._log("Mission Finished", "info")
        self._save_mission_json()

    # ── Timeline ──────────────────────────────────────────────────────────────
    def _log(self, message: str, level: str = "info"):
        entry = TimelineEntry(message, level)
        with self._lock:
            self._timeline.append(entry)
        if self._on_timeline_update:
            try:
                self._on_timeline_update(entry)
            except Exception:
                pass

    def log_event(self, message: str, level: str = "info"):
        self._log(message, level)

    def get_timeline(self) -> List[TimelineEntry]:
        with self._lock:
            return list(self._timeline)

    # ── Statistics ────────────────────────────────────────────────────────────
    def _reset_stats(self):
        for k in self._stats:
            self._stats[k] = 0 if isinstance(self._stats[k], int) else 0.0

    def record_detection(self, class_name: str, confidence: float):
        """Call for every confirmed detection during an active mission."""
        if self._status != MissionStatus.ACTIVE:
            return
        name = class_name.lower()
        with self._lock:
            s = self._stats
            s["total_detections"] += 1
            s["conf_sum"]         += confidence
            s["avg_confidence"]    = s["conf_sum"] / s["total_detections"]
            if name == "person":
                s["persons"] += 1
            elif name in {"car", "truck", "bus", "motorcycle", "bicycle", "vehicle"}:
                s["vehicles"] += 1
            elif name in {"cat", "dog", "horse", "animal"}:
                s["animals"] += 1
            elif name in {"fire", "smoke"}:
                s["fire_smoke"] += 1
            s["frames_processed"] += 1
            elapsed = max(self.elapsed_seconds, 1)
            s["detection_rate"] = s["total_detections"] / elapsed

        if self._on_stats_update:
            try:
                self._on_stats_update(dict(self._stats))
            except Exception:
                pass

    def increment_screenshots(self):
        with self._lock:
            self._stats["screenshots"] += 1

    def get_stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    # ── Persistence ───────────────────────────────────────────────────────────
    def _gen_id(self) -> str:
        date_str = datetime.now().strftime("%Y%m%d")
        uid      = str(uuid.uuid4())[:8].upper()
        return f"MSN-{date_str}-{uid}"

    def _create_folder(self) -> Path:
        date_str = datetime.now().strftime("%Y-%m-%d")
        safe_name = "".join(c if c.isalnum() or c in "-_ " else "_"
                            for c in self.mission_name).strip().replace(" ", "_")
        folder_name = f"Mission_{date_str}_{safe_name}"
        root = self.DB_ROOT / folder_name
        for sub in ("recordings", "screenshots", "logs", "report", "evidence"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        # Create empty CSV / JSON placeholders
        (root / "detections.csv").write_text(
            "detection_id,timestamp,class,confidence,track_id,priority,camera,frame_id,verified,notes\n",
            encoding="utf-8")
        (root / "detections.json").write_text("[]", encoding="utf-8")
        return root

    def _save_mission_json(self):
        if self.mission_dir is None:
            return
        data = {
            "mission_id":    self.mission_id,
            "mission_name":  self.mission_name,
            "operator":      self.operator_name,
            "drone":         self.drone_name,
            "search_area":   self.search_area,
            "mission_type":  str(self.mission_type.value),
            "status":        str(self._status.value),
            "start_time":    self.start_time.isoformat() if self.start_time else None,
            "end_time":      self.end_time.isoformat()   if self.end_time   else None,
            "stats":         dict(self._stats),
            "timeline":      [e.to_dict() for e in self._timeline],
        }
        try:
            path = self.mission_dir / "mission.json"
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    # ── Mission History DB ────────────────────────────────────────────────────
    @classmethod
    def list_missions(cls) -> List[Dict]:
        """Return list of past mission summaries from the missions/ folder."""
        missions = []
        if not cls.DB_ROOT.exists():
            return missions
        for folder in sorted(cls.DB_ROOT.iterdir(), reverse=True):
            mj = folder / "mission.json"
            if mj.exists():
                try:
                    data = json.loads(mj.read_text(encoding="utf-8"))
                    data["_folder"] = str(folder)
                    missions.append(data)
                except Exception:
                    pass
        return missions
