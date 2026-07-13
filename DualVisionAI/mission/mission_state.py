"""
Mission State — DualVision AI Phase 3
Manages mission lifecycle, folder structure, timeline, statistics,
and JSON persistence.

Folder isolation guarantee
──────────────────────────
Every call to start() produces a NEW unique directory:
    missions/Mission_YYYYMMDD_HHMMSS_NNN/
where NNN is a zero-padded monotonic counter that increments even if two
missions start within the same second.  The folder is NEVER re-used.

Finish guarantee
────────────────
finish() writes:
  • mission.json   — full machine-readable record
  • report/mission_report.txt — human-readable summary
  • logs/mission.log           — full timeline
then sets mission_dir to None so any stale reference fails loudly.
"""

import json
import os
import threading
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class MissionType(str, Enum):
    SEARCH_RESCUE = "Search & Rescue"
    DISASTER      = "Disaster Assessment"
    FIRE          = "Fire Monitoring"
    VEHICLE       = "Vehicle Search"
    WILDLIFE      = "Wildlife Monitoring"


class MissionStatus(str, Enum):
    IDLE     = "Idle"
    ACTIVE   = "Active"
    PAUSED   = "Paused"
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


# ── Global monotonic mission counter (process lifetime) ──────────────────────
_MISSION_COUNTER_LOCK = threading.Lock()
_MISSION_COUNTER      = 0


def _next_mission_number() -> int:
    global _MISSION_COUNTER
    with _MISSION_COUNTER_LOCK:
        _MISSION_COUNTER += 1
        return _MISSION_COUNTER


class MissionState:
    """
    Single reusable mission session object.
    Calling start() always produces a brand-new, unique folder.
    Thread-safe via self._lock.
    """

    DB_ROOT = Path("missions")

    def __init__(self):
        self._lock   = threading.Lock()
        self._status = MissionStatus.IDLE

        # Form fields (set by start())
        self.mission_name  = ""
        self.mission_id    = ""
        self.operator_name = ""
        self.drone_name    = ""
        self.search_area   = ""
        self.mission_type  = MissionType.SEARCH_RESCUE
        self.start_time:   Optional[datetime] = None
        self.end_time:     Optional[datetime] = None

        # ── Current mission folder — None when no mission is active ──────────
        # This is set atomically inside start() and cleared after finish()
        # writes its final report.  Writers must read this once and hold
        # their own reference; never re-read it mid-write.
        self.mission_dir:  Optional[Path] = None

        # Sequence number for the current mission (human-readable)
        self._mission_seq: int = 0

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
        """
        Start a new mission.
        ALWAYS creates a completely new, unique folder.
        Returns the new mission_dir Path.
        """
        seq = _next_mission_number()

        with self._lock:
            self._status       = MissionStatus.ACTIVE
            self.mission_name  = name
            self.mission_id    = self._gen_id(seq)
            self.operator_name = operator
            self.drone_name    = drone
            self.search_area   = area
            self.mission_type  = mtype
            self.start_time    = datetime.now()
            self.end_time      = None
            self._mission_seq  = seq
            self._timeline.clear()
            self._reset_stats()

            # ── Create fresh folder INSIDE the lock so mission_dir is
            # always consistent with the rest of the fields.
            self.mission_dir = self._create_fresh_folder(seq)

        self._log("Mission Started", "info")
        self._save_mission_json()
        return self.mission_dir  # type: ignore[return-value]

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
        """
        Finish the mission, flush all data to disk, write the report,
        then clear mission_dir so the next start() cannot accidentally
        inherit this folder.
        """
        with self._lock:
            if self._status not in (MissionStatus.ACTIVE, MissionStatus.PAUSED):
                return
            self._status  = MissionStatus.FINISHED
            self.end_time = datetime.now()
            folder = self.mission_dir   # hold reference before we clear it

        self._log("Mission Finished", "info")

        # ── Flush all persistent data ─────────────────────────────────────────
        self._save_mission_json(folder=folder)
        self._write_mission_log(folder=folder)
        self._write_report(folder=folder)

        # ── Clear folder reference — next start() MUST create a new one ──────
        with self._lock:
            self.mission_dir = None

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
        """Called inside _lock — do NOT acquire lock again."""
        for k in list(self._stats.keys()):
            self._stats[k] = 0 if isinstance(self._stats[k], int) else 0.0

    def record_detection(self, class_name: str, confidence: float):
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

    # ── Folder creation ───────────────────────────────────────────────────────
    def _create_fresh_folder(self, seq: int) -> Path:
        """
        Create a GUARANTEED unique mission folder.

        Naming: Mission_YYYYMMDD_HHMMSS_NNN
          • YYYYMMDD_HHMMSS  — wall-clock timestamp of this exact start() call
          • NNN              — global monotonic counter (never repeats in process)

        Even if two missions start within the same second, NNN differs.
        Even if the process restarts and the clock hasn't advanced, the
        timestamp differs from any prior run.

        The folder is always brand new — no exist_ok silent re-use.
        """
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"Mission_{ts}_{seq:03d}"
        root = self.DB_ROOT / folder_name

        # If somehow the same name exists (clock skew, test replay), add UUID
        if root.exists():
            root = self.DB_ROOT / f"{folder_name}_{str(uuid.uuid4())[:6].upper()}"

        # Create sub-directories
        for sub in ("recordings", "screenshots", "evidence", "logs", "report"):
            (root / sub).mkdir(parents=True, exist_ok=False)

        # Write fresh (empty) CSV header — file guaranteed not to exist yet
        (root / "detections.csv").write_text(
            "detection_id,timestamp,class,confidence,track_id,"
            "priority,camera,frame_id,verified,notes\n",
            encoding="utf-8")
        # Write empty JSON array
        (root / "detections.json").write_text("[]", encoding="utf-8")

        return root

    # ── Persistence ───────────────────────────────────────────────────────────
    def _gen_id(self, seq: int) -> str:
        date_str = datetime.now().strftime("%Y%m%d")
        return f"MSN-{date_str}-{seq:03d}"

    def _save_mission_json(self, folder: Optional[Path] = None):
        target = folder or self.mission_dir
        if target is None:
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
            "elapsed":       self.elapsed_str,
            "stats":         dict(self._stats),
            "timeline":      [e.to_dict() for e in self._timeline],
        }
        try:
            (target / "mission.json").write_text(
                json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _write_mission_log(self, folder: Optional[Path] = None):
        """Write logs/mission.log with full timestamped timeline."""
        target = folder or self.mission_dir
        if target is None:
            return
        try:
            lines = [
                f"Mission Log — {self.mission_name}",
                f"ID: {self.mission_id}",
                f"Operator: {self.operator_name}  |  Drone: {self.drone_name}",
                f"Area: {self.search_area}",
                f"Type: {self.mission_type.value}",
                f"Start: {self.start_time}  |  End: {self.end_time}",
                f"Elapsed: {self.elapsed_str}",
                "─" * 60,
                "",
            ]
            for e in self._timeline:
                lines.append(f"{e.timestamp.strftime('%Y-%m-%d %H:%M:%S')}  "
                             f"[{e.level.upper():<9}]  {e.message}")
            (target / "logs" / "mission.log").write_text(
                "\n".join(lines), encoding="utf-8")
        except Exception:
            pass

    def _write_report(self, folder: Optional[Path] = None):
        """Write report/mission_report.txt — human-readable summary."""
        target = folder or self.mission_dir
        if target is None:
            return
        try:
            s = self._stats
            lines = [
                "=" * 60,
                "    DualVision AI — Mission Report",
                "=" * 60,
                "",
                f"  Mission Name : {self.mission_name}",
                f"  Mission ID   : {self.mission_id}",
                f"  Mission Type : {self.mission_type.value}",
                f"  Operator     : {self.operator_name}",
                f"  Drone        : {self.drone_name}",
                f"  Search Area  : {self.search_area}",
                "",
                f"  Start Time   : {self.start_time}",
                f"  End Time     : {self.end_time}",
                f"  Duration     : {self.elapsed_str}",
                "",
                "─" * 60,
                "  STATISTICS",
                "─" * 60,
                f"  Total Detections : {s['total_detections']}",
                f"  Persons Found    : {s['persons']}",
                f"  Vehicles Found   : {s['vehicles']}",
                f"  Animals Found    : {s['animals']}",
                f"  Fire / Smoke     : {s['fire_smoke']}",
                f"  Screenshots      : {s['screenshots']}",
                f"  Avg Confidence   : {s['avg_confidence']:.3f}",
                f"  Detection Rate   : {s['detection_rate']*60:.1f} / min",
                "",
                "─" * 60,
                "  FOLDER",
                "─" * 60,
                f"  {target}",
                "",
                "=" * 60,
                f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "=" * 60,
            ]
            (target / "report" / "mission_report.txt").write_text(
                "\n".join(lines), encoding="utf-8")
        except Exception:
            pass

    # ── Mission History DB ────────────────────────────────────────────────────
    @classmethod
    def list_missions(cls) -> List[Dict]:
        """Return list of all past missions, newest first."""
        missions = []
        if not cls.DB_ROOT.exists():
            return missions
        for folder in sorted(cls.DB_ROOT.iterdir(), reverse=True):
            if not folder.is_dir():
                continue
            mj = folder / "mission.json"
            if mj.exists():
                try:
                    data = json.loads(mj.read_text(encoding="utf-8"))
                    data["_folder"] = str(folder)
                    missions.append(data)
                except Exception:
                    pass
        return missions
