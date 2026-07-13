"""
Alert System — DualVision AI Phase 3
Manages detection alerts: popup, sound, mission log.
High-priority objects get visual highlight + sound.
"""

import threading
import time
from typing import Callable, Optional

from mission.mission_state import get_priority, priority_label

# ── Sound support (optional — tkinter bell fallback) ──────────────────────────
_HAS_WINSOUND = False
try:
    import winsound as _ws
    _HAS_WINSOUND = True
except ImportError:
    pass


def _beep(freq: int = 800, duration: int = 200):
    """Non-blocking beep — falls back to no sound if unavailable."""
    def _do():
        try:
            if _HAS_WINSOUND:
                _ws.Beep(freq, duration)
        except Exception:
            pass
    threading.Thread(target=_do, daemon=True).start()


class Alert:
    __slots__ = ("class_name", "confidence", "track_id", "camera",
                 "priority", "label", "timestamp")

    def __init__(self, class_name: str, confidence: float,
                 track_id: int, camera: str):
        import datetime
        self.class_name  = class_name
        self.confidence  = confidence
        self.track_id    = track_id
        self.camera      = camera
        self.priority    = get_priority(class_name)
        self.label       = priority_label(self.priority)
        self.timestamp   = datetime.datetime.now().strftime("%H:%M:%S")


class AlertSystem:
    """
    Call `check(class_name, confidence, track_id, camera)` from the
    detection loop.  Only fires when a **new** track_id is first seen,
    so each track triggers at most one alert.
    """

    def __init__(self):
        self._seen_tracks: set = set()
        self._lock = threading.Lock()
        self._enabled      = True
        self._high_only    = False
        self._sound_enabled = True

        # Callbacks — set by MissionDialog
        self._on_alert: Optional[Callable] = None     # fn(Alert)
        self._bell_fn:  Optional[Callable] = None     # tkinter root.bell()

    def set_callbacks(self, on_alert: Callable, bell_fn: Callable = None):
        self._on_alert = on_alert
        self._bell_fn  = bell_fn

    def configure(self, enabled: bool = True,
                  high_only: bool = False, sound: bool = True):
        self._enabled       = enabled
        self._high_only     = high_only
        self._sound_enabled = sound

    def reset(self):
        with self._lock:
            self._seen_tracks.clear()

    def check(self, class_name: str, confidence: float,
              track_id: int, camera: str):
        if not self._enabled:
            return
        with self._lock:
            if track_id in self._seen_tracks:
                return
            self._seen_tracks.add(track_id)

        a = Alert(class_name, confidence, track_id, camera)

        if self._high_only and a.priority != "high":
            return

        # Sound
        if self._sound_enabled:
            if a.priority == "high":
                _beep(1000, 300)
            elif a.priority == "medium":
                _beep(700, 200)
            # low — no sound
            if self._bell_fn and a.priority == "high":
                try:
                    self._bell_fn()
                except Exception:
                    pass

        # Callback → UI thread will show popup
        if self._on_alert:
            try:
                self._on_alert(a)
            except Exception:
                pass
