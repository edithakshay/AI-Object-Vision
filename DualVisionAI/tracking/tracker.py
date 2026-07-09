"""
DualVision AI Detector — ByteTrack Tracker  (Phase 2 — Stable CPU Edition)

Phase 2 additions over the Phase 1 regression-fixed version:
  • Track history — each track stores its centre-point history (deque)
  • Motion estimation — velocity (vx, vy) and direction computed per frame
  • Trail lines — draw_trails() renders polylines on an OpenCV frame
  • Event callbacks — "created", "lost", "recovered", "removed" events
    emitted via a registered callable so callers can log them without
    coupling the tracker to any file-system dependency
  • Confirmed-tracks stat — tracks that have exceeded min_hits
  • Longest-active-track stat
  • Empty-detection handling — update([]) ages tracks instead of reset()
    so re-identification after brief disappearance works correctly

Kalman stability (from Phase 1 fixes, unchanged):
  • float64 throughout
  • Joseph form covariance update
  • np.linalg.solve instead of np.linalg.inv
  • NaN / Inf health guard with re-initiation
"""

import cv2
import numpy as np
import time
from collections import deque
from typing import List, Dict, Optional, Callable


# ── Kalman Filter ─────────────────────────────────────────────────────────────

class KalmanBoxFilter:
    """
    Constant-velocity Kalman filter — state [cx, cy, w, h, vx, vy, vw, vh].
    Numerically stable: float64, Joseph form, solve not inv, NaN guard.
    """

    _DT = np.float64

    def __init__(self):
        D = self._DT
        self.F = np.eye(8, dtype=D)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        self.H = np.eye(4, 8, dtype=D)
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0,
                          0.01, 0.01, 0.0001, 0.0001]).astype(D)
        self.R = np.diag([1.0, 1.0, 10.0, 10.0]).astype(D)
        self.x = np.zeros((8, 1), dtype=D)
        self.P = np.eye(8, dtype=D) * 10.0
        self._last_good_box: list = [0.0, 0.0, 1.0, 1.0]

    @staticmethod
    def _box_to_cx(box):
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        return np.array([[
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
            max(x2 - x1, 1.0),
            max(y2 - y1, 1.0),
        ]], dtype=KalmanBoxFilter._DT).T

    @staticmethod
    def _cx_to_box(cx):
        vals = cx.ravel()
        cx_, cy_ = float(vals[0]), float(vals[1])
        w, h = max(float(vals[2]), 1.0), max(float(vals[3]), 1.0)
        return [cx_ - w / 2, cy_ - h / 2, cx_ + w / 2, cy_ + h / 2]

    def _is_healthy(self) -> bool:
        return np.all(np.isfinite(self.x)) and np.all(np.isfinite(self.P))

    def initiate(self, box):
        self._last_good_box = list(box)
        self.x[:] = 0.0
        self.x[:4] = self._box_to_cx(box)
        self.P = np.eye(8, dtype=self._DT) * 10.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        if not self._is_healthy():
            self.initiate(self._last_good_box)
        return self._cx_to_box(self.x[:4])

    def update(self, box):
        self._last_good_box = list(box)
        z = self._box_to_cx(box)
        S = self.H @ self.P @ self.H.T + self.R
        PH = self.P @ self.H.T
        try:
            K = np.linalg.solve(S.T, PH.T).T
        except np.linalg.LinAlgError:
            return
        y = z - self.H @ self.x
        self.x = self.x + K @ y
        I8 = np.eye(8, dtype=self._DT)
        IKH = I8 - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T
        if not self._is_healthy():
            self.initiate(box)

    @property
    def predicted_box(self):
        if not self._is_healthy():
            return list(self._last_good_box)
        return self._cx_to_box(self.x[:4])


# ── IoU helpers ───────────────────────────────────────────────────────────────

def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def _iou_matrix(tracks, detections) -> np.ndarray:
    boxes_t = [t.predicted_box for t in tracks]
    boxes_d = [d["box"] for d in detections]
    mat = np.zeros((len(boxes_t), len(boxes_d)), dtype=np.float32)
    for i, bt in enumerate(boxes_t):
        for j, bd in enumerate(boxes_d):
            mat[i, j] = _iou(bt, bd)
    return mat


def _greedy_match(iou_mat: np.ndarray, threshold: float) -> List[tuple]:
    mat = iou_mat.copy()
    pairs = []
    while mat.size > 0:
        best = mat.max()
        if best < threshold:
            break
        ti, di = np.unravel_index(mat.argmax(), mat.shape)
        pairs.append((int(ti), int(di)))
        mat[ti, :] = -1.0
        mat[:, di] = -1.0
    return pairs


# ── Direction estimation ───────────────────────────────────────────────────────

_DIR_THRESHOLD = 0.5   # pixels/frame below which a track is "stationary"

def _compute_direction(vx: float, vy: float) -> str:
    spd = (vx ** 2 + vy ** 2) ** 0.5
    if spd < _DIR_THRESHOLD:
        return "stationary"
    ax, ay = abs(vx), abs(vy)
    if ax > ay * 2:
        return "right" if vx > 0 else "left"
    if ay > ax * 2:
        return "down" if vy > 0 else "up"
    if vx > 0 and vy > 0:
        return "down-right"
    if vx > 0 and vy < 0:
        return "up-right"
    if vx < 0 and vy > 0:
        return "down-left"
    return "up-left"


# ── Track ─────────────────────────────────────────────────────────────────────

class _Track:
    _counter = 0

    def __init__(self, box, class_id: int, confidence: float,
                 max_trail_length: int = 30):
        _Track._counter += 1
        self.track_id   = _Track._counter
        self.class_id   = class_id
        self.confidence = confidence
        self.hits       = 1
        self.age        = 1
        self.missed     = 0
        self.state      = "active"

        _now = time.time()
        self.first_seen  = _now
        self.last_seen   = _now

        # Lifecycle counters
        self.lost_count      = 0   # times this track went "lost"
        self.recovered_count = 0   # times this track was recovered

        # Motion
        self.velocity  = (0.0, 0.0)    # (vx, vy) pixels/frame
        self.direction = "stationary"

        # Trail — deque of (cx, cy) centre points
        self.center_history: deque = deque(maxlen=max_trail_length)

        # Kalman
        self.kf = KalmanBoxFilter()
        self.kf.initiate(box)
        self.box = list(box)

        # Seed history with first observation
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        self.center_history.append((cx, cy))

    def predict(self):
        self.age    += 1
        self.missed += 1
        return self.kf.predict()

    def update(self, box, class_id: int, confidence: float):
        prev_cx = (self.box[0] + self.box[2]) / 2.0
        prev_cy = (self.box[1] + self.box[3]) / 2.0

        self.box        = list(box)
        self.class_id   = class_id
        self.confidence = confidence
        self.hits      += 1
        self.missed     = 0
        self.state      = "active"
        self.last_seen  = time.time()

        # Update Kalman
        self.kf.update(box)

        # Velocity and direction
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        vx = cx - prev_cx
        vy = cy - prev_cy
        self.velocity  = (vx, vy)
        self.direction = _compute_direction(vx, vy)

        # Trail history
        self.center_history.append((cx, cy))

    @property
    def predicted_box(self):
        return self.kf.predicted_box

    @property
    def track_age_sec(self) -> float:
        return time.time() - self.first_seen

    @property
    def confirmed(self) -> bool:
        return self.hits >= 1   # min_hits checked by ByteTracker.update()

    @property
    def center(self):
        return ((self.box[0] + self.box[2]) / 2.0,
                (self.box[1] + self.box[3]) / 2.0)


# ── Trail drawing ─────────────────────────────────────────────────────────────

def draw_trails(
    frame,
    trails: Dict[int, list],
    line_color: tuple = (100, 200, 255),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw trail polylines onto *frame* (in-place copy returned).

    trails: {track_id: [(cx, cy), ...]}  — oldest point first.
    Brightness fades from dim (old) to bright (newest) to show motion direction.
    """
    if frame is None or not trails:
        return frame
    out = frame.copy()
    for _tid, points in trails.items():
        n = len(points)
        if n < 2:
            continue
        for i in range(1, n):
            alpha = i / n           # 0.0 = oldest, 1.0 = newest
            c = tuple(int(v * alpha) for v in line_color)
            w = max(1, int(thickness * alpha))
            pt1 = (int(points[i - 1][0]), int(points[i - 1][1]))
            pt2 = (int(points[i][0]),     int(points[i][1]))
            cv2.line(out, pt1, pt2, c, w, cv2.LINE_AA)
        # Bright dot at current position
        if n >= 1:
            tip = (int(points[-1][0]), int(points[-1][1]))
            cv2.circle(out, tip, thickness + 1, line_color, -1, cv2.LINE_AA)
    return out


# ── ByteTracker ───────────────────────────────────────────────────────────────

class ByteTracker:
    """
    Enhanced ByteTrack-inspired tracker — Phase 2 (Stable CPU Edition).

    Parameters
    ----------
    max_age              : frames a lost track is kept before removal.
    min_hits             : minimum hits before a track is emitted.
    iou_threshold        : stage-1 IoU threshold (active tracks).
    low_iou              : stage-2 IoU threshold (lost tracks).
    high_conf            : confidence split between "high" and "low" dets.
    max_trail_length     : maximum centre-point history per track.

    Event callback
    --------------
    Register with set_event_callback(fn) where fn(event_type, track):
      • "created"   — new track spawned
      • "lost"      — active track missed for the first time this frame
      • "recovered" — lost track re-identified
      • "removed"   — track expired (missed > max_age)
    """

    def __init__(
        self,
        max_age:             int   = 5,
        min_hits:            int   = 1,
        iou_threshold:       float = 0.35,
        low_iou:             float = 0.20,
        high_conf:           float = 0.45,
        max_trail_length:    int   = 30,
    ):
        self.max_age          = max_age
        self.min_hits         = min_hits
        self.iou_threshold    = iou_threshold
        self.low_iou          = low_iou
        self.high_conf        = high_conf
        self.max_trail_length = max_trail_length

        self._tracks: List[_Track] = []

        self._new_tracks_total       = 0
        self._recovered_tracks_total = 0
        self._tracking_fps           = 0.0
        self._tracking_latency_ms    = 0.0
        self._fps_counter            = 0
        self._fps_t0                 = time.time()

        self._event_callback: Optional[Callable] = None

    # ── Event callbacks ───────────────────────────────────────────────────────

    def set_event_callback(self, fn: Callable):
        """Register fn(event_type: str, track: _Track) for lifecycle events."""
        self._event_callback = fn

    def _emit(self, event_type: str, track: _Track):
        if self._event_callback is not None:
            try:
                self._event_callback(event_type, track)
            except Exception:
                pass

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self):
        self._tracks.clear()
        _Track._counter = 0
        self._new_tracks_total       = 0
        self._recovered_tracks_total = 0
        self._tracking_fps           = 0.0
        self._tracking_latency_ms    = 0.0
        self._fps_counter            = 0
        self._fps_t0                 = time.time()

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        detections: [{"box": [x1,y1,x2,y2], "class_id": int, "confidence": float}]
        Returns list of confirmed tracked objects with "track_id" added.

        Calling update([]) on an empty frame ages all tracks correctly so that
        re-identification works after brief disappearances — do NOT reset().
        """
        t_start = time.perf_counter()

        # ── Step 1: Predict existing tracks ───────────────────────────────────
        for t in self._tracks:
            t.predict()

        # ── Step 2: Split detections by confidence ────────────────────────────
        high_dets = [d for d in detections if d["confidence"] >= self.high_conf]
        low_dets  = [d for d in detections if d["confidence"] <  self.high_conf]

        active_tracks = [t for t in self._tracks
                         if t.state == "active" or t.missed <= 2]
        lost_tracks   = [t for t in self._tracks if t.state == "lost"]

        matched_track_ids:   set = set()
        matched_det_indices: set = set()

        # ── Stage 1a: High-conf dets vs active/recent tracks ──────────────────
        if active_tracks and high_dets:
            mat   = _iou_matrix(active_tracks, high_dets)
            pairs = _greedy_match(mat, self.iou_threshold)
            for ti, di in pairs:
                track    = active_tracks[ti]
                det      = high_dets[di]
                was_lost = track.state == "lost"
                track.update(det["box"], det["class_id"], det["confidence"])
                matched_track_ids.add(track.track_id)
                matched_det_indices.add(("high", di))
                if was_lost:
                    self._recovered_tracks_total += 1
                    self._emit("recovered", track)

        # ── Stage 1b: Low-conf dets vs unmatched active tracks ────────────────
        unmatched_active = [t for t in active_tracks
                            if t.track_id not in matched_track_ids]
        if unmatched_active and low_dets:
            mat   = _iou_matrix(unmatched_active, low_dets)
            pairs = _greedy_match(mat, self.low_iou)
            for ti, di in pairs:
                track    = unmatched_active[ti]
                det      = low_dets[di]
                was_lost = track.state == "lost"
                track.update(det["box"], det["class_id"], det["confidence"])
                matched_track_ids.add(track.track_id)
                matched_det_indices.add(("low", di))
                if was_lost:
                    self._recovered_tracks_total += 1
                    self._emit("recovered", track)

        # ── Stage 2a: Unmatched high-conf dets vs lost tracks ────────────────
        unmatched_high = [(i, d) for i, d in enumerate(high_dets)
                          if ("high", i) not in matched_det_indices]
        if lost_tracks and unmatched_high:
            mat   = _iou_matrix(lost_tracks, [d for _, d in unmatched_high])
            pairs = _greedy_match(mat, self.low_iou)
            for ti, di in pairs:
                track  = lost_tracks[ti]
                orig_i = unmatched_high[di][0]
                det    = high_dets[orig_i]
                track.update(det["box"], det["class_id"], det["confidence"])
                track.recovered_count += 1
                matched_track_ids.add(track.track_id)
                matched_det_indices.add(("high", orig_i))
                self._recovered_tracks_total += 1
                self._emit("recovered", track)

        # ── Stage 2b: Unmatched low-conf dets vs still-unmatched lost tracks ─
        # Gap fix: a returning object detected at low confidence would otherwise
        # spawn a NEW track instead of recovering the existing lost one.
        unmatched_low = [(i, d) for i, d in enumerate(low_dets)
                         if ("low", i) not in matched_det_indices]
        still_lost = [t for t in lost_tracks
                      if t.track_id not in matched_track_ids]
        if still_lost and unmatched_low:
            mat   = _iou_matrix(still_lost, [d for _, d in unmatched_low])
            pairs = _greedy_match(mat, self.low_iou)
            for ti, di in pairs:
                track  = still_lost[ti]
                orig_i = unmatched_low[di][0]
                det    = low_dets[orig_i]
                track.update(det["box"], det["class_id"], det["confidence"])
                track.recovered_count += 1
                matched_track_ids.add(track.track_id)
                matched_det_indices.add(("low", orig_i))
                self._recovered_tracks_total += 1
                self._emit("recovered", track)

        # ── Step 3: Spawn new tracks for ALL unmatched detections ─────────────
        new_track_ids: set = set()

        for i, det in enumerate(high_dets):
            if ("high", i) not in matched_det_indices:
                t = _Track(det["box"], det["class_id"], det["confidence"],
                           self.max_trail_length)
                self._tracks.append(t)
                self._new_tracks_total += 1
                new_track_ids.add(t.track_id)
                self._emit("created", t)

        for i, det in enumerate(low_dets):
            if ("low", i) not in matched_det_indices:
                t = _Track(det["box"], det["class_id"], det["confidence"],
                           self.max_trail_length)
                self._tracks.append(t)
                self._new_tracks_total += 1
                new_track_ids.add(t.track_id)
                self._emit("created", t)

        # ── Step 4: Classify surviving tracks ─────────────────────────────────
        surviving: List[_Track] = []
        for t in self._tracks:
            if t.track_id in matched_track_ids or t.track_id in new_track_ids:
                t.state = "active"
                surviving.append(t)
            elif t.missed <= self.max_age:
                if t.state == "active":          # just became lost
                    t.lost_count += 1
                    self._emit("lost", t)
                t.state = "lost"
                surviving.append(t)
            else:
                self._emit("removed", t)          # track expires

        self._tracks = surviving

        # ── Step 5: Collect output — confirmed active tracks ──────────────────
        results: List[Dict] = []
        for t in self._tracks:
            if t.state == "active" and t.hits >= self.min_hits and t.missed == 0:
                results.append({
                    "box":             t.box,
                    "class_id":        t.class_id,
                    "confidence":      t.confidence,
                    "track_id":        t.track_id,
                    "velocity":        t.velocity,
                    "direction":       t.direction,
                    "age_sec":         t.track_age_sec,
                    "hits":            t.hits,
                    "lost_count":      t.lost_count,
                    "recovered_count": t.recovered_count,
                })

        # ── FPS / latency ─────────────────────────────────────────────────────
        self._tracking_latency_ms = (time.perf_counter() - t_start) * 1000.0
        self._fps_counter += 1
        _now = time.time()
        elapsed = _now - self._fps_t0
        if elapsed >= 1.0:
            self._tracking_fps = self._fps_counter / elapsed
            self._fps_counter  = 0
            self._fps_t0       = _now

        return results

    # ── Trail data ────────────────────────────────────────────────────────────

    def get_trails(self) -> Dict[int, list]:
        """Return {track_id: [(cx, cy), ...]} for all active tracks."""
        trails = {}
        for t in self._tracks:
            if t.state == "active" and len(t.center_history) > 1:
                trails[t.track_id] = list(t.center_history)
        return trails

    # ── Statistics ────────────────────────────────────────────────────────────

    @property
    def active_tracks(self) -> int:
        return sum(1 for t in self._tracks if t.state == "active")

    @property
    def confirmed_tracks(self) -> int:
        return sum(1 for t in self._tracks
                   if t.state == "active" and t.hits >= self.min_hits)

    @property
    def lost_tracks(self) -> int:
        return sum(1 for t in self._tracks if t.state == "lost")

    @property
    def recovered_tracks_total(self) -> int:
        return self._recovered_tracks_total

    @property
    def new_tracks_total(self) -> int:
        return self._new_tracks_total

    @property
    def avg_track_age_sec(self) -> float:
        ages = [t.track_age_sec for t in self._tracks if t.state == "active"]
        return (sum(ages) / len(ages)) if ages else 0.0

    @property
    def longest_active_track_sec(self) -> float:
        ages = [t.track_age_sec for t in self._tracks if t.state == "active"]
        return max(ages) if ages else 0.0

    @property
    def tracking_fps(self) -> float:
        return self._tracking_fps

    @property
    def tracking_latency_ms(self) -> float:
        return self._tracking_latency_ms

    @property
    def track_count(self) -> int:
        return len(self._tracks)

    def get_stats(self) -> Dict:
        return {
            "active_tracks":      self.active_tracks,
            "confirmed_tracks":   self.confirmed_tracks,
            "lost_tracks":        self.lost_tracks,
            "recovered_total":    self._recovered_tracks_total,
            "new_total":          self._new_tracks_total,
            "avg_age_sec":        self.avg_track_age_sec,
            "longest_active_sec": self.longest_active_track_sec,
            "tracking_fps":       self._tracking_fps,
            "latency_ms":         self._tracking_latency_ms,
        }
