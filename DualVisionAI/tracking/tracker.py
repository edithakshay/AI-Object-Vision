"""
DualVision AI Detector — Enhanced ByteTrack-inspired Tracker
v1.3 Stable CPU Edition  (regression-fixed)

Regression fixes applied:
  1. New tracks created in Step 3 are now correctly marked "active" (not "lost")
     on the same frame they are born — they are added to new_track_ids which is
     treated like matched_track_ids in Step 4.
  2. New tracks are spawned from ALL unmatched detections (not only high-conf),
     so objects with confidence below 0.50 still get a track ID.
  3. main_window.py overlay rule: if tracker returns empty, raw detections are
     drawn without track IDs (handled in _process_result).

Architecture:
  • Kalman filter for smooth bbox prediction and occlusion handling
  • Two-stage matching (high-confidence first, then low-confidence)
  • Lost track buffer — objects can be re-identified after brief disappearance
  • Persistent, non-reassigned track IDs
  • Full statistics: active, lost, recovered, new tracks, avg age, FPS, latency
"""

import numpy as np
import time
from typing import List, Dict


# ── Kalman Filter ─────────────────────────────────────────────────────────────

class KalmanBoxFilter:
    """
    Constant-velocity Kalman filter for bounding-box tracking.
    State vector: [cx, cy, w, h, vx, vy, vw, vh]
    Measurement:  [cx, cy, w, h]
    """

    def __init__(self):
        dt = 1.0

        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = dt

        self.H = np.eye(4, 8, dtype=np.float32)

        self.Q = np.diag([
            1.0, 1.0, 1.0, 1.0,
            0.01, 0.01, 0.0001, 0.0001
        ]).astype(np.float32)

        self.R = np.diag([
            1.0, 1.0, 10.0, 10.0
        ]).astype(np.float32)

        self.x = np.zeros((8, 1), dtype=np.float32)
        self.P = np.eye(8, dtype=np.float32) * 10.0

    @staticmethod
    def _box_to_cx(box):
        x1, y1, x2, y2 = box
        return np.array([[
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
            x2 - x1,
            y2 - y1,
        ]], dtype=np.float32).T

    @staticmethod
    def _cx_to_box(cx):
        cx_, cy_, w, h = float(cx[0]), float(cx[1]), float(cx[2]), float(cx[3])
        w = max(w, 1.0)
        h = max(h, 1.0)
        return [cx_ - w/2, cy_ - h/2, cx_ + w/2, cy_ + h/2]

    def initiate(self, box):
        self.x[:4] = self._box_to_cx(box)
        self.x[4:] = 0.0
        self.P = np.eye(8, dtype=np.float32) * 10.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self._cx_to_box(self.x[:4])

    def update(self, box):
        z = self._box_to_cx(box)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = z - self.H @ self.x
        self.x = self.x + K @ y
        self.P = (np.eye(8, dtype=np.float32) - K @ self.H) @ self.P

    @property
    def predicted_box(self):
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
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
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


# ── Track ─────────────────────────────────────────────────────────────────────

class _Track:
    _counter = 0

    def __init__(self, box, class_id: int, confidence: float):
        _Track._counter += 1
        self.track_id   = _Track._counter
        self.class_id   = class_id
        self.confidence = confidence
        self.hits       = 1
        self.age        = 1
        self.missed     = 0
        self.state      = "active"
        self._created_at = time.time()

        self.kf = KalmanBoxFilter()
        self.kf.initiate(box)
        self.box = list(box)

    def predict(self):
        self.age    += 1
        self.missed += 1
        return self.kf.predict()

    def update(self, box, class_id: int, confidence: float):
        self.box        = list(box)
        self.class_id   = class_id
        self.confidence = confidence
        self.hits      += 1
        self.missed     = 0
        self.state      = "active"
        self.kf.update(box)

    @property
    def predicted_box(self):
        return self.kf.predicted_box

    @property
    def track_age_sec(self) -> float:
        return time.time() - self._created_at


# ── ByteTracker ───────────────────────────────────────────────────────────────

class ByteTracker:
    """
    Enhanced ByteTrack-inspired tracker — regression-fixed.

    Parameters
    ----------
    max_age      : frames a lost track is kept before removal.
    min_hits     : minimum detections before track is emitted (1 = immediate).
    iou_threshold: stage-1 IoU threshold for active tracks.
    low_iou      : stage-2 IoU threshold for lost tracks.
    high_conf    : confidence threshold for "high" vs "low" split.
                   Should match or be slightly below the detector's conf.
    """

    def __init__(
        self,
        max_age:       int   = 5,
        min_hits:      int   = 1,
        iou_threshold: float = 0.35,
        low_iou:       float = 0.20,
        high_conf:     float = 0.45,   # matches default detector conf
    ):
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_threshold = iou_threshold
        self.low_iou       = low_iou
        self.high_conf     = high_conf

        self._tracks: List[_Track] = []

        self._new_tracks_total       = 0
        self._recovered_tracks_total = 0
        self._tracking_fps           = 0.0
        self._tracking_latency_ms    = 0.0
        self._fps_counter            = 0
        self._fps_t0                 = time.time()

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
        detections: [{"box": [x1,y1,x2,y2], "class_id": int,
                       "confidence": float}]
        returns:    same list with "track_id" added for all confirmed tracks.

        IMPORTANT: Returns a result for every matched/new detection on the
        very first frame it is seen (min_hits=1).  New tracks are NEVER
        suppressed on their birth frame.
        """
        t_start = time.perf_counter()

        # ── Step 1: Predict all existing tracks one frame forward ─────────────
        for t in self._tracks:
            t.predict()

        # ── Step 2: Split detections by confidence ────────────────────────────
        high_dets = [d for d in detections if d["confidence"] >= self.high_conf]
        low_dets  = [d for d in detections if d["confidence"] <  self.high_conf]

        active_tracks = [t for t in self._tracks
                         if t.state == "active" or t.missed <= 2]
        lost_tracks   = [t for t in self._tracks if t.state == "lost"]

        # IDs matched this frame — tracks in here keep state="active"
        matched_track_ids: set   = set()
        matched_det_indices: set = set()

        # ── Stage 1a: High-conf dets vs active/recent tracks ─────────────────
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

        # ── Stage 2: Remaining dets vs lost tracks ────────────────────────────
        unmatched_high = [(i, d) for i, d in enumerate(high_dets)
                          if ("high", i) not in matched_det_indices]
        if lost_tracks and unmatched_high:
            mat   = _iou_matrix(lost_tracks, [d for _, d in unmatched_high])
            pairs = _greedy_match(mat, self.low_iou)
            for ti, di in pairs:
                track   = lost_tracks[ti]
                orig_i  = unmatched_high[di][0]
                det     = high_dets[orig_i]
                track.update(det["box"], det["class_id"], det["confidence"])
                matched_track_ids.add(track.track_id)
                matched_det_indices.add(("high", orig_i))
                self._recovered_tracks_total += 1

        # ── Step 3: Create new tracks for ALL still-unmatched detections ──────
        # BUG-FIX: both high AND low unmatched dets spawn new tracks.
        # BUG-FIX: new track IDs are recorded in new_track_ids so Step 4
        #          marks them "active" instead of "lost".
        new_track_ids: set = set()

        for i, det in enumerate(high_dets):
            if ("high", i) not in matched_det_indices:
                t = _Track(det["box"], det["class_id"], det["confidence"])
                self._tracks.append(t)
                self._new_tracks_total += 1
                new_track_ids.add(t.track_id)   # ← FIX: remember new IDs

        for i, det in enumerate(low_dets):
            if ("low", i) not in matched_det_indices:
                t = _Track(det["box"], det["class_id"], det["confidence"])
                self._tracks.append(t)
                self._new_tracks_total += 1
                new_track_ids.add(t.track_id)   # ← FIX: remember new IDs

        # ── Step 4: Classify surviving tracks ─────────────────────────────────
        # BUG-FIX: tracks in new_track_ids are treated exactly like matched
        # tracks — they are marked "active", not "lost".
        surviving: List[_Track] = []
        for t in self._tracks:
            if t.track_id in matched_track_ids or t.track_id in new_track_ids:
                t.state = "active"          # ← FIX: new tracks stay active
                surviving.append(t)
            elif t.missed <= self.max_age:
                t.state = "lost"
                surviving.append(t)
            # else: missed > max_age → track removed

        self._tracks = surviving

        # ── Step 5: Collect output — all active confirmed tracks ──────────────
        results: List[Dict] = []
        for t in self._tracks:
            if t.state == "active" and t.hits >= self.min_hits and t.missed == 0:
                results.append({
                    "box":        t.box,
                    "class_id":   t.class_id,
                    "confidence": t.confidence,
                    "track_id":   t.track_id,
                })

        # ── Update FPS / latency ──────────────────────────────────────────────
        self._tracking_latency_ms = (time.perf_counter() - t_start) * 1000.0
        self._fps_counter += 1
        now = time.time()
        elapsed = now - self._fps_t0
        if elapsed >= 1.0:
            self._tracking_fps = self._fps_counter / elapsed
            self._fps_counter  = 0
            self._fps_t0       = now

        return results

    # ── Statistics ────────────────────────────────────────────────────────────

    @property
    def active_tracks(self) -> int:
        return sum(1 for t in self._tracks if t.state == "active")

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
            "active_tracks":   self.active_tracks,
            "lost_tracks":     self.lost_tracks,
            "recovered_total": self._recovered_tracks_total,
            "new_total":       self._new_tracks_total,
            "avg_age_sec":     self.avg_track_age_sec,
            "tracking_fps":    self._tracking_fps,
            "latency_ms":      self._tracking_latency_ms,
        }
