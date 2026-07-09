"""
Confidence Smoother — DualVision AI v1.3 Stable CPU Edition.

Prevents the "disappears for 1-3 frames" problem by applying temporal
smoothing to raw ONNX detections BEFORE they reach the ByteTracker.

Algorithm per frame:
  1. IoU-match current frame detections to the previous-frame history.
  2. For matched detections, apply EMA: smooth_conf = α·raw + (1-α)·prev
  3. For unmatched history entries (disappeared):
       – If within max_ghost_frames, synthesise a ghost detection at the
         last known position with decaying confidence.
       – Ghost confidence = prev_smooth · decay_factor^missed_frames
       – Drop ghost when confidence falls below min_ghost_conf.
  4. Return the merged list (real + ghost) for the tracker to consume.

Key design rules:
  • Pure Python / NumPy — no CV2, no ONNX, no PyTorch.
  • Thread-safe: one instance per stream (rgb / thermal).
  • All parameters hot-configurable (update_params).
  • Does NOT replace the tracker — sits upstream of it.
"""

import time
import numpy as np
from typing import List, Dict


def _iou(a: list, b: list) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw  = max(0, ix2 - ix1)
    ih  = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class _Entry:
    """Per-detection history entry."""
    __slots__ = ("box", "class_id", "smooth_conf", "missed",
                 "last_seen", "first_seen", "frames_seen")

    def __init__(self, box, class_id, conf):
        self.box         = list(box)
        self.class_id    = int(class_id)
        self.smooth_conf = float(conf)
        self.missed      = 0
        t = time.time()
        self.last_seen   = t
        self.first_seen  = t
        self.frames_seen = 1


class ConfidenceSmoother:
    """
    IoU-based temporal confidence smoother.

    Parameters
    ----------
    ema_alpha        : EMA weight for current frame (0–1; lower = smoother)
    iou_threshold    : minimum IoU to match a detection to a history entry
    max_ghost_frames : frames a detection is kept alive after disappearing
    ghost_decay      : confidence multiplier per missed frame (0–1)
    min_ghost_conf   : minimum confidence for ghost detections
    min_box_area     : ignore detections smaller than this (px²), 0=disabled
    max_box_area     : ignore detections larger than this (px²), 0=disabled
    """

    def __init__(
        self,
        ema_alpha:        float = 0.35,
        iou_threshold:    float = 0.40,
        max_ghost_frames: int   = 3,
        ghost_decay:      float = 0.70,
        min_ghost_conf:   float = 0.25,
        min_box_area:     int   = 0,
        max_box_area:     int   = 0,
    ):
        self.ema_alpha        = ema_alpha
        self.iou_threshold    = iou_threshold
        self.max_ghost_frames = max_ghost_frames
        self.ghost_decay      = ghost_decay
        self.min_ghost_conf   = min_ghost_conf
        self.min_box_area     = min_box_area
        self.max_box_area     = max_box_area
        self._history: List[_Entry] = []

    # ── Hot-reconfiguration ────────────────────────────────────────────────────
    def update_params(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def reset(self):
        self._history.clear()

    # ── Main API ───────────────────────────────────────────────────────────────
    def process(self, detections: List[Dict]) -> List[Dict]:
        """
        Smooth a list of raw detections.

        Input / output format:
          {"box": [x1,y1,x2,y2], "class_id": int, "confidence": float}

        Ghost detections are tagged with  "ghost": True.
        """
        # ── 0. Filter by box area ────────────────────────────────────────────
        filtered = []
        for d in detections:
            b = d["box"]
            w = max(0, b[2] - b[0])
            h = max(0, b[3] - b[1])
            area = w * h
            if self.min_box_area > 0 and area < self.min_box_area:
                continue
            if self.max_box_area > 0 and area > self.max_box_area:
                continue
            filtered.append(d)

        # ── 1. Predict all history entries forward (age by 1) ────────────────
        for e in self._history:
            e.missed += 1

        # ── 2. Match current detections to history entries ───────────────────
        matched_entry_indices: set = set()
        matched_det_indices:   set = set()
        output: List[Dict] = []

        for di, det in enumerate(filtered):
            best_iou  = self.iou_threshold - 1e-6  # must exceed threshold
            best_ei   = -1
            for ei, entry in enumerate(self._history):
                if ei in matched_entry_indices:
                    continue
                if entry.class_id != det["class_id"]:
                    continue
                iou = _iou(entry.box, det["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_ei  = ei

            if best_ei >= 0:
                # Matched → EMA update
                e = self._history[best_ei]
                e.smooth_conf = (self.ema_alpha * det["confidence"]
                                 + (1 - self.ema_alpha) * e.smooth_conf)
                e.box       = det["box"]
                e.missed    = 0
                e.last_seen = time.time()
                e.frames_seen += 1
                matched_entry_indices.add(best_ei)
                matched_det_indices.add(di)
                output.append({
                    "box":        e.box,
                    "class_id":   e.class_id,
                    "confidence": e.smooth_conf,
                    "ghost":      False,
                })
            else:
                # New detection → create history entry
                e = _Entry(det["box"], det["class_id"], det["confidence"])
                self._history.append(e)
                matched_det_indices.add(di)
                output.append({
                    "box":        e.box,
                    "class_id":   e.class_id,
                    "confidence": e.smooth_conf,
                    "ghost":      False,
                })

        # ── 3. Unmatched history entries → ghost detections ──────────────────
        surviving: List[_Entry] = []
        for ei, entry in enumerate(self._history):
            if ei in matched_entry_indices:
                # Already matched above — keep entry
                surviving.append(entry)
                continue

            if entry.missed > self.max_ghost_frames:
                # Expired — drop
                continue

            # Ghost: decay confidence and synthesise a detection
            entry.smooth_conf = entry.smooth_conf * self.ghost_decay
            if entry.smooth_conf < self.min_ghost_conf:
                continue   # too faint — drop

            surviving.append(entry)
            if entry.missed <= self.max_ghost_frames:
                output.append({
                    "box":        entry.box,
                    "class_id":   entry.class_id,
                    "confidence": entry.smooth_conf,
                    "ghost":      True,
                })

        self._history = surviving
        return output

    # ── Stats ─────────────────────────────────────────────────────────────────
    def get_stats(self) -> dict:
        active  = [e for e in self._history if e.missed == 0]
        ghosted = [e for e in self._history if e.missed > 0]
        confs   = [e.smooth_conf for e in active]
        return {
            "active_entries":  len(active),
            "ghost_entries":   len(ghosted),
            "avg_smooth_conf": round(sum(confs) / len(confs), 3) if confs else 0.0,
        }
