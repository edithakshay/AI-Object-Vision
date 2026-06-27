"""
Lightweight IoU-based tracker.

max_age=0  → track deleted the moment it is NOT matched by a detection.
             This eliminates ghost bounding boxes completely.
max_age>0  → track stays alive for N extra inference cycles after last match.
"""
import numpy as np
import time


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


class _Track:
    _counter = 0

    def __init__(self, box, class_id, confidence):
        _Track._counter += 1
        self.track_id = _Track._counter
        self.box = box
        self.class_id = class_id
        self.confidence = confidence
        self.hits = 1
        self.missed = 0       # consecutive missed inference cycles

    def matched(self, box, class_id, confidence):
        self.box = box
        self.class_id = class_id
        self.confidence = confidence
        self.hits += 1
        self.missed = 0

    def miss(self):
        self.missed += 1


class ByteTracker:
    def __init__(self, max_age: int = 0, min_hits: int = 1,
                 iou_threshold: float = 0.30):
        """
        max_age : how many consecutive missed cycles before a track is removed.
                  0 = remove immediately when not matched (no ghost boxes).
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._tracks: list[_Track] = []

    def reset(self):
        self._tracks.clear()
        _Track._counter = 0

    def update(self, detections: list[dict]) -> list[dict]:
        """
        detections : [{"box": [x1,y1,x2,y2], "class_id": int, "confidence": float}]
        returns    : same list with "track_id" added for each matched/new detection
        """
        # Mark all tracks as missed initially; matched ones will be reset below
        for t in self._tracks:
            t.miss()

        matched_det_indices: set[int] = set()

        # --- Hungarian-lite: greedily match highest-IoU pairs ---
        if self._tracks and detections:
            iou_mat = np.array(
                [[_iou(t.box, d["box"]) for d in detections]
                 for t in self._tracks],
                dtype=np.float32
            )
            while True:
                if iou_mat.size == 0:
                    break
                best = iou_mat.max()
                if best < self.iou_threshold:
                    break
                ti, di = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
                self._tracks[ti].matched(
                    detections[di]["box"],
                    detections[di]["class_id"],
                    detections[di]["confidence"]
                )
                matched_det_indices.add(int(di))
                iou_mat[ti, :] = -1.0
                iou_mat[:, di] = -1.0

        # --- Create new tracks for unmatched detections ---
        for di, det in enumerate(detections):
            if di not in matched_det_indices:
                self._tracks.append(
                    _Track(det["box"], det["class_id"], det["confidence"])
                )

        # --- Cull dead tracks; collect output ---
        results: list[dict] = []
        alive: list[_Track] = []
        for t in self._tracks:
            if t.missed > self.max_age:
                continue                 # track is dead — drop it
            alive.append(t)
            if t.hits >= self.min_hits and t.missed == 0:
                # Only emit boxes that were matched THIS cycle
                results.append({
                    "box": t.box,
                    "class_id": t.class_id,
                    "confidence": t.confidence,
                    "track_id": t.track_id,
                })

        self._tracks = alive
        return results

    @property
    def track_count(self) -> int:
        return len(self._tracks)
