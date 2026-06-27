"""
Lightweight ByteTrack-inspired tracker using IoU matching.
No external dependency — pure NumPy implementation.
"""
import numpy as np
import time


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


class Track:
    _id_counter = 0

    def __init__(self, box, class_id, confidence):
        Track._id_counter += 1
        self.track_id = Track._id_counter
        self.box = box
        self.class_id = class_id
        self.confidence = confidence
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.created_at = time.time()

    def update(self, box, class_id, confidence):
        self.box = box
        self.class_id = class_id
        self.confidence = confidence
        self.hits += 1
        self.time_since_update = 0

    def predict(self):
        self.age += 1
        self.time_since_update += 1


class ByteTracker:
    def __init__(self, max_age: int = 30, min_hits: int = 1, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._tracks: list[Track] = []

    def reset(self):
        self._tracks = []
        Track._id_counter = 0

    def update(self, detections: list[dict]) -> list[dict]:
        """
        detections: list of {"box": [x1,y1,x2,y2], "class_id": int, "confidence": float}
        returns: list with added "track_id" field
        """
        for t in self._tracks:
            t.predict()

        unmatched_tracks = list(range(len(self._tracks)))
        matched_det_indices = set()
        results = []

        if detections and self._tracks:
            iou_matrix = np.zeros((len(self._tracks), len(detections)))
            for ti, track in enumerate(self._tracks):
                for di, det in enumerate(detections):
                    iou_matrix[ti, di] = iou(track.box, det["box"])

            while True:
                if iou_matrix.size == 0:
                    break
                max_val = iou_matrix.max()
                if max_val < self.iou_threshold:
                    break
                ti, di = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                det = detections[di]
                self._tracks[ti].update(det["box"], det["class_id"], det["confidence"])
                if ti in unmatched_tracks:
                    unmatched_tracks.remove(ti)
                matched_det_indices.add(di)
                iou_matrix[ti, :] = -1
                iou_matrix[:, di] = -1

        for di, det in enumerate(detections):
            if di not in matched_det_indices:
                new_track = Track(det["box"], det["class_id"], det["confidence"])
                self._tracks.append(new_track)

        surviving = []
        for track in self._tracks:
            if track.time_since_update <= self.max_age:
                if track.hits >= self.min_hits or track.time_since_update == 0:
                    results.append({
                        "box": track.box,
                        "class_id": track.class_id,
                        "confidence": track.confidence,
                        "track_id": track.track_id
                    })
                surviving.append(track)

        self._tracks = surviving
        return results

    @property
    def track_count(self) -> int:
        return len(self._tracks)
