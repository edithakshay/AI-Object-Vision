import csv
import json
import logging
import threading
from pathlib import Path
from datetime import datetime
from collections import deque

logger = logging.getLogger("DualVisionAI.detlog")


class DetectionLog:
    def __init__(self, output_dir: str = "logs", max_entries: int = 100000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries
        self._entries: deque = deque(maxlen=max_entries)
        self._lock = threading.Lock()
        self._csv_path = self.output_dir / f"detections_{datetime.now():%Y%m%d_%H%M%S}.csv"
        self._init_csv()

    def _init_csv(self):
        with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "camera", "class", "confidence", "track_id",
                             "x1", "y1", "x2", "y2"])

    def log(self, camera: str, class_name: str, confidence: float,
            track_id: int, box: list):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        entry = {
            "timestamp": ts,
            "camera": camera,
            "class": class_name,
            "confidence": round(confidence, 4),
            "track_id": track_id,
            "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]
        }
        with self._lock:
            self._entries.append(entry)
            try:
                with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([entry["timestamp"], entry["camera"],
                                     entry["class"], entry["confidence"],
                                     entry["track_id"], entry["x1"],
                                     entry["y1"], entry["x2"], entry["y2"]])
            except Exception as e:
                logger.error(f"CSV write error: {e}")

    def export_csv(self, path: str):
        with self._lock:
            entries = list(self._entries)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "camera", "class", "confidence", "track_id",
                "x1", "y1", "x2", "y2"])
            writer.writeheader()
            writer.writerows(entries)

    def export_json(self, path: str):
        with self._lock:
            entries = list(self._entries)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)

    def export_txt(self, path: str):
        with self._lock:
            entries = list(self._entries)
        with open(path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(
                    f"[{e['timestamp']}] Cam:{e['camera']} "
                    f"Class:{e['class']} Conf:{e['confidence']:.3f} "
                    f"ID:{e['track_id']} Box:[{e['x1']},{e['y1']},{e['x2']},{e['y2']}]\n"
                )

    def get_recent(self, n: int = 100) -> list:
        with self._lock:
            return list(self._entries)[-n:]

    def total_count(self) -> int:
        with self._lock:
            return len(self._entries)
