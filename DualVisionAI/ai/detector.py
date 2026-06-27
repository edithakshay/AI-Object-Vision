"""
AI Detector — wraps Ultralytics YOLO (PT) with optional ONNX Runtime fallback.
Runs inference in a dedicated thread; callers push frames and pull results.
"""
import cv2
import numpy as np
import threading
import queue
import time
import logging
from pathlib import Path

logger = logging.getLogger("DualVisionAI.detector")

COCO_PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]


def class_color(class_id: int) -> tuple:
    return COCO_PALETTE[class_id % len(COCO_PALETTE)]


class DetectionResult:
    __slots__ = ("boxes", "class_ids", "confidences", "class_names",
                 "track_ids", "inference_ms")

    def __init__(self):
        self.boxes: list = []
        self.class_ids: list = []
        self.confidences: list = []
        self.class_names: list = []
        self.track_ids: list = []
        self.inference_ms: float = 0.0


class Detector:
    def __init__(self, model_path: str, conf: float = 0.45,
                 iou: float = 0.45, use_gpu: bool = True,
                 input_size: int = 640, frame_skip: int = 1):
        self.model_path = Path(model_path)
        self.conf = conf
        self.iou = iou
        self.use_gpu = use_gpu
        self.input_size = input_size
        self.frame_skip = max(1, frame_skip)

        self._model = None
        self._class_names: list[str] = []
        self._loaded = False
        self._lock = threading.Lock()

        self._rgb_queue: queue.Queue = queue.Queue(maxsize=2)
        self._thermal_queue: queue.Queue = queue.Queue(maxsize=2)
        self._rgb_result: DetectionResult | None = None
        self._thermal_result: DetectionResult | None = None
        self._result_lock = threading.Lock()

        self._running = False
        self._paused = False
        self._thread: threading.Thread | None = None
        self._skip_counter = 0

        self.fps_inference = 0.0
        self._inf_count = 0
        self._inf_timer = time.time()

    def load(self) -> bool:
        try:
            logger.info(f"Loading model: {self.model_path}")
            from ultralytics import YOLO
            device = "0" if self.use_gpu else "cpu"
            self._model = YOLO(str(self.model_path))
            self._model.to(device if self.use_gpu else "cpu")
            self._class_names = list(self._model.names.values())
            self._loaded = True
            logger.info(f"Model loaded: {len(self._class_names)} classes, device={device}")
            return True
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            self._loaded = False
            return False

    def start(self):
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        if self._running:
            return
        self._running = True
        self._paused = False
        self._thread = threading.Thread(target=self._inference_loop,
                                        daemon=True, name="Detector")
        self._thread.start()
        logger.info("Detector thread started.")

    def stop(self):
        self._running = False
        for q in (self._rgb_queue, self._thermal_queue):
            try:
                q.put_nowait(None)
            except queue.Full:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def push_rgb(self, frame):
        self._push(self._rgb_queue, frame)

    def push_thermal(self, frame):
        self._push(self._thermal_queue, frame)

    def _push(self, q: queue.Queue, frame):
        if frame is None:
            return
        if q.full():
            try:
                q.get_nowait()
            except queue.Empty:
                pass
        try:
            q.put_nowait(frame)
        except queue.Full:
            pass

    def get_rgb_result(self) -> DetectionResult | None:
        with self._result_lock:
            return self._rgb_result

    def get_thermal_result(self) -> DetectionResult | None:
        with self._result_lock:
            return self._thermal_result

    def update_params(self, conf: float = None, iou: float = None,
                      frame_skip: int = None, input_size: int = None):
        if conf is not None:
            self.conf = conf
        if iou is not None:
            self.iou = iou
        if frame_skip is not None:
            self.frame_skip = max(1, frame_skip)
        if input_size is not None:
            self.input_size = input_size

    @property
    def class_names(self) -> list[str]:
        return self._class_names

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _inference_loop(self):
        while self._running:
            if self._paused:
                time.sleep(0.05)
                continue

            rgb_frame = None
            thermal_frame = None

            try:
                rgb_frame = self._rgb_queue.get(timeout=0.1)
            except queue.Empty:
                pass

            try:
                thermal_frame = self._thermal_queue.get_nowait()
            except queue.Empty:
                pass

            if rgb_frame is None and thermal_frame is None:
                continue

            self._skip_counter += 1
            if self._skip_counter < self.frame_skip:
                continue
            self._skip_counter = 0

            t0 = time.time()
            results = {}

            frames_to_run = {}
            if rgb_frame is not None:
                frames_to_run["rgb"] = rgb_frame
            if thermal_frame is not None:
                frames_to_run["thermal"] = thermal_frame

            for key, frame in frames_to_run.items():
                res = self._run_inference(frame)
                results[key] = res

            inf_ms = (time.time() - t0) * 1000
            self._update_fps()

            with self._result_lock:
                if "rgb" in results:
                    results["rgb"].inference_ms = inf_ms
                    self._rgb_result = results["rgb"]
                if "thermal" in results:
                    results["thermal"].inference_ms = inf_ms
                    self._thermal_result = results["thermal"]

    def _run_inference(self, frame) -> DetectionResult:
        result = DetectionResult()
        if frame is None or self._model is None:
            return result

        try:
            resized = cv2.resize(frame, (self.input_size, self.input_size))
            preds = self._model.predict(
                resized,
                conf=self.conf,
                iou=self.iou,
                verbose=False,
                stream=False
            )
            if not preds:
                return result

            pred = preds[0]
            if pred.boxes is None or len(pred.boxes) == 0:
                return result

            h_orig, w_orig = frame.shape[:2]
            sx = w_orig / self.input_size
            sy = h_orig / self.input_size

            for box in pred.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(float)
                x1 = float(xyxy[0] * sx)
                y1 = float(xyxy[1] * sy)
                x2 = float(xyxy[2] * sx)
                y2 = float(xyxy[3] * sy)

                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = self._class_names[cls_id] if cls_id < len(self._class_names) else str(cls_id)

                result.boxes.append([x1, y1, x2, y2])
                result.class_ids.append(cls_id)
                result.confidences.append(conf)
                result.class_names.append(name)
                result.track_ids.append(0)
        except Exception as e:
            logger.error(f"Inference error: {e}")

        return result

    def _update_fps(self):
        self._inf_count += 1
        now = time.time()
        elapsed = now - self._inf_timer
        if elapsed >= 1.0:
            self.fps_inference = self._inf_count / elapsed
            self._inf_count = 0
            self._inf_timer = now


def draw_detections(frame, result: DetectionResult,
                    draw_tracking: bool = True) -> np.ndarray:
    if frame is None or result is None:
        return frame
    out = frame.copy()
    for i, box in enumerate(result.boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        cls_id = result.class_ids[i] if i < len(result.class_ids) else 0
        conf = result.confidences[i] if i < len(result.confidences) else 0.0
        name = result.class_names[i] if i < len(result.class_names) else ""
        tid = result.track_ids[i] if i < len(result.track_ids) else 0

        color = class_color(cls_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label_parts = [f"{name} {conf:.2f}"]
        if draw_tracking and tid > 0:
            label_parts.append(f"ID:{tid}")
        label = " | ".join(label_parts)

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        by1 = max(y1 - th - baseline - 4, 0)
        cv2.rectangle(out, (x1, by1), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return out
