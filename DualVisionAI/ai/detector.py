"""
AI Detector — threaded YOLO inference, GPU-first with CPU fallback.
Every result carries a timestamp; stale results suppress bounding boxes.
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
    (255, 56,  56),  (255,157,151),  (255,112, 31),  (255,178, 29),
    (207,210, 49),   ( 72,249, 10),  (146,204, 23),  ( 61,219,134),
    ( 26,147, 52),   (  0,212,187),  ( 44,153,168),  (  0,194,255),
    ( 52, 69,147),   (100,115,255),  (  0, 24,236),  (132, 56,255),
    ( 82,  0,133),   (203, 56,255),  (255,149,200),  (255, 55,199),
]


def class_color(cls: int) -> tuple:
    return COCO_PALETTE[cls % len(COCO_PALETTE)]


# ------------------------------------------------------------------
# GPU detection
# ------------------------------------------------------------------
def _pick_device(prefer_gpu: bool) -> str:
    if not prefer_gpu:
        logger.info("GPU disabled by user — using CPU.")
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory // (1024**2)
            logger.info(f"GPU selected: {name}  ({vram} MB VRAM)")
            return "0"                       # Ultralytics expects "0", not "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Apple MPS selected.")
            return "mps"
        logger.warning("PyTorch installed but no CUDA GPU found — using CPU.")
        logger.warning("Install CUDA PyTorch:  pip install torch --index-url "
                       "https://download.pytorch.org/whl/cu121")
    except ImportError:
        logger.warning("PyTorch not installed — GPU unavailable.  "
                       "Install: pip install torch --index-url "
                       "https://download.pytorch.org/whl/cu121")
    return "cpu"


# ------------------------------------------------------------------
# Result object
# ------------------------------------------------------------------
class DetectionResult:
    __slots__ = ("boxes", "class_ids", "confidences", "class_names",
                 "track_ids", "inference_ms", "timestamp")

    def __init__(self):
        self.boxes: list        = []
        self.class_ids: list    = []
        self.confidences: list  = []
        self.class_names: list  = []
        self.track_ids: list    = []
        self.inference_ms: float = 0.0
        self.timestamp: float   = time.time()

    def is_fresh(self, max_age: float = 0.8) -> bool:
        return (time.time() - self.timestamp) < max_age

    def is_empty(self) -> bool:
        return len(self.boxes) == 0


# ------------------------------------------------------------------
# Detector
# ------------------------------------------------------------------
class Detector:
    def __init__(self, model_path: str, conf: float = 0.45,
                 iou: float = 0.45, use_gpu: bool = True,
                 input_size: int = 640, frame_skip: int = 1):
        self.model_path  = Path(model_path)
        self.model_name  = self.model_path.name
        self.conf        = conf
        self.iou         = iou
        self.use_gpu     = use_gpu
        self.input_size  = input_size
        self.frame_skip  = max(1, frame_skip)
        self.device      = "cpu"          # resolved in load()

        self._model         = None
        self._class_names:list[str] = []
        self._loaded        = False

        # Queue size=1 → inference always works on the LATEST frame
        self._rgb_q:     queue.Queue = queue.Queue(maxsize=1)
        self._thermal_q: queue.Queue = queue.Queue(maxsize=1)
        self._rgb_result:     DetectionResult | None = None
        self._thermal_result: DetectionResult | None = None
        self._result_lock = threading.Lock()

        self._running = False
        self._paused  = False
        self._thread: threading.Thread | None = None
        self._skip    = 0

        self.fps_inference = 0.0
        self._inf_n    = 0
        self._inf_t0   = time.time()

    # ---------- public API ----------
    def load(self) -> bool:
        try:
            from ultralytics import YOLO

            self.device = _pick_device(self.use_gpu)
            logger.info(f"Loading {self.model_name} → device={self.device}")

            if self.model_path.exists() and self.model_path.stat().st_size > 100_000:
                self._model = YOLO(str(self.model_path))
                logger.info(f"Loaded from disk: {self.model_path}")
            else:
                logger.info("Auto-downloading via Ultralytics …")
                self._model = YOLO(self.model_name)

            # Move to GPU (if applicable) — use Ultralytics' native device arg
            # during predict() instead of .to() which can cause dtype issues
            self._class_names = list(self._model.names.values())

            # Warm-up — first inference is always slow
            dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
            self._model.predict(dummy, verbose=False,
                                conf=self.conf, iou=self.iou,
                                device=self.device)

            self._loaded = True
            logger.info(f"Model ready — {len(self._class_names)} classes  "
                        f"device={self.device}")
            return True
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            self._loaded = False
            return False

    def start(self):
        if not self._loaded:
            raise RuntimeError("Model not loaded — call load() first.")
        if self._running:
            return
        self._running = True
        self._paused  = False
        self._thread  = threading.Thread(target=self._loop,
                                         daemon=True, name="InferenceThread")
        self._thread.start()
        logger.info("Inference thread started.")

    def stop(self):
        self._running = False
        for q in (self._rgb_q, self._thermal_q):
            try:
                q.put_nowait(None)
            except queue.Full:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=4)

    def pause(self):  self._paused = True
    def resume(self): self._paused = False

    def push_rgb(self, frame):     self._push(self._rgb_q,     frame)
    def push_thermal(self, frame): self._push(self._thermal_q, frame)

    def _push(self, q: queue.Queue, frame):
        if frame is None:
            return
        try:
            q.get_nowait()          # discard old frame
        except queue.Empty:
            pass
        try:
            q.put_nowait(frame)
        except queue.Full:
            pass

    def get_rgb_result(self)     -> DetectionResult | None:
        with self._result_lock: return self._rgb_result
    def get_thermal_result(self) -> DetectionResult | None:
        with self._result_lock: return self._thermal_result

    def clear_results(self):
        with self._result_lock:
            self._rgb_result     = None
            self._thermal_result = None

    def update_params(self, conf=None, iou=None, frame_skip=None, input_size=None):
        if conf is not None:        self.conf = conf
        if iou  is not None:        self.iou  = iou
        if frame_skip is not None:  self.frame_skip = max(1, frame_skip)
        if input_size is not None:  self.input_size = input_size

    @property
    def class_names(self) -> list[str]: return self._class_names
    @property
    def is_loaded(self)   -> bool:      return self._loaded

    # ---------- inference thread ----------
    def _loop(self):
        while self._running:
            if self._paused:
                time.sleep(0.05)
                continue

            rgb = thermal = None
            try:
                rgb = self._rgb_q.get(timeout=0.05)
            except queue.Empty:
                pass
            try:
                thermal = self._thermal_q.get_nowait()
            except queue.Empty:
                pass

            if rgb is None and thermal is None:
                continue

            self._skip += 1
            if self._skip < self.frame_skip:
                continue
            self._skip = 0

            t0 = time.time()
            rgb_res     = self._infer(rgb)     if rgb     is not None else None
            thermal_res = self._infer(thermal) if thermal is not None else None
            inf_ms = (time.time() - t0) * 1000.0

            with self._result_lock:
                if rgb_res is not None:
                    rgb_res.inference_ms     = inf_ms
                    self._rgb_result         = rgb_res
                if thermal_res is not None:
                    thermal_res.inference_ms = inf_ms
                    self._thermal_result     = thermal_res

            self._tick_fps()

    def _infer(self, frame) -> DetectionResult:
        res = DetectionResult()
        if frame is None or self._model is None:
            return res
        try:
            h0, w0 = frame.shape[:2]
            small  = cv2.resize(frame, (self.input_size, self.input_size))
            preds  = self._model.predict(
                small, conf=self.conf, iou=self.iou,
                verbose=False, device=self.device
            )
            res.timestamp = time.time()

            if not preds or preds[0].boxes is None or len(preds[0].boxes) == 0:
                return res          # empty result, fresh timestamp → clears boxes

            sx, sy = w0 / self.input_size, h0 / self.input_size
            for box in preds[0].boxes:
                x = box.xyxy[0].cpu().numpy().astype(float)
                res.boxes.append([x[0]*sx, x[1]*sy, x[2]*sx, x[3]*sy])
                cls = int(box.cls[0])
                res.class_ids.append(cls)
                res.confidences.append(float(box.conf[0]))
                res.class_names.append(
                    self._class_names[cls] if cls < len(self._class_names) else str(cls))
                res.track_ids.append(0)
        except Exception as e:
            logger.error(f"Inference error: {e}")
            res.timestamp = time.time()
        return res

    def _tick_fps(self):
        self._inf_n += 1
        now = time.time()
        if (now - self._inf_t0) >= 1.0:
            self.fps_inference = self._inf_n / (now - self._inf_t0)
            self._inf_n  = 0
            self._inf_t0 = now


# ------------------------------------------------------------------
# Drawing helper — no stale boxes, no ghost boxes
# ------------------------------------------------------------------
def draw_detections(frame, result: DetectionResult | None) -> np.ndarray:
    """Draw boxes from result onto frame.
    Pass result=None to get a clean frame (no boxes).
    Freshness / ghost-box logic is handled by the caller (main_window worker).
    """
    if frame is None:
        return frame
    if result is None or result.is_empty():
        return frame          # ← clean frame: object gone or not yet detected
    out = frame.copy()
    for i, box in enumerate(result.boxes):
        x1, y1, x2, y2 = (int(v) for v in box)
        cls  = result.class_ids[i]  if i < len(result.class_ids)  else 0
        conf = result.confidences[i] if i < len(result.confidences) else 0.0
        name = result.class_names[i] if i < len(result.class_names) else ""
        tid  = result.track_ids[i]  if i < len(result.track_ids)  else 0

        color = class_color(cls)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = f"{name} {conf:.2f}" + (f" | ID:{tid}" if tid > 0 else "")
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        by = max(y1 - th - bl - 4, 0)
        cv2.rectangle(out, (x1, by), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - bl - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return out
