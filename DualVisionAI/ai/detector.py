"""
AI Detector — YOLO26 inference engine with GPU/CUDA support via BackendManager.

Backend priority (automatic):
  1. ONNX Runtime + CUDAExecutionProvider  — RTX / GPU, fastest
  2. PyTorch CUDA                           — if ORT CUDA unavailable
  3. ONNX Runtime + CPUExecutionProvider   — CPU ONNX
  4. PyTorch CPU                            — last resort

Architecture:
  • BackendManager selects and configures the inference backend at load time.
  • Two independent inference threads (RGB + Thermal) run concurrently.
  • ONNX InferenceSession.run() is thread-safe; PT predict() uses a lock.
  • FP16 preprocessing when use_fp16=True and CUDA is active.
  • Per-phase timing: preprocess / inference / postprocess exposed as properties.
"""

import cv2
import numpy as np
import threading
import queue
import time
import os
import logging
from pathlib import Path

from ai.backend_manager import (
    BackendManager,
    BACKEND_ONNX_CUDA, BACKEND_ONNX_CPU,
    BACKEND_PT_CUDA,   BACKEND_PT_CPU,
)

logger = logging.getLogger("DualVisionAI.detector")

# ── Colour palette (COCO 80-class) ────────────────────────────────────────────
COCO_PALETTE = [
    (255, 56,  56), (255,157,151), (255,112, 31), (255,178, 29),
    (207,210, 49),  ( 72,249, 10), (146,204, 23), ( 61,219,134),
    ( 26,147, 52),  (  0,212,187), ( 44,153,168), (  0,194,255),
    ( 52, 69,147),  (100,115,255), (  0, 24,236), (132, 56,255),
    ( 82,  0,133),  (203, 56,255), (255,149,200), (255, 55,199),
]

def class_color(cls: int) -> tuple:
    return COCO_PALETTE[cls % len(COCO_PALETTE)]


# ── Result container ──────────────────────────────────────────────────────────
class DetectionResult:
    __slots__ = ("boxes", "class_ids", "confidences", "class_names",
                 "track_ids", "inference_ms", "timestamp",
                 "preprocess_ms", "postprocess_ms")

    def __init__(self):
        self.boxes:          list  = []
        self.class_ids:      list  = []
        self.confidences:    list  = []
        self.class_names:    list  = []
        self.track_ids:      list  = []
        self.inference_ms:   float = 0.0
        self.preprocess_ms:  float = 0.0
        self.postprocess_ms: float = 0.0
        self.timestamp:      float = time.time()

    def is_fresh(self, max_age: float = 0.8) -> bool:
        return (time.time() - self.timestamp) < max_age

    def is_empty(self) -> bool:
        return len(self.boxes) == 0


# ── Detector ──────────────────────────────────────────────────────────────────
class Detector:
    """
    YOLO26 inference engine — GPU-first via BackendManager.

    Pass a BackendManager instance (or None to auto-create one).
    """

    def __init__(
        self,
        model_path:      str,
        conf:            float = 0.45,
        iou:             float = 0.45,
        use_gpu:         bool  = True,
        input_size:      int   = 640,
        frame_skip:      int   = 1,
        cpu_threads:     int   = 0,
        use_fp16:        bool  = False,
        onnx_path:       str   = "",
        backend_manager: BackendManager | None = None,
    ):
        self.model_path  = Path(model_path)
        self.model_name  = self.model_path.name
        self.conf        = conf
        self.iou         = iou
        self.use_gpu     = use_gpu
        self.input_size  = input_size
        self.frame_skip  = max(1, frame_skip)
        self.cpu_threads = cpu_threads
        self.use_fp16    = use_fp16
        self.onnx_path   = Path(onnx_path) if onnx_path else None

        # Use provided BackendManager or create one
        if backend_manager is not None:
            self._bm = backend_manager
        else:
            self._bm = BackendManager(
                use_gpu=use_gpu, use_fp16=use_fp16)

        self.device     = self._bm.device
        self.backend    = self._bm.backend

        self._model       = None          # ultralytics YOLO (PT fallback)
        self._ort_session = None          # ONNX Runtime session
        self._ort_input   = ""
        self._class_names: list[str] = []
        self._loaded      = False
        self.onnx_active  = False

        # Frame queues (maxsize=1 → always latest frame)
        self._rgb_q:     queue.Queue = queue.Queue(maxsize=1)
        self._thermal_q: queue.Queue = queue.Queue(maxsize=1)

        # Results
        self._rgb_result:     DetectionResult | None = None
        self._thermal_result: DetectionResult | None = None
        self._result_lock = threading.Lock()
        self._infer_lock  = threading.Lock()   # for non-thread-safe PT predict()

        self._running = False
        self._paused  = False
        self._rgb_thread:     threading.Thread | None = None
        self._thermal_thread: threading.Thread | None = None

        # Performance counters
        self.fps_inference = 0.0
        self.fps_rgb       = 0.0
        self.fps_thermal   = 0.0
        self.avg_fps       = 0.0
        self.frame_drops   = 0

        # Per-phase timing (ms) — averaged over last few inferences
        self.preprocess_ms:  float = 0.0
        self.infer_ms:       float = 0.0
        self.postprocess_ms: float = 0.0

        self._fps_history: list[float] = []

        if cpu_threads > 0:
            os.environ["OMP_NUM_THREADS"]      = str(cpu_threads)
            os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_threads)
            cv2.setNumThreads(cpu_threads)

    # ── public API ─────────────────────────────────────────────────────────────
    def load(self) -> bool:
        try:
            logger.info(f"Loading YOLO26 — backend={self.backend}  "
                        f"device={self.device}")

            backend = self._bm.backend

            # ── ONNX path ─────────────────────────────────────────────────────
            if self.onnx_path and self.onnx_path.exists() \
                    and self.onnx_path.stat().st_size > 100_000:
                try:
                    self._ort_session = self._bm.build_ort_session(
                        str(self.onnx_path), self.cpu_threads)
                    self._setup_ort_meta()
                    self.onnx_active = True
                    self.device      = self._bm.device
                    self.backend     = self._bm.backend
                    logger.info(f"ONNX Runtime active: {self.onnx_path.name}  "
                                f"provider={self._bm.ort_active_provider}")
                except Exception as e:
                    logger.warning(f"ONNX load failed ({e}) — falling back to .pt")
                    self._ort_session = None

            # ── PyTorch fallback ──────────────────────────────────────────────
            if not self.onnx_active:
                from ultralytics import YOLO
                pt = self.model_path
                self._model = YOLO(str(pt) if pt.exists() and
                                   pt.stat().st_size > 100_000 else self.model_name)
                self._class_names = list(self._model.names.values())
                # Warm-up on the correct device
                dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
                self._model.predict(dummy, verbose=False, conf=self.conf,
                                    iou=self.iou, device=self.device)
                self.onnx_active = False
                self.backend     = (BACKEND_PT_CUDA if self.device != "cpu"
                                    else BACKEND_PT_CPU)

            if not self._class_names:
                self._class_names = [str(i) for i in range(80)]

            self._loaded = True
            logger.info(f"Model ready — {len(self._class_names)} classes  "
                        f"backend={self.backend}  device={self.device}  "
                        f"onnx={self.onnx_active}  fp16={self.use_fp16}")
            return True

        except Exception as e:
            logger.error(f"Model load failed: {e}")
            self._loaded = False
            return False

    def _setup_ort_meta(self):
        """Extract class names and input node name from ORT session."""
        try:
            meta = self._ort_session.get_modelmeta()
            names_str = (meta.custom_metadata_map or {}).get("names", "")
            if names_str:
                import ast
                self._class_names = list(ast.literal_eval(names_str).values())
        except Exception:
            pass
        self._ort_input = self._ort_session.get_inputs()[0].name

    def start(self):
        if not self._loaded:
            raise RuntimeError("Model not loaded — call load() first.")
        if self._running:
            return
        self._running = True
        self._paused  = False

        self._rgb_thread = threading.Thread(
            target=self._inference_loop, args=(self._rgb_q, "rgb"),
            daemon=True, name="Infer-RGB")
        self._thermal_thread = threading.Thread(
            target=self._inference_loop, args=(self._thermal_q, "thermal"),
            daemon=True, name="Infer-Thermal")
        self._rgb_thread.start()
        self._thermal_thread.start()
        logger.info("Inference threads started.")

    def stop(self):
        self._running = False
        for q in (self._rgb_q, self._thermal_q):
            try: q.put_nowait(None)
            except queue.Full: pass
        for t in (self._rgb_thread, self._thermal_thread):
            if t and t.is_alive():
                t.join(timeout=4)

    def pause(self):  self._paused = True
    def resume(self): self._paused = False

    def push_rgb(self, frame):     self._push(self._rgb_q, frame)
    def push_thermal(self, frame): self._push(self._thermal_q, frame)

    def _push(self, q: queue.Queue, frame):
        if frame is None:
            return
        try: q.get_nowait()
        except queue.Empty: pass
        try: q.put_nowait(frame)
        except queue.Full: self.frame_drops += 1

    def get_rgb_result(self)     -> DetectionResult | None:
        with self._result_lock: return self._rgb_result

    def get_thermal_result(self) -> DetectionResult | None:
        with self._result_lock: return self._thermal_result

    def clear_results(self):
        with self._result_lock:
            self._rgb_result     = None
            self._thermal_result = None

    def update_params(self, conf=None, iou=None, frame_skip=None,
                      input_size=None, cpu_threads=None):
        if conf       is not None: self.conf       = conf
        if iou        is not None: self.iou        = iou
        if frame_skip is not None: self.frame_skip = max(1, frame_skip)
        if input_size is not None: self.input_size = input_size
        if cpu_threads is not None:
            self.cpu_threads = cpu_threads
            if cpu_threads > 0:
                os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
                cv2.setNumThreads(cpu_threads)

    @property
    def class_names(self) -> list[str]: return self._class_names
    @property
    def is_loaded(self)   -> bool:      return self._loaded
    @property
    def queue_size(self)  -> int:
        return self._rgb_q.qsize() + self._thermal_q.qsize()
    @property
    def active_threads(self) -> int:
        return sum(1 for t in (self._rgb_thread, self._thermal_thread)
                   if t and t.is_alive())
    @property
    def ort_provider(self) -> str:
        return self._bm.ort_active_provider

    # ── inference loop (per stream) ────────────────────────────────────────────
    def _inference_loop(self, q: queue.Queue, stream: str):
        skip_counter = 0
        inf_n  = 0
        inf_t0 = time.time()

        while self._running:
            if self._paused:
                time.sleep(0.05)
                continue
            try:
                frame = q.get(timeout=0.05)
            except queue.Empty:
                continue
            if frame is None:
                break

            skip_counter += 1
            if skip_counter < self.frame_skip:
                continue
            skip_counter = 0

            t0  = time.perf_counter()
            res = (self._infer_onnx(frame) if self.onnx_active
                   else self._infer_pt(frame))
            res.inference_ms = (time.perf_counter() - t0) * 1000.0

            # Running average of phase timings
            a = 0.2
            self.preprocess_ms  = (1 - a) * self.preprocess_ms  + a * res.preprocess_ms
            self.infer_ms       = (1 - a) * self.infer_ms       + a * res.inference_ms
            self.postprocess_ms = (1 - a) * self.postprocess_ms + a * res.postprocess_ms

            with self._result_lock:
                if stream == "rgb":
                    self._rgb_result = res
                else:
                    self._thermal_result = res

            # FPS
            inf_n += 1
            now     = time.time()
            elapsed = now - inf_t0
            if elapsed >= 1.0:
                fps = inf_n / elapsed
                if stream == "rgb":
                    self.fps_rgb = fps
                else:
                    self.fps_thermal = fps
                self.fps_inference = (self.fps_rgb + self.fps_thermal) / 2
                self._update_avg_fps(fps)
                inf_n  = 0
                inf_t0 = now

    # ── ONNX inference (GPU or CPU) ────────────────────────────────────────────
    def _infer_onnx(self, frame) -> DetectionResult:
        res = DetectionResult()
        if frame is None or self._ort_session is None:
            return res
        try:
            h0, w0 = frame.shape[:2]
            sz     = self.input_size

            # ── Pre-process ───────────────────────────────────────────────────
            t_pre = time.perf_counter()
            blob  = cv2.resize(frame, (sz, sz))
            blob  = blob[:, :, ::-1].astype(np.float32) / 255.0   # BGR→RGB /255
            blob  = blob.transpose(2, 0, 1)[np.newaxis]            # NCHW

            if self.use_fp16 and self._bm.cuda_available:
                blob = blob.astype(np.float16)

            res.preprocess_ms = (time.perf_counter() - t_pre) * 1000.0

            # ── Inference ─────────────────────────────────────────────────────
            t_inf   = time.perf_counter()
            outputs = self._ort_session.run(None, {self._ort_input: blob})
            res.timestamp    = time.time()
            raw_inf_ms       = (time.perf_counter() - t_inf) * 1000.0

            # ── Post-process ──────────────────────────────────────────────────
            t_post = time.perf_counter()
            pred   = outputs[0][0]   # (4+nc, num_det) or (num_det, 4+nc)
            if pred.shape[0] < pred.shape[1]:
                pass
            else:
                pred = pred.T

            boxes_xywh = pred[:4].T
            scores     = pred[4:].T.astype(np.float32)

            best_cls  = scores.argmax(1)
            best_conf = scores.max(1)
            mask      = best_conf >= self.conf
            if not mask.any():
                res.postprocess_ms = (time.perf_counter() - t_post) * 1000.0
                return res

            boxes_f = boxes_xywh[mask]
            confs_f = best_conf[mask].astype(float)
            cls_f   = best_cls[mask]

            sx, sy = w0 / sz, h0 / sz
            for b, c, cl in zip(boxes_f, confs_f, cls_f):
                cx, cy, bw, bh = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                x1 = (cx - bw / 2) * sz * sx
                y1 = (cy - bh / 2) * sz * sy
                x2 = (cx + bw / 2) * sz * sx
                y2 = (cy + bh / 2) * sz * sy
                ci = int(cl)
                res.boxes.append([x1, y1, x2, y2])
                res.class_ids.append(ci)
                res.confidences.append(float(c))
                res.class_names.append(
                    self._class_names[ci] if ci < len(self._class_names) else str(ci))
                res.track_ids.append(0)

            # NMS
            if len(res.boxes) > 1:
                rects   = [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in res.boxes]
                indices = cv2.dnn.NMSBoxes(rects, res.confidences,
                                           self.conf, self.iou)
                if indices is not None and len(indices):
                    idx = [int(i) for i in indices.flatten()]
                    res.boxes       = [res.boxes[i]       for i in idx]
                    res.class_ids   = [res.class_ids[i]   for i in idx]
                    res.confidences = [res.confidences[i] for i in idx]
                    res.class_names = [res.class_names[i] for i in idx]
                    res.track_ids   = [res.track_ids[i]   for i in idx]

            res.postprocess_ms = (time.perf_counter() - t_post) * 1000.0
            res.inference_ms   = raw_inf_ms

        except Exception as e:
            logger.error(f"ONNX inference error: {e}")
            res.timestamp = time.time()
        return res

    # ── PyTorch / ultralytics fallback ─────────────────────────────────────────
    def _infer_pt(self, frame) -> DetectionResult:
        res = DetectionResult()
        if frame is None or self._model is None:
            return res
        try:
            h0, w0 = frame.shape[:2]

            t_pre  = time.perf_counter()
            small  = cv2.resize(frame, (self.input_size, self.input_size))
            res.preprocess_ms = (time.perf_counter() - t_pre) * 1000.0

            t_inf  = time.perf_counter()
            with self._infer_lock:
                preds = self._model.predict(
                    small, conf=self.conf, iou=self.iou,
                    verbose=False, device=self.device,
                    half=self.use_fp16 and self.device != "cpu")
            res.timestamp    = time.time()
            res.inference_ms = (time.perf_counter() - t_inf) * 1000.0

            t_post = time.perf_counter()
            if preds and preds[0].boxes is not None and len(preds[0].boxes):
                sx, sy = w0 / self.input_size, h0 / self.input_size
                for box in preds[0].boxes:
                    x = box.xyxy[0].cpu().numpy().astype(float)
                    cls = int(box.cls[0])
                    res.boxes.append([x[0]*sx, x[1]*sy, x[2]*sx, x[3]*sy])
                    res.class_ids.append(cls)
                    res.confidences.append(float(box.conf[0]))
                    res.class_names.append(
                        self._class_names[cls]
                        if cls < len(self._class_names) else str(cls))
                    res.track_ids.append(0)
            res.postprocess_ms = (time.perf_counter() - t_post) * 1000.0

        except Exception as e:
            logger.error(f"PT inference error: {e}")
            res.timestamp = time.time()
        return res

    def _update_avg_fps(self, fps: float):
        self._fps_history.append(fps)
        if len(self._fps_history) > 10:
            self._fps_history.pop(0)
        self.avg_fps = sum(self._fps_history) / len(self._fps_history)


# ── Drawing helper ─────────────────────────────────────────────────────────────
def draw_detections(frame, result: DetectionResult | None) -> np.ndarray:
    if frame is None:
        return frame
    if result is None or result.is_empty():
        return frame
    out = frame.copy()
    for i, box in enumerate(result.boxes):
        x1, y1, x2, y2 = (int(v) for v in box)
        cls  = result.class_ids[i]   if i < len(result.class_ids)   else 0
        conf = result.confidences[i] if i < len(result.confidences) else 0.0
        name = result.class_names[i] if i < len(result.class_names) else ""
        tid  = result.track_ids[i]   if i < len(result.track_ids)   else 0
        color = class_color(cls)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {conf:.2f}" + (f" | ID:{tid}" if tid > 0 else "")
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        by = max(y1 - th - bl - 4, 0)
        cv2.rectangle(out, (x1, by), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - bl - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return out
