"""
AI Detector — DualVision AI v1.3 Stable CPU Edition.

Inference pipeline (always CPU):
  Camera → Frame Queue (size=1) → Resize → ONNX Runtime (CPUExecutionProvider)
         → Post-process → NMS → Bounding Boxes → Tracking → Overlay → Display

Rules:
  • ONNX Runtime CPUExecutionProvider ONLY.
  • PyTorch is NEVER used for inference.
  • Queue size = 1 — always infer the latest frame, never old ones.
  • Two independent inference threads: RGB and Thermal.
  • Per-phase timing: preprocess / inference / postprocess exposed as properties.
  • No silent failure — every exception is logged with full traceback.
  • Numpy array reuse to minimise per-frame allocation.
"""

import cv2
import numpy as np
import threading
import queue
import time
import logging
import traceback
from pathlib import Path

from ai.backend_manager import BackendManager

logger    = logging.getLogger("DualVisionAI.detector")
inf_logger = logging.getLogger("DualVisionAI.detector.inference")

# ── COCO colour palette ────────────────────────────────────────────────────────
_PALETTE = [
    (255, 56,  56), (255,157,151), (255,112, 31), (255,178, 29),
    (207,210, 49),  ( 72,249, 10), (146,204, 23), ( 61,219,134),
    ( 26,147, 52),  (  0,212,187), ( 44,153,168), (  0,194,255),
    ( 52, 69,147),  (100,115,255), (  0, 24,236), (132, 56,255),
    ( 82,  0,133),  (203, 56,255), (255,149,200), (255, 55,199),
]

def _class_color(cls: int) -> tuple:
    return _PALETTE[cls % len(_PALETTE)]


# ── Result ────────────────────────────────────────────────────────────────────
class DetectionResult:
    __slots__ = ("boxes", "class_ids", "confidences", "class_names",
                 "track_ids", "inference_ms", "preprocess_ms",
                 "postprocess_ms", "timestamp")

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

    def is_empty(self) -> bool:
        return len(self.boxes) == 0

    def is_fresh(self, max_age: float = 0.8) -> bool:
        return (time.time() - self.timestamp) < max_age


# ── Detector ──────────────────────────────────────────────────────────────────
class Detector:
    """
    YOLO26n inference engine — ONNX Runtime CPU only.

    Requires a BackendManager built with CPUExecutionProvider.
    Load once, run indefinitely. Thread-safe.
    """

    def __init__(
        self,
        onnx_path:       str,
        conf:            float = 0.45,
        iou:             float = 0.45,
        input_size:      int   = 640,
        frame_skip:      int   = 1,
        backend_manager: BackendManager | None = None,
    ):
        self._onnx_path  = Path(onnx_path)
        self.conf        = conf
        self.iou         = iou
        self.input_size  = input_size
        self.frame_skip  = max(1, frame_skip)

        if backend_manager is None:
            backend_manager = BackendManager()
        self._bm = backend_manager

        self._session     = None
        self._input_name  = ""
        self._class_names: list[str] = []
        self._loaded      = False

        # Queue size = 1: always process the latest frame
        self._rgb_q:     queue.Queue = queue.Queue(maxsize=1)
        self._thermal_q: queue.Queue = queue.Queue(maxsize=1)

        self._rgb_result:     DetectionResult | None = None
        self._thermal_result: DetectionResult | None = None
        self._result_lock = threading.Lock()

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

        # Smoothed per-phase timing (ms)
        self.preprocess_ms:  float = 0.0
        self.infer_ms:       float = 0.0
        self.postprocess_ms: float = 0.0

        self._fps_history: list[float] = []

        # Pre-allocated input buffer for memory reuse (NCHW float32)
        self._blob: np.ndarray | None = None

        # Inference log interval
        self._inf_log_counter = 0
        self._INF_LOG_EVERY   = 100   # log every N inferences

        # Diagnostic counters (reset on each load)
        self._diag_count      = 0   # counts how many times we've logged raw output
        self._debug_img_saved = False

    # ── Load ──────────────────────────────────────────────────────────────────
    def load(self) -> bool:
        """
        Load ONNX model.  Raises on failure — never silent.
        Returns True on success.
        """
        try:
            logger.info(f"Loading ONNX model: {self._onnx_path.name}")
            self._session    = self._bm.build_ort_session(str(self._onnx_path))
            self._input_name = self._session.get_inputs()[0].name
            self._parse_class_names()
            self._log_onnx_io()   # STEP 3: log exact input/output shapes
            self._prewarm()       # STEP 4: warm-up also logs output shape + dtype
            self._loaded      = True
            self._diag_count      = 0
            self._debug_img_saved = False
            logger.info(
                f"Model loaded — input='{self._input_name}'  "
                f"classes={len(self._class_names)}  "
                f"input_size={self.input_size}")
            inf_logger.info(
                f"Model ready: {self._onnx_path.name}  "
                f"classes={len(self._class_names)}  "
                f"provider=CPUExecutionProvider")
            return True
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error(f"Model load failed:\n{tb}")
            inf_logger.error(f"Model load failed: {exc}\n{tb}")
            self._loaded = False
            raise   # caller shows popup with the exact message

    def _parse_class_names(self):
        try:
            meta = self._session.get_modelmeta()
            names_str = (meta.custom_metadata_map or {}).get("names", "")
            if names_str:
                import ast
                self._class_names = list(ast.literal_eval(names_str).values())
                return
        except Exception:
            pass
        # fallback: 80-class COCO
        self._class_names = [str(i) for i in range(80)]

    def _log_onnx_io(self):
        """Log ONNX input/output shapes to startup.log and debug.log."""
        try:
            inp = self._session.get_inputs()[0]
            inf_logger.info(
                f"ONNX INPUT  — name='{inp.name}'  "
                f"shape={inp.shape}  type={inp.type}")
            for i, out in enumerate(self._session.get_outputs()):
                inf_logger.info(
                    f"ONNX OUTPUT[{i}] — name='{out.name}'  "
                    f"shape={out.shape}  type={out.type}")
        except Exception as e:
            logger.warning(f"Could not log ONNX I/O info: {e}")

    def _prewarm(self):
        """One warm-up inference to let ORT JIT its kernels."""
        try:
            sz   = self.input_size
            blob = np.zeros((1, 3, sz, sz), dtype=np.float32)
            out  = self._session.run(None, {self._input_name: blob})
            inf_logger.info(
                f"ORT warm-up complete — "
                f"output[0].shape={out[0].shape}  dtype={out[0].dtype}")
        except Exception as e:
            logger.warning(f"Warm-up failed (non-fatal): {e}")

    # ── Start / Stop ──────────────────────────────────────────────────────────
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
        logger.info("Inference threads started (RGB + Thermal).")

    def stop(self):
        self._running = False
        # Unblock queue.get()
        for q in (self._rgb_q, self._thermal_q):
            try: q.put_nowait(None)
            except queue.Full: pass
        for t in (self._rgb_thread, self._thermal_thread):
            if t and t.is_alive():
                t.join(timeout=5)
        self._rgb_thread     = None
        self._thermal_thread = None
        logger.info("Inference threads stopped.")

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    # ── Frame push ────────────────────────────────────────────────────────────
    def push_rgb(self, frame):     self._push(self._rgb_q, frame)
    def push_thermal(self, frame): self._push(self._thermal_q, frame)

    def _push(self, q: queue.Queue, frame):
        if frame is None:
            return
        # Discard stale frame to keep queue at max 1
        try: q.get_nowait()
        except queue.Empty: pass
        try: q.put_nowait(frame)
        except queue.Full: self.frame_drops += 1

    # ── Results ───────────────────────────────────────────────────────────────
    def get_rgb_result(self)     -> DetectionResult | None:
        with self._result_lock: return self._rgb_result

    def get_thermal_result(self) -> DetectionResult | None:
        with self._result_lock: return self._thermal_result

    def clear_results(self):
        with self._result_lock:
            self._rgb_result     = None
            self._thermal_result = None

    # ── Parameters ────────────────────────────────────────────────────────────
    def update_params(self, conf=None, iou=None, frame_skip=None,
                      input_size=None):
        if conf        is not None: self.conf       = conf
        if iou         is not None: self.iou        = iou
        if frame_skip  is not None: self.frame_skip = max(1, frame_skip)
        if input_size  is not None:
            if input_size != self.input_size:
                self.input_size = input_size
                self._blob = None   # force re-allocate

    # ── Properties ────────────────────────────────────────────────────────────
    @property
    def class_names(self)   -> list[str]: return self._class_names
    @property
    def is_loaded(self)     -> bool:      return self._loaded
    @property
    def queue_size(self)    -> int:
        return self._rgb_q.qsize() + self._thermal_q.qsize()
    @property
    def active_threads(self) -> int:
        return sum(1 for t in (self._rgb_thread, self._thermal_thread)
                   if t and t.is_alive())
    @property
    def onnx_active(self) -> bool:   return self._loaded
    @property
    def backend(self)     -> str:    return self._bm.backend
    @property
    def device(self)      -> str:    return self._bm.device

    # ── Inference loop (per stream) ───────────────────────────────────────────
    def _inference_loop(self, q: queue.Queue, stream: str):
        skip  = 0
        inf_n = 0
        t0    = time.time()

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

            skip += 1
            if skip < self.frame_skip:
                continue
            skip = 0

            try:
                res = self._infer(frame)
            except Exception:
                tb = traceback.format_exc()
                logger.error(f"[{stream}] Inference error:\n{tb}")
                inf_logger.error(f"[{stream}] Inference error:\n{tb}")
                continue

            with self._result_lock:
                if stream == "rgb":
                    self._rgb_result = res
                else:
                    self._thermal_result = res

            # Smoothed timing
            a = 0.15
            self.preprocess_ms  = (1-a)*self.preprocess_ms  + a*res.preprocess_ms
            self.infer_ms       = (1-a)*self.infer_ms       + a*res.inference_ms
            self.postprocess_ms = (1-a)*self.postprocess_ms + a*res.postprocess_ms

            # FPS
            inf_n += 1
            now   = time.time()
            if now - t0 >= 1.0:
                fps = inf_n / (now - t0)
                if stream == "rgb":
                    self.fps_rgb = fps
                else:
                    self.fps_thermal = fps
                self.fps_inference = (self.fps_rgb + self.fps_thermal) / 2
                self._fps_history.append(fps)
                if len(self._fps_history) > 15:
                    self._fps_history.pop(0)
                self.avg_fps = sum(self._fps_history) / len(self._fps_history)
                inf_n = 0
                t0    = now

            # Periodic inference log
            self._inf_log_counter += 1
            if self._inf_log_counter >= self._INF_LOG_EVERY:
                self._inf_log_counter = 0
                inf_logger.info(
                    f"[{stream}] fps={self.fps_inference:.1f}  "
                    f"pre={self.preprocess_ms:.1f}ms  "
                    f"inf={self.infer_ms:.1f}ms  "
                    f"post={self.postprocess_ms:.1f}ms  "
                    f"dets={len(res.boxes)}")

    # ── ONNX CPU Inference ────────────────────────────────────────────────────
    def _infer(self, frame) -> DetectionResult:
        res    = DetectionResult()
        h0, w0 = frame.shape[:2]
        sz     = self.input_size

        # ── Pre-process ───────────────────────────────────────────────────────
        t_pre = time.perf_counter()

        small = cv2.resize(frame, (sz, sz), interpolation=cv2.INTER_LINEAR)
        rgb   = small[:, :, ::-1]   # BGR → RGB in-place view (no copy)

        # Reuse buffer if shape unchanged
        if self._blob is None or self._blob.shape != (1, 3, sz, sz):
            self._blob = np.empty((1, 3, sz, sz), dtype=np.float32)

        # Fill buffer: HWC → NCHW /255
        np.divide(
            rgb.transpose(2, 0, 1).astype(np.float32),
            255.0,
            out=self._blob[0])

        res.preprocess_ms = (time.perf_counter() - t_pre) * 1000.0

        # ── Debug: save preprocessed image once ───────────────────────────────
        if not self._debug_img_saved:
            try:
                import os
                os.makedirs("debug", exist_ok=True)
                dbg = (self._blob[0].transpose(1, 2, 0) * 255).astype(np.uint8)[:, :, ::-1]
                cv2.imwrite("debug/preprocessed.png", dbg)
                inf_logger.info(
                    f"STEP 2 — Saved debug/preprocessed.png  "
                    f"frame_shape={frame.shape}  "
                    f"blob_shape={self._blob.shape}  "
                    f"blob_dtype={self._blob.dtype}  "
                    f"blob_min={self._blob.min():.4f}  "
                    f"blob_max={self._blob.max():.4f}")
                self._debug_img_saved = True
            except Exception as _de:
                logger.warning(f"Debug image save failed (non-fatal): {_de}")

        # ── ONNX inference ────────────────────────────────────────────────────
        t_inf   = time.perf_counter()
        outputs = self._session.run(None, {self._input_name: self._blob})
        res.timestamp    = time.time()
        res.inference_ms = (time.perf_counter() - t_inf) * 1000.0

        # ── Post-process ──────────────────────────────────────────────────────
        t_post = time.perf_counter()
        self._postprocess(outputs, res, w0, h0, sz)
        res.postprocess_ms = (time.perf_counter() - t_post) * 1000.0

        return res

    # ────────────────────────────────────────────────────────────────────────
    # YOLO output-format auto-detector
    # ────────────────────────────────────────────────────────────────────────
    # Three known YOLO ONNX output layouts:
    #
    #  FORMAT A — NMS already applied (e.g. YOLOv10 export)
    #    shape: (1, N≤300, 6)   values: [x1, y1, x2, y2, conf, cls_id]
    #    → decode directly, no NMS needed.
    #
    #  FORMAT B — YOLOv8/v11 style, no objectness
    #    shape: (1, 84, 8400) or transposed   channels: [cx,cy,w,h, cls0..79]
    #    → conf = max(cls_scores), NMS required.
    #
    #  FORMAT C — YOLOv5/YOLOv7/YOLO26 style, WITH objectness
    #    shape: (1, 85, 8400) or transposed   channels: [cx,cy,w,h, obj, cls0..79]
    #    → conf = obj × max(cls_scores), NMS required.
    #    BUG HISTORY: reading pred[:,4:] as pure class scores (skipping objectness)
    #    shifts every class ID up by 1, turning person detections into "bicycle".
    # ────────────────────────────────────────────────────────────────────────

    def _postprocess(self, outputs, res: DetectionResult,
                     w0: int, h0: int, sz: int):
        raw = outputs[0]      # (1, ?, ?) — batch dimension first

        # ── Diagnostic: log first 5 inferences ──────────────────────────────
        do_diag = self._diag_count < 5
        if do_diag:
            self._diag_count += 1
            lines = [
                f"STEP4[{self._diag_count}]  raw_shape={raw.shape}  "
                f"num_outputs={len(outputs)}  dtype={raw.dtype}  "
                f"val_range=[{float(raw.min()):.4f}, {float(raw.max()):.4f}]"
            ]
            for i, o in enumerate(outputs):
                lines.append(f"  output[{i}]: shape={o.shape}  dtype={o.dtype}")
            inf_logger.info("\n".join(lines))

        pred = raw[0]         # strip batch dim

        # ── Format A: NMS-included ───────────────────────────────────────────
        # Signature: second dim ≤ 1000, third dim == 6
        if pred.ndim == 2 and pred.shape[1] == 6 and pred.shape[0] <= 1000:
            if do_diag:
                inf_logger.info("  → FORMAT A detected (NMS-included)")
            self._decode_nms_format(pred, res, w0, h0, sz)
            return

        # ── Normalise to (N_anchors, n_channels) ────────────────────────────
        # Raw is either (n_channels, N_anchors) or (N_anchors, n_channels).
        # n_channels is always the smaller dimension.
        if pred.ndim == 2 and pred.shape[0] < pred.shape[1]:
            pred = pred.T     # → (N_anchors, n_channels)

        n_anchors, n_ch = pred.shape

        if do_diag:
            sample_row = pred[n_anchors // 2, :10].tolist()
            inf_logger.info(
                f"  pred_shape=({n_anchors}, {n_ch})  "
                f"mid_row_first10={[round(v,4) for v in sample_row]}")

        # ── Format C: objectness present — channels = 4 + 1 + n_cls ────────
        # Exact match: n_ch == n_classes + 5  (85 for COCO-80)
        n_cls_c = n_ch - 5
        if n_cls_c > 0:
            obj_col = pred[:, 4].astype(np.float32)
            # Objectness is sigmoid-activated → values genuinely in [0, 1].
            # If that column is truly objectness, its maximum will be ≤ 1.
            if obj_col.max() <= 1.01:
                if do_diag:
                    inf_logger.info(
                        f"  → FORMAT C detected (objectness col4)  "
                        f"n_cls={n_cls_c}  "
                        f"obj_max={obj_col.max():.4f}  "
                        f"obj_mean={obj_col.mean():.6f}")
                self._decode_v5_format(pred, n_cls_c, res, w0, h0, sz, do_diag)
                return

        # ── Format B: no objectness — channels = 4 + n_cls ──────────────────
        n_cls_b = n_ch - 4
        if n_cls_b > 0:
            if do_diag:
                inf_logger.info(
                    f"  → FORMAT B detected (no objectness)  n_cls={n_cls_b}")
            self._decode_v8_format(pred, n_cls_b, res, w0, h0, sz, do_diag)
            return

        inf_logger.error(
            f"_postprocess: unrecognised output  "
            f"pred_shape={pred.shape} — no detections decoded.")

    # ── Format A: NMS already applied ──────────────────────────────────────────
    def _decode_nms_format(self, pred, res: DetectionResult,
                           w0: int, h0: int, sz: int):
        """pred shape (N, 6): [x1, y1, x2, y2, conf, cls_id]  xyxy in input px space."""
        sx, sy = w0 / sz, h0 / sz
        n_kept = 0
        for row in pred:
            x1, y1, x2, y2, conf, cls_f = (float(v) for v in row)
            if conf < self.conf:
                continue
            ci = int(cls_f)
            res.boxes.append([x1 * sx, y1 * sy, x2 * sx, y2 * sy])
            res.class_ids.append(ci)
            res.confidences.append(conf)
            res.class_names.append(
                self._class_names[ci] if ci < len(self._class_names) else str(ci))
            res.track_ids.append(0)
            n_kept += 1
        inf_logger.debug(f"NMS-format: {n_kept} detections after conf filter.")

    # ── Format C: YOLOv5/YOLO26 — WITH objectness ──────────────────────────────
    def _decode_v5_format(self, pred, n_cls: int, res: DetectionResult,
                          w0: int, h0: int, sz: int, do_diag: bool):
        """
        pred shape (N, 5+n_cls): [cx, cy, w, h, obj, cls0..cls_{n_cls-1}]
        Coordinates in INPUT PIXEL SPACE [0, sz].
        Final confidence = objectness × max(class_scores).
        """
        boxes_xywh = pred[:, :4].astype(np.float32)
        obj_conf   = pred[:, 4].astype(np.float32)          # objectness ∈ [0,1]
        cls_scores = pred[:, 5:5 + n_cls].astype(np.float32)# class probs ∈ [0,1]

        # Combined confidence: same calculation as official Ultralytics YOLOv5 post-proc
        best_cls  = cls_scores.argmax(axis=1)
        best_cls_score = cls_scores[np.arange(len(cls_scores)), best_cls]
        best_conf = obj_conf * best_cls_score

        if do_diag:
            inf_logger.info(
                f"  v5-decode  obj_max={obj_conf.max():.4f}  "
                f"cls_max={best_cls_score.max():.4f}  "
                f"combined_max={best_conf.max():.4f}  "
                f"above_thresh={int((best_conf >= self.conf).sum())}  "
                f"threshold={self.conf}")

        self._nms_and_fill(boxes_xywh, best_conf, best_cls, res, w0, h0, sz)

    # ── Format B: YOLOv8/v11 — NO objectness ───────────────────────────────────
    def _decode_v8_format(self, pred, n_cls: int, res: DetectionResult,
                          w0: int, h0: int, sz: int, do_diag: bool):
        """
        pred shape (N, 4+n_cls): [cx, cy, w, h, cls0..cls_{n_cls-1}]
        Coordinates in INPUT PIXEL SPACE [0, sz].
        Final confidence = max(class_scores) directly.
        """
        boxes_xywh = pred[:, :4].astype(np.float32)
        cls_scores = pred[:, 4:4 + n_cls].astype(np.float32)

        best_cls  = cls_scores.argmax(axis=1)
        best_conf = cls_scores[np.arange(len(cls_scores)), best_cls]

        if do_diag:
            inf_logger.info(
                f"  v8-decode  conf_max={best_conf.max():.4f}  "
                f"above_thresh={int((best_conf >= self.conf).sum())}  "
                f"threshold={self.conf}")

        self._nms_and_fill(boxes_xywh, best_conf, best_cls, res, w0, h0, sz)

    # ── Shared: threshold → NMS → fill result ──────────────────────────────────
    def _nms_and_fill(self, boxes_xywh, best_conf, best_cls,
                      res: DetectionResult, w0: int, h0: int, sz: int):
        """
        Apply confidence mask, scale coords from input-px to original-frame-px,
        run NMS once, then populate res.
        Coordinates: cx/cy/w/h in [0, sz] → scale by (w0/sz), (h0/sz).
        """
        mask = best_conf >= self.conf
        if not mask.any():
            return

        bx = boxes_xywh[mask]
        cf = best_conf[mask]
        cl = best_cls[mask]

        sx = w0 / sz   # scale input-px → original frame px
        sy = h0 / sz

        raw_boxes: list = []
        raw_confs: list = []
        raw_cls:   list = []

        for i in range(len(bx)):
            cx, cy, bw, bh = float(bx[i,0]), float(bx[i,1]), \
                              float(bx[i,2]), float(bx[i,3])
            x1 = (cx - bw / 2) * sx
            y1 = (cy - bh / 2) * sy
            x2 = (cx + bw / 2) * sx
            y2 = (cy + bh / 2) * sy
            raw_boxes.append([x1, y1, x2, y2])
            raw_confs.append(float(cf[i]))
            raw_cls.append(int(cl[i]))

        # NMS — applied exactly once (not for Format A which already has NMS)
        if len(raw_boxes) > 1:
            rects = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in raw_boxes]
            idx   = cv2.dnn.NMSBoxes(rects, raw_confs, self.conf, self.iou)
            idx   = ([int(i) for i in idx.flatten()]
                     if idx is not None and len(idx) else [])
        else:
            idx = list(range(len(raw_boxes)))

        for i in idx:
            ci = raw_cls[i]
            res.boxes.append(raw_boxes[i])
            res.class_ids.append(ci)
            res.confidences.append(raw_confs[i])
            res.class_names.append(
                self._class_names[ci] if ci < len(self._class_names) else str(ci))
            res.track_ids.append(0)


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
        color = _class_color(cls)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {conf:.2f}" + (f" #{tid}" if tid > 0 else "")
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        by = max(y1 - th - bl - 4, 0)
        cv2.rectangle(out, (x1, by), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1+2, y1-bl-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return out
