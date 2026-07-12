"""
DualVision AI Detector — Main Window
v1.3 Stable CPU Edition

Rules enforced here:
  • Single Active Camera — only one RTSPStream runs at a time.
  • ONNX Runtime CPUExecutionProvider — no GPU, no CUDA, no PyTorch inference.
  • Fixed model: YOLO26n → yolo26n.onnx.
  • Frame queue size = 1 on the capture side; detector queue = 1 internally.
  • Full logging to logs/startup.log, inference.log, camera.log, debug.log,
    fps_debug.log (per-second FPS diagnostics).
  • FPS counters are fully independent: Capture, Inference, Display, Avg.
  • FPS resets on: Start, camera switch, stream reconnect.
  • FPS freezes (not zeroed) on Stop — last live values remain visible.
  • No silent failure — popups show exact error messages.
"""

import customtkinter as ctk
import threading
import time
import gc
import cv2
import logging
import traceback
from datetime import datetime
from tkinter import filedialog, messagebox

from config.settings import Settings
from camera.stream import RTSPStream, StreamStatus
from ai.detector import Detector, draw_detections, DetectionResult
from ai.model_manager import ModelManager
from ai.backend_manager import BackendManager
from tracking.tracker import ByteTracker, draw_trails
from ai.confidence_smoother import ConfidenceSmoother
from utils.screenshot import ScreenshotUtil
from utils.recorder import VideoRecorder
from utils.detection_log import DetectionLog
from utils.app_logger import setup_logging

from ui.toolbar import Toolbar
from ui.statusbar import StatusBar
from ui.camera_panel import CameraPanel
from ui.control_panel import ControlPanel
from ui.settings_dialog import SettingsDialog
from ui.about_dialog import AboutDialog
from ui.debug_dashboard import DebugDashboard

logger    = logging.getLogger("DualVisionAI.mainwindow")
fps_logger = logging.getLogger("DualVisionAI.fps")

VERSION  = "1.3"
EDITION  = "Stable CPU Edition"
UI_FPS   = 30


class MainWindow(ctk.CTk):
    def __init__(self, settings: Settings):
        super().__init__()
        self._settings = settings

        # ── Logging must be first ─────────────────────────────────────────────
        log_dir = settings.get("logging", "output_dir", "logs")
        setup_logging(log_dir=log_dir)
        logger.info(f"DualVision AI Detector v{VERSION} {EDITION} starting …")

        # ── State ─────────────────────────────────────────────────────────────
        self._detecting   = False
        self._paused      = False
        self._camera_mode = "rgb"
        self._switching   = False

        self._session_detection_count  = 0
        self._current_device           = "CPU"

        self._rgb_draw_result:     DetectionResult | None = None
        self._thermal_draw_result: DetectionResult | None = None
        self._rgb_last_inf_ts:     float = 0.0
        self._thermal_last_inf_ts: float = 0.0
        self._last_inf_ms:         float = 0.0

        self._rgb_display_frame     = None
        self._thermal_display_frame = None
        self._frame_lock = threading.Lock()

        self._debug_dashboard: DebugDashboard | None = None
        self._model_manager_dlg = None

        self._pending_log: list[tuple] = []
        self._log_lock = threading.Lock()

        # ── Display FPS (independent of inference FPS) ────────────────────────
        self._display_frame_count = 0
        self._display_fps_timer   = time.time()
        self._display_fps         = 0.0

        # ── Frozen Capture FPS — snapshot taken at Stop; shown while stopped ──
        self._frozen_cap_fps = 0.0

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # ── BackendManager (CPU-only, no GPU detection) ───────────────────────
        cpu_threads = settings.get("inference", "cpu_threads", 0)
        self._backend_manager = BackendManager(num_threads=cpu_threads)
        self._backend_manager.write_startup_log(log_dir)

        self._init_services()
        self._build_window()
        self._build_ui()
        self._start_streams()
        self._start_worker()
        self._start_ui_tick()
        self._setup_shortcuts()
        self._start_fps_debug_thread()

        # Wire BackendManager into ControlPanel after UI is ready
        self.after(300, lambda: self._control_panel.set_backend_manager(
            self._backend_manager))
        # Startup UI self-test — runs after all widgets settle
        self.after(500, self._run_ui_selftest)
        logger.info("Main window ready.")

    # ── Services ──────────────────────────────────────────────────────────────
    def _init_services(self):
        s = self._settings
        self._rgb_stream = RTSPStream(
            name="RGB",
            url=s.get("rtsp", "rgb_url", ""),
            buffer_size=1,   # queue=1: always latest frame
            reconnect_delay=s.get("rtsp", "reconnect_delay", 3),
            timeout=s.get("rtsp", "timeout", 10),
        )
        self._thermal_stream = RTSPStream(
            name="Thermal",
            url=s.get("rtsp", "thermal_url", ""),
            buffer_size=1,
            reconnect_delay=s.get("rtsp", "reconnect_delay", 3),
            timeout=s.get("rtsp", "timeout", 10),
        )
        self._rgb_stream.set_status_callback(self._on_stream_status)
        self._thermal_stream.set_status_callback(self._on_stream_status)
        self._rgb_stream.set_reconnect_callback(self._on_stream_reconnect)
        self._thermal_stream.set_reconnect_callback(self._on_stream_reconnect)

        self._model_manager = ModelManager(model_dir="models")
        self._detector: Detector | None = None

        self._rgb_tracker     = self._make_tracker()
        self._thermal_tracker = self._make_tracker()

        self._rgb_smoother     = self._make_smoother()
        self._thermal_smoother = self._make_smoother()

        # ── Tracking event logger ─────────────────────────────────────────────
        import os
        _log_dir = s.get("logging", "output_dir", "logs")
        os.makedirs(_log_dir, exist_ok=True)
        _trk_handler = logging.FileHandler(
            os.path.join(_log_dir, "tracking.log"), mode="a", encoding="utf-8")
        _trk_handler.setFormatter(logging.Formatter(
            "%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        _trk_handler.setLevel(logging.INFO)
        self._trk_logger = logging.getLogger("DualVisionAI.tracking_events")
        if not self._trk_logger.handlers:
            self._trk_logger.addHandler(_trk_handler)
        self._trk_logger.setLevel(logging.INFO)
        self._trk_logger.propagate = False

        self._rgb_tracker.set_event_callback(self._on_tracking_event)
        self._thermal_tracker.set_event_callback(self._on_tracking_event)

        self._screenshot_util = ScreenshotUtil(
            output_dir=s.get("screenshots", "output_dir", "screenshots"))
        self._recorder = VideoRecorder(
            output_dir=s.get("recording", "output_dir", "recordings"),
            fps=s.get("recording", "fps", 25))
        self._det_log = DetectionLog(
            output_dir=s.get("logging", "output_dir", "logs"),
            max_entries=s.get("logging", "max_log_entries", 100_000))

        self._worker_running = False
        self._ui_running     = False

    # ── Tracker factory ───────────────────────────────────────────────────────
    def _make_tracker(self) -> ByteTracker:
        """Create a ByteTracker from current tracking settings."""
        ts = self._settings.section("tracking")
        return ByteTracker(
            max_age=int(ts.get("track_timeout", 30)),
            min_hits=int(ts.get("min_confirmation_hits", 3)),
            iou_threshold=float(ts.get("association_threshold", 0.20)),
            low_iou=float(ts.get("low_association_threshold", 0.10)),
            high_conf=float(ts.get("tracking_confidence", 0.40)),
            max_trail_length=int(ts.get("max_trail_length", 30)),
            persistence_frames=int(ts.get("persistence_frames", 5)),
        )

    # ── Smoother factory ──────────────────────────────────────────────────────
    def _make_smoother(self) -> ConfidenceSmoother:
        """Create a ConfidenceSmoother from current smoothing settings."""
        sm = self._settings.section("smoothing")
        return ConfidenceSmoother(
            ema_alpha=float(sm.get("ema_alpha", 0.35)),
            iou_threshold=float(sm.get("iou_threshold", 0.40)),
            max_ghost_frames=int(sm.get("max_ghost_frames", 3)),
            ghost_decay=float(sm.get("ghost_decay", 0.70)),
            min_ghost_conf=float(sm.get("min_ghost_conf", 0.25)),
            min_box_area=int(sm.get("min_box_area", 0)),
            max_box_area=int(sm.get("max_box_area", 0)),
        )

    # ── Tracking event callback ───────────────────────────────────────────────
    def _on_tracking_event(self, event_type: str, track):
        """Write track lifecycle events to logs/tracking.log."""
        try:
            det = self._detector
            names = det.class_names if det else []
            name = names[track.class_id] if track.class_id < len(names) else "?"
            vx, vy = track.velocity
            speed = (vx ** 2 + vy ** 2) ** 0.5
            self._trk_logger.info(
                f"{event_type.upper():<10}  "
                f"Track#{track.track_id:4d}  "
                f"class={name:<12}  "
                f"hits={track.hits:3d}  "
                f"age={track.track_age_sec:5.1f}s  "
                f"spd={speed:4.1f}px  "
                f"dir={track.direction:<12}  "
                f"box=({track.box[0]:.0f},{track.box[1]:.0f},"
                f"{track.box[2]:.0f},{track.box[3]:.0f})")
        except Exception:
            pass

    # ── Window ────────────────────────────────────────────────────────────────
    def _build_window(self):
        s = self._settings
        self.title(f"DualVision AI Detector  v{VERSION} — {EDITION}")
        w = s.get("ui", "window_width",  1280)
        h = s.get("ui", "window_height",  900)
        x = s.get("ui", "window_x",       100)
        y = s.get("ui", "window_y",       100)
        self.geometry(f"{w}x{h}+{x}+{y}")
        self.minsize(900, 600)
        if s.get("ui", "maximized", False):
            self.state("zoomed")
        self.configure(fg_color="#050A14")
        self.protocol("WM_DELETE_WINDOW", self._on_exit)
        self._set_window_icon()

    def _set_window_icon(self):
        try:
            from pathlib import Path
            assets = Path(__file__).parent.parent / "assets"
            ico = assets / "icon.ico"
            png = assets / "logo.png"
            if not ico.exists() or not png.exists():
                from assets.make_icon import generate
                generate(assets)
            if ico.exists():
                self.iconbitmap(str(ico))
            elif png.exists():
                from PIL import Image, ImageTk
                img   = Image.open(str(png)).resize((64, 64))
                photo = ImageTk.PhotoImage(img)
                self.iconphoto(True, photo)
                self._icon_ref = photo
        except Exception as e:
            logger.debug(f"Icon not set: {e}")

    # ── UI Layout ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        callbacks = {
            "start":        self._on_start,
            "stop":         self._on_stop,
            "pause":        self._on_pause,
            "resume":       self._on_resume,
            "screenshot":   self._on_screenshot,
            "record_start": self._on_record_start,
            "record_stop":  self._on_record_stop,
            "settings":     self._on_settings,
            "export_csv":   self._on_export_csv,
            "export_json":  self._on_export_json,
            "about":        self._on_about,
            "exit":         self._on_exit,
            "debug":         self._on_debug,
            "model_manager": self._on_model_manager,
        }
        self._toolbar = Toolbar(self, callbacks=callbacks)
        self._toolbar.pack(fill="x", side="top")

        self._content = ctk.CTkFrame(self, fg_color="transparent")
        self._content.pack(fill="both", expand=True, padx=6, pady=4)

        self._rgb_panel     = CameraPanel(self._content, title="RGB Camera")
        self._thermal_panel = CameraPanel(self._content, title="Thermal Camera")

        self._control_panel = ControlPanel(
            self._content,
            on_camera_switch=self._on_camera_switch_requested,
            backend_manager=self._backend_manager)
        self._control_panel.pack(side="right", fill="y", padx=(3, 0))

        self._apply_camera_layout("rgb", animate=False)

        self._statusbar = StatusBar(self)
        self._statusbar.pack(fill="x", side="bottom")

    def _apply_camera_layout(self, mode: str, animate: bool = True):
        if mode == "rgb":
            self._thermal_panel.pack_forget()
            self._rgb_panel.pack(side="left", fill="both", expand=True,
                                 padx=(0, 3))
        else:
            self._rgb_panel.pack_forget()
            self._thermal_panel.pack(side="left", fill="both", expand=True,
                                     padx=(0, 3))

    # ── Single-camera switching ────────────────────────────────────────────────
    def _on_camera_switch_requested(self, mode: str):
        if mode == self._camera_mode or self._switching:
            return
        self._control_panel.set_camera_buttons_enabled(False)
        self._switching = True
        threading.Thread(target=self._do_camera_switch,
                         args=(mode,), daemon=True,
                         name="CamSwitch").start()

    def _do_camera_switch(self, new_mode: str):
        """
        Hard-stop the inactive camera, clear all its resources, then
        start the new one.  Only one stream runs at any time.
        """
        old_mode = self._camera_mode
        logger.info(f"Camera switch: {old_mode} → {new_mode}")

        was_detecting = self._detecting
        if was_detecting and self._detector:
            self._detector.pause()

        self._recorder.stop_all()
        self.after(0, self._control_panel.set_recording_status, False, "")

        # ── Fully stop old stream ─────────────────────────────────────────────
        if old_mode == "rgb":
            self._rgb_stream.stop()
            with self._frame_lock:
                self._rgb_display_frame = None
            self._rgb_draw_result  = None
            self._rgb_last_inf_ts  = 0.0
            self._rgb_tracker.reset()
            self._flush_queue(self._detector, "rgb")
        else:
            self._thermal_stream.stop()
            with self._frame_lock:
                self._thermal_display_frame = None
            self._thermal_draw_result  = None
            self._thermal_last_inf_ts  = 0.0
            self._thermal_tracker.reset()
            self._flush_queue(self._detector, "thermal")

        gc.collect()
        self._camera_mode = new_mode
        self._last_inf_ms = 0.0

        self.after(0, self._apply_camera_layout, new_mode)
        self.after(0, self._control_panel.set_camera_mode_label, new_mode)

        # ── Start new stream (fresh RTSPStream instance) ───────────────────────
        s = self._settings
        if new_mode == "rgb":
            self._rgb_stream = RTSPStream(
                name="RGB",
                url=s.get("rtsp", "rgb_url", ""),
                buffer_size=1,
                reconnect_delay=s.get("rtsp", "reconnect_delay", 3),
                timeout=s.get("rtsp", "timeout", 10),
            )
            self._rgb_stream.set_status_callback(self._on_stream_status)
            self._rgb_stream.set_reconnect_callback(self._on_stream_reconnect)
            self._rgb_stream.start()
            logger.info("RGB stream started after camera switch.")
        else:
            self._thermal_stream = RTSPStream(
                name="Thermal",
                url=s.get("rtsp", "thermal_url", ""),
                buffer_size=1,
                reconnect_delay=s.get("rtsp", "reconnect_delay", 3),
                timeout=s.get("rtsp", "timeout", 10),
            )
            self._thermal_stream.set_status_callback(self._on_stream_status)
            self._thermal_stream.set_reconnect_callback(self._on_stream_reconnect)
            self._thermal_stream.start()
            logger.info("Thermal stream started after camera switch.")

        if was_detecting and self._detector:
            self._detector.clear_results()
            self._detector.reset_fps()        # prevent stale FPS from old stream
            self._display_frame_count = 0     # reset display FPS window too
            self._display_fps_timer   = time.time()
            self._display_fps         = 0.0
            self._frozen_cap_fps      = 0.0   # clear frozen snapshot from any prior stop
            self._detector.resume()

        self._switching = False
        self.after(0, self._control_panel.set_camera_buttons_enabled, True)
        logger.info(f"Camera switch complete → {new_mode}")

    @staticmethod
    def _flush_queue(detector: "Detector | None", stream: str):
        if detector is None:
            return
        q = detector._rgb_q if stream == "rgb" else detector._thermal_q
        try:
            while not q.empty():
                q.get_nowait()
        except Exception:
            pass

    # ── Streams ───────────────────────────────────────────────────────────────
    def _start_fps_debug_thread(self):
        """Background thread — writes one line per second to fps_debug.log."""
        def _log_loop():
            while self._ui_running or not hasattr(self, "_ui_running"):
                try:
                    mode    = self._camera_mode
                    det_on  = self._detecting
                    det     = self._detector
                    stream  = (self._rgb_stream if mode == "rgb"
                               else self._thermal_stream)
                    cap_fps = stream.fps_actual   if stream  else 0.0
                    inf_fps = det.fps_inference   if det else 0.0
                    inf_ms  = det.infer_ms        if det else 0.0
                    q_size  = det.queue_size      if det else 0
                    fps_logger.debug(
                        f"cam={mode.upper():<7}  det={'ON' if det_on else 'OFF'}  "
                        f"cap_fps={cap_fps:5.1f}  "
                        f"infer_fps={inf_fps:5.1f}  "
                        f"display_fps={self._display_fps:5.1f}  "
                        f"inf_ms={inf_ms:6.1f}  "
                        f"queue={q_size}")
                except Exception:
                    pass
                time.sleep(1.0)
        threading.Thread(target=_log_loop, daemon=True,
                         name="FPS-Debug-Logger").start()

    def _start_streams(self):
        """Only start the active camera stream."""
        if self._camera_mode == "rgb":
            self._rgb_stream.start()
        else:
            self._thermal_stream.start()

    # ── Worker ────────────────────────────────────────────────────────────────
    def _start_worker(self):
        self._worker_running = True
        threading.Thread(target=self._worker_loop,
                         daemon=True, name="Worker").start()

    def _worker_loop(self):
        while self._worker_running:
            t0 = time.perf_counter()
            try:
                self._process_one_frame()
            except Exception:
                logger.error(f"Worker error:\n{traceback.format_exc()}")
            elapsed = time.perf_counter() - t0
            sleep   = max(0.001, 1/60 - elapsed)
            time.sleep(sleep)

    def _process_one_frame(self):
        if self._switching:
            time.sleep(0.02)
            return

        mode = self._camera_mode
        if mode == "rgb":
            stream = self._rgb_stream
        else:
            stream = self._thermal_stream

        frame = stream.read()

        if not self._detecting or self._paused or self._detector is None:
            if frame is not None:
                with self._frame_lock:
                    if mode == "rgb":
                        self._rgb_display_frame = frame
                    else:
                        self._thermal_display_frame = frame
                # ── Recording without detection — always write raw frame ───────
                rec_label = "RGB" if mode == "rgb" else "Thermal"
                if self._recorder.is_recording(rec_label):
                    self._recorder.write(rec_label, frame)
            return

        if frame is not None:
            if mode == "rgb":
                self._detector.push_rgb(frame)
            else:
                self._detector.push_thermal(frame)

        track = self._settings.get("detection", "enable_tracking", True)

        enable_trails  = (track and
                          self._settings.get("tracking", "enable_trails", False))
        rec_mode       = self._settings.get("recording", "recording_mode", "overlay")

        if mode == "rgb":
            self._process_result(frame, "rgb", track)
            display = draw_detections(frame, self._rgb_draw_result) if frame is not None else None
            if display is not None and enable_trails:
                display = draw_trails(display, self._rgb_tracker.get_trails())
            if self._recorder.is_recording("RGB"):
                rec_frame = frame if (rec_mode == "raw" and frame is not None) else display
                if rec_frame is not None:
                    self._recorder.write("RGB", rec_frame)
            if display is not None:
                with self._frame_lock:
                    self._rgb_display_frame = display
        else:
            self._process_result(frame, "thermal", track)
            display = draw_detections(frame, self._thermal_draw_result) if frame is not None else None
            if display is not None and enable_trails:
                display = draw_trails(display, self._thermal_tracker.get_trails())
            if self._recorder.is_recording("Thermal"):
                rec_frame = frame if (rec_mode == "raw" and frame is not None) else display
                if rec_frame is not None:
                    self._recorder.write("Thermal", rec_frame)
            if display is not None:
                with self._frame_lock:
                    self._thermal_display_frame = display

    def _process_result(self, frame, stream: str, enable_tracking: bool):
        if stream == "rgb":
            result = self._detector.get_rgb_result()
            last   = self._rgb_last_inf_ts
        else:
            result = self._detector.get_thermal_result()
            last   = self._thermal_last_inf_ts

        if result is None or result.timestamp == last:
            return

        if stream == "rgb":
            self._rgb_last_inf_ts = result.timestamp
        else:
            self._thermal_last_inf_ts = result.timestamp
        self._last_inf_ms = result.inference_ms

        if result.is_empty():
            if enable_tracking:
                # Age tracks naturally — do NOT reset; preserves re-identification
                (self._rgb_tracker if stream == "rgb"
                 else self._thermal_tracker).update([])
            if stream == "rgb":
                self._rgb_draw_result = None
            else:
                self._thermal_draw_result = None
            return

        # ── STEP 1: Pre-fill class names from raw ONNX output ────────────────
        # This guarantees boxes always have labels, even if tracking is skipped
        # or throws.  Tracking will OVERWRITE these if it succeeds.
        _names = self._detector.class_names
        result.class_names = [
            _names[cid] if cid < len(_names) else "?"
            for cid in result.class_ids]
        result.track_ids = []

        # ── STEP 2: Optional tracking — wrapped so exceptions NEVER prevent ──
        # the draw result from being set.  If tracking fails, raw ONNX
        # detections (set in STEP 1) are drawn instead.
        if enable_tracking:
            try:
                tracker = (self._rgb_tracker if stream == "rgb"
                           else self._thermal_tracker)
                dets = [{"box": b, "class_id": c, "confidence": cf}
                        for b, c, cf in zip(result.boxes, result.class_ids,
                                            result.confidences)]
                # ── Confidence smoother — sits UPSTREAM of the tracker ────────
                # Applies EMA smoothing and synthesises ghost detections for
                # recently-seen objects to prevent 1–3 frame disappearances.
                if self._settings.get("smoothing", "enable_smoother", True):
                    smoother = (self._rgb_smoother if stream == "rgb"
                                else self._thermal_smoother)
                    dets = smoother.process(dets)
                tracked = tracker.update(dets)

                if tracked:
                    result.boxes       = [t["box"]       for t in tracked]
                    result.class_ids   = [t["class_id"]  for t in tracked]
                    result.confidences = [t["confidence"] for t in tracked]
                    result.class_names = [
                        _names[t["class_id"]] if t["class_id"] < len(_names) else "?"
                        for t in tracked]
                    result.track_ids   = [t["track_id"]  for t in tracked]
                # else: tracked is empty — STEP 1 raw detections remain
            except Exception as _trk_err:
                # Tracking error: log once and fall back to raw detections.
                # NEVER propagate — this block must not prevent STEP 3.
                logger.warning(
                    f"[{stream}] Tracking error (raw detections used): {_trk_err}")

        # ── STEP 3: Set draw result — ALWAYS reached regardless of tracking ──
        # This is the line that was previously skipped when tracking threw.
        if stream == "rgb":
            self._rgb_draw_result = result
        else:
            self._thermal_draw_result = result

        self._session_detection_count += len(result.boxes)

        for i in range(len(result.boxes)):
            name = result.class_names[i]  if i < len(result.class_names)  else ""
            conf = result.confidences[i]  if i < len(result.confidences)  else 0.0
            tid  = result.track_ids[i]    if i < len(result.track_ids)    else 0
            box  = result.boxes[i]        if i < len(result.boxes)        else [0,0,0,0]
            cam_label = "RGB" if stream == "rgb" else "Thermal"
            self._det_log.log(cam_label, name, conf, tid, box)
            with self._log_lock:
                self._pending_log.append((cam_label, name, conf, tid))

    # ── UI Tick ───────────────────────────────────────────────────────────────
    def _start_ui_tick(self):
        self._ui_running = True
        self.after(max(1, int(1000/UI_FPS)), self._ui_tick)

    def _ui_tick(self):
        try:
            self._do_ui_tick()
        except Exception:
            logger.error(f"UI tick error:\n{traceback.format_exc()}")
        finally:
            if self._ui_running:
                self.after(max(1, int(1000/UI_FPS)), self._ui_tick)

    def _do_ui_tick(self):
        mode = self._camera_mode
        det  = self._detector

        with self._frame_lock:
            rgb_frame     = self._rgb_display_frame
            thermal_frame = self._thermal_display_frame

        fps_inf = det.fps_inference if det else 0.0
        avg_fps = det.avg_fps       if det else 0.0

        rgb_det     = len(self._rgb_draw_result.boxes)     if self._rgb_draw_result     else 0
        thermal_det = len(self._thermal_draw_result.boxes) if self._thermal_draw_result else 0
        active_det  = rgb_det if mode == "rgb" else thermal_det

        # Update active camera panel and count display frames
        frame_shown = False
        if mode == "rgb" and rgb_frame is not None:
            self._rgb_panel.update_frame(
                rgb_frame, self._rgb_stream.fps_actual,
                rgb_det, self._last_inf_ms)
            frame_shown = True
        elif mode == "thermal" and thermal_frame is not None:
            self._thermal_panel.update_frame(
                thermal_frame, self._thermal_stream.fps_actual,
                thermal_det, self._last_inf_ms)
            frame_shown = True

        # Display FPS — count only while detecting; freeze when stopped
        if frame_shown and self._detecting:
            self._display_frame_count += 1
            _now = time.time()
            _elapsed = _now - self._display_fps_timer
            if _elapsed >= 1.0:
                self._display_fps       = self._display_frame_count / _elapsed
                self._display_frame_count = 0
                self._display_fps_timer = _now

        # Capture FPS: live while detecting, frozen at last Stop value otherwise
        if self._detecting:
            cap_fps = (self._rgb_stream.fps_actual if mode == "rgb"
                       else self._thermal_stream.fps_actual)
        else:
            cap_fps = self._frozen_cap_fps

        # Collect tracking stats from the active tracker
        tracker = (self._rgb_tracker if mode == "rgb"
                   else self._thermal_tracker)
        trk_stats = tracker.get_stats()
        # Build compact track-ID string for the Detection panel
        active_result = (self._rgb_draw_result if mode == "rgb"
                         else self._thermal_draw_result)
        if active_result and active_result.track_ids:
            ids_str = ", ".join(
                f"#{tid}" for tid in active_result.track_ids[:8])
            if len(active_result.track_ids) > 8:
                ids_str += " …"
        else:
            ids_str = "—"

        self._control_panel.update_stats(
            fps_inf=fps_inf,
            avg_fps=avg_fps,
            cap_fps=cap_fps,
            display_fps=self._display_fps,
            inf_ms=self._last_inf_ms,
            preprocess_ms=det.preprocess_ms   if det else 0.0,
            infer_ms=det.infer_ms             if det else 0.0,
            postprocess_ms=det.postprocess_ms if det else 0.0,
            active_threads=det.active_threads if det else 0,
            queue_size=det.queue_size         if det else 0,
            frame_drops=det.frame_drops       if det else 0,
            active_dets=active_det,
            session_total=self._session_detection_count,
            camera_mode=mode,
            trk_active=trk_stats["active_tracks"],
            trk_confirmed=trk_stats["confirmed_tracks"],
            trk_lost=trk_stats["lost_tracks"],
            trk_recovered=trk_stats["recovered_total"],
            trk_new=trk_stats["new_total"],
            trk_avg_age=trk_stats["avg_age_sec"],
            trk_longest=trk_stats["longest_active_sec"],
            trk_fps=trk_stats["tracking_fps"],
            trk_latency=trk_stats["latency_ms"],
            trk_ids=ids_str,
        )

        # Flush detection log
        with self._log_lock:
            batch = self._pending_log[:20]
            self._pending_log = self._pending_log[20:]
        for cam, name, conf, tid in batch:
            self._control_panel.log_detection(cam, name, conf, tid)

        # Status bar
        active_status = (self._rgb_stream.status.value if mode == "rgb"
                         else self._thermal_stream.status.value)
        rgb_s    = active_status if mode == "rgb"     else "Inactive"
        therm_s  = active_status if mode == "thermal" else "Inactive"
        active_variant = self._settings.get("inference", "active_model", "yolo26n")
        self._statusbar.update(
            fps=fps_inf, model=active_variant.upper(), device="CPU",
            detecting=self._detecting,
            rgb_status=rgb_s, thermal_status=therm_s,
            timestamp=datetime.now().strftime("%H:%M:%S"))

        # ── Feed Debug Dashboard if open ──────────────────────────────────────
        if (self._debug_dashboard is not None and
                self._debug_dashboard.winfo_exists()):
            try:
                smoother    = (self._rgb_smoother if mode == "rgb"
                               else self._thermal_smoother)
                tracker_obj = (self._rgb_tracker if mode == "rgb"
                               else self._thermal_tracker)
                active_result = (self._rgb_draw_result if mode == "rgb"
                                 else self._thermal_draw_result)

                # Build per-object list from the active result
                tracked_objs = []
                if active_result:
                    for _i in range(len(active_result.boxes)):
                        tracked_objs.append({
                            "track_id":        (active_result.track_ids[_i]
                                                if _i < len(active_result.track_ids) else 0),
                            "class_id":        (active_result.class_ids[_i]
                                                if _i < len(active_result.class_ids) else 0),
                            "confidence":      (active_result.confidences[_i]
                                                if _i < len(active_result.confidences) else 0.0),
                            "direction":       "—",
                            "age_sec":         0.0,
                            "hits":            1,
                            "lost_count":      0,
                            "recovered_count": 0,
                            "ghost":           (active_result.ghost_flags[_i]
                                                if _i < len(active_result.ghost_flags) else False),
                        })

                sm_stats  = smoother.get_stats()
                trk_stats = tracker_obj.get_stats()
                confs     = [o["confidence"] for o in tracked_objs if not o["ghost"]]
                avg_conf  = sum(confs) / len(confs) if confs else 0.0

                import psutil as _ps
                self._debug_dashboard.update_data({
                    "fps_inf":          det.fps_inference   if det else 0.0,
                    "avg_fps":          det.avg_fps          if det else 0.0,
                    "inf_ms":           det.infer_ms         if det else 0.0,
                    "pre_ms":           det.preprocess_ms    if det else 0.0,
                    "post_ms":          det.postprocess_ms   if det else 0.0,
                    "frame_drops":      det.frame_drops      if det else 0,
                    "cpu_pct":          _ps.cpu_percent(interval=None),
                    "ram_mb":           _ps.virtual_memory().used // (1024 * 1024),
                    "model_name":       active_variant,
                    "input_size":       self._settings.get("detection", "input_width", 640),
                    "num_classes":      len(det.class_names) if det else 0,
                    "avg_conf":         avg_conf,
                    "tracker_stats":    trk_stats,
                    "tracked_objects":  tracked_objs,
                    "class_names":      ({i: n for i, n in enumerate(det.class_names)}
                                        if det else {}),
                    "smoother_stats":   sm_stats,
                    "smoother_avg_conf": sm_stats.get("avg_smooth_conf", 0.0),
                    "trk_ghost":        tracker_obj.get_ghost_count(),
                    "trk_lifetime_avg": trk_stats.get("avg_age_sec", 0.0),
                    "trk_id_changes":   0,
                    "trk_lost":         trk_stats.get("lost_tracks", 0),
                    "conf_current":     avg_conf,
                })
            except Exception:
                pass

    # ── Actions ───────────────────────────────────────────────────────────────
    def _on_start(self):
        if self._detecting:
            return

        s            = self._settings
        conf         = s.get("detection", "confidence",    0.45)
        iou_val      = s.get("detection", "iou",           0.45)
        input_size   = s.get("detection", "input_width",   640)
        frame_skip   = s.get("detection", "frame_skip",    1)
        cpu_th       = s.get("inference", "cpu_threads",   0)
        active_model = s.get("inference", "active_model",  "yolo26n")

        def _load_and_start():
            try:
                # ── Select model variant ──────────────────────────────────────
                try:
                    self._model_manager.set_variant(active_model)
                except ValueError:
                    self._model_manager.set_variant("yolo26n")
                variant_info = self._model_manager.get_model_info()
                pt_name   = variant_info["name"]
                onnx_name = f"{active_model}.onnx"

                # ── Ensure .pt ───────────────────────────────────────────────
                if not self._model_manager.is_pt_ready():
                    self.after(0, messagebox.showinfo, "Downloading Model",
                               f"{pt_name} not found locally.\n\n"
                               f"Ultralytics will download it "
                               f"(~{variant_info.get('pt_mb', '?')} MB or less).\n"
                               "Internet is required only this one time.\n\n"
                               "Click OK — detection will start once the "
                               "download completes.")

                # ── Ensure .onnx ──────────────────────────────────────────────
                if not self._model_manager.is_onnx_ready():
                    self.after(0, lambda n=onnx_name: logger.info(
                        f"Exporting {n} …"))
                    try:
                        self._model_manager.export_onnx()
                    except Exception as exc:
                        tb = traceback.format_exc()
                        def _show_err(e=str(exc), t=tb, n=onnx_name):
                            messagebox.showerror(
                                "ONNX Export Failed",
                                f"Could not export {n}:\n\n{e}\n\n"
                                "Full traceback written to logs/inference.log.\n\n"
                                "Fix the error and click Start again.")
                        self.after(0, _show_err)
                        logger.error(f"ONNX export failed:\n{tb}")
                        return

                onnx_path = self._model_manager.get_onnx_path()
                if not self._model_manager.is_onnx_ready():
                    self.after(0, messagebox.showerror, "Model Error",
                               f"{onnx_name} not found.\n\n"
                               "Run Setup → Export to ONNX, then try again.")
                    return

                # ── Rebuild BackendManager if threads changed ─────────────────
                bm = self._backend_manager
                if bm.num_threads != (cpu_th if cpu_th > 0 else bm.num_threads):
                    bm = BackendManager(num_threads=cpu_th)
                    bm.write_startup_log(
                        self._settings.get("logging", "output_dir", "logs"))
                    self._backend_manager = bm
                    self.after(0, lambda: self._control_panel.set_backend_manager(bm))

                # ── Build Detector ────────────────────────────────────────────
                detector = Detector(
                    onnx_path=str(onnx_path),
                    conf=conf, iou=iou_val,
                    input_size=input_size,
                    frame_skip=frame_skip,
                    backend_manager=self._backend_manager,
                )

                detector.load()   # raises on failure with exact message

                # ── Swap out old detector ─────────────────────────────────────
                if self._detector is not None:
                    try:
                        self._detector.stop()
                        self._detector.clear_results()
                    except Exception:
                        pass

                self._detector = detector
                # Full counter reset on every Start press
                self._detector.reset_fps()
                self._display_frame_count = 0
                self._display_fps_timer   = time.time()
                self._display_fps         = 0.0
                self._frozen_cap_fps      = 0.0
                # Re-create trackers so any changed tracking settings take effect
                self._rgb_tracker     = self._make_tracker()
                self._thermal_tracker = self._make_tracker()
                self._rgb_tracker.set_event_callback(self._on_tracking_event)
                self._thermal_tracker.set_event_callback(self._on_tracking_event)
                # Re-create smoothers from current Optimization settings
                self._rgb_smoother     = self._make_smoother()
                self._thermal_smoother = self._make_smoother()
                self._rgb_draw_result      = None
                self._thermal_draw_result  = None
                self._rgb_last_inf_ts      = 0.0
                self._thermal_last_inf_ts  = 0.0
                self._last_inf_ms          = 0.0
                self._session_detection_count = 0
                self._detecting = True
                self._paused    = False
                self._detector.start()

                logger.info(
                    f"Detection started — model={onnx_path.name}  "
                    f"variant={active_model}  "
                    f"backend=ONNX Runtime CPU  "
                    f"provider=CPUExecutionProvider  "
                    f"camera={self._camera_mode}  "
                    f"input_size={input_size}  "
                    f"conf={conf}  iou={iou_val}  skip={frame_skip}")

            except Exception as exc:
                tb = traceback.format_exc()
                logger.error(f"Start failed:\n{tb}")
                def _show(e=str(exc), t=tb):
                    messagebox.showerror(
                        "Startup Error",
                        f"{e}\n\nFull traceback written to logs/debug.log")
                self.after(0, _show)

        threading.Thread(target=_load_and_start, daemon=True,
                         name="ModelLoader").start()

    def _on_stop(self):
        # Snapshot Capture FPS before stopping so the dashboard keeps showing
        # the last real value instead of jumping to whatever the live stream
        # reports after detection ends.
        mode = self._camera_mode
        self._frozen_cap_fps = (self._rgb_stream.fps_actual if mode == "rgb"
                                else self._thermal_stream.fps_actual)
        self._detecting = False
        self._paused    = False
        if self._detector:
            self._detector.stop()
            self._detector.clear_results()
            # Do NOT call reset_fps() here — requirement says "freeze values,
            # do not continue updating". Inference threads are stopped so FPS
            # values naturally freeze at their last reading.
        # Freeze display FPS too (stop incrementing counter)
        self._display_frame_count = 0   # prevents stale partial-window math
        self._rgb_tracker.reset()
        self._thermal_tracker.reset()
        self._rgb_smoother.reset()
        self._thermal_smoother.reset()
        self._rgb_draw_result      = None
        self._thermal_draw_result  = None
        self._rgb_last_inf_ts      = 0.0
        self._thermal_last_inf_ts  = 0.0
        self._last_inf_ms          = 0.0
        self._recorder.stop_all()
        self._control_panel.set_recording_status(False, "")
        logger.info("Detection stopped.")

    def _on_pause(self):
        if self._detecting and not self._paused:
            self._paused = True
            if self._detector:
                self._detector.pause()
                self._detector.clear_results()
            if self._camera_mode == "rgb":
                self._rgb_stream.pause()
            else:
                self._thermal_stream.pause()
            logger.info("Detection paused.")

    def _on_resume(self):
        if self._detecting and self._paused:
            self._paused = False
            if self._detector:
                self._detector.resume()
            if self._camera_mode == "rgb":
                self._rgb_stream.resume()
            else:
                self._thermal_stream.resume()
            logger.info("Detection resumed.")

    def _on_screenshot(self):
        with self._frame_lock:
            rgb_f   = self._rgb_display_frame
            therm_f = self._thermal_display_frame
        frame = rgb_f if self._camera_mode == "rgb" else therm_f
        paths = self._screenshot_util.save_both(
            frame.copy() if frame is not None else None, None)
        if paths:
            messagebox.showinfo("Screenshot Saved", "Saved:\n" + "\n".join(paths))
        else:
            messagebox.showwarning("Screenshot", "No frame available yet.")

    def _on_record_start(self):
        with self._frame_lock:
            rgb_f   = self._rgb_display_frame
            therm_f = self._thermal_display_frame
        frame = rgb_f if self._camera_mode == "rgb" else therm_f
        label = "RGB" if self._camera_mode == "rgb" else "Thermal"
        if frame is not None:
            h, w = frame.shape[:2]
            p = self._recorder.start(label, w, h)
            if p:
                self._control_panel.set_recording_status(True, p)
                return
        messagebox.showwarning("Recording", "No frame available to record.")

    def _on_record_stop(self):
        self._recorder.stop_all()
        self._control_panel.set_recording_status(False, "")

    def _on_settings(self):
        def on_save():
            if self._detector:
                self._detector.update_params(
                    conf=self._settings.get("detection", "confidence", 0.45),
                    iou=self._settings.get("detection", "iou",         0.45),
                    frame_skip=self._settings.get("detection", "frame_skip", 1),
                    input_size=self._settings.get("detection", "input_width", 640),
                )
        SettingsDialog(self, self._settings, on_save=on_save,
                       detector=self._detector,
                       backend_manager=self._backend_manager)

    def _on_export_csv(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")],
            title="Export Detection Log as CSV")
        if path:
            try:
                self._det_log.export_csv(path)
                messagebox.showinfo("Export", f"CSV exported:\n{path}")
            except Exception as exc:
                messagebox.showerror("Export Error", str(exc))

    def _on_export_json(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
            title="Export Detection Log as JSON")
        if path:
            try:
                self._det_log.export_json(path)
                messagebox.showinfo("Export", f"JSON exported:\n{path}")
            except Exception as exc:
                messagebox.showerror("Export Error", str(exc))

    def _on_about(self):
        AboutDialog(self)

    def _on_model_manager(self):
        """Open or focus the dedicated Model Manager window (PART 5)."""
        try:
            if (self._model_manager_dlg is None or
                    not self._model_manager_dlg.winfo_exists()):
                from ui.model_manager_dialog import ModelManagerDialog
                self._model_manager_dlg = ModelManagerDialog(
                    self,
                    self._settings,
                    on_load_model=self._on_load_model)
                self._model_manager_dlg.lift()
            else:
                self._model_manager_dlg.lift()
                self._model_manager_dlg.focus_force()
        except Exception as exc:
            logger.warning(f"Model Manager error: {exc}")

    def _on_load_model(self, variant: str):
        """
        Hot-swap active model (PART 5).
        Called by ModelManagerDialog when the user clicks 'Load'.
        If detection is running: stop it, swap, restart automatically.
        If idle: save setting and show a brief info message.
        """
        import tkinter.messagebox as _mb
        was_detecting = self._detecting
        if was_detecting:
            logger.info(f"Hot-swap: stopping detection to load {variant}")
            self._on_stop()

        self._settings.set("inference", "active_model", variant)
        self._settings.save()
        logger.info(f"Hot-swap: active_model → {variant}")

        if was_detecting:
            # Small delay so inference threads finish cleanly before restart
            logger.info(f"Hot-swap: restarting detection with {variant}")
            self.after(700, self._on_start)
        else:
            try:
                _mb.showinfo(
                    "Model Loaded",
                    f"Active model set to  {variant}.\n\n"
                    "Click  ▶ Start  to begin detection with this model.",
                    parent=self)
            except Exception:
                pass

    def _on_debug(self):
        """Open or focus the Debug Dashboard floating window."""
        try:
            if (self._debug_dashboard is None or
                    not self._debug_dashboard.winfo_exists()):
                self._debug_dashboard = DebugDashboard(self)
                self._debug_dashboard.lift()
            else:
                self._debug_dashboard.lift()
                self._debug_dashboard.focus_force()
        except Exception as exc:
            logger.warning(f"Debug dashboard error: {exc}")

    def _on_stream_reconnect(self, name: str):
        """Called by RTSPStream each time it successfully (re)connects.
        Reset inference FPS so stale pre-disconnect values don't contaminate
        the dashboard.
        """
        logger.info(f"[{name}] Reconnected — resetting FPS counters.")
        if self._detector and self._detecting:
            self._detector.reset_fps()
        # Reset display FPS — zero value and counter so reconnect doesn't
        # show a stale reading until the next full 1-second window.
        self._display_fps         = 0.0
        self._display_frame_count = 0
        self._display_fps_timer   = time.time()

    def _on_stream_status(self, name: str, status: StreamStatus):
        mode = self._camera_mode
        if (name == "RGB" and mode == "rgb") or \
           (name == "Thermal" and mode == "thermal"):
            connected = status == StreamStatus.CONNECTED
            panel     = self._rgb_panel if mode == "rgb" else self._thermal_panel
            self.after(0, panel.set_status, connected, status.value)

    def _run_ui_selftest(self):
        """Startup UI self-test — called once after all widgets settle."""
        try:
            from utils.ui_self_test import run_ui_selftest
            result = run_ui_selftest(self)
            if result.ok:
                logger.info(
                    f"UI self-test PASSED — "
                    f"{result.passed} checks OK  (see logs/ui_check.log)")
            else:
                logger.warning(
                    f"UI self-test: {result.failed} FAILED, "
                    f"{result.passed} passed  (see logs/ui_check.log)")
        except Exception:
            logger.warning("UI self-test could not run (non-fatal).")

    def _setup_shortcuts(self):
        self.bind("<space>",     lambda e: self._on_pause() if self._detecting else None)
        self.bind("<Control-s>", lambda e: self._on_screenshot())
        self.bind("<Control-r>", lambda e: self._on_record_start())
        self.bind("<Escape>",    lambda e: self._on_exit())

    def _on_exit(self):
        if messagebox.askyesno("Exit", "Exit DualVision AI Detector?"):
            logger.info("Application exit requested.")
            self._ui_running     = False
            self._worker_running = False
            self._detecting      = False
            if self._detector:
                try: self._detector.stop()
                except Exception: pass
            self._recorder.stop_all()
            try: self._rgb_stream.stop()
            except Exception: pass
            try: self._thermal_stream.stop()
            except Exception: pass
            self._settings.set("ui", "window_width",  self.winfo_width())
            self._settings.set("ui", "window_height", self.winfo_height())
            self._settings.set("ui", "window_x",      self.winfo_x())
            self._settings.set("ui", "window_y",      self.winfo_y())
            self._settings.set("ui", "maximized",     self.state() == "zoomed")
            self._settings.save()
            logger.info("Settings saved. Goodbye.")
            self.destroy()
