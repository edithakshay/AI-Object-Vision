import customtkinter as ctk
import threading
import time
import cv2
import logging
from datetime import datetime
from tkinter import filedialog, messagebox

from config.settings import Settings
from camera.stream import RTSPStream, StreamStatus
from ai.detector import Detector, draw_detections, DetectionResult
from ai.model_manager import ModelManager
from tracking.tracker import ByteTracker
from utils.screenshot import ScreenshotUtil
from utils.recorder import VideoRecorder
from utils.detection_log import DetectionLog

from ui.toolbar import Toolbar
from ui.statusbar import StatusBar
from ui.camera_panel import CameraPanel
from ui.control_panel import ControlPanel
from ui.settings_dialog import SettingsDialog
from ui.about_dialog import AboutDialog

logger = logging.getLogger("DualVisionAI.mainwindow")
VERSION = "1.0.0"

# How old (seconds) a result can be before bounding boxes are hidden.
# Keep short so stale boxes from slow inference disappear quickly.
RESULT_MAX_AGE = 0.8

# UI refresh rate — independent of inference speed
UI_FPS = 30


class MainWindow(ctk.CTk):
    def __init__(self, settings: Settings):
        super().__init__()
        self._settings = settings
        self._detecting = False
        self._paused = False
        self._session_detection_count = 0
        self._current_device = "CPU"
        self._current_model_name = ""

        # Last result snapshot used for drawing — updated from inference thread
        self._rgb_draw_result: DetectionResult | None = None
        self._thermal_draw_result: DetectionResult | None = None
        self._result_swap_lock = threading.Lock()

        # Buffered frames for display (latest only)
        self._rgb_display_frame = None
        self._thermal_display_frame = None
        self._frame_lock = threading.Lock()

        # Pending log entries accumulated between UI ticks
        self._pending_log: list[tuple] = []
        self._log_lock = threading.Lock()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self._init_services()
        self._build_window()
        self._build_ui()
        self._start_streams()
        self._start_worker_thread()
        self._start_ui_tick()
        self._setup_shortcuts()

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def _init_services(self):
        s = self._settings
        self._rgb_stream = RTSPStream(
            name="RGB",
            url=s.get("rtsp", "rgb_url"),
            buffer_size=2,
            reconnect_delay=s.get("rtsp", "reconnect_delay", 3),
            timeout=s.get("rtsp", "timeout", 10)
        )
        self._thermal_stream = RTSPStream(
            name="Thermal",
            url=s.get("rtsp", "thermal_url"),
            buffer_size=2,
            reconnect_delay=s.get("rtsp", "reconnect_delay", 3),
            timeout=s.get("rtsp", "timeout", 10)
        )
        self._rgb_stream.set_status_callback(self._on_stream_status)
        self._thermal_stream.set_status_callback(self._on_stream_status)

        self._model_manager = ModelManager(model_dir="models")
        self._detector: Detector | None = None
        # max_age=0 → track is REMOVED immediately if not matched in next
        # inference cycle. This is the key fix for ghost bounding boxes:
        # the moment YOLO stops seeing an object, its box disappears.
        self._rgb_tracker = ByteTracker(max_age=0, iou_threshold=0.35)
        self._thermal_tracker = ByteTracker(max_age=0, iou_threshold=0.35)

        self._screenshot_util = ScreenshotUtil(
            output_dir=s.get("screenshots", "output_dir", "screenshots"))
        self._recorder = VideoRecorder(
            output_dir=s.get("recording", "output_dir", "recordings"),
            fps=s.get("recording", "fps", 25))
        self._det_log = DetectionLog(
            output_dir=s.get("logging", "output_dir", "logs"),
            max_entries=s.get("logging", "max_log_entries", 100000))

        self._worker_running = False
        self._ui_running = False

    def _build_window(self):
        s = self._settings
        self.title(f"DualVision AI Detector v{VERSION}")
        w = s.get("ui", "window_width", 1600)
        h = s.get("ui", "window_height", 950)
        x = s.get("ui", "window_x", 100)
        y = s.get("ui", "window_y", 100)
        self.geometry(f"{w}x{h}+{x}+{y}")
        self.minsize(1100, 700)
        if s.get("ui", "maximized", False):
            self.state("zoomed")
        self.configure(fg_color="#050A14")
        self.protocol("WM_DELETE_WINDOW", self._on_exit)

    def _build_ui(self):
        callbacks = {
            "start": self._on_start,
            "stop": self._on_stop,
            "pause": self._on_pause,
            "resume": self._on_resume,
            "screenshot": self._on_screenshot,
            "record_start": self._on_record_start,
            "record_stop": self._on_record_stop,
            "settings": self._on_settings,
            "export_csv": self._on_export_csv,
            "export_json": self._on_export_json,
            "about": self._on_about,
            "exit": self._on_exit,
        }
        self._toolbar = Toolbar(self, callbacks=callbacks)
        self._toolbar.pack(fill="x", side="top")

        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=6, pady=4)

        self._rgb_panel = CameraPanel(content, title="RGB Camera")
        self._rgb_panel.pack(side="left", fill="both", expand=True, padx=(0, 3))

        self._thermal_panel = CameraPanel(content, title="Thermal Camera")
        self._thermal_panel.pack(side="left", fill="both", expand=True, padx=(3, 3))

        self._control_panel = ControlPanel(content)
        self._control_panel.pack(side="left", fill="y", padx=(3, 0))

        self._statusbar = StatusBar(self)
        self._statusbar.pack(fill="x", side="bottom")

    def _start_streams(self):
        self._rgb_stream.start()
        self._thermal_stream.start()

    def _setup_shortcuts(self):
        self.bind("<space>", lambda e: self._on_pause() if self._detecting else None)
        self.bind("<Control-s>", lambda e: self._on_screenshot())
        self.bind("<Control-r>", lambda e: self._on_record_start())
        self.bind("<Escape>", lambda e: self._on_exit())

    # ------------------------------------------------------------------
    # Worker thread — reads frames, applies tracking, prepares display data
    # Runs at camera FPS; completely decoupled from Tkinter
    # ------------------------------------------------------------------
    def _start_worker_thread(self):
        self._worker_running = True
        threading.Thread(target=self._worker_loop,
                         daemon=True, name="Worker").start()

    def _worker_loop(self):
        while self._worker_running:
            t0 = time.perf_counter()
            try:
                self._process_one_frame()
            except Exception as e:
                logger.error(f"Worker error: {e}")
            # Run at up to 60 Hz; don't busy-spin
            elapsed = time.perf_counter() - t0
            sleep = max(0.001, 1 / 60 - elapsed)
            time.sleep(sleep)

    def _process_one_frame(self):
        rgb_frame = self._rgb_stream.read()
        thermal_frame = self._thermal_stream.read()

        if not self._detecting or self._paused or self._detector is None:
            # Store raw frames for display even when not detecting
            with self._frame_lock:
                if rgb_frame is not None:
                    self._rgb_display_frame = rgb_frame
                if thermal_frame is not None:
                    self._thermal_display_frame = thermal_frame
            return

        # Push raw frames to inference thread
        if rgb_frame is not None:
            self._detector.push_rgb(rgb_frame)
        if thermal_frame is not None:
            self._detector.push_thermal(thermal_frame)

        enable_tracking = self._settings.get("detection", "enable_tracking", True)

        # --- RGB result ---
        rgb_result = self._detector.get_rgb_result()
        rgb_display = rgb_frame  # default: raw frame

        if rgb_result is not None and rgb_result.is_fresh(RESULT_MAX_AGE):
            if not rgb_result.is_empty() and enable_tracking:
                dets = [{"box": b, "class_id": c, "confidence": cf}
                        for b, c, cf in zip(rgb_result.boxes,
                                            rgb_result.class_ids,
                                            rgb_result.confidences)]
                tracked = self._rgb_tracker.update(dets)
                rgb_result.boxes = [t["box"] for t in tracked]
                rgb_result.class_ids = [t["class_id"] for t in tracked]
                rgb_result.confidences = [t["confidence"] for t in tracked]
                rgb_result.class_names = [
                    (self._detector.class_names[t["class_id"]]
                     if t["class_id"] < len(self._detector.class_names) else "?")
                    for t in tracked]
                rgb_result.track_ids = [t["track_id"] for t in tracked]

            if rgb_frame is not None:
                rgb_display = draw_detections(rgb_frame, rgb_result)

            # Log new detections (batched, won't spam the UI thread)
            for i in range(len(rgb_result.boxes)):
                name = rgb_result.class_names[i] if i < len(rgb_result.class_names) else ""
                conf = rgb_result.confidences[i] if i < len(rgb_result.confidences) else 0
                tid = rgb_result.track_ids[i] if i < len(rgb_result.track_ids) else 0
                box = rgb_result.boxes[i] if i < len(rgb_result.boxes) else [0, 0, 0, 0]
                self._det_log.log("RGB", name, conf, tid, box)
                with self._log_lock:
                    self._pending_log.append(("RGB", name, conf, tid))
        else:
            # Stale or no result — draw raw frame (no ghost boxes)
            if rgb_frame is not None:
                rgb_display = rgb_frame
            if enable_tracking:
                self._rgb_tracker.reset()

        # --- Thermal result ---
        thermal_result = self._detector.get_thermal_result()
        thermal_display = thermal_frame

        if thermal_result is not None and thermal_result.is_fresh(RESULT_MAX_AGE):
            if not thermal_result.is_empty() and enable_tracking:
                dets = [{"box": b, "class_id": c, "confidence": cf}
                        for b, c, cf in zip(thermal_result.boxes,
                                            thermal_result.class_ids,
                                            thermal_result.confidences)]
                tracked = self._thermal_tracker.update(dets)
                thermal_result.boxes = [t["box"] for t in tracked]
                thermal_result.class_ids = [t["class_id"] for t in tracked]
                thermal_result.confidences = [t["confidence"] for t in tracked]
                thermal_result.class_names = [
                    (self._detector.class_names[t["class_id"]]
                     if t["class_id"] < len(self._detector.class_names) else "?")
                    for t in tracked]
                thermal_result.track_ids = [t["track_id"] for t in tracked]

            if thermal_frame is not None:
                thermal_display = draw_detections(thermal_frame, thermal_result)
        else:
            if thermal_frame is not None:
                thermal_display = thermal_frame
            if enable_tracking:
                self._thermal_tracker.reset()

        # Update session count only when result is new (use inference timestamp)
        if rgb_result and rgb_result.is_fresh(0.15):
            self._session_detection_count += len(rgb_result.boxes)
        if thermal_result and thermal_result.is_fresh(0.15):
            self._session_detection_count += len(thermal_result.boxes)

        # Write to recorder
        if rgb_display is not None and self._recorder.is_recording("RGB"):
            self._recorder.write("RGB", rgb_display)
        if thermal_display is not None and self._recorder.is_recording("Thermal"):
            self._recorder.write("Thermal", thermal_display)

        # Swap in display frames atomically
        with self._frame_lock:
            if rgb_display is not None:
                self._rgb_display_frame = rgb_display
            if thermal_display is not None:
                self._thermal_display_frame = thermal_display

        with self._result_swap_lock:
            if rgb_result is not None:
                self._rgb_draw_result = rgb_result
            if thermal_result is not None:
                self._thermal_draw_result = thermal_result

    # ------------------------------------------------------------------
    # UI tick — runs on Tkinter's main thread via after()
    # Single scheduled call per tick; updates all panels at once
    # ------------------------------------------------------------------
    def _start_ui_tick(self):
        self._ui_running = True
        self._schedule_tick()

    def _schedule_tick(self):
        if self._ui_running:
            interval_ms = max(1, int(1000 / UI_FPS))
            self.after(interval_ms, self._ui_tick)

    def _ui_tick(self):
        try:
            self._do_ui_tick()
        except Exception as e:
            logger.error(f"UI tick error: {e}")
        finally:
            self._schedule_tick()

    def _do_ui_tick(self):
        with self._frame_lock:
            rgb_frame = self._rgb_display_frame
            thermal_frame = self._thermal_display_frame

        with self._result_swap_lock:
            rgb_result = self._rgb_draw_result
            thermal_result = self._thermal_draw_result

        det = self._detector
        fps_inf = det.fps_inference if det else 0.0
        inf_ms = 0.0
        rgb_det_count = 0
        thermal_det_count = 0

        if rgb_result and rgb_result.is_fresh(RESULT_MAX_AGE):
            rgb_det_count = len(rgb_result.boxes)
            inf_ms = rgb_result.inference_ms
        if thermal_result and thermal_result.is_fresh(RESULT_MAX_AGE):
            thermal_det_count = len(thermal_result.boxes)
            if inf_ms == 0:
                inf_ms = thermal_result.inference_ms

        # Camera panels
        fps_rgb = self._rgb_stream.fps_actual
        fps_th = self._thermal_stream.fps_actual
        if rgb_frame is not None:
            self._rgb_panel.update_frame(rgb_frame, fps_rgb, rgb_det_count, inf_ms)
        if thermal_frame is not None:
            self._thermal_panel.update_frame(thermal_frame, fps_th, thermal_det_count, inf_ms)

        # Control panel stats (one call)
        model_name = self._current_model_name
        device = self._current_device
        cls_count = len(det.class_names) if det else 0
        self._control_panel.update_stats(
            fps_inf, rgb_det_count, thermal_det_count,
            inf_ms, self._session_detection_count,
            model_name, device, cls_count)

        # Flush pending detection log entries (batched — at most once per tick)
        with self._log_lock:
            log_batch = self._pending_log[:20]  # max 20 entries per tick
            self._pending_log = self._pending_log[20:]
        for camera, name, conf, tid in log_batch:
            self._control_panel.log_detection(camera, name, conf, tid)

        # Status bar (one call)
        rgb_status = self._rgb_stream.status.value
        thermal_status = self._thermal_stream.status.value
        now = datetime.now().strftime("%H:%M:%S")
        self._statusbar.update(
            fps=fps_inf,
            model=model_name,
            device=device,
            detecting=self._detecting,
            rgb_status=rgb_status,
            thermal_status=thermal_status,
            timestamp=now)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _on_start(self):
        if self._detecting:
            return
        s = self._settings
        model_name = s.get("inference", "model_name", "yolov8n.pt")
        use_gpu = s.get("inference", "use_gpu", True)
        conf = s.get("detection", "confidence", 0.45)
        iou_val = s.get("detection", "iou", 0.45)
        input_size = s.get("detection", "input_width", 640)
        frame_skip = s.get("detection", "frame_skip", 1)
        self._current_model_name = model_name

        def _load_and_start():
            model_path = self._model_manager.get_model_path(model_name)
            if not self._model_manager.is_downloaded(model_name):
                self.after(0, messagebox.showinfo,
                           "Downloading Model",
                           f"Model '{model_name}' not found locally.\n\n"
                           "Ultralytics will auto-download it (~5 MB).\n"
                           "Internet required only this once.\n\n"
                           "Click OK and wait — the app will start detecting shortly.")

            detector = Detector(
                model_path=str(model_path),
                conf=conf,
                iou=iou_val,
                use_gpu=use_gpu,
                input_size=input_size,
                frame_skip=frame_skip
            )

            if not detector.load():
                self.after(0, messagebox.showerror,
                           "Model Error",
                           f"Failed to load model: {model_name}\n\n"
                           "Run  python setup.py  first to download the model,\n"
                           "or check your YOLO / ultralytics installation.")
                return

            self._current_device = detector.device

            if self._detector is not None:
                try:
                    self._detector.stop()
                    self._detector.clear_results()
                except Exception:
                    pass

            self._detector = detector
            self._rgb_tracker.reset()
            self._thermal_tracker.reset()
            with self._result_swap_lock:
                self._rgb_draw_result = None
                self._thermal_draw_result = None
            self._session_detection_count = 0
            self._detecting = True
            self._paused = False
            self._detector.start()
            logger.info(f"Detection started — model={model_name} device={self._current_device}")

        threading.Thread(target=_load_and_start, daemon=True, name="ModelLoader").start()

    def _on_stop(self):
        self._detecting = False
        self._paused = False
        if self._detector:
            self._detector.stop()
            self._detector.clear_results()
        self._rgb_tracker.reset()
        self._thermal_tracker.reset()
        with self._result_swap_lock:
            self._rgb_draw_result = None
            self._thermal_draw_result = None
        self._recorder.stop_all()
        self._control_panel.set_recording_status(False, "")
        logger.info("Detection stopped.")

    def _on_pause(self):
        if self._detecting and not self._paused:
            self._paused = True
            if self._detector:
                self._detector.pause()
                self._detector.clear_results()
            self._rgb_stream.pause()
            self._thermal_stream.pause()

    def _on_resume(self):
        if self._detecting and self._paused:
            self._paused = False
            if self._detector:
                self._detector.resume()
            self._rgb_stream.resume()
            self._thermal_stream.resume()

    def _on_screenshot(self):
        with self._frame_lock:
            rgb = self._rgb_display_frame
            thermal = self._thermal_display_frame
        paths = self._screenshot_util.save_both(
            rgb.copy() if rgb is not None else None,
            thermal.copy() if thermal is not None else None)
        if paths:
            messagebox.showinfo("Screenshot Saved", "Saved:\n" + "\n".join(paths))
        else:
            messagebox.showwarning("Screenshot", "No frames available yet.")

    def _on_record_start(self):
        with self._frame_lock:
            rgb = self._rgb_display_frame
            thermal = self._thermal_display_frame
        started = []
        if rgb is not None:
            h, w = rgb.shape[:2]
            p = self._recorder.start("RGB", w, h)
            if p:
                started.append(p)
        if thermal is not None:
            h, w = thermal.shape[:2]
            p = self._recorder.start("Thermal", w, h)
            if p:
                started.append(p)
        if started:
            self._control_panel.set_recording_status(True, "\n".join(started))
        else:
            messagebox.showwarning("Recording", "No frames available to record.")

    def _on_record_stop(self):
        self._recorder.stop_all()
        self._control_panel.set_recording_status(False, "")

    def _on_settings(self):
        def on_save():
            if self._detector:
                self._detector.update_params(
                    conf=self._settings.get("detection", "confidence", 0.45),
                    iou=self._settings.get("detection", "iou", 0.45),
                    frame_skip=self._settings.get("detection", "frame_skip", 1),
                    input_size=self._settings.get("detection", "input_width", 640))
        SettingsDialog(self, self._settings, on_save=on_save)

    def _on_export_csv(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")],
            title="Export Detection Log as CSV")
        if path:
            try:
                self._det_log.export_csv(path)
                messagebox.showinfo("Export", f"CSV exported:\n{path}")
            except Exception as e:
                messagebox.showerror("Export Error", str(e))

    def _on_export_json(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
            title="Export Detection Log as JSON")
        if path:
            try:
                self._det_log.export_json(path)
                messagebox.showinfo("Export", f"JSON exported:\n{path}")
            except Exception as e:
                messagebox.showerror("Export Error", str(e))

    def _on_about(self):
        AboutDialog(self)

    def _on_stream_status(self, name: str, status: StreamStatus):
        connected = status == StreamStatus.CONNECTED
        status_text = status.value
        if name == "RGB":
            self.after(0, self._rgb_panel.set_status, connected, status_text)
        elif name == "Thermal":
            self.after(0, self._thermal_panel.set_status, connected, status_text)

    def _on_exit(self):
        if messagebox.askyesno("Exit", "Exit DualVision AI Detector?"):
            self._ui_running = False
            self._worker_running = False
            self._detecting = False
            if self._detector:
                try:
                    self._detector.stop()
                except Exception:
                    pass
            self._recorder.stop_all()
            self._rgb_stream.stop()
            self._thermal_stream.stop()
            self._settings.set("ui", "window_width", self.winfo_width())
            self._settings.set("ui", "window_height", self.winfo_height())
            self._settings.set("ui", "window_x", self.winfo_x())
            self._settings.set("ui", "window_y", self.winfo_y())
            self._settings.set("ui", "maximized", self.state() == "zoomed")
            self._settings.save()
            self.destroy()
