"""
DualVision AI Detector — Main Window
Supports Single Active Camera Mode + GPU/CUDA inference via BackendManager.
"""
import customtkinter as ctk
import threading
import time
import gc
import cv2
import logging
from datetime import datetime
from tkinter import filedialog, messagebox

from config.settings import Settings
from camera.stream import RTSPStream, StreamStatus
from ai.detector import Detector, draw_detections, DetectionResult
from ai.model_manager import ModelManager
from ai.backend_manager import BackendManager
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
VERSION = "1.2.0"

UI_FPS = 30


class MainWindow(ctk.CTk):
    def __init__(self, settings: Settings):
        super().__init__()
        self._settings = settings
        self._detecting   = False
        self._paused      = False
        self._camera_mode = "rgb"
        self._switching   = False

        self._session_detection_count = 0
        self._current_device     = "CPU"
        self._current_model_name = ""

        self._rgb_draw_result:     DetectionResult | None = None
        self._thermal_draw_result: DetectionResult | None = None
        self._rgb_last_inf_ts:     float = 0.0
        self._thermal_last_inf_ts: float = 0.0
        self._last_inf_ms:         float = 0.0

        self._rgb_display_frame     = None
        self._thermal_display_frame = None
        self._frame_lock = threading.Lock()

        self._pending_log: list[tuple] = []
        self._log_lock = threading.Lock()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # BackendManager runs GPU detection synchronously at startup
        self._backend_manager = BackendManager(
            use_gpu  = settings.get("inference", "use_gpu",  True),
            use_fp16 = settings.get("inference", "use_fp16", False),
        )
        self._backend_manager.write_startup_log("logs")

        self._init_services()
        self._build_window()
        self._build_ui()
        self._start_streams()
        self._start_worker_thread()
        self._start_ui_tick()
        self._setup_shortcuts()

        # Update GPU static info in the control panel after UI is up
        self.after(300, lambda: self._control_panel.set_backend_manager(
            self._backend_manager))

    # ──────────────────────────────────────────────────────────────────────────
    # Init
    # ──────────────────────────────────────────────────────────────────────────
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

        self._rgb_tracker     = ByteTracker(max_age=0, iou_threshold=0.35)
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
        self._ui_running     = False

    def _build_window(self):
        s = self._settings
        self.title(f"DualVision AI Detector v{VERSION}")
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

    # ── camera layout ─────────────────────────────────────────────────────────
    def _apply_camera_layout(self, mode: str, animate: bool = True):
        if mode == "rgb":
            self._thermal_panel.pack_forget()
            self._rgb_panel.pack(side="left", fill="both", expand=True,
                                 padx=(0, 3))
        else:
            self._rgb_panel.pack_forget()
            self._thermal_panel.pack(side="left", fill="both", expand=True,
                                     padx=(0, 3))

    # ── camera switch ─────────────────────────────────────────────────────────
    def _on_camera_switch_requested(self, mode: str):
        if mode == self._camera_mode or self._switching:
            return
        self._control_panel.set_camera_buttons_enabled(False)
        self._switching = True
        threading.Thread(target=self._do_camera_switch,
                         args=(mode,), daemon=True,
                         name="CamSwitch").start()

    def _do_camera_switch(self, new_mode: str):
        old_mode = self._camera_mode
        logger.info(f"Camera switch: {old_mode} → {new_mode}")

        was_detecting = self._detecting
        if was_detecting and self._detector:
            self._detector.pause()

        self._recorder.stop_all()
        self.after(0, self._control_panel.set_recording_status, False, "")

        if old_mode == "rgb":
            self._rgb_stream.stop()
            with self._frame_lock:
                self._rgb_display_frame = None
            self._rgb_draw_result  = None
            self._rgb_last_inf_ts  = 0.0
            self._rgb_tracker.reset()
            if self._detector:
                try:
                    while not self._detector._rgb_q.empty():
                        try: self._detector._rgb_q.get_nowait()
                        except Exception: break
                except Exception:
                    pass
        else:
            self._thermal_stream.stop()
            with self._frame_lock:
                self._thermal_display_frame = None
            self._thermal_draw_result  = None
            self._thermal_last_inf_ts  = 0.0
            self._thermal_tracker.reset()
            if self._detector:
                try:
                    while not self._detector._thermal_q.empty():
                        try: self._detector._thermal_q.get_nowait()
                        except Exception: break
                except Exception:
                    pass

        gc.collect()

        self._camera_mode = new_mode
        self._last_inf_ms = 0.0

        self.after(0, self._apply_camera_layout, new_mode)
        self.after(0, self._control_panel.set_camera_mode_label, new_mode)

        s = self._settings
        if new_mode == "rgb":
            self._rgb_stream = RTSPStream(
                name="RGB", url=s.get("rtsp", "rgb_url"),
                buffer_size=2,
                reconnect_delay=s.get("rtsp", "reconnect_delay", 3),
                timeout=s.get("rtsp", "timeout", 10))
            self._rgb_stream.set_status_callback(self._on_stream_status)
            self._rgb_stream.start()
        else:
            self._thermal_stream = RTSPStream(
                name="Thermal", url=s.get("rtsp", "thermal_url"),
                buffer_size=2,
                reconnect_delay=s.get("rtsp", "reconnect_delay", 3),
                timeout=s.get("rtsp", "timeout", 10))
            self._thermal_stream.set_status_callback(self._on_stream_status)
            self._thermal_stream.start()

        if was_detecting and self._detector:
            self._detector.clear_results()
            self._detector.resume()

        self._switching = False
        self.after(0, self._control_panel.set_camera_buttons_enabled, True)
        logger.info(f"Camera switch complete → {new_mode}")

    # ── streams ───────────────────────────────────────────────────────────────
    def _start_streams(self):
        if self._camera_mode == "rgb":
            self._rgb_stream.start()
        else:
            self._thermal_stream.start()

    # ── worker thread ─────────────────────────────────────────────────────────
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
            time.sleep(max(0.001, 1 / 60 - (time.perf_counter() - t0)))

    def _process_one_frame(self):
        if self._switching:
            time.sleep(0.02)
            return

        mode  = self._camera_mode
        frame = (self._rgb_stream.read() if mode == "rgb"
                 else self._thermal_stream.read())

        if not self._detecting or self._paused or self._detector is None:
            with self._frame_lock:
                if frame is not None:
                    if mode == "rgb":
                        self._rgb_display_frame = frame
                    else:
                        self._thermal_display_frame = frame
            return

        if frame is not None:
            if mode == "rgb":
                self._detector.push_rgb(frame)
            else:
                self._detector.push_thermal(frame)

        enable_tracking = self._settings.get("detection", "enable_tracking", True)

        if mode == "rgb":
            self._process_rgb_result(frame, enable_tracking)
            display = (draw_detections(frame, self._rgb_draw_result)
                       if frame is not None else None)
            if display is not None and self._recorder.is_recording("RGB"):
                self._recorder.write("RGB", display)
            with self._frame_lock:
                if display is not None:
                    self._rgb_display_frame = display
        else:
            self._process_thermal_result(frame, enable_tracking)
            display = (draw_detections(frame, self._thermal_draw_result)
                       if frame is not None else None)
            if display is not None and self._recorder.is_recording("Thermal"):
                self._recorder.write("Thermal", display)
            with self._frame_lock:
                if display is not None:
                    self._thermal_display_frame = display

    def _process_rgb_result(self, frame, enable_tracking: bool):
        result = self._detector.get_rgb_result()
        if result is None or result.timestamp == self._rgb_last_inf_ts:
            return
        self._rgb_last_inf_ts = result.timestamp
        self._last_inf_ms     = result.inference_ms
        if result.is_empty():
            self._rgb_draw_result = None
            if enable_tracking:
                self._rgb_tracker.reset()
            return
        if enable_tracking:
            dets = [{"box": b, "class_id": c, "confidence": cf}
                    for b, c, cf in zip(result.boxes, result.class_ids,
                                        result.confidences)]
            tracked = self._rgb_tracker.update(dets)
            result.boxes       = [t["box"]       for t in tracked]
            result.class_ids   = [t["class_id"]  for t in tracked]
            result.confidences = [t["confidence"] for t in tracked]
            result.class_names = [
                (self._detector.class_names[t["class_id"]]
                 if t["class_id"] < len(self._detector.class_names) else "?")
                for t in tracked]
            result.track_ids = [t["track_id"] for t in tracked]
        self._rgb_draw_result = result
        self._session_detection_count += len(result.boxes)
        for i in range(len(result.boxes)):
            name = result.class_names[i]  if i < len(result.class_names)  else ""
            conf = result.confidences[i]  if i < len(result.confidences)  else 0.0
            tid  = result.track_ids[i]    if i < len(result.track_ids)    else 0
            box  = result.boxes[i]        if i < len(result.boxes)        else [0,0,0,0]
            self._det_log.log("RGB", name, conf, tid, box)
            with self._log_lock:
                self._pending_log.append(("RGB", name, conf, tid))

    def _process_thermal_result(self, frame, enable_tracking: bool):
        result = self._detector.get_thermal_result()
        if result is None or result.timestamp == self._thermal_last_inf_ts:
            return
        self._thermal_last_inf_ts = result.timestamp
        self._last_inf_ms         = result.inference_ms
        if result.is_empty():
            self._thermal_draw_result = None
            if enable_tracking:
                self._thermal_tracker.reset()
            return
        if enable_tracking:
            dets = [{"box": b, "class_id": c, "confidence": cf}
                    for b, c, cf in zip(result.boxes, result.class_ids,
                                        result.confidences)]
            tracked = self._thermal_tracker.update(dets)
            result.boxes       = [t["box"]       for t in tracked]
            result.class_ids   = [t["class_id"]  for t in tracked]
            result.confidences = [t["confidence"] for t in tracked]
            result.class_names = [
                (self._detector.class_names[t["class_id"]]
                 if t["class_id"] < len(self._detector.class_names) else "?")
                for t in tracked]
            result.track_ids = [t["track_id"] for t in tracked]
        self._thermal_draw_result = result
        self._session_detection_count += len(result.boxes)
        for i in range(len(result.boxes)):
            name = result.class_names[i]  if i < len(result.class_names)  else ""
            conf = result.confidences[i]  if i < len(result.confidences)  else 0.0
            tid  = result.track_ids[i]    if i < len(result.track_ids)    else 0
            box  = result.boxes[i]        if i < len(result.boxes)        else [0,0,0,0]
            self._det_log.log("Thermal", name, conf, tid, box)
            with self._log_lock:
                self._pending_log.append(("Thermal", name, conf, tid))

    # ── UI tick ───────────────────────────────────────────────────────────────
    def _start_ui_tick(self):
        self._ui_running = True
        self._schedule_tick()

    def _schedule_tick(self):
        if self._ui_running:
            self.after(max(1, int(1000 / UI_FPS)), self._ui_tick)

    def _ui_tick(self):
        try:
            self._do_ui_tick()
        except Exception as e:
            logger.error(f"UI tick error: {e}")
        finally:
            self._schedule_tick()

    def _do_ui_tick(self):
        mode = self._camera_mode
        det  = self._detector

        with self._frame_lock:
            rgb_frame     = self._rgb_display_frame
            thermal_frame = self._thermal_display_frame

        fps_inf = det.fps_inference if det else 0.0
        inf_ms  = self._last_inf_ms

        rgb_det_count     = len(self._rgb_draw_result.boxes)     if self._rgb_draw_result     else 0
        thermal_det_count = len(self._thermal_draw_result.boxes) if self._thermal_draw_result else 0

        # Update only the visible panel
        if mode == "rgb" and rgb_frame is not None:
            fps_d = self._rgb_stream.fps_actual
            self._rgb_panel.update_frame(rgb_frame, fps_d, rgb_det_count, inf_ms)
        elif mode == "thermal" and thermal_frame is not None:
            fps_d = self._thermal_stream.fps_actual
            self._thermal_panel.update_frame(thermal_frame, fps_d,
                                             thermal_det_count, inf_ms)

        model_name = self._current_model_name
        device     = self._current_device
        cls_count  = len(det.class_names) if det else 0

        self._control_panel.update_stats(
            fps_inf, rgb_det_count, thermal_det_count,
            inf_ms, self._session_detection_count,
            model_name, device, cls_count,
            avg_fps        = det.avg_fps        if det else 0.0,
            active_threads = det.active_threads if det else 0,
            queue_size     = det.queue_size     if det else 0,
            frame_drops    = det.frame_drops    if det else 0,
            onnx_active    = det.onnx_active    if det else False,
            camera_mode    = mode,
            preprocess_ms  = det.preprocess_ms  if det else 0.0,
            infer_ms       = det.infer_ms       if det else 0.0,
            postprocess_ms = det.postprocess_ms if det else 0.0,
            ort_provider   = det.ort_provider   if det else "",
        )

        with self._log_lock:
            log_batch = self._pending_log[:20]
            self._pending_log = self._pending_log[20:]
        for camera, name, conf, tid in log_batch:
            self._control_panel.log_detection(camera, name, conf, tid)

        if mode == "rgb":
            active_status = self._rgb_stream.status.value
            self._statusbar.update(
                fps=fps_inf, model=model_name, device=device,
                detecting=self._detecting,
                rgb_status=active_status, thermal_status="Inactive",
                timestamp=datetime.now().strftime("%H:%M:%S"))
        else:
            active_status = self._thermal_stream.status.value
            self._statusbar.update(
                fps=fps_inf, model=model_name, device=device,
                detecting=self._detecting,
                rgb_status="Inactive", thermal_status=active_status,
                timestamp=datetime.now().strftime("%H:%M:%S"))

    # ── Actions ───────────────────────────────────────────────────────────────
    def _on_start(self):
        if self._detecting:
            return
        s          = self._settings
        model_name = s.get("inference", "model_name", "yolo26n.pt")
        use_gpu    = s.get("inference", "use_gpu",    True)
        use_fp16   = s.get("inference", "use_fp16",   False)
        conf       = s.get("detection", "confidence",  0.45)
        iou_val    = s.get("detection", "iou",         0.45)
        input_size = s.get("detection", "input_width", 640)
        frame_skip = s.get("detection", "frame_skip",  1)
        cpu_th     = s.get("inference", "cpu_threads", 0)
        self._current_model_name = model_name

        # Rebuild BackendManager if GPU preference changed
        bm = self._backend_manager
        if bm.use_gpu != use_gpu or bm.use_fp16 != use_fp16:
            bm = BackendManager(use_gpu=use_gpu, use_fp16=use_fp16)
            bm.write_startup_log("logs")
            self._backend_manager = bm
            self.after(0, lambda: self._control_panel.set_backend_manager(bm))

        def _load_and_start():
            model_path = self._model_manager.get_model_path(model_name)
            if not self._model_manager.is_downloaded(model_name):
                self.after(0, messagebox.showinfo, "Downloading Model",
                           f"Model '{model_name}' not found locally.\n\n"
                           "Ultralytics will auto-download it (~6 MB).\n"
                           "Internet required only this once.\n\n"
                           "Click OK and wait — detection will start shortly.")

            # Determine ONNX path
            use_onnx   = s.get("inference", "use_onnx", True)
            onnx_path  = ""
            if use_onnx:
                onnx_p = self._model_manager.get_onnx_path(model_name)
                if onnx_p and onnx_p.exists() and onnx_p.stat().st_size > 100_000:
                    onnx_path = str(onnx_p)
                else:
                    # Auto-export
                    exported = self._model_manager.export_onnx(model_name)
                    if exported:
                        onnx_path = str(exported)

            detector = Detector(
                model_path=str(model_path),
                conf=conf, iou=iou_val, use_gpu=use_gpu,
                input_size=input_size, frame_skip=frame_skip,
                cpu_threads=cpu_th, use_fp16=use_fp16,
                onnx_path=onnx_path,
                backend_manager=self._backend_manager,
            )

            if not detector.load():
                self.after(0, messagebox.showerror, "Model Error",
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

            self._detector    = detector
            self._rgb_tracker.reset()
            self._thermal_tracker.reset()
            self._rgb_draw_result     = None
            self._thermal_draw_result = None
            self._rgb_last_inf_ts     = 0.0
            self._thermal_last_inf_ts = 0.0
            self._last_inf_ms         = 0.0
            self._session_detection_count = 0
            self._detecting = True
            self._paused    = False
            self._detector.start()
            logger.info(f"Detection started — model={model_name}  "
                        f"backend={detector.backend}  "
                        f"device={self._current_device}  "
                        f"camera={self._camera_mode}")

        threading.Thread(target=_load_and_start, daemon=True,
                         name="ModelLoader").start()

    def _on_stop(self):
        self._detecting = False
        self._paused    = False
        if self._detector:
            self._detector.stop()
            self._detector.clear_results()
        self._rgb_tracker.reset()
        self._thermal_tracker.reset()
        self._rgb_draw_result     = None
        self._thermal_draw_result = None
        self._rgb_last_inf_ts     = 0.0
        self._thermal_last_inf_ts = 0.0
        self._last_inf_ms         = 0.0
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

    def _on_resume(self):
        if self._detecting and self._paused:
            self._paused = False
            if self._detector:
                self._detector.resume()
            if self._camera_mode == "rgb":
                self._rgb_stream.resume()
            else:
                self._thermal_stream.resume()

    def _on_screenshot(self):
        with self._frame_lock:
            rgb_f   = self._rgb_display_frame
            therm_f = self._thermal_display_frame
        if self._camera_mode == "rgb":
            paths = self._screenshot_util.save_both(
                rgb_f.copy() if rgb_f is not None else None, None)
        else:
            paths = self._screenshot_util.save_both(
                None, therm_f.copy() if therm_f is not None else None)
        if paths:
            messagebox.showinfo("Screenshot Saved", "Saved:\n" + "\n".join(paths))
        else:
            messagebox.showwarning("Screenshot", "No frame available yet.")

    def _on_record_start(self):
        with self._frame_lock:
            rgb_f   = self._rgb_display_frame
            therm_f = self._thermal_display_frame
        started = []
        if self._camera_mode == "rgb" and rgb_f is not None:
            h, w = rgb_f.shape[:2]
            p = self._recorder.start("RGB", w, h)
            if p: started.append(p)
        elif self._camera_mode == "thermal" and therm_f is not None:
            h, w = therm_f.shape[:2]
            p = self._recorder.start("Thermal", w, h)
            if p: started.append(p)
        if started:
            self._control_panel.set_recording_status(True, "\n".join(started))
        else:
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
                    input_size=self._settings.get("detection", "input_width", 640))
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
        connected   = status == StreamStatus.CONNECTED
        status_text = status.value
        if name == "RGB" and self._camera_mode == "rgb":
            self.after(0, self._rgb_panel.set_status, connected, status_text)
        elif name == "Thermal" and self._camera_mode == "thermal":
            self.after(0, self._thermal_panel.set_status, connected, status_text)

    def _setup_shortcuts(self):
        self.bind("<space>",     lambda e: self._on_pause() if self._detecting else None)
        self.bind("<Control-s>", lambda e: self._on_screenshot())
        self.bind("<Control-r>", lambda e: self._on_record_start())
        self.bind("<Escape>",    lambda e: self._on_exit())

    def _on_exit(self):
        if messagebox.askyesno("Exit", "Exit DualVision AI Detector?"):
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
            self.destroy()
