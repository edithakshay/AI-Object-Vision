"""
Right control panel — camera selector, GPU diagnostics, system stats,
model info, performance dashboard, detection stats, detection log.
"""
import customtkinter as ctk
import psutil
import threading
import time
from datetime import datetime


class ControlPanel(ctk.CTkFrame):
    def __init__(self, parent, on_camera_switch=None,
                 backend_manager=None, **kwargs):
        super().__init__(parent, fg_color="#0D1626",
                         corner_radius=10, border_width=1,
                         border_color="#1E3A5F", width=300, **kwargs)
        self.pack_propagate(False)
        self._on_camera_switch = on_camera_switch
        self._bm = backend_manager   # BackendManager reference (may be None)
        self._build()
        self._monitor_thread = threading.Thread(
            target=self._system_monitor, daemon=True, name="SysMonitor")
        self._monitor_thread.start()

    def _build(self):
        # ── CAMERA MODE ───────────────────────────────────────────────────────
        self._section("CAMERA MODE")
        cam_frame = ctk.CTkFrame(self, fg_color="#0A0F1E", corner_radius=8)
        cam_frame.pack(fill="x", padx=10, pady=(0, 6))

        self._cam_mode_var = ctk.StringVar(value="rgb")

        row_rgb = ctk.CTkFrame(cam_frame, fg_color="transparent")
        row_rgb.pack(fill="x", padx=8, pady=(6, 2))
        self._rb_rgb = ctk.CTkRadioButton(
            row_rgb, text="RGB Camera",
            variable=self._cam_mode_var, value="rgb",
            font=("Segoe UI", 11, "bold"), text_color="#3B82F6",
            fg_color="#2563EB", hover_color="#1D4ED8",
            border_color="#1E3A5F",
            command=lambda: self._on_cam_select("rgb"))
        self._rb_rgb.pack(side="left")

        row_th = ctk.CTkFrame(cam_frame, fg_color="transparent")
        row_th.pack(fill="x", padx=8, pady=(2, 6))
        self._rb_thermal = ctk.CTkRadioButton(
            row_th, text="Thermal Camera",
            variable=self._cam_mode_var, value="thermal",
            font=("Segoe UI", 11, "bold"), text_color="#F97316",
            fg_color="#EA580C", hover_color="#C2410C",
            border_color="#1E3A5F",
            command=lambda: self._on_cam_select("thermal"))
        self._rb_thermal.pack(side="left")

        self._cam_status_label = ctk.CTkLabel(
            cam_frame, text="Active: RGB Camera",
            font=("Segoe UI", 9), text_color="#22C55E")
        self._cam_status_label.pack(anchor="w", padx=10, pady=(0, 6))

        # ── GPU DIAGNOSTICS ───────────────────────────────────────────────────
        self._section("GPU DIAGNOSTICS")
        gpu_frame = ctk.CTkFrame(self, fg_color="#0A0F1E", corner_radius=8)
        gpu_frame.pack(fill="x", padx=10, pady=(0, 6))

        self._gpu_name_var    = ctk.StringVar(value="Detecting …")
        self._cuda_var        = ctk.StringVar(value="…")
        self._backend_var     = ctk.StringVar(value="…")
        self._ort_prov_var    = ctk.StringVar(value="…")
        self._inf_device_var  = ctk.StringVar(value="…")
        self._gpu_load_var    = ctk.StringVar(value="0%")
        self._vram_var        = ctk.StringVar(value="—")
        self._gpu_temp_var    = ctk.StringVar(value="—")
        self._stat_row(gpu_frame, "GPU",          self._gpu_name_var,   "#A78BFA")
        self._stat_row(gpu_frame, "CUDA",         self._cuda_var,       "#22C55E")
        self._stat_row(gpu_frame, "Backend",      self._backend_var,    "#3B82F6")
        self._stat_row(gpu_frame, "ORT Provider", self._ort_prov_var,   "#A78BFA")
        self._stat_row(gpu_frame, "Infer Device", self._inf_device_var, "#F97316")
        self._stat_row(gpu_frame, "GPU Load",     self._gpu_load_var,   "#F59E0B")
        self._stat_row(gpu_frame, "VRAM",         self._vram_var,       "#3B82F6")
        self._stat_row(gpu_frame, "GPU Temp",     self._gpu_temp_var,   "#EF4444")

        # ── SYSTEM ────────────────────────────────────────────────────────────
        self._section("SYSTEM")
        sys_frame = ctk.CTkFrame(self, fg_color="#0A0F1E", corner_radius=8)
        sys_frame.pack(fill="x", padx=10, pady=(0, 6))

        self._cpu_var = ctk.StringVar(value="0%")
        self._ram_var = ctk.StringVar(value="0 MB")
        self._stat_row(sys_frame, "CPU Usage", self._cpu_var, "#F59E0B")
        self._stat_row(sys_frame, "RAM Usage", self._ram_var, "#3B82F6")

        # ── MODEL ─────────────────────────────────────────────────────────────
        self._section("MODEL")
        mdl_frame = ctk.CTkFrame(self, fg_color="#0A0F1E", corner_radius=8)
        mdl_frame.pack(fill="x", padx=10, pady=(0, 6))

        self._model_var   = ctk.StringVar(value="Not loaded")
        self._device_var  = ctk.StringVar(value="—")
        self._classes_var = ctk.StringVar(value="—")
        self._inf_ms_var  = ctk.StringVar(value="0 ms")
        self._onnx_var    = ctk.StringVar(value="—")
        self._stat_row(mdl_frame, "Model",     self._model_var,   "#CBD5E1")
        self._stat_row(mdl_frame, "Device",    self._device_var,  "#22C55E")
        self._stat_row(mdl_frame, "Classes",   self._classes_var, "#94A3B8")
        self._stat_row(mdl_frame, "Inference", self._inf_ms_var,  "#F59E0B")
        self._stat_row(mdl_frame, "ONNX",      self._onnx_var,    "#A78BFA")

        # ── PERFORMANCE ───────────────────────────────────────────────────────
        self._section("PERFORMANCE")
        perf_frame = ctk.CTkFrame(self, fg_color="#0A0F1E", corner_radius=8)
        perf_frame.pack(fill="x", padx=10, pady=(0, 6))

        self._fps_var       = ctk.StringVar(value="0.0")
        self._avg_fps_var   = ctk.StringVar(value="0.0")
        self._pre_ms_var    = ctk.StringVar(value="0 ms")
        self._infer_ms_var  = ctk.StringVar(value="0 ms")
        self._post_ms_var   = ctk.StringVar(value="0 ms")
        self._threads_var   = ctk.StringVar(value="0")
        self._qsize_var     = ctk.StringVar(value="0")
        self._drops_var     = ctk.StringVar(value="0")
        self._stat_row(perf_frame, "FPS",         self._fps_var,      "#22C55E")
        self._stat_row(perf_frame, "Avg FPS",     self._avg_fps_var,  "#16A34A")
        self._stat_row(perf_frame, "Preprocess",  self._pre_ms_var,   "#64748B")
        self._stat_row(perf_frame, "Infer ms",    self._infer_ms_var, "#F59E0B")
        self._stat_row(perf_frame, "Postproc",    self._post_ms_var,  "#64748B")
        self._stat_row(perf_frame, "Threads",     self._threads_var,  "#A78BFA")
        self._stat_row(perf_frame, "Queue",       self._qsize_var,    "#64748B")
        self._stat_row(perf_frame, "Drops",       self._drops_var,    "#EF4444")

        # ── DETECTION STATS ───────────────────────────────────────────────────
        self._section("DETECTION STATS")
        det_frame = ctk.CTkFrame(self, fg_color="#0A0F1E", corner_radius=8)
        det_frame.pack(fill="x", padx=10, pady=(0, 6))

        self._active_det_var = ctk.StringVar(value="0")
        self._session_var    = ctk.StringVar(value="0")
        self._stat_row(det_frame, "Active Dets", self._active_det_var, "#3B82F6")
        self._stat_row(det_frame, "Session",     self._session_var,    "#64748B")

        # ── DETECTION LOG ─────────────────────────────────────────────────────
        self._section("DETECTION LOG")
        log_frame = ctk.CTkFrame(self, fg_color="#050A14", corner_radius=8)
        log_frame.pack(fill="both", expand=True, padx=10, pady=(0, 6))

        self._log_text = ctk.CTkTextbox(
            log_frame, fg_color="#050A14", text_color="#64748B",
            font=("Consolas", 9), wrap="none", state="disabled")
        self._log_text.pack(fill="both", expand=True, padx=4, pady=4)

        btn_row = ctk.CTkFrame(log_frame, fg_color="transparent")
        btn_row.pack(fill="x", padx=4, pady=(0, 4))
        ctk.CTkButton(btn_row, text="Clear", width=60, height=22,
                      fg_color="#1E293B", hover_color="#334155",
                      font=("Segoe UI", 9),
                      command=self._clear_log).pack(side="right")

        # ── RECORDING ─────────────────────────────────────────────────────────
        self._section("RECORDING")
        rec_frame = ctk.CTkFrame(self, fg_color="#0A0F1E", corner_radius=8)
        rec_frame.pack(fill="x", padx=10, pady=(0, 10))

        self._rec_status = ctk.CTkLabel(rec_frame, text="● Not recording",
                                        font=("Segoe UI", 10),
                                        text_color="#64748B")
        self._rec_status.pack(anchor="w", padx=10, pady=6)
        self._rec_path = ctk.CTkLabel(rec_frame, text="",
                                      font=("Consolas", 8),
                                      text_color="#334155", wraplength=260)
        self._rec_path.pack(anchor="w", padx=10)

        # Populate GPU statics immediately if BackendManager is available
        self.after(200, self._refresh_gpu_statics)

    # ── camera selector ───────────────────────────────────────────────────────
    def _on_cam_select(self, mode: str):
        label = "RGB Camera" if mode == "rgb" else "Thermal Camera"
        self._cam_status_label.configure(
            text=f"Switching to {label} …", text_color="#F59E0B")
        if self._on_camera_switch:
            self._on_camera_switch(mode)

    def set_camera_mode_label(self, mode: str):
        label = "RGB Camera" if mode == "rgb" else "Thermal Camera"
        self._cam_status_label.configure(
            text=f"Active: {label}", text_color="#22C55E")

    def set_camera_buttons_enabled(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        self._rb_rgb.configure(state=state)
        self._rb_thermal.configure(state=state)

    def set_backend_manager(self, bm):
        """Attach a BackendManager after construction."""
        self._bm = bm
        self.after(0, self._refresh_gpu_statics)

    # ── section / row helpers ─────────────────────────────────────────────────
    def _section(self, text):
        f = ctk.CTkFrame(self, fg_color="transparent")
        f.pack(fill="x", padx=10, pady=(8, 2))
        ctk.CTkLabel(f, text=text, font=("Segoe UI", 9, "bold"),
                     text_color="#2563EB").pack(side="left")
        ctk.CTkFrame(f, height=1, fg_color="#1E3A5F").pack(
            side="left", fill="x", expand=True, padx=4)

    def _stat_row(self, parent, label, var, color="#CBD5E1"):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=1)
        ctk.CTkLabel(row, text=label, font=("Segoe UI", 10),
                     text_color="#64748B", width=88, anchor="w").pack(side="left")
        ctk.CTkLabel(row, textvariable=var, font=("Segoe UI", 10, "bold"),
                     text_color=color, anchor="e").pack(side="right")

    # ── GPU static refresh (called once after BackendManager is ready) ─────────
    def _refresh_gpu_statics(self):
        if self._bm is None:
            return
        d = self._bm.get_diagnostics()
        self._gpu_name_var.set(d["gpu_name"])
        self._cuda_var.set("YES ✓" if d["cuda_available"] else "NO")
        self._backend_var.set(d["backend"])
        ort_short = (d["ort_provider"].replace("ExecutionProvider", "")
                     if d["ort_provider"] else "—")
        self._ort_prov_var.set(ort_short)
        dev = d["inference_device"]
        # Shorten long GPU name to fit
        if len(dev) > 22:
            dev = dev[:22] + "…"
        self._inf_device_var.set(dev)

    # ── public update API ─────────────────────────────────────────────────────
    def update_stats(self, fps_inf: float, rgb_det: int, thermal_det: int,
                     inf_ms: float, session_total: int,
                     model_name: str = "", device: str = "", classes: int = 0,
                     avg_fps: float = 0.0, active_threads: int = 0,
                     queue_size: int = 0, frame_drops: int = 0,
                     onnx_active: bool = False, camera_mode: str = "rgb",
                     preprocess_ms: float = 0.0, infer_ms: float = 0.0,
                     postprocess_ms: float = 0.0, ort_provider: str = ""):
        self._fps_var.set(f"{fps_inf:.1f}")
        self._avg_fps_var.set(f"{avg_fps:.1f}")
        self._pre_ms_var.set(f"{preprocess_ms:.1f} ms")
        self._infer_ms_var.set(f"{infer_ms:.1f} ms")
        self._post_ms_var.set(f"{postprocess_ms:.1f} ms")
        self._threads_var.set(str(active_threads))
        self._qsize_var.set(str(queue_size))
        self._drops_var.set(str(frame_drops))

        active_det = rgb_det if camera_mode == "rgb" else thermal_det
        self._active_det_var.set(str(active_det))
        self._session_var.set(str(session_total))
        self._inf_ms_var.set(f"{inf_ms:.0f} ms")

        if model_name:
            self._model_var.set(model_name)
        if device:
            self._device_var.set(device.upper())
        if classes:
            self._classes_var.set(str(classes))

        onnx_label = "Active" if onnx_active else "PT fallback"
        if ort_provider:
            short = ort_provider.replace("ExecutionProvider", "")
            onnx_label = f"{short}" if onnx_active else "PT fallback"
        self._onnx_var.set(onnx_label)

        # Update ORT provider in GPU section too
        if ort_provider:
            short = ort_provider.replace("ExecutionProvider", "")
            self._ort_prov_var.set(short)

    def log_detection(self, camera: str, name: str, conf: float, tid: int):
        ts   = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {camera:<8} {name:<12} {conf:.2f} ID:{tid}\n"
        try:
            self._log_text.configure(state="normal")
            self._log_text.insert("end", line)
            self._log_text.see("end")
            self._log_text.configure(state="disabled")
        except Exception:
            pass

    def _clear_log(self):
        try:
            self._log_text.configure(state="normal")
            self._log_text.delete("1.0", "end")
            self._log_text.configure(state="disabled")
        except Exception:
            pass

    def set_recording_status(self, recording: bool, path: str = ""):
        if recording:
            self._rec_status.configure(text="● Recording …", text_color="#EF4444")
            self._rec_path.configure(text=path)
        else:
            self._rec_status.configure(text="● Not recording", text_color="#64748B")
            self._rec_path.configure(text="")

    # ── background system + GPU monitor ──────────────────────────────────────
    def _system_monitor(self):
        while True:
            try:
                cpu = psutil.cpu_percent(interval=1)
                ram = psutil.virtual_memory().used // (1024 * 1024)
                self._cpu_var.set(f"{cpu:.0f}%")
                self._ram_var.set(f"{ram} MB")

                if self._bm is not None and self._bm.cuda_available:
                    m = self._bm.get_live_gpu_metrics()
                    self._gpu_load_var.set(f"{m['gpu_load_pct']:.0f}%")
                    self._vram_var.set(
                        f"{m['gpu_mem_used']} / {m['gpu_mem_total']} MB")
                    if m["gpu_temp"] > 0:
                        self._gpu_temp_var.set(f"{m['gpu_temp']:.0f} °C")
            except Exception:
                pass
            time.sleep(2)
