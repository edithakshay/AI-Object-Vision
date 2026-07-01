"""
Right control panel — System stats, Model info, Performance Dashboard,
Detection stats, Detection log, Recording status.
"""
import customtkinter as ctk
import psutil
import threading
import time
from datetime import datetime


class ControlPanel(ctk.CTkFrame):
    """Right panel: live metrics, model info, performance dashboard, detection log."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color="#0D1626",
                         corner_radius=10, border_width=1,
                         border_color="#1E3A5F", width=290, **kwargs)
        self.pack_propagate(False)
        self._build()
        self._monitor_thread = threading.Thread(target=self._system_monitor,
                                                daemon=True, name="SysMonitor")
        self._monitor_thread.start()

    def _build(self):
        # ── SYSTEM ──────────────────────────────────────────────────────────
        self._section("SYSTEM")
        sys_frame = ctk.CTkFrame(self, fg_color="#0A0F1E", corner_radius=8)
        sys_frame.pack(fill="x", padx=10, pady=(0, 6))

        self._cpu_var = ctk.StringVar(value="0%")
        self._ram_var = ctk.StringVar(value="0 MB")
        self._gpu_var = ctk.StringVar(value="N/A")
        self._stat_row(sys_frame, "CPU Usage",  self._cpu_var, "#F59E0B")
        self._stat_row(sys_frame, "RAM Usage",  self._ram_var, "#3B82F6")
        self._stat_row(sys_frame, "GPU Status", self._gpu_var, "#A78BFA")

        # ── MODEL ────────────────────────────────────────────────────────────
        self._section("MODEL")
        mdl_frame = ctk.CTkFrame(self, fg_color="#0A0F1E", corner_radius=8)
        mdl_frame.pack(fill="x", padx=10, pady=(0, 6))

        self._model_var    = ctk.StringVar(value="Not loaded")
        self._device_var   = ctk.StringVar(value="CPU")
        self._classes_var  = ctk.StringVar(value="—")
        self._inf_ms_var   = ctk.StringVar(value="0 ms")
        self._onnx_var     = ctk.StringVar(value="—")
        self._stat_row(mdl_frame, "Model",     self._model_var,   "#CBD5E1")
        self._stat_row(mdl_frame, "Device",    self._device_var,  "#22C55E")
        self._stat_row(mdl_frame, "Classes",   self._classes_var, "#94A3B8")
        self._stat_row(mdl_frame, "Inference", self._inf_ms_var,  "#F59E0B")
        self._stat_row(mdl_frame, "ONNX",      self._onnx_var,    "#A78BFA")

        # ── PERFORMANCE DASHBOARD ────────────────────────────────────────────
        self._section("PERFORMANCE")
        perf_frame = ctk.CTkFrame(self, fg_color="#0A0F1E", corner_radius=8)
        perf_frame.pack(fill="x", padx=10, pady=(0, 6))

        self._fps_var     = ctk.StringVar(value="0.0")
        self._avg_fps_var = ctk.StringVar(value="0.0")
        self._fps_rgb_var = ctk.StringVar(value="0.0")
        self._fps_th_var  = ctk.StringVar(value="0.0")
        self._threads_var = ctk.StringVar(value="0")
        self._qsize_var   = ctk.StringVar(value="0")
        self._drops_var   = ctk.StringVar(value="0")
        self._stat_row(perf_frame, "FPS (Inf)",     self._fps_var,     "#22C55E")
        self._stat_row(perf_frame, "Avg FPS",       self._avg_fps_var, "#16A34A")
        self._stat_row(perf_frame, "RGB FPS",       self._fps_rgb_var, "#3B82F6")
        self._stat_row(perf_frame, "Thermal FPS",   self._fps_th_var,  "#F97316")
        self._stat_row(perf_frame, "Threads",       self._threads_var, "#A78BFA")
        self._stat_row(perf_frame, "Queue",         self._qsize_var,   "#64748B")
        self._stat_row(perf_frame, "Frame Drops",   self._drops_var,   "#EF4444")

        # ── DETECTION STATS ──────────────────────────────────────────────────
        self._section("DETECTION STATS")
        det_frame = ctk.CTkFrame(self, fg_color="#0A0F1E", corner_radius=8)
        det_frame.pack(fill="x", padx=10, pady=(0, 6))

        self._rgb_det_var     = ctk.StringVar(value="0")
        self._thermal_det_var = ctk.StringVar(value="0")
        self._total_var       = ctk.StringVar(value="0")
        self._session_var     = ctk.StringVar(value="0")
        self._stat_row(det_frame, "RGB Dets",     self._rgb_det_var,     "#3B82F6")
        self._stat_row(det_frame, "Thermal Dets", self._thermal_det_var, "#F97316")
        self._stat_row(det_frame, "Current",      self._total_var,       "#E2E8F0")
        self._stat_row(det_frame, "Session",      self._session_var,     "#64748B")

        # ── DETECTION LOG ────────────────────────────────────────────────────
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

        # ── RECORDING ────────────────────────────────────────────────────────
        self._section("RECORDING")
        rec_frame = ctk.CTkFrame(self, fg_color="#0A0F1E", corner_radius=8)
        rec_frame.pack(fill="x", padx=10, pady=(0, 10))

        self._rec_status = ctk.CTkLabel(rec_frame, text="● Not recording",
                                        font=("Segoe UI", 10),
                                        text_color="#64748B")
        self._rec_status.pack(anchor="w", padx=10, pady=6)
        self._rec_path = ctk.CTkLabel(rec_frame, text="",
                                      font=("Consolas", 8),
                                      text_color="#334155", wraplength=250)
        self._rec_path.pack(anchor="w", padx=10)

    # ── section / row helpers ─────────────────────────────────────────────────
    def _section(self, text):
        f = ctk.CTkFrame(self, fg_color="transparent")
        f.pack(fill="x", padx=10, pady=(8, 2))
        ctk.CTkLabel(f, text=text, font=("Segoe UI", 9, "bold"),
                     text_color="#2563EB").pack(side="left")
        ctk.CTkFrame(f, height=1, fg_color="#1E3A5F").pack(
            side="left", fill="x", expand=True, padx=4)

    def _stat_row(self, parent, label, var, color):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=1)
        ctk.CTkLabel(row, text=label, font=("Segoe UI", 10),
                     text_color="#64748B", width=100, anchor="w").pack(side="left")
        ctk.CTkLabel(row, textvariable=var, font=("Segoe UI", 10, "bold"),
                     text_color=color, anchor="e").pack(side="right")

    # ── public update API ─────────────────────────────────────────────────────
    def update_stats(self, fps_inf: float, rgb_det: int, thermal_det: int,
                     inf_ms: float, session_total: int,
                     model_name: str = "", device: str = "", classes: int = 0,
                     avg_fps: float = 0.0, fps_rgb: float = 0.0,
                     fps_thermal: float = 0.0, active_threads: int = 0,
                     queue_size: int = 0, frame_drops: int = 0,
                     onnx_active: bool = False):
        self._fps_var.set(f"{fps_inf:.1f}")
        self._avg_fps_var.set(f"{avg_fps:.1f}")
        self._fps_rgb_var.set(f"{fps_rgb:.1f}")
        self._fps_th_var.set(f"{fps_thermal:.1f}")
        self._threads_var.set(str(active_threads))
        self._qsize_var.set(str(queue_size))
        self._drops_var.set(str(frame_drops))
        self._rgb_det_var.set(str(rgb_det))
        self._thermal_det_var.set(str(thermal_det))
        self._total_var.set(str(rgb_det + thermal_det))
        self._session_var.set(str(session_total))
        self._inf_ms_var.set(f"{inf_ms:.0f} ms")
        if model_name:
            self._model_var.set(model_name)
        if device:
            self._device_var.set(device)
        if classes:
            self._classes_var.set(str(classes))
        self._onnx_var.set("Active" if onnx_active else "PT fallback")

    def log_detection(self, camera: str, name: str, conf: float, tid: int):
        ts = datetime.now().strftime("%H:%M:%S")
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
            self._rec_status.configure(text="● Recording...", text_color="#EF4444")
            self._rec_path.configure(text=path)
        else:
            self._rec_status.configure(text="● Not recording", text_color="#64748B")
            self._rec_path.configure(text="")

    # ── background system monitor ─────────────────────────────────────────────
    def _system_monitor(self):
        while True:
            try:
                cpu = psutil.cpu_percent(interval=1)
                mem = psutil.virtual_memory()
                ram_mb = mem.used // (1024 * 1024)
                self._cpu_var.set(f"{cpu:.0f}%")
                self._ram_var.set(f"{ram_mb} MB")
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        g = gpus[0]
                        self._gpu_var.set(f"{g.load*100:.0f}% {g.memoryUsed:.0f}MB")
                    else:
                        self._gpu_var.set("No GPU")
                except Exception:
                    self._gpu_var.set("N/A")
            except Exception:
                pass
            time.sleep(2)
