"""
Right control panel — v1.3 Stable CPU Edition.
  • Fully scrollable — no content ever clipped
  • Camera Mode selector (Single Active Camera)
  • CPU Backend Diagnostics
  • System stats (CPU%, RAM)
  • Model info
  • Performance dashboard
  • TRACKING STATS (new)
  • Detection stats (complete — no hidden fields)
  • Detection log
  • Recording status
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
        self._bm = backend_manager
        self._build()
        threading.Thread(target=self._system_monitor,
                         daemon=True, name="SysMonitor").start()

    def _build(self):
        # ── Scrollable container — prevents any content from being clipped ────
        self._scroll = ctk.CTkScrollableFrame(
            self, fg_color="transparent",
            scrollbar_button_color="#1E3A5F",
            scrollbar_button_hover_color="#2563EB")
        self._scroll.pack(fill="both", expand=True, padx=0, pady=0)

        s = self._scroll   # short alias for all child widgets

        # ── CAMERA MODE ───────────────────────────────────────────────────────
        self._section(s, "CAMERA MODE")
        cam_frame = ctk.CTkFrame(s, fg_color="#0A0F1E", corner_radius=8)
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

        # ── BACKEND ───────────────────────────────────────────────────────────
        self._section(s, "BACKEND")
        be_frame = ctk.CTkFrame(s, fg_color="#0A0F1E", corner_radius=8)
        be_frame.pack(fill="x", padx=10, pady=(0, 6))

        self._backend_var      = ctk.StringVar(value="ONNX Runtime CPU")
        self._provider_var     = ctk.StringVar(value="CPUExecutionProvider")
        self._model_label_var  = ctk.StringVar(value="YOLO26n")
        self._device_label_var = ctk.StringVar(value="CPU")
        self._ort_ver_var      = ctk.StringVar(value="—")
        self._threads_var      = ctk.StringVar(value="—")
        self._stat_row(be_frame, "Backend",   self._backend_var,     "#22C55E")
        self._stat_row(be_frame, "Provider",  self._provider_var,    "#22C55E")
        self._stat_row(be_frame, "Model",     self._model_label_var, "#A78BFA")
        self._stat_row(be_frame, "Device",    self._device_label_var,"#3B82F6")
        self._stat_row(be_frame, "ORT Ver",   self._ort_ver_var,     "#64748B")
        self._stat_row(be_frame, "CPU Thds",  self._threads_var,     "#64748B")

        # ── SYSTEM ────────────────────────────────────────────────────────────
        self._section(s, "SYSTEM")
        sys_frame = ctk.CTkFrame(s, fg_color="#0A0F1E", corner_radius=8)
        sys_frame.pack(fill="x", padx=10, pady=(0, 6))

        self._cpu_var = ctk.StringVar(value="0%")
        self._ram_var = ctk.StringVar(value="0 MB")
        self._stat_row(sys_frame, "CPU Usage", self._cpu_var, "#F59E0B")
        self._stat_row(sys_frame, "RAM Usage", self._ram_var, "#3B82F6")

        # ── PERFORMANCE ───────────────────────────────────────────────────────
        self._section(s, "PERFORMANCE")
        perf_frame = ctk.CTkFrame(s, fg_color="#0A0F1E", corner_radius=8)
        perf_frame.pack(fill="x", padx=10, pady=(0, 6))

        self._fps_var       = ctk.StringVar(value="0.0")
        self._avg_fps_var   = ctk.StringVar(value="0.0")
        self._cap_fps_var   = ctk.StringVar(value="0.0")
        self._disp_fps_var  = ctk.StringVar(value="0.0")
        self._inf_ms_var    = ctk.StringVar(value="0 ms")
        self._pre_ms_var    = ctk.StringVar(value="0 ms")
        self._infer_ms_var  = ctk.StringVar(value="0 ms")
        self._post_ms_var   = ctk.StringVar(value="0 ms")
        self._thr_count_var = ctk.StringVar(value="0")
        self._q_size_var    = ctk.StringVar(value="0")
        self._drops_var     = ctk.StringVar(value="0")
        self._stat_row(perf_frame, "Infer FPS",   self._fps_var,       "#22C55E")
        self._stat_row(perf_frame, "Avg FPS",     self._avg_fps_var,   "#16A34A")
        self._stat_row(perf_frame, "Capture FPS", self._cap_fps_var,   "#3B82F6")
        self._stat_row(perf_frame, "Display FPS", self._disp_fps_var,  "#818CF8")
        self._stat_row(perf_frame, "Total ms",    self._inf_ms_var,    "#F59E0B")
        self._stat_row(perf_frame, "Preprocess",  self._pre_ms_var,    "#64748B")
        self._stat_row(perf_frame, "Infer ms",    self._infer_ms_var,  "#F59E0B")
        self._stat_row(perf_frame, "Postproc",    self._post_ms_var,   "#64748B")
        self._stat_row(perf_frame, "Threads",     self._thr_count_var, "#A78BFA")
        self._stat_row(perf_frame, "Frame Queue", self._q_size_var,    "#64748B")
        self._stat_row(perf_frame, "Drops",       self._drops_var,     "#EF4444")

        # ── TRACKING STATS ────────────────────────────────────────────────────
        self._section(s, "TRACKING")
        trk_frame = ctk.CTkFrame(s, fg_color="#0A0F1E", corner_radius=8)
        trk_frame.pack(fill="x", padx=10, pady=(0, 6))

        self._trk_active_var    = ctk.StringVar(value="0")
        self._trk_lost_var      = ctk.StringVar(value="0")
        self._trk_recovered_var = ctk.StringVar(value="0")
        self._trk_new_var       = ctk.StringVar(value="0")
        self._trk_age_var       = ctk.StringVar(value="0.0 s")
        self._trk_fps_var       = ctk.StringVar(value="0.0")
        self._trk_latency_var   = ctk.StringVar(value="0.0 ms")
        self._stat_row(trk_frame, "Active Tracks",  self._trk_active_var,    "#22C55E")
        self._stat_row(trk_frame, "Lost Tracks",    self._trk_lost_var,      "#F59E0B")
        self._stat_row(trk_frame, "Recovered",      self._trk_recovered_var, "#A78BFA")
        self._stat_row(trk_frame, "New Tracks",     self._trk_new_var,       "#3B82F6")
        self._stat_row(trk_frame, "Avg Track Age",  self._trk_age_var,       "#64748B")
        self._stat_row(trk_frame, "Tracking FPS",   self._trk_fps_var,       "#22C55E")
        self._stat_row(trk_frame, "Track Latency",  self._trk_latency_var,   "#64748B")

        # ── DETECTION STATS ───────────────────────────────────────────────────
        self._section(s, "DETECTION")
        det_frame = ctk.CTkFrame(s, fg_color="#0A0F1E", corner_radius=8)
        det_frame.pack(fill="x", padx=10, pady=(0, 6))

        self._active_det_var  = ctk.StringVar(value="0")
        self._session_var     = ctk.StringVar(value="0")
        self._camera_var      = ctk.StringVar(value="RGB")
        self._track_ids_var   = ctk.StringVar(value="—")
        self._stat_row(det_frame, "Active Dets",   self._active_det_var, "#3B82F6")
        self._stat_row(det_frame, "Session Total", self._session_var,    "#64748B")
        self._stat_row(det_frame, "Camera",        self._camera_var,     "#CBD5E1")
        self._stat_row(det_frame, "Track IDs",     self._track_ids_var,  "#A78BFA")

        # ── DETECTION LOG ─────────────────────────────────────────────────────
        self._section(s, "DETECTION LOG")
        log_outer = ctk.CTkFrame(s, fg_color="#050A14", corner_radius=8)
        log_outer.pack(fill="x", padx=10, pady=(0, 6))

        self._log_text = ctk.CTkTextbox(
            log_outer, fg_color="#050A14", text_color="#64748B",
            font=("Consolas", 9), wrap="none", state="disabled",
            height=140)
        self._log_text.pack(fill="x", expand=False, padx=4, pady=4)

        btn_row = ctk.CTkFrame(log_outer, fg_color="transparent")
        btn_row.pack(fill="x", padx=4, pady=(0, 4))
        ctk.CTkButton(btn_row, text="Clear", width=60, height=22,
                      fg_color="#1E293B", hover_color="#334155",
                      font=("Segoe UI", 9),
                      command=self._clear_log).pack(side="right")

        # ── RECORDING ─────────────────────────────────────────────────────────
        self._section(s, "RECORDING")
        rec_frame = ctk.CTkFrame(s, fg_color="#0A0F1E", corner_radius=8)
        rec_frame.pack(fill="x", padx=10, pady=(0, 10))

        self._rec_status = ctk.CTkLabel(
            rec_frame, text="● Not recording",
            font=("Segoe UI", 10), text_color="#64748B")
        self._rec_status.pack(anchor="w", padx=10, pady=6)
        self._rec_path = ctk.CTkLabel(
            rec_frame, text="",
            font=("Consolas", 8), text_color="#334155", wraplength=260)
        self._rec_path.pack(anchor="w", padx=10, pady=(0, 6))

        # Populate backend statics immediately
        self.after(200, self._refresh_backend_statics)

    # ── Camera selector ───────────────────────────────────────────────────────
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
        self._camera_var.set("RGB" if mode == "rgb" else "Thermal")

    def set_camera_buttons_enabled(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        self._rb_rgb.configure(state=state)
        self._rb_thermal.configure(state=state)

    def set_backend_manager(self, bm):
        self._bm = bm
        self.after(0, self._refresh_backend_statics)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _section(self, parent, text):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.pack(fill="x", padx=10, pady=(8, 2))
        ctk.CTkLabel(f, text=text, font=("Segoe UI", 9, "bold"),
                     text_color="#2563EB").pack(side="left")
        ctk.CTkFrame(f, height=1, fg_color="#1E3A5F").pack(
            side="left", fill="x", expand=True, padx=4)

    def _stat_row(self, parent, label, var, color="#CBD5E1"):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=1)
        ctk.CTkLabel(row, text=label, font=("Segoe UI", 10),
                     text_color="#64748B", width=92, anchor="w").pack(side="left")
        ctk.CTkLabel(row, textvariable=var, font=("Segoe UI", 10, "bold"),
                     text_color=color, anchor="e").pack(side="right")

    def _refresh_backend_statics(self):
        if self._bm is None:
            return
        d = self._bm.get_diagnostics()
        self._backend_var.set(d["backend"])
        self._provider_var.set(d["provider"])
        self._model_label_var.set(d["model"])
        self._device_label_var.set(d["device"])
        self._ort_ver_var.set(d["ort_version"])
        self._threads_var.set(f"{d['intra_threads']} intra / {d['inter_threads']} inter")

    # ── Public update API ─────────────────────────────────────────────────────
    def update_stats(
        self,
        fps_inf:        float = 0.0,
        avg_fps:        float = 0.0,
        cap_fps:        float = 0.0,
        display_fps:    float = 0.0,
        inf_ms:         float = 0.0,
        preprocess_ms:  float = 0.0,
        infer_ms:       float = 0.0,
        postprocess_ms: float = 0.0,
        active_threads: int   = 0,
        queue_size:     int   = 0,
        frame_drops:    int   = 0,
        active_dets:    int   = 0,
        session_total:  int   = 0,
        camera_mode:    str   = "rgb",
        # Tracking stats
        trk_active:     int   = 0,
        trk_lost:       int   = 0,
        trk_recovered:  int   = 0,
        trk_new:        int   = 0,
        trk_avg_age:    float = 0.0,
        trk_fps:        float = 0.0,
        trk_latency:    float = 0.0,
        trk_ids:        str   = "—",
    ):
        self._fps_var.set(f"{fps_inf:.1f}")
        self._avg_fps_var.set(f"{avg_fps:.1f}")
        self._cap_fps_var.set(f"{cap_fps:.1f}")
        self._disp_fps_var.set(f"{display_fps:.1f}")
        self._inf_ms_var.set(f"{inf_ms:.0f} ms")
        self._pre_ms_var.set(f"{preprocess_ms:.1f} ms")
        self._infer_ms_var.set(f"{infer_ms:.1f} ms")
        self._post_ms_var.set(f"{postprocess_ms:.1f} ms")
        self._thr_count_var.set(str(active_threads))
        self._q_size_var.set(str(queue_size))
        self._drops_var.set(str(frame_drops))
        self._active_det_var.set(str(active_dets))
        self._session_var.set(str(session_total))
        self._camera_var.set("RGB" if camera_mode == "rgb" else "Thermal")
        # Tracking
        self._trk_active_var.set(str(trk_active))
        self._trk_lost_var.set(str(trk_lost))
        self._trk_recovered_var.set(str(trk_recovered))
        self._trk_new_var.set(str(trk_new))
        self._trk_age_var.set(f"{trk_avg_age:.1f} s")
        self._trk_fps_var.set(f"{trk_fps:.1f}")
        self._trk_latency_var.set(f"{trk_latency:.1f} ms")
        self._track_ids_var.set(trk_ids if trk_ids else "—")

    def update_tracking_stats(self, stats: dict):
        """Accept a dict from ByteTracker.get_stats()."""
        self._trk_active_var.set(str(stats.get("active_tracks", 0)))
        self._trk_lost_var.set(str(stats.get("lost_tracks", 0)))
        self._trk_recovered_var.set(str(stats.get("recovered_total", 0)))
        self._trk_new_var.set(str(stats.get("new_total", 0)))
        self._trk_age_var.set(f"{stats.get('avg_age_sec', 0.0):.1f} s")
        self._trk_fps_var.set(f"{stats.get('tracking_fps', 0.0):.1f}")
        self._trk_latency_var.set(f"{stats.get('latency_ms', 0.0):.1f} ms")

    def log_detection(self, camera: str, name: str, conf: float, tid: int):
        ts   = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {camera:<8} {name:<14} {conf:.2f}"
        if tid > 0:
            line += f"  #{tid}"
        line += "\n"
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

    # ── Background system monitor ─────────────────────────────────────────────
    def _system_monitor(self):
        while True:
            try:
                cpu = psutil.cpu_percent(interval=1)
                ram = psutil.virtual_memory().used // (1024 * 1024)
                cpu_str = f"{cpu:.0f}%"
                ram_str = f"{ram} MB"
                try:
                    self.after(0, self._cpu_var.set, cpu_str)
                    self.after(0, self._ram_var.set, ram_str)
                except Exception:
                    pass
            except Exception:
                pass
            time.sleep(2)
