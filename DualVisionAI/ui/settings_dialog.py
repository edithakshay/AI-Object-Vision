"""
Settings dialog — DualVision AI v1.3 Stable CPU Edition.
4-tab layout:
  Tab 1: General     (RTSP URLs, Detection, Inference)
  Tab 2: ONNX / CPU  (Model info, ONNX status, CPU optimisation)
  Tab 3: Dashboard   (Live performance metrics — auto-refreshed)
  Tab 4: Tracking    (Trail lines, track parameters, event logging)
"""
import threading
import time
import customtkinter as ctk
from config.settings import Settings
from ai.model_manager import ModelManager, MODEL_NAME, MODEL_VERSION, MODEL_VARIANTS


class SettingsDialog(ctk.CTkToplevel):
    def __init__(self, parent, settings: Settings, on_save=None,
                 detector=None, backend_manager=None):
        super().__init__(parent)
        self.title("Settings — v1.3 Stable CPU Edition")
        self.resizable(False, False)
        self.configure(fg_color="#0D1626")
        self.grab_set()

        self._settings       = settings
        self._on_save        = on_save
        self._detector       = detector
        self._bm             = backend_manager
        self._model_mgr      = ModelManager(model_dir="models")
        self._refresh_id     = None

        w, h = 620, 780
        pw = parent.winfo_rootx() + parent.winfo_width()  // 2
        ph = parent.winfo_rooty() + parent.winfo_height() // 2
        self.geometry(f"{w}x{h}+{pw - w//2}+{ph - h//2}")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build()
        self._start_refresh()

    # ── Layout ────────────────────────────────────────────────────────────────
    def _build(self):
        ctk.CTkLabel(self, text="Settings", font=("Segoe UI", 16, "bold"),
                     text_color="#E2E8F0").pack(pady=(18, 6))

        self._tabs = ctk.CTkTabview(
            self, fg_color="#0D1626",
            segmented_button_fg_color="#131F35",
            segmented_button_selected_color="#2563EB",
            segmented_button_unselected_color="#1E293B",
            segmented_button_selected_hover_color="#1D4ED8")
        self._tabs.pack(fill="both", expand=True, padx=16, pady=0)

        for name in ("General", "ONNX / CPU", "Dashboard", "Tracking",
                     "Optimization"):
            self._tabs.add(name)

        self._build_general(self._tabs.tab("General"))
        self._build_onnx_cpu(self._tabs.tab("ONNX / CPU"))
        self._build_dashboard(self._tabs.tab("Dashboard"))
        self._build_tracking(self._tabs.tab("Tracking"))
        self._build_optimization(self._tabs.tab("Optimization"))

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=16, pady=12)
        ctk.CTkButton(btn_frame, text="Save Settings", command=self._save,
                      fg_color="#2563EB", hover_color="#1D4ED8",
                      height=36).pack(side="right", padx=4)
        ctk.CTkButton(btn_frame, text="Cancel", command=self._on_close,
                      fg_color="#1E293B", hover_color="#334155",
                      height=36).pack(side="right", padx=4)

    # ── Tab 1: General ─────────────────────────────────────────────────────────
    def _build_general(self, tab):
        scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        self._add_section(scroll, "RTSP Streams")
        self._rgb_url = self._add_entry(
            scroll, "RGB Stream URL",
            self._settings.get("rtsp", "rgb_url", ""))
        self._thermal_url = self._add_entry(
            scroll, "Thermal Stream URL",
            self._settings.get("rtsp", "thermal_url", ""))
        self._reconnect_delay = self._add_entry(
            scroll, "Reconnect Delay (s)",
            str(self._settings.get("rtsp", "reconnect_delay", 3)))

        self._add_section(scroll, "Detection")
        self._conf = self._add_slider(
            scroll, "Confidence Threshold",
            self._settings.get("detection", "confidence", 0.45), 0.05, 0.95)
        self._iou = self._add_slider(
            scroll, "IOU Threshold",
            self._settings.get("detection", "iou", 0.45), 0.05, 0.95)
        self._input_size = self._add_option(
            scroll, "Input Resolution",
            ["320", "416", "512", "640"],
            str(self._settings.get("detection", "input_width", 640)))
        self._frame_skip = self._add_option(
            scroll, "Frame Skip (1 = every frame)",
            ["1", "2", "3", "4", "5"],
            str(self._settings.get("detection", "frame_skip", 1)))
        self._max_fps = self._add_entry(
            scroll, "Maximum UI FPS",
            str(self._settings.get("detection", "max_fps", 30)))
        self._tracking = self._add_switch(
            scroll, "Enable Object Tracking",
            self._settings.get("detection", "enable_tracking", True))

        self._add_section(scroll, "Recording")
        ctk.CTkLabel(scroll,
            text="Recording mode applies when Detection is ON.\n"
                 "When Detection is OFF, raw camera frames are always recorded.",
            font=("Segoe UI", 9), text_color="#64748B",
            justify="left", wraplength=540).pack(anchor="w", pady=(0, 4))
        self._recording_mode = self._add_option(
            scroll, "Recording Mode",
            ["overlay", "raw"],
            str(self._settings.get("recording", "recording_mode", "overlay")))

        self._add_section(scroll, "CPU Threads")
        ctk.CTkLabel(scroll,
            text="0 = auto-detect (recommended).\n"
                 "Set manually only to limit CPU usage on low-power machines.",
            font=("Segoe UI", 9), text_color="#64748B",
            justify="left", wraplength=540).pack(anchor="w", pady=(0, 4))
        self._cpu_threads = self._add_entry(
            scroll, "Intra-Op Threads (0 = auto)",
            str(self._settings.get("inference", "cpu_threads", 0)))

    # ── Tab 2: ONNX / CPU ─────────────────────────────────────────────────────
    def _build_onnx_cpu(self, tab):
        scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        # Model info
        self._add_section(scroll, "Model")
        info = ctk.CTkFrame(scroll, fg_color="#0A0F1E", corner_radius=8)
        info.pack(fill="x", pady=(0, 8))
        self._m_name_var   = ctk.StringVar(value=MODEL_NAME)
        self._m_ver_var    = ctk.StringVar(value=MODEL_VERSION)
        self._m_pt_var     = ctk.StringVar(value="Checking …")
        self._m_onnx_var   = ctk.StringVar(value="Checking …")
        self._m_pt_mb_var  = ctk.StringVar(value="—")
        self._m_onnx_mb_var = ctk.StringVar(value="—")
        self._info_row(info, "Model",       self._m_name_var,    "#A78BFA")
        self._info_row(info, "Version",     self._m_ver_var,     "#CBD5E1")
        self._info_row(info, ".pt Status",  self._m_pt_var,      "#22C55E")
        self._info_row(info, ".pt Size",    self._m_pt_mb_var,   "#64748B")
        self._info_row(info, "ONNX Status", self._m_onnx_var,    "#22C55E")
        self._info_row(info, "ONNX Size",   self._m_onnx_mb_var, "#64748B")

        # Download
        self._add_section(scroll, "Download Model")
        self._dl_desc_lbl = ctk.CTkLabel(scroll,
            text="",
            font=("Segoe UI", 9), text_color="#64748B",
            justify="left", wraplength=540)
        self._dl_desc_lbl.pack(anchor="w", pady=(0, 4))

        dl_row = ctk.CTkFrame(scroll, fg_color="transparent")
        dl_row.pack(fill="x", pady=(0, 4))
        self._dl_btn = ctk.CTkButton(
            dl_row, text="⬇  Download Model", width=180,
            fg_color="#1E3A5F", hover_color="#334155",
            height=30, command=self._on_download)
        self._dl_btn.pack(side="left")
        self._dl_bar = ctk.CTkProgressBar(
            dl_row, width=200, fg_color="#131F35", progress_color="#2563EB")
        self._dl_bar.set(0)
        self._dl_bar.pack(side="left", padx=10)

        self._dl_status = ctk.CTkLabel(
            scroll, text="", font=("Segoe UI", 10),
            text_color="#64748B", anchor="w")
        self._dl_status.pack(fill="x", pady=(0, 6))

        # ONNX Export
        self._add_section(scroll, "Export ONNX")
        self._exp_desc_lbl = ctk.CTkLabel(scroll,
            text="",
            font=("Segoe UI", 9), text_color="#64748B",
            justify="left", wraplength=540)
        self._exp_desc_lbl.pack(anchor="w", pady=(0, 4))
        self._export_btn = ctk.CTkButton(
            scroll, text="Export to ONNX Now",
            fg_color="#1E3A5F", hover_color="#334155",
            height=30, command=self._on_export)
        self._export_btn.pack(anchor="w", pady=(0, 4))
        self._export_status = ctk.CTkLabel(
            scroll, text="", font=("Segoe UI", 10),
            text_color="#64748B", anchor="w")
        self._export_status.pack(fill="x")

        # ONNX Runtime
        self._add_section(scroll, "ONNX Runtime Diagnostics")
        ort_frame = ctk.CTkFrame(scroll, fg_color="#0A0F1E", corner_radius=8)
        ort_frame.pack(fill="x", pady=(0, 8))
        self._ort_ver_var  = ctk.StringVar(value="—")
        self._ort_prov_var = ctk.StringVar(value="CPUExecutionProvider")
        self._cpu_name_var = ctk.StringVar(value="—")
        self._cpu_log_var  = ctk.StringVar(value="—")
        self._cpu_phy_var  = ctk.StringVar(value="—")
        self._intra_var    = ctk.StringVar(value="—")
        self._inter_var    = ctk.StringVar(value="—")
        self._info_row(ort_frame, "ORT Version",   self._ort_ver_var,  "#CBD5E1")
        self._info_row(ort_frame, "Provider",      self._ort_prov_var, "#22C55E")
        self._info_row(ort_frame, "CPU",           self._cpu_name_var, "#CBD5E1")
        self._info_row(ort_frame, "Logical Cores", self._cpu_log_var,  "#94A3B8")
        self._info_row(ort_frame, "Phys Cores",    self._cpu_phy_var,  "#94A3B8")
        self._info_row(ort_frame, "Intra Threads", self._intra_var,    "#A78BFA")
        self._info_row(ort_frame, "Inter Threads", self._inter_var,    "#A78BFA")

        # Populate statics immediately
        self.after(100, self._refresh_onnx_tab)

    # ── Tab 3: Dashboard ───────────────────────────────────────────────────────
    def _build_dashboard(self, tab):
        scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        self._add_section(scroll, "Inference Backend")
        be_frame = ctk.CTkFrame(scroll, fg_color="#0A0F1E", corner_radius=8)
        be_frame.pack(fill="x", pady=(0, 8))
        self._d_backend   = ctk.StringVar(value="ONNX Runtime CPU")
        self._d_provider  = ctk.StringVar(value="CPUExecutionProvider")
        active_key = self._settings.get("inference", "active_model", "yolo26n")
        self._d_model     = ctk.StringVar(value=MODEL_VARIANTS.get(active_key, MODEL_VARIANTS["yolo26n"])["label"])
        self._d_device    = ctk.StringVar(value="CPU")
        self._d_onnx_st   = ctk.StringVar(value="—")
        self._d_model_ld  = ctk.StringVar(value="No")
        self._info_row(be_frame, "Backend",       self._d_backend,  "#22C55E")
        self._info_row(be_frame, "Provider",      self._d_provider, "#22C55E")
        self._info_row(be_frame, "Model",         self._d_model,    "#A78BFA")
        self._info_row(be_frame, "Device",        self._d_device,   "#3B82F6")
        self._info_row(be_frame, "ONNX Active",   self._d_onnx_st,  "#A78BFA")
        self._info_row(be_frame, "Model Loaded",  self._d_model_ld, "#22C55E")

        self._add_section(scroll, "FPS")
        fps_frame = ctk.CTkFrame(scroll, fg_color="#0A0F1E", corner_radius=8)
        fps_frame.pack(fill="x", pady=(0, 8))
        self._d_fps_cur = ctk.StringVar(value="0.0")
        self._d_fps_avg = ctk.StringVar(value="0.0")
        self._d_cap_fps = ctk.StringVar(value="0.0")
        self._info_row(fps_frame, "Infer FPS",   self._d_fps_cur, "#22C55E")
        self._info_row(fps_frame, "Avg FPS",     self._d_fps_avg, "#22C55E")
        self._info_row(fps_frame, "Capture FPS", self._d_cap_fps, "#3B82F6")

        self._add_section(scroll, "Frame Pipeline (ms)")
        pipe_frame = ctk.CTkFrame(scroll, fg_color="#0A0F1E", corner_radius=8)
        pipe_frame.pack(fill="x", pady=(0, 8))
        self._d_pre_ms   = ctk.StringVar(value="—")
        self._d_inf_ms   = ctk.StringVar(value="—")
        self._d_post_ms  = ctk.StringVar(value="—")
        self._info_row(pipe_frame, "Preprocess",  self._d_pre_ms,  "#64748B")
        self._info_row(pipe_frame, "Inference",   self._d_inf_ms,  "#F59E0B")
        self._info_row(pipe_frame, "Postprocess", self._d_post_ms, "#64748B")

        self._add_section(scroll, "System")
        sys_frame = ctk.CTkFrame(scroll, fg_color="#0A0F1E", corner_radius=8)
        sys_frame.pack(fill="x", pady=(0, 8))
        self._d_cpu      = ctk.StringVar(value="0%")
        self._d_ram      = ctk.StringVar(value="0 MB")
        self._d_threads  = ctk.StringVar(value="0")
        self._d_queue    = ctk.StringVar(value="0")
        self._d_drops    = ctk.StringVar(value="0")
        self._d_dets     = ctk.StringVar(value="0")
        self._d_session  = ctk.StringVar(value="0")
        self._d_camera   = ctk.StringVar(value="RGB")
        self._info_row(sys_frame, "CPU Usage",      self._d_cpu,     "#F59E0B")
        self._info_row(sys_frame, "RAM Usage",      self._d_ram,     "#3B82F6")
        self._info_row(sys_frame, "Active Threads", self._d_threads, "#A78BFA")
        self._info_row(sys_frame, "Frame Queue",    self._d_queue,   "#64748B")
        self._info_row(sys_frame, "Frame Drops",    self._d_drops,   "#EF4444")
        self._info_row(sys_frame, "Active Dets",    self._d_dets,    "#3B82F6")
        self._info_row(sys_frame, "Session Dets",   self._d_session, "#64748B")
        self._info_row(sys_frame, "Current Camera", self._d_camera,  "#CBD5E1")

        ctk.CTkLabel(scroll, text="⟳ Auto-refreshes every second",
                     font=("Segoe UI", 9), text_color="#334155").pack(
                     anchor="w", pady=(4, 0))

    # ── Tab 4: Tracking ────────────────────────────────────────────────────────
    def _build_tracking(self, tab):
        scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        self._add_section(scroll, "Trail Lines")
        ctk.CTkLabel(scroll,
            text="Trail lines draw the centre-point path of each tracked\n"
                 "object. Enabling trails has a small CPU cost (~0.5 ms/frame).",
            font=("Segoe UI", 9), text_color="#64748B",
            justify="left", wraplength=540).pack(anchor="w", pady=(0, 4))
        self._trk_trails = self._add_switch(
            scroll, "Enable Trail Lines",
            self._settings.get("tracking", "enable_trails", False))
        self._trk_trail_len = self._add_slider(
            scroll, "Max Trail Length (frames)",
            float(self._settings.get("tracking", "max_trail_length", 30)),
            5.0, 80.0)

        self._add_section(scroll, "Track Parameters")
        ctk.CTkLabel(scroll,
            text="Changes take effect the next time detection is started\n"
                 "(click Stop → Start to apply).",
            font=("Segoe UI", 9), text_color="#64748B",
            justify="left", wraplength=540).pack(anchor="w", pady=(0, 4))
        self._trk_timeout = self._add_slider(
            scroll, "Track Timeout (missed frames before removal)",
            float(self._settings.get("tracking", "track_timeout", 5)),
            1.0, 30.0)
        self._trk_min_hits = self._add_option(
            scroll, "Min Confirmation Frames",
            ["1", "2", "3", "4", "5"],
            str(self._settings.get("tracking", "min_confirmation_hits", 1)))
        self._trk_assoc = self._add_slider(
            scroll, "Association IoU Threshold",
            float(self._settings.get("tracking", "association_threshold", 0.35)),
            0.10, 0.80)
        self._trk_conf_split = self._add_slider(
            scroll, "Confidence Split (high vs low)",
            float(self._settings.get("tracking", "tracking_confidence", 0.45)),
            0.10, 0.95)

        self._add_section(scroll, "Tracking Log")
        ctk.CTkLabel(scroll,
            text="Track lifecycle events (created / lost / recovered / removed)\n"
                 "are written to logs/tracking.log when detection is running.",
            font=("Segoe UI", 9), text_color="#64748B",
            justify="left", wraplength=540).pack(anchor="w", pady=(0, 6))

        restore_btn = ctk.CTkButton(
            scroll, text="Restore Tracking Defaults",
            fg_color="#1E293B", hover_color="#334155",
            height=30, command=self._restore_tracking_defaults)
        restore_btn.pack(anchor="w", pady=(4, 8))

    # ── Tab 5: Optimization ────────────────────────────────────────────────────
    def _build_optimization(self, tab):
        from ai.model_manager import MODEL_VARIANTS
        scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        # ── Model Selection ──────────────────────────────────────────────────
        self._add_section(scroll, "Model Selection")
        ctk.CTkLabel(scroll,
            text="Larger models detect objects more reliably but run slower on CPU.\n"
                 "Switching requires Stop → Start to reload the model.",
            font=("Segoe UI", 9), text_color="#64748B",
            justify="left", wraplength=540).pack(anchor="w", pady=(0, 4))
        variant_labels = [v["label"] for v in MODEL_VARIANTS.values()]
        variant_keys   = list(MODEL_VARIANTS.keys())
        cur_key        = self._settings.get("inference", "active_model", "yolo26n")
        cur_label      = MODEL_VARIANTS.get(cur_key, MODEL_VARIANTS["yolo26n"])["label"]
        self._opt_model_var = ctk.StringVar(value=cur_label)
        self._opt_model_keys = variant_keys
        self._opt_model_labels = variant_labels
        ctk.CTkOptionMenu(scroll, values=variant_labels,
                          variable=self._opt_model_var,
                          fg_color="#1E293B", button_color="#2563EB",
                          button_hover_color="#1D4ED8",
                          font=("Segoe UI", 10)).pack(
                          anchor="w", pady=(0, 8))
        # Keep ONNX/CPU tab in sync whenever the user changes the model dropdown
        self._opt_model_var.trace_add(
            "write", lambda *_: self.after(60, self._refresh_onnx_tab))

        # ── Confidence Smoother ──────────────────────────────────────────────
        self._add_section(scroll, "Confidence Smoother")
        ctk.CTkLabel(scroll,
            text="Prevents 1-3 frame disappearances by applying temporal\n"
                 "EMA smoothing and synthesising ghost detections for recently-\n"
                 "seen objects. Requires Detection ON.",
            font=("Segoe UI", 9), text_color="#64748B",
            justify="left", wraplength=540).pack(anchor="w", pady=(0, 4))
        self._opt_smoother_en = self._add_switch(
            scroll, "Enable Confidence Smoother",
            self._settings.get("smoothing", "enable_smoother", True))
        self._opt_ema = self._add_slider(
            scroll, "EMA Alpha (lower = smoother, slower response)",
            float(self._settings.get("smoothing", "ema_alpha", 0.35)),
            0.05, 0.95)
        self._opt_sm_iou = self._add_slider(
            scroll, "Smoother IoU Threshold",
            float(self._settings.get("smoothing", "iou_threshold", 0.40)),
            0.10, 0.90)
        self._opt_ghost_frames = self._add_option(
            scroll, "Max Ghost Frames (keep alive N frames after disappear)",
            ["0", "1", "2", "3", "4", "5", "6", "8", "10"],
            str(self._settings.get("smoothing", "max_ghost_frames", 3)))
        self._opt_ghost_decay = self._add_slider(
            scroll, "Ghost Confidence Decay per Frame",
            float(self._settings.get("smoothing", "ghost_decay", 0.70)),
            0.30, 1.00)
        self._opt_min_ghost_conf = self._add_slider(
            scroll, "Minimum Ghost Confidence (below = drop)",
            float(self._settings.get("smoothing", "min_ghost_conf", 0.25)),
            0.05, 0.70)

        # ── Detection Persistence ────────────────────────────────────────────
        self._add_section(scroll, "Detection Persistence (Tracker)")
        ctk.CTkLabel(scroll,
            text="Keep showing a track's last known position (dashed box)\n"
                 "for up to N frames after the detector loses it.\n"
                 "0 = disabled. 5 = ~330 ms at 15 fps.",
            font=("Segoe UI", 9), text_color="#64748B",
            justify="left", wraplength=540).pack(anchor="w", pady=(0, 4))
        self._opt_persist = self._add_slider(
            scroll, "Persistence Frames",
            float(self._settings.get("tracking", "persistence_frames", 5)),
            0.0, 15.0)

        # ── Box Area Filter ──────────────────────────────────────────────────
        self._add_section(scroll, "Box Area Filter (pixels²)")
        ctk.CTkLabel(scroll,
            text="Ignore detections that are too small (noise) or too large\n"
                 "(full-frame false positives). 0 = disabled.",
            font=("Segoe UI", 9), text_color="#64748B",
            justify="left", wraplength=540).pack(anchor="w", pady=(0, 4))
        self._opt_min_area = self._add_entry(
            scroll, "Minimum Box Area (px²)",
            str(self._settings.get("smoothing", "min_box_area", 0)))
        self._opt_max_area = self._add_entry(
            scroll, "Maximum Box Area (px²)",
            str(self._settings.get("smoothing", "max_box_area", 0)))

        restore_btn = ctk.CTkButton(
            scroll, text="Restore Optimization Defaults",
            fg_color="#1E293B", hover_color="#334155",
            height=30, command=self._restore_optimization_defaults)
        restore_btn.pack(anchor="w", pady=(8, 8))

    def _restore_optimization_defaults(self):
        from ai.model_manager import MODEL_VARIANTS
        try:
            self._opt_model_var.set(MODEL_VARIANTS["yolo26n"]["label"])
            self._opt_smoother_en.set(True)
            self._opt_ema.set(0.35)
            self._opt_sm_iou.set(0.40)
            self._opt_ghost_frames.set("3")
            self._opt_ghost_decay.set(0.70)
            self._opt_min_ghost_conf.set(0.25)
            self._opt_persist.set(5.0)
            self._opt_min_area.delete(0, "end"); self._opt_min_area.insert(0, "0")
            self._opt_max_area.delete(0, "end"); self._opt_max_area.insert(0, "0")
        except Exception:
            pass

    def _restore_tracking_defaults(self):
        defaults = {
            "enable_trails":              False,
            "max_trail_length":           30,
            "track_timeout":              5,
            "min_confirmation_hits":      1,
            "tracking_confidence":        0.45,
            "association_threshold":      0.35,
            "low_association_threshold":  0.20,
        }
        try:
            self._trk_trails.set(False)
            self._trk_trail_len.set(defaults["max_trail_length"])
            self._trk_timeout.set(defaults["track_timeout"])
            self._trk_min_hits.set(str(defaults["min_confirmation_hits"]))
            self._trk_assoc.set(defaults["association_threshold"])
            self._trk_conf_split.set(defaults["tracking_confidence"])
        except Exception:
            pass

    # ── Variant helper ─────────────────────────────────────────────────────────
    def _get_active_variant(self) -> str:
        """Return the variant key currently selected in the Optimization dropdown."""
        from ai.model_manager import MODEL_VARIANTS
        try:
            sel_label = self._opt_model_var.get()
            key = next(
                (k for k, v in MODEL_VARIANTS.items() if v["label"] == sel_label),
                None)
            if key:
                return key
        except AttributeError:
            pass
        return self._settings.get("inference", "active_model", "yolo26n")

    # ── Download / Export ──────────────────────────────────────────────────────
    def _on_download(self):
        variant = self._get_active_variant()
        self._model_mgr.set_variant(variant)
        label = variant.upper()
        self._dl_btn.configure(state="disabled", text=f"Downloading {label} …")
        self._dl_status.configure(text="Starting …", text_color="#94A3B8")
        self._model_mgr.set_callbacks(
            progress=self._on_dl_progress, status=self._on_dl_status)
        threading.Thread(target=lambda v=variant: self._do_download(v),
                         daemon=True).start()

    def _do_download(self, variant: str):
        self._model_mgr.ensure_pt(blocking=True, variant=variant)
        try:
            self.after(0, self._refresh_onnx_tab)
            self.after(0, lambda v=variant: self._dl_btn.configure(
                state="normal", text=f"⬇  Download {v.upper()}"))
        except Exception:
            pass

    def _on_dl_progress(self, pct: float):
        try: self.after(0, lambda p=pct: self._dl_bar.set(p / 100.0))
        except Exception: pass

    def _on_dl_status(self, msg: str):
        try: self.after(0, lambda m=msg: self._dl_status.configure(
            text=m, text_color="#94A3B8"))
        except Exception: pass

    def _on_export(self):
        variant = self._get_active_variant()
        self._model_mgr.set_variant(variant)
        self._export_btn.configure(state="disabled", text="Exporting …")
        self._export_status.configure(
            text=f"Exporting {variant.upper()} — may take 30–60 s …",
            text_color="#F59E0B")

        def _do(v=variant):
            try:
                result = self._model_mgr.export_onnx(variant=v)
                def _ok():
                    self._export_btn.configure(
                        state="normal", text="Export to ONNX Now")
                    if result:
                        self._export_status.configure(
                            text=f"Exported: {result.name}", text_color="#22C55E")
                    else:
                        self._export_status.configure(
                            text="Export failed — check logs/inference.log",
                            text_color="#EF4444")
                    self._refresh_onnx_tab()
                try: self.after(0, _ok)
                except Exception: pass
            except Exception as exc:
                def _err():
                    self._export_btn.configure(
                        state="normal", text="Export to ONNX Now")
                    self._export_status.configure(
                        text=f"Error: {exc}", text_color="#EF4444")
                try: self.after(0, _err)
                except Exception: pass

        threading.Thread(target=_do, daemon=True).start()

    # ── Refresh helpers ────────────────────────────────────────────────────────
    def _refresh_onnx_tab(self):
        try:
            from ai.model_manager import MODEL_VARIANTS
            # ── Determine the variant currently selected in the UI ──────────
            variant  = self._get_active_variant()
            vlabel   = variant.upper()          # e.g. "YOLO26M"
            vmeta    = MODEL_VARIANTS.get(variant, MODEL_VARIANTS["yolo26n"])
            pt_fname = f"{variant}.pt"
            ox_fname = f"{variant}.onnx"

            # Approximate PT sizes (MB) per variant for the description label
            _pt_sizes = {
                "yolo26n": 6, "yolo26s": 22, "yolo26m": 52,
                "yolo26l": 87, "yolo26x": 136,
            }
            approx_mb = _pt_sizes.get(variant, "?")

            # Update description labels to reflect the selected model
            self._dl_desc_lbl.configure(
                text=f"Downloads {pt_fname} once (~{approx_mb} MB).\n"
                     "Internet required only this one time.")
            self._exp_desc_lbl.configure(
                text=f"Exports {pt_fname} → {ox_fname}.\n"
                     "PyTorch is used only during export — never during inference.\n"
                     "Export runs automatically on first Start.")

            # Point model_mgr at the selected variant before reading status
            self._model_mgr.set_variant(variant)
            info = self._model_mgr.get_model_info()

            self._m_name_var.set(vmeta["label"])
            self._m_ver_var.set(info.get("version", MODEL_VERSION))
            self._m_pt_var.set("Ready ✓" if info["pt_ready"] else "Not downloaded")
            self._m_onnx_var.set("Ready ✓" if info["onnx_ready"] else "Not exported")
            self._m_pt_mb_var.set(f"{info['pt_mb']} MB" if info['pt_mb'] else "—")
            self._m_onnx_mb_var.set(f"{info['onnx_mb']} MB" if info['onnx_mb'] else "—")

            # Update download button and status to match selected model
            if info["pt_ready"]:
                self._dl_btn.configure(
                    state="disabled", text=f"✓ {vlabel} Downloaded")
                self._dl_status.configure(
                    text=f"{pt_fname} is ready.", text_color="#22C55E")
            else:
                self._dl_btn.configure(
                    state="normal", text=f"⬇  Download {vlabel}")
                self._dl_status.configure(
                    text=f"{pt_fname} not found — click Download.",
                    text_color="#F59E0B")

            # Backend diagnostics
            if self._bm is not None:
                d = self._bm.get_diagnostics()
                self._ort_ver_var.set(d["ort_version"])
                self._ort_prov_var.set(d["provider"])
                self._cpu_name_var.set(d["cpu_name"])
                self._cpu_log_var.set(str(d["cpu_logical"]))
                self._cpu_phy_var.set(str(d["cpu_physical"]))
                self._intra_var.set(str(d["intra_threads"]))
                self._inter_var.set(str(d["inter_threads"]))
        except Exception:
            pass

    # ── Dashboard auto-refresh ─────────────────────────────────────────────────
    def _start_refresh(self):
        self._refresh_dashboard()

    def _refresh_dashboard(self):
        try:
            self._do_refresh()
        except Exception:
            pass
        try:
            self._refresh_id = self.after(1000, self._refresh_dashboard)
        except Exception:
            pass

    def _do_refresh(self):
        det = self._detector

        # Static backend (always CPU)
        self._d_backend.set("ONNX Runtime CPU")
        self._d_provider.set("CPUExecutionProvider")
        # Show whichever variant is currently active
        _ak = self._settings.get("inference", "active_model", "yolo26n")
        self._d_model.set(MODEL_VARIANTS.get(_ak, MODEL_VARIANTS["yolo26n"])["label"])
        self._d_device.set("CPU")

        if det is not None:
            self._d_model_ld.set("Yes" if det.is_loaded else "No")
            self._d_onnx_st.set("Active ✓" if det.onnx_active else "—")
            self._d_fps_cur.set(f"{det.fps_inference:.1f}")
            self._d_fps_avg.set(f"{det.avg_fps:.1f}")
            self._d_pre_ms.set(f"{det.preprocess_ms:.1f} ms")
            self._d_inf_ms.set(f"{det.infer_ms:.1f} ms")
            self._d_post_ms.set(f"{det.postprocess_ms:.1f} ms")
            self._d_threads.set(str(det.active_threads))
            self._d_queue.set(str(det.queue_size))
            self._d_drops.set(str(det.frame_drops))

        try:
            import psutil
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().used // (1024 * 1024)
            self._d_cpu.set(f"{cpu:.0f}%")
            self._d_ram.set(f"{ram} MB")
        except Exception:
            pass

    # ── Field builders ─────────────────────────────────────────────────────────
    def _add_section(self, parent, text):
        ctk.CTkLabel(parent, text=text.upper(),
                     font=("Segoe UI", 9, "bold"),
                     text_color="#2563EB").pack(anchor="w", pady=(12, 2))
        ctk.CTkFrame(parent, height=1,
                     fg_color="#1E3A5F").pack(fill="x", pady=(0, 6))

    def _add_entry(self, parent, label, value):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=3)
        ctk.CTkLabel(row, text=label, font=("Segoe UI", 11),
                     text_color="#94A3B8", width=200, anchor="w").pack(side="left")
        entry = ctk.CTkEntry(row, width=260,
                             fg_color="#131F35", border_color="#1E3A5F")
        entry.insert(0, value)
        entry.pack(side="right")
        return entry

    def _add_slider(self, parent, label, value, from_, to):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=3)
        ctk.CTkLabel(row, text=label, font=("Segoe UI", 11),
                     text_color="#94A3B8", width=200, anchor="w").pack(side="left")
        val_lbl = ctk.CTkLabel(row, text=f"{value:.2f}",
                               font=("Segoe UI", 11), text_color="#CBD5E1", width=40)
        val_lbl.pack(side="right")
        slider = ctk.CTkSlider(row, from_=from_, to=to, width=160,
                               fg_color="#1E293B", progress_color="#2563EB",
                               button_color="#3B82F6")
        slider.set(value)
        slider.configure(command=lambda v: val_lbl.configure(text=f"{v:.2f}"))
        slider.pack(side="right", padx=6)
        return slider

    def _add_option(self, parent, label, values, current):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=3)
        ctk.CTkLabel(row, text=label, font=("Segoe UI", 11),
                     text_color="#94A3B8", width=200, anchor="w").pack(side="left")
        var = ctk.StringVar(value=current)
        ctk.CTkOptionMenu(row, values=values, variable=var, width=160,
                          fg_color="#131F35", button_color="#1E3A5F",
                          dropdown_fg_color="#131F35").pack(side="right")
        return var

    def _add_switch(self, parent, label, value):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=3)
        ctk.CTkLabel(row, text=label, font=("Segoe UI", 11),
                     text_color="#94A3B8", width=200, anchor="w").pack(side="left")
        var = ctk.BooleanVar(value=value)
        ctk.CTkSwitch(row, text="", variable=var, onvalue=True, offvalue=False,
                      fg_color="#1E293B", progress_color="#2563EB").pack(side="right")
        return var

    def _info_row(self, parent, label, var, color="#CBD5E1"):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=2)
        ctk.CTkLabel(row, text=label, font=("Segoe UI", 10),
                     text_color="#64748B", width=140, anchor="w").pack(side="left")
        ctk.CTkLabel(row, textvariable=var, font=("Segoe UI", 10, "bold"),
                     text_color=color, anchor="w",
                     wraplength=380).pack(side="left", fill="x", expand=True)

    # ── Save ───────────────────────────────────────────────────────────────────
    def _save(self):
        try:
            self._settings.set("rtsp", "rgb_url",       self._rgb_url.get())
            self._settings.set("rtsp", "thermal_url",   self._thermal_url.get())
            self._settings.set("rtsp", "reconnect_delay",
                               float(self._reconnect_delay.get()))
            self._settings.set("detection", "confidence", float(self._conf.get()))
            self._settings.set("detection", "iou",        float(self._iou.get()))
            self._settings.set("detection", "input_width",
                               int(self._input_size.get()))
            self._settings.set("detection", "frame_skip",
                               int(self._frame_skip.get()))
            self._settings.set("detection", "max_fps",    int(self._max_fps.get()))
            self._settings.set("detection", "enable_tracking",
                               bool(self._tracking.get()))
            try:
                self._settings.set("inference", "cpu_threads",
                                   int(self._cpu_threads.get()))
            except Exception:
                pass
            self._settings.set("recording", "recording_mode",
                               self._recording_mode.get())
            # Tracking tab
            try:
                self._settings.set("tracking", "enable_trails",
                                   bool(self._trk_trails.get()))
                self._settings.set("tracking", "max_trail_length",
                                   int(float(self._trk_trail_len.get())))
                self._settings.set("tracking", "track_timeout",
                                   int(float(self._trk_timeout.get())))
                self._settings.set("tracking", "min_confirmation_hits",
                                   int(self._trk_min_hits.get()))
                self._settings.set("tracking", "association_threshold",
                                   float(self._trk_assoc.get()))
                self._settings.set("tracking", "tracking_confidence",
                                   float(self._trk_conf_split.get()))
            except Exception:
                pass
            # Optimization tab
            try:
                from ai.model_manager import MODEL_VARIANTS
                sel_label = self._opt_model_var.get()
                sel_key   = next(
                    (k for k, v in MODEL_VARIANTS.items()
                     if v["label"] == sel_label), "yolo26n")
                self._settings.set("inference", "active_model", sel_key)
                self._settings.set("smoothing", "enable_smoother",
                                   bool(self._opt_smoother_en.get()))
                self._settings.set("smoothing", "ema_alpha",
                                   float(self._opt_ema.get()))
                self._settings.set("smoothing", "iou_threshold",
                                   float(self._opt_sm_iou.get()))
                self._settings.set("smoothing", "max_ghost_frames",
                                   int(self._opt_ghost_frames.get()))
                self._settings.set("smoothing", "ghost_decay",
                                   float(self._opt_ghost_decay.get()))
                self._settings.set("smoothing", "min_ghost_conf",
                                   float(self._opt_min_ghost_conf.get()))
                self._settings.set("tracking", "persistence_frames",
                                   int(float(self._opt_persist.get())))
                self._settings.set("smoothing", "min_box_area",
                                   int(self._opt_min_area.get() or 0))
                self._settings.set("smoothing", "max_box_area",
                                   int(self._opt_max_area.get() or 0))
            except Exception:
                pass
            self._settings.save()
            if self._on_save:
                self._on_save()
        except Exception as e:
            import tkinter.messagebox as mb
            mb.showerror("Save Error", str(e), parent=self)
        self._on_close()

    def _on_close(self):
        if self._refresh_id:
            try: self.after_cancel(self._refresh_id)
            except Exception: pass
        self.destroy()
