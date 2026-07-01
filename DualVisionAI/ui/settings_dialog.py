"""
Settings dialog — tabbed layout.
  Tab 1: General  (RTSP, Detection, Inference, Model)
  Tab 2: YOLO26   (model info, ONNX status, advanced inference params)
  Tab 3: Performance Dashboard (live metrics — read-only, refreshed every second)
"""
import threading
import time
import customtkinter as ctk
from config.settings import Settings
from ai.model_manager import ModelManager, SUPPORTED_MODELS


class SettingsDialog(ctk.CTkToplevel):
    def __init__(self, parent, settings: Settings, on_save=None, detector=None):
        super().__init__(parent)
        self.title("Settings")
        self.resizable(False, False)
        self.configure(fg_color="#0D1626")
        self.grab_set()
        self._settings  = settings
        self._on_save   = on_save
        self._detector  = detector          # live Detector reference (may be None)
        self._model_mgr = ModelManager(model_dir="models")
        self._refresh_after_id = None

        w, h = 600, 760
        pw = parent.winfo_rootx() + parent.winfo_width()  // 2
        ph = parent.winfo_rooty() + parent.winfo_height() // 2
        self.geometry(f"{w}x{h}+{pw - w//2}+{ph - h//2}")

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build()
        self._start_refresh()

    # ── top-level build ────────────────────────────────────────────────────────
    def _build(self):
        ctk.CTkLabel(self, text="Settings", font=("Segoe UI", 16, "bold"),
                     text_color="#E2E8F0").pack(pady=(18, 6))

        self._tabs = ctk.CTkTabview(self, fg_color="#0D1626",
                                    segmented_button_fg_color="#131F35",
                                    segmented_button_selected_color="#2563EB",
                                    segmented_button_unselected_color="#1E293B",
                                    segmented_button_selected_hover_color="#1D4ED8")
        self._tabs.pack(fill="both", expand=True, padx=16, pady=0)

        self._tabs.add("General")
        self._tabs.add("YOLO26")
        self._tabs.add("Dashboard")

        self._build_general(self._tabs.tab("General"))
        self._build_yolo26(self._tabs.tab("YOLO26"))
        self._build_dashboard(self._tabs.tab("Dashboard"))

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

        # RTSP
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

        # Detection
        self._add_section(scroll, "Detection")
        self._conf = self._add_slider(
            scroll, "Confidence Threshold",
            self._settings.get("detection", "confidence", 0.45), 0.05, 0.95)
        self._iou = self._add_slider(
            scroll, "IOU Threshold",
            self._settings.get("detection", "iou", 0.45), 0.05, 0.95)
        self._input_size = self._add_option(
            scroll, "Input Resolution",
            ["320", "416", "512", "640", "736", "832"],
            str(self._settings.get("detection", "input_width", 640)))
        self._frame_skip = self._add_option(
            scroll, "Frame Skip",
            ["1", "2", "3", "4", "5"],
            str(self._settings.get("detection", "frame_skip", 1)))
        self._max_fps = self._add_entry(
            scroll, "Maximum FPS",
            str(self._settings.get("detection", "max_fps", 60)))
        self._tracking = self._add_switch(
            scroll, "Enable Tracking",
            self._settings.get("detection", "enable_tracking", True))

        # Inference
        self._add_section(scroll, "Inference")
        self._use_gpu  = self._add_switch(
            scroll, "Enable GPU",
            self._settings.get("inference", "use_gpu", False))
        self._use_fp16 = self._add_switch(
            scroll, "FP16 Precision",
            self._settings.get("inference", "use_fp16", False))
        self._use_onnx = self._add_switch(
            scroll, "Use ONNX Runtime",
            self._settings.get("inference", "use_onnx", True))

        # Model selector
        self._add_section(scroll, "YOLO26 Model")
        self._model_name = self._add_model_selector(scroll)

    # ── Tab 2: YOLO26 Settings ─────────────────────────────────────────────────
    def _build_yolo26(self, tab):
        scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        self._add_section(scroll, "Model Information")

        self._y26_model_var   = ctk.StringVar(value="—")
        self._y26_version_var = ctk.StringVar(value=self._settings.get("inference", "model_name", "yolo26n.pt"))
        self._y26_size_var    = ctk.StringVar(value="—")
        self._y26_path_var    = ctk.StringVar(value="—")
        self._y26_cache_var   = ctk.StringVar(value="models/")
        self._y26_onnx_var    = ctk.StringVar(value="—")

        info_frame = ctk.CTkFrame(scroll, fg_color="#0A0F1E", corner_radius=8)
        info_frame.pack(fill="x", pady=(0, 8))
        self._info_row(info_frame, "Model Name",    self._y26_version_var)
        self._info_row(info_frame, "Version",       ctk.StringVar(value="YOLO26 (Ultralytics v8.4.0)"))
        self._info_row(info_frame, "Model Size",    self._y26_size_var)
        self._info_row(info_frame, "Model Path",    self._y26_path_var)
        self._info_row(info_frame, "Cache Folder",  self._y26_cache_var)
        self._info_row(info_frame, "ONNX File",     self._y26_onnx_var)

        self._add_section(scroll, "Inference Parameters")
        self._cpu_threads = self._add_entry(
            scroll, "CPU Threads (0 = auto)",
            str(self._settings.get("inference", "cpu_threads", 0)))

        self._add_section(scroll, "ONNX Runtime")
        self._y26_ort_status_var = ctk.StringVar(value="Checking …")
        ort_frame = ctk.CTkFrame(scroll, fg_color="#0A0F1E", corner_radius=8)
        ort_frame.pack(fill="x", pady=(0, 8))
        self._info_row(ort_frame, "ORT Status",  self._y26_ort_status_var)
        self._y26_device_var = ctk.StringVar(value="CPU")
        self._info_row(ort_frame, "Device", self._y26_device_var)

        # Refresh model info immediately
        self._refresh_yolo26_info()

        self._add_section(scroll, "Export ONNX")
        exp_note = ctk.CTkLabel(scroll,
            text="ONNX export happens automatically on first Start.\n"
                 "Use the button below to force a re-export.",
            font=("Segoe UI", 10), text_color="#64748B",
            justify="left", wraplength=520)
        exp_note.pack(anchor="w", pady=(0, 6))

        self._export_btn = ctk.CTkButton(
            scroll, text="Export to ONNX Now",
            fg_color="#1E3A5F", hover_color="#334155",
            height=30, command=self._on_export_onnx)
        self._export_btn.pack(anchor="w", pady=(0, 4))
        self._export_status = ctk.CTkLabel(scroll, text="",
            font=("Segoe UI", 10), text_color="#64748B", anchor="w")
        self._export_status.pack(fill="x")

    # ── Tab 3: Performance Dashboard ───────────────────────────────────────────
    def _build_dashboard(self, tab):
        scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        self._add_section(scroll, "Live FPS")
        fps_frame = ctk.CTkFrame(scroll, fg_color="#0A0F1E", corner_radius=8)
        fps_frame.pack(fill="x", pady=(0, 8))

        self._d_fps_cur  = ctk.StringVar(value="0.0")
        self._d_fps_avg  = ctk.StringVar(value="0.0")
        self._d_fps_rgb  = ctk.StringVar(value="0.0")
        self._d_fps_th   = ctk.StringVar(value="0.0")
        self._info_row(fps_frame, "Current FPS",    self._d_fps_cur,  "#22C55E")
        self._info_row(fps_frame, "Average FPS",    self._d_fps_avg,  "#22C55E")
        self._info_row(fps_frame, "RGB Stream FPS", self._d_fps_rgb,  "#3B82F6")
        self._info_row(fps_frame, "Thermal FPS",    self._d_fps_th,   "#F97316")

        self._add_section(scroll, "Inference")
        inf_frame = ctk.CTkFrame(scroll, fg_color="#0A0F1E", corner_radius=8)
        inf_frame.pack(fill="x", pady=(0, 8))
        self._d_inf_ms   = ctk.StringVar(value="0 ms")
        self._d_model_ld = ctk.StringVar(value="No")
        self._d_onnx_st  = ctk.StringVar(value="—")
        self._d_device   = ctk.StringVar(value="CPU")
        self._info_row(inf_frame, "Inference Time",  self._d_inf_ms,   "#F59E0B")
        self._info_row(inf_frame, "Model Loaded",    self._d_model_ld, "#22C55E")
        self._info_row(inf_frame, "ONNX Active",     self._d_onnx_st,  "#A78BFA")
        self._info_row(inf_frame, "Device",          self._d_device,   "#CBD5E1")

        self._add_section(scroll, "System")
        sys_frame = ctk.CTkFrame(scroll, fg_color="#0A0F1E", corner_radius=8)
        sys_frame.pack(fill="x", pady=(0, 8))
        self._d_cpu      = ctk.StringVar(value="0%")
        self._d_ram      = ctk.StringVar(value="0 MB")
        self._d_threads  = ctk.StringVar(value="0")
        self._d_qsize    = ctk.StringVar(value="0")
        self._d_drops    = ctk.StringVar(value="0")
        self._info_row(sys_frame, "CPU Usage",      self._d_cpu,     "#F59E0B")
        self._info_row(sys_frame, "RAM Usage",      self._d_ram,     "#3B82F6")
        self._info_row(sys_frame, "Active Threads", self._d_threads, "#A78BFA")
        self._info_row(sys_frame, "Queue Size",     self._d_qsize,   "#64748B")
        self._info_row(sys_frame, "Frame Drops",    self._d_drops,   "#EF4444")

        ctk.CTkLabel(scroll, text="⟳ Auto-refreshes every second",
                     font=("Segoe UI", 9), text_color="#334155").pack(
                     anchor="w", pady=(4, 0))

    # ── Model selector ─────────────────────────────────────────────────────────
    def _add_model_selector(self, parent):
        current = self._settings.get("inference", "model_name", "yolo26n.pt")
        var = ctk.StringVar(value=current)

        row1 = ctk.CTkFrame(parent, fg_color="transparent")
        row1.pack(fill="x", pady=(2, 0))
        ctk.CTkLabel(row1, text="Model", font=("Segoe UI", 11),
                     text_color="#94A3B8", width=200, anchor="w").pack(side="left")
        menu = ctk.CTkOptionMenu(row1, values=SUPPORTED_MODELS, variable=var,
                                 width=220, fg_color="#131F35",
                                 button_color="#1E3A5F",
                                 dropdown_fg_color="#131F35",
                                 command=self._on_model_change)
        menu.pack(side="right")

        row2 = ctk.CTkFrame(parent, fg_color="transparent")
        row2.pack(fill="x", pady=(4, 0))
        self._dl_btn = ctk.CTkButton(
            row2, text="⬇ Download Model", width=160,
            fg_color="#1E3A5F", hover_color="#334155",
            font=("Segoe UI", 10, "bold"), height=28,
            command=self._on_download)
        self._dl_btn.pack(side="left")
        self._dl_bar = ctk.CTkProgressBar(row2, width=180,
                                          fg_color="#131F35",
                                          progress_color="#2563EB")
        self._dl_bar.set(0)
        self._dl_bar.pack(side="left", padx=(8, 0))

        self._dl_status = ctk.CTkLabel(parent, text="",
                                       font=("Segoe UI", 10),
                                       text_color="#64748B", anchor="w")
        self._dl_status.pack(fill="x", pady=(2, 6))

        self._model_var = var
        self._update_download_ui(current)
        return var

    def _on_model_change(self, name: str):
        self._update_download_ui(name)
        self._y26_version_var.set(name)
        self._refresh_yolo26_info()

    def _update_download_ui(self, name: str):
        if self._model_mgr.is_downloaded(name):
            self._dl_btn.configure(state="disabled", text="✓ Already Downloaded")
            self._dl_status.configure(
                text=f"{name} is ready — no download needed.",
                text_color="#22C55E")
        else:
            self._dl_btn.configure(state="normal", text="⬇ Download Model")
            self._dl_status.configure(
                text=f"{name} not found locally — click Download.",
                text_color="#F59E0B")
        self._dl_bar.set(0)

    def _on_download(self):
        name = self._model_var.get()
        self._dl_btn.configure(state="disabled", text="Downloading …")
        self._dl_status.configure(text="Starting download …", text_color="#94A3B8")
        self._dl_bar.set(0)
        self._model_mgr.set_callbacks(
            progress=self._on_dl_progress,
            status=self._on_dl_status)
        threading.Thread(target=self._do_download, args=(name,),
                         daemon=True).start()

    def _do_download(self, name: str):
        self._model_mgr.ensure_model(name, blocking=True)
        try:
            self.after(0, lambda: self._update_download_ui(name))
            self.after(0, self._refresh_yolo26_info)
        except Exception:
            pass

    def _on_dl_progress(self, pct: float):
        try:
            self.after(0, lambda p=pct: self._dl_bar.set(p / 100.0))
        except Exception:
            pass

    def _on_dl_status(self, msg: str):
        try:
            self.after(0, lambda m=msg: self._dl_status.configure(
                text=m, text_color="#94A3B8"))
        except Exception:
            pass

    # ── YOLO26 tab helpers ─────────────────────────────────────────────────────
    def _refresh_yolo26_info(self):
        try:
            name = self._settings.get("inference", "model_name", "yolo26n.pt")
            try:
                name = self._model_var.get()
            except Exception:
                pass
            info = self._model_mgr.get_model_info(name)
            self._y26_version_var.set(name)
            self._y26_size_var.set(f"{info['size_mb']} MB")
            self._y26_path_var.set(info["path"])
            self._y26_cache_var.set(info["cache"])
            self._y26_onnx_var.set(info["onnx"])

            try:
                import onnxruntime as ort
                ver = ort.__version__
                self._y26_ort_status_var.set(f"Available  (v{ver})")
            except ImportError:
                self._y26_ort_status_var.set("Not installed")

            if self._detector is not None:
                self._y26_device_var.set(getattr(self._detector, "device", "cpu").upper())
        except Exception:
            pass

    def _on_export_onnx(self):
        name = self._settings.get("inference", "model_name", "yolo26n.pt")
        try:
            name = self._model_var.get()
        except Exception:
            pass
        self._export_btn.configure(state="disabled", text="Exporting …")
        self._export_status.configure(text="Exporting — this may take 30–60 s …",
                                      text_color="#F59E0B")

        def _do():
            result = self._model_mgr.export_onnx(name)
            def _done():
                self._export_btn.configure(state="normal",
                                           text="Export to ONNX Now")
                if result:
                    self._export_status.configure(
                        text=f"Exported: {result.name}", text_color="#22C55E")
                    self._refresh_yolo26_info()
                else:
                    self._export_status.configure(
                        text="Export failed — check logs.", text_color="#EF4444")
            try:
                self.after(0, _done)
            except Exception:
                pass
        threading.Thread(target=_do, daemon=True).start()

    # ── Dashboard auto-refresh ─────────────────────────────────────────────────
    def _start_refresh(self):
        self._refresh_dashboard()

    def _refresh_dashboard(self):
        try:
            self._do_refresh_dashboard()
        except Exception:
            pass
        try:
            self._refresh_after_id = self.after(1000, self._refresh_dashboard)
        except Exception:
            pass

    def _do_refresh_dashboard(self):
        det = self._detector

        if det is not None:
            self._d_fps_cur.set(f"{det.fps_inference:.1f}")
            self._d_fps_avg.set(f"{det.avg_fps:.1f}")
            self._d_fps_rgb.set(f"{det.fps_rgb:.1f}")
            self._d_fps_th.set(f"{det.fps_thermal:.1f}")

            rgb_res = det.get_rgb_result()
            inf_ms = rgb_res.inference_ms if rgb_res else 0.0
            self._d_inf_ms.set(f"{inf_ms:.1f} ms")
            self._d_model_ld.set("Yes" if det.is_loaded else "No")
            self._d_onnx_st.set("Active" if det.onnx_active else "PyTorch fallback")
            self._d_device.set(det.device.upper())
            self._d_threads.set(str(det.active_threads))
            self._d_qsize.set(str(det.queue_size))
            self._d_drops.set(str(det.frame_drops))
        else:
            self._d_fps_cur.set("—")
            self._d_fps_avg.set("—")
            self._d_fps_rgb.set("—")
            self._d_fps_th.set("—")
            self._d_inf_ms.set("—")
            self._d_model_ld.set("No")
            self._d_onnx_st.set("—")
            self._d_device.set("—")
            self._d_threads.set("—")
            self._d_qsize.set("—")
            self._d_drops.set("—")

        try:
            import psutil
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().used // (1024 * 1024)
            self._d_cpu.set(f"{cpu:.0f}%")
            self._d_ram.set(f"{ram} MB")
        except Exception:
            pass

    # ── Field helpers ──────────────────────────────────────────────────────────
    def _add_section(self, parent, text):
        ctk.CTkLabel(parent, text=text.upper(), font=("Segoe UI", 9, "bold"),
                     text_color="#2563EB").pack(anchor="w", pady=(12, 2))
        ctk.CTkFrame(parent, height=1, fg_color="#1E3A5F").pack(
            fill="x", pady=(0, 6))

    def _add_entry(self, parent, label, value):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=3)
        ctk.CTkLabel(row, text=label, font=("Segoe UI", 11),
                     text_color="#94A3B8", width=200, anchor="w").pack(side="left")
        entry = ctk.CTkEntry(row, width=240,
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
        ctk.CTkSwitch(row, text="", variable=var,
                      onvalue=True, offvalue=False,
                      fg_color="#1E293B",
                      progress_color="#2563EB").pack(side="right")
        return var

    def _info_row(self, parent, label, var, color="#CBD5E1"):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=2)
        ctk.CTkLabel(row, text=label, font=("Segoe UI", 10),
                     text_color="#64748B", width=140, anchor="w").pack(side="left")
        ctk.CTkLabel(row, textvariable=var, font=("Segoe UI", 10, "bold"),
                     text_color=color, anchor="w", wraplength=360).pack(
                     side="left", fill="x", expand=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    def _save(self):
        try:
            self._settings.set("rtsp", "rgb_url",         self._rgb_url.get())
            self._settings.set("rtsp", "thermal_url",     self._thermal_url.get())
            self._settings.set("rtsp", "reconnect_delay", float(self._reconnect_delay.get()))
            self._settings.set("detection", "confidence",      float(self._conf.get()))
            self._settings.set("detection", "iou",             float(self._iou.get()))
            size = int(self._input_size.get())
            self._settings.set("detection", "input_width",     size)
            self._settings.set("detection", "input_height",    size)
            self._settings.set("detection", "frame_skip",      int(self._frame_skip.get()))
            self._settings.set("detection", "max_fps",         int(self._max_fps.get()))
            self._settings.set("detection", "enable_tracking", self._tracking.get())
            self._settings.set("inference", "use_gpu",         self._use_gpu.get())
            self._settings.set("inference", "use_fp16",        self._use_fp16.get())
            self._settings.set("inference", "use_onnx",        self._use_onnx.get())
            self._settings.set("inference", "model_name",      self._model_name.get())
            try:
                threads = int(self._cpu_threads.get())
                self._settings.set("inference", "cpu_threads", threads)
            except Exception:
                pass
            self._settings.save()
            if self._on_save:
                self._on_save()
            self._on_close()
        except Exception as e:
            ctk.CTkLabel(self, text=f"Error: {e}", text_color="#EF4444").pack(pady=4)

    def _on_close(self):
        if self._refresh_after_id:
            try:
                self.after_cancel(self._refresh_after_id)
            except Exception:
                pass
        self.destroy()
