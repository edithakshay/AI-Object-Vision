"""
Settings dialog — includes an inline model download button so users can
download any model without leaving the app.
"""
import threading
import customtkinter as ctk
from config.settings import Settings
from ai.model_manager import ModelManager, SUPPORTED_MODELS, _DOWNLOADABLE


class SettingsDialog(ctk.CTkToplevel):
    def __init__(self, parent, settings: Settings, on_save=None):
        super().__init__(parent)
        self.title("Settings")
        self.resizable(False, False)
        self.configure(fg_color="#0D1626")
        self.grab_set()
        self._settings = settings
        self._on_save  = on_save
        self._model_mgr = ModelManager(model_dir="models")

        w, h = 560, 720
        pw = parent.winfo_rootx() + parent.winfo_width()  // 2
        ph = parent.winfo_rooty() + parent.winfo_height() // 2
        self.geometry(f"{w}x{h}+{pw - w//2}+{ph - h//2}")

        self._build()

    # ── UI construction ────────────────────────────────────────────────────────
    def _build(self):
        ctk.CTkLabel(self, text="Settings", font=("Segoe UI", 16, "bold"),
                     text_color="#E2E8F0").pack(pady=(18, 8))

        scroll = ctk.CTkScrollableFrame(self, fg_color="#0D1626")
        scroll.pack(fill="both", expand=True, padx=16, pady=0)

        # ── RTSP ──────────────────────────────────────────────────────────────
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

        # ── Detection ─────────────────────────────────────────────────────────
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

        # ── Inference ─────────────────────────────────────────────────────────
        self._add_section(scroll, "Inference")
        self._use_gpu = self._add_switch(
            scroll, "Enable GPU",
            self._settings.get("inference", "use_gpu", True))
        self._use_fp16 = self._add_switch(
            scroll, "FP16 Precision",
            self._settings.get("inference", "use_fp16", True))

        # Model selector + download button
        self._add_section(scroll, "Model")
        self._model_name = self._add_model_selector(scroll)

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=16, pady=12)
        ctk.CTkButton(btn_frame, text="Save Settings", command=self._save,
                      fg_color="#2563EB", hover_color="#1D4ED8",
                      height=36).pack(side="right", padx=4)
        ctk.CTkButton(btn_frame, text="Cancel", command=self.destroy,
                      fg_color="#1E293B", hover_color="#334155",
                      height=36).pack(side="right", padx=4)

    def _add_model_selector(self, parent):
        """Model dropdown + inline Download button + status label."""
        # Split into downloadable and future groups for display
        yolov8  = [m for m in SUPPORTED_MODELS if m.startswith("yolov8")]
        yolo11  = [m for m in SUPPORTED_MODELS if m.startswith("yolo11")]
        yolo26  = [m for m in SUPPORTED_MODELS if m.startswith("yolo26")]
        all_models = yolov8 + yolo11 + yolo26

        current = self._settings.get("inference", "model_name", "yolov8n.pt")
        var = ctk.StringVar(value=current)

        # Row 1: label + dropdown
        row1 = ctk.CTkFrame(parent, fg_color="transparent")
        row1.pack(fill="x", pady=(2, 0))
        ctk.CTkLabel(row1, text="Model", font=("Segoe UI", 11),
                     text_color="#94A3B8", width=200, anchor="w").pack(side="left")
        menu = ctk.CTkOptionMenu(row1, values=all_models, variable=var, width=220,
                                 fg_color="#131F35", button_color="#1E3A5F",
                                 dropdown_fg_color="#131F35",
                                 command=self._on_model_change)
        menu.pack(side="right")

        # Row 2: download button + progress bar
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

        # Row 3: status label
        self._dl_status = ctk.CTkLabel(parent, text="",
                                       font=("Segoe UI", 10),
                                       text_color="#64748B", anchor="w")
        self._dl_status.pack(fill="x", pady=(2, 6))

        self._model_var = var
        self._update_download_ui(current)
        return var

    def _on_model_change(self, name: str):
        self._update_download_ui(name)

    def _update_download_ui(self, name: str):
        if name.startswith("yolo26"):
            self._dl_btn.configure(state="disabled",
                                   text="⬇ Not Released Yet")
            self._dl_status.configure(
                text="YOLO26 is not yet released by Ultralytics.",
                text_color="#F59E0B")
        elif self._model_mgr.is_downloaded(name):
            self._dl_btn.configure(state="disabled",
                                   text="✓ Already Downloaded")
            self._dl_status.configure(
                text=f"{name} is ready — no download needed.",
                text_color="#22C55E")
        else:
            self._dl_btn.configure(state="normal",
                                   text="⬇ Download Model")
            self._dl_status.configure(
                text=f"{name} not found locally — click Download.",
                text_color="#F59E0B")
        self._dl_bar.set(0)

    def _on_download(self):
        name = self._model_var.get()
        if name.startswith("yolo26"):
            return
        self._dl_btn.configure(state="disabled", text="Downloading …")
        self._dl_status.configure(text="Starting download …",
                                  text_color="#94A3B8")
        self._dl_bar.set(0)

        self._model_mgr.set_callbacks(
            progress=self._on_dl_progress,
            status=self._on_dl_status)

        threading.Thread(target=self._do_download, args=(name,),
                         daemon=True).start()

    def _do_download(self, name: str):
        self._model_mgr.ensure_model(name, blocking=True)
        # Refresh UI on Tkinter thread
        try:
            self.after(0, lambda: self._update_download_ui(name))
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

    # ── Save ──────────────────────────────────────────────────────────────────
    def _save(self):
        try:
            self._settings.set("rtsp", "rgb_url",      self._rgb_url.get())
            self._settings.set("rtsp", "thermal_url",   self._thermal_url.get())
            self._settings.set("rtsp", "reconnect_delay",
                               float(self._reconnect_delay.get()))
            self._settings.set("detection", "confidence", float(self._conf.get()))
            self._settings.set("detection", "iou",        float(self._iou.get()))
            size = int(self._input_size.get())
            self._settings.set("detection", "input_width",  size)
            self._settings.set("detection", "input_height", size)
            self._settings.set("detection", "frame_skip",
                               int(self._frame_skip.get()))
            self._settings.set("detection", "max_fps",
                               int(self._max_fps.get()))
            self._settings.set("detection", "enable_tracking", self._tracking.get())
            self._settings.set("inference", "use_gpu",    self._use_gpu.get())
            self._settings.set("inference", "use_fp16",   self._use_fp16.get())
            self._settings.set("inference", "model_name", self._model_name.get())
            self._settings.save()
            if self._on_save:
                self._on_save()
            self.destroy()
        except Exception as e:
            ctk.CTkLabel(self, text=f"Error: {e}",
                         text_color="#EF4444").pack(pady=4)
