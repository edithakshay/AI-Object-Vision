import customtkinter as ctk
from config.settings import Settings


class SettingsDialog(ctk.CTkToplevel):
    def __init__(self, parent, settings: Settings, on_save=None):
        super().__init__(parent)
        self.title("Settings")
        self.resizable(False, False)
        self.configure(fg_color="#0D1626")
        self.grab_set()
        self._settings = settings
        self._on_save = on_save

        w, h = 520, 640
        pw = parent.winfo_rootx() + parent.winfo_width() // 2
        ph = parent.winfo_rooty() + parent.winfo_height() // 2
        self.geometry(f"{w}x{h}+{pw - w//2}+{ph - h//2}")

        self._build()

    def _build(self):
        ctk.CTkLabel(self, text="Settings", font=("Segoe UI", 16, "bold"),
                     text_color="#E2E8F0").pack(pady=(18, 8))

        scroll = ctk.CTkScrollableFrame(self, fg_color="#0D1626")
        scroll.pack(fill="both", expand=True, padx=16, pady=0)

        self._add_section(scroll, "RTSP Streams")
        self._rgb_url = self._add_entry(scroll, "RGB Stream URL",
                                        self._settings.get("rtsp", "rgb_url", ""))
        self._thermal_url = self._add_entry(scroll, "Thermal Stream URL",
                                            self._settings.get("rtsp", "thermal_url", ""))
        self._reconnect_delay = self._add_entry(scroll, "Reconnect Delay (s)",
                                                str(self._settings.get("rtsp", "reconnect_delay", 3)))

        self._add_section(scroll, "Detection")
        self._conf = self._add_slider(scroll, "Confidence Threshold",
                                      self._settings.get("detection", "confidence", 0.45),
                                      0.05, 0.95)
        self._iou = self._add_slider(scroll, "IOU Threshold",
                                     self._settings.get("detection", "iou", 0.45),
                                     0.05, 0.95)
        self._input_size = self._add_option(scroll, "Input Resolution",
                                            ["320", "416", "512", "640", "736", "832"],
                                            str(self._settings.get("detection", "input_width", 640)))
        self._frame_skip = self._add_option(scroll, "Frame Skip",
                                            ["1", "2", "3", "4", "5"],
                                            str(self._settings.get("detection", "frame_skip", 1)))
        self._max_fps = self._add_entry(scroll, "Maximum FPS",
                                        str(self._settings.get("detection", "max_fps", 60)))
        self._tracking = self._add_switch(scroll, "Enable Tracking",
                                          self._settings.get("detection", "enable_tracking", True))

        self._add_section(scroll, "Inference")
        self._use_gpu = self._add_switch(scroll, "Enable GPU",
                                         self._settings.get("inference", "use_gpu", True))
        self._use_fp16 = self._add_switch(scroll, "FP16 Precision",
                                          self._settings.get("inference", "use_fp16", True))
        self._model_name = self._add_option(scroll, "Model",
                                            ["yolov8n.pt", "yolov8s.pt", "yolo11n.pt"],
                                            self._settings.get("inference", "model_name", "yolov8n.pt"))

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=16, pady=12)
        ctk.CTkButton(btn_frame, text="Save Settings", command=self._save,
                      fg_color="#2563EB", hover_color="#1D4ED8").pack(side="right", padx=4)
        ctk.CTkButton(btn_frame, text="Cancel", command=self.destroy,
                      fg_color="#1E293B", hover_color="#334155").pack(side="right", padx=4)

    def _add_section(self, parent, text):
        ctk.CTkLabel(parent, text=text.upper(), font=("Segoe UI", 9, "bold"),
                     text_color="#2563EB").pack(anchor="w", pady=(12, 2))
        ctk.CTkFrame(parent, height=1, fg_color="#1E3A5F").pack(fill="x", pady=(0, 6))

    def _add_entry(self, parent, label, value):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=3)
        ctk.CTkLabel(row, text=label, font=("Segoe UI", 11), text_color="#94A3B8",
                     width=200, anchor="w").pack(side="left")
        entry = ctk.CTkEntry(row, width=240, fg_color="#131F35", border_color="#1E3A5F")
        entry.insert(0, value)
        entry.pack(side="right")
        return entry

    def _add_slider(self, parent, label, value, from_, to):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=3)
        ctk.CTkLabel(row, text=label, font=("Segoe UI", 11), text_color="#94A3B8",
                     width=200, anchor="w").pack(side="left")
        val_label = ctk.CTkLabel(row, text=f"{value:.2f}", font=("Segoe UI", 11),
                                 text_color="#CBD5E1", width=40)
        val_label.pack(side="right")
        slider = ctk.CTkSlider(row, from_=from_, to=to, width=160,
                               fg_color="#1E293B", progress_color="#2563EB",
                               button_color="#3B82F6")
        slider.set(value)
        slider.configure(command=lambda v: val_label.configure(text=f"{v:.2f}"))
        slider.pack(side="right", padx=6)
        return slider

    def _add_option(self, parent, label, values, current):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=3)
        ctk.CTkLabel(row, text=label, font=("Segoe UI", 11), text_color="#94A3B8",
                     width=200, anchor="w").pack(side="left")
        var = ctk.StringVar(value=current)
        menu = ctk.CTkOptionMenu(row, values=values, variable=var, width=160,
                                 fg_color="#131F35", button_color="#1E3A5F",
                                 dropdown_fg_color="#131F35")
        menu.pack(side="right")
        return var

    def _add_switch(self, parent, label, value):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=3)
        ctk.CTkLabel(row, text=label, font=("Segoe UI", 11), text_color="#94A3B8",
                     width=200, anchor="w").pack(side="left")
        var = ctk.BooleanVar(value=value)
        switch = ctk.CTkSwitch(row, text="", variable=var,
                               onvalue=True, offvalue=False,
                               fg_color="#1E293B", progress_color="#2563EB")
        switch.pack(side="right")
        return var

    def _save(self):
        try:
            self._settings.set("rtsp", "rgb_url", self._rgb_url.get())
            self._settings.set("rtsp", "thermal_url", self._thermal_url.get())
            self._settings.set("rtsp", "reconnect_delay", float(self._reconnect_delay.get()))
            self._settings.set("detection", "confidence", float(self._conf.get()))
            self._settings.set("detection", "iou", float(self._iou.get()))
            size = int(self._input_size.get())
            self._settings.set("detection", "input_width", size)
            self._settings.set("detection", "input_height", size)
            self._settings.set("detection", "frame_skip", int(self._frame_skip.get()))
            self._settings.set("detection", "max_fps", int(self._max_fps.get()))
            self._settings.set("detection", "enable_tracking", self._tracking.get())
            self._settings.set("inference", "use_gpu", self._use_gpu.get())
            self._settings.set("inference", "use_fp16", self._use_fp16.get())
            self._settings.set("inference", "model_name", self._model_name.get())
            self._settings.save()
            if self._on_save:
                self._on_save()
            self.destroy()
        except Exception as e:
            ctk.CTkLabel(self, text=f"Error: {e}", text_color="#EF4444").pack(pady=4)
