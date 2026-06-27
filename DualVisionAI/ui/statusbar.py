import customtkinter as ctk


class StatusBar(ctk.CTkFrame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color="#050A14", height=28,
                         corner_radius=0, **kwargs)
        self.pack_propagate(False)
        self._vars = {}
        self._build()

    def _build(self):
        items = [
            ("fps", "FPS: 0.0", "#22C55E"),
            ("model", "Model: —", "#CBD5E1"),
            ("device", "Device: CPU", "#A78BFA"),
            ("resolution", "Res: —", "#64748B"),
            ("det_status", "Stopped", "#EF4444"),
            ("rgb_status", "RGB: Disconnected", "#94A3B8"),
            ("thermal_status", "Thermal: Disconnected", "#94A3B8"),
        ]

        for key, text, color in items:
            var = ctk.StringVar(value=text)
            self._vars[key] = var
            lbl = ctk.CTkLabel(self, textvariable=var,
                               font=("Consolas", 9),
                               text_color=color)
            lbl.pack(side="left", padx=12)
            ctk.CTkFrame(self, width=1, fg_color="#1E293B").pack(side="left", fill="y")

        self._vars["time"] = ctk.StringVar(value="")
        ctk.CTkLabel(self, textvariable=self._vars["time"],
                     font=("Consolas", 9), text_color="#334155").pack(side="right", padx=10)

    def update(self, fps: float = None, model: str = None, device: str = None,
               resolution: str = None, detecting: bool = None,
               rgb_connected: bool = None, thermal_connected: bool = None,
               rgb_status: str = None, thermal_status: str = None,
               timestamp: str = None):
        if fps is not None:
            self._vars["fps"].set(f"FPS: {fps:.1f}")
        if model is not None:
            self._vars["model"].set(f"Model: {model}")
        if device is not None:
            self._vars["device"].set(f"Device: {device}")
        if resolution is not None:
            self._vars["resolution"].set(f"Res: {resolution}")
        if detecting is not None:
            txt = "Detecting" if detecting else "Stopped"
            color = "#22C55E" if detecting else "#EF4444"
            self._vars["det_status"].set(txt)
        if rgb_connected is not None or rgb_status is not None:
            s = rgb_status or ("Connected" if rgb_connected else "Disconnected")
            self._vars["rgb_status"].set(f"RGB: {s}")
        if thermal_connected is not None or thermal_status is not None:
            s = thermal_status or ("Connected" if thermal_connected else "Disconnected")
            self._vars["thermal_status"].set(f"Thermal: {s}")
        if timestamp is not None:
            self._vars["time"].set(timestamp)
