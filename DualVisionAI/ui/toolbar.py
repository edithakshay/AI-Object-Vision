import customtkinter as ctk
from tkinter import filedialog, messagebox


class Toolbar(ctk.CTkFrame):
    def __init__(self, parent, callbacks: dict, **kwargs):
        super().__init__(parent, fg_color="#080E1C", height=48,
                         corner_radius=0, **kwargs)
        self.pack_propagate(False)
        self._callbacks = callbacks
        self._is_detecting = False
        self._is_paused = False
        self._is_recording = False
        self._build()

    def _build(self):
        left = ctk.CTkFrame(self, fg_color="transparent")
        left.pack(side="left", padx=8, pady=6)

        logo_label = ctk.CTkLabel(left, text="◈ DualVision AI",
                                   font=("Segoe UI", 13, "bold"),
                                   text_color="#2563EB")
        logo_label.pack(side="left", padx=(0, 20))

        self._btn_start = self._btn(left, "▶  Start", "#22C55E", "#16A34A",
                                    self._on_start)
        self._btn_stop = self._btn(left, "■  Stop", "#EF4444", "#DC2626",
                                   self._on_stop, state="disabled")
        self._btn_pause = self._btn(left, "⏸  Pause", "#F59E0B", "#D97706",
                                    self._on_pause, state="disabled")

        ctk.CTkFrame(left, width=1, fg_color="#1E3A5F").pack(side="left",
                                                               fill="y", padx=10)

        self._btn(left, "📷  Screenshot", "#1E3A5F", "#334155",
                  lambda: self._call("screenshot"))
        self._btn_rec = self._btn(left, "⏺  Record", "#1E3A5F", "#334155",
                                   self._on_record)

        ctk.CTkFrame(left, width=1, fg_color="#1E3A5F").pack(side="left",
                                                               fill="y", padx=10)
        self._btn(left, "⚙  Settings", "#1E3A5F", "#334155",
                  lambda: self._call("settings"))

        right = ctk.CTkFrame(self, fg_color="transparent")
        right.pack(side="right", padx=8, pady=6)

        self._btn(right, "Export CSV", "#1E3A5F", "#334155",
                  lambda: self._call("export_csv"))
        self._btn(right, "Export JSON", "#1E3A5F", "#334155",
                  lambda: self._call("export_json"))
        ctk.CTkFrame(right, width=1, fg_color="#1E3A5F").pack(side="left",
                                                                fill="y", padx=6)
        self._btn(right, "About", "#1E3A5F", "#334155",
                  lambda: self._call("about"))
        self._btn(right, "✕  Exit", "#7F1D1D", "#991B1B",
                  lambda: self._call("exit"))

    def _btn(self, parent, text, fg, hover, cmd, state="normal"):
        b = ctk.CTkButton(parent, text=text, command=cmd,
                          fg_color=fg, hover_color=hover,
                          font=("Segoe UI", 10, "bold"),
                          height=32, corner_radius=6,
                          state=state)
        b.pack(side="left", padx=3)
        return b

    def _call(self, name, *args):
        cb = self._callbacks.get(name)
        if cb:
            cb(*args)

    def _on_start(self):
        self._is_detecting = True
        self._is_paused = False
        self._btn_start.configure(state="disabled")
        self._btn_stop.configure(state="normal")
        self._btn_pause.configure(state="normal")
        self._call("start")

    def _on_stop(self):
        self._is_detecting = False
        self._is_paused = False
        self._is_recording = False
        self._btn_start.configure(state="normal")
        self._btn_stop.configure(state="disabled")
        self._btn_pause.configure(state="disabled")
        self._btn_rec.configure(text="⏺  Record", fg_color="#1E3A5F")
        self._call("stop")

    def _on_pause(self):
        if not self._is_paused:
            self._is_paused = True
            self._btn_pause.configure(text="▶  Resume")
            self._call("pause")
        else:
            self._is_paused = False
            self._btn_pause.configure(text="⏸  Pause")
            self._call("resume")

    def _on_record(self):
        if not self._is_recording:
            self._is_recording = True
            self._btn_rec.configure(text="⏹  Stop Rec", fg_color="#DC2626")
            self._call("record_start")
        else:
            self._is_recording = False
            self._btn_rec.configure(text="⏺  Record", fg_color="#1E3A5F")
            self._call("record_stop")

    def set_detecting(self, detecting: bool):
        self._is_detecting = detecting
        if detecting:
            self._btn_start.configure(state="disabled")
            self._btn_stop.configure(state="normal")
            self._btn_pause.configure(state="normal")
        else:
            self._btn_start.configure(state="normal")
            self._btn_stop.configure(state="disabled")
            self._btn_pause.configure(state="disabled")
