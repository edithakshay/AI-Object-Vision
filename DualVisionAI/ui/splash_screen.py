import customtkinter as ctk
import threading
import time


class SplashScreen(ctk.CTkToplevel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.title("")
        self.overrideredirect(True)
        self.configure(fg_color="#0A0F1E")

        w, h = 520, 320
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")
        self.lift()
        self.attributes("-topmost", True)

        self._build()

    def _build(self):
        frame = ctk.CTkFrame(self, fg_color="#0A0F1E", corner_radius=16,
                             border_width=1, border_color="#1E3A5F")
        frame.pack(fill="both", expand=True, padx=2, pady=2)

        ctk.CTkLabel(
            frame,
            text="◈",
            font=("Segoe UI", 56),
            text_color="#2563EB"
        ).pack(pady=(36, 4))

        ctk.CTkLabel(
            frame,
            text="DualVision AI Detector",
            font=("Segoe UI", 22, "bold"),
            text_color="#E2E8F0"
        ).pack()

        ctk.CTkLabel(
            frame,
            text="High-FPS Dual RTSP AI Object Detection",
            font=("Segoe UI", 12),
            text_color="#64748B"
        ).pack(pady=(4, 24))

        self._status_label = ctk.CTkLabel(
            frame,
            text="Initializing...",
            font=("Segoe UI", 11),
            text_color="#94A3B8"
        )
        self._status_label.pack()

        self._progress = ctk.CTkProgressBar(
            frame, width=380, height=6,
            fg_color="#1E293B",
            progress_color="#2563EB"
        )
        self._progress.set(0)
        self._progress.pack(pady=(10, 0))

        ctk.CTkLabel(
            frame,
            text="v1.0.0 • Python 3.12 • YOLO + CustomTkinter",
            font=("Segoe UI", 9),
            text_color="#334155"
        ).pack(side="bottom", pady=14)

    def set_status(self, text: str, progress: float = None):
        try:
            self._status_label.configure(text=text)
            if progress is not None:
                self._progress.set(min(1.0, progress))
            self.update()
        except Exception:
            pass

    def close(self):
        try:
            self.destroy()
        except Exception:
            pass
