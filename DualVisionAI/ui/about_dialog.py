import customtkinter as ctk


class AboutDialog(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("About DualVision AI Detector")
        self.resizable(False, False)
        self.configure(fg_color="#0D1626")
        self.grab_set()

        w, h = 460, 400
        pw = parent.winfo_rootx() + parent.winfo_width() // 2
        ph = parent.winfo_rooty() + parent.winfo_height() // 2
        self.geometry(f"{w}x{h}+{pw - w//2}+{ph - h//2}")

        self._build()

    def _build(self):
        ctk.CTkLabel(self, text="◈", font=("Segoe UI", 44), text_color="#2563EB").pack(pady=(28, 4))
        ctk.CTkLabel(self, text="DualVision AI Detector", font=("Segoe UI", 20, "bold"),
                     text_color="#E2E8F0").pack()
        ctk.CTkLabel(self, text="Version 1.0.0", font=("Segoe UI", 11),
                     text_color="#64748B").pack(pady=(2, 16))

        info = [
            ("Platform", "Windows 10/11 (x64)"),
            ("Python", "3.12"),
            ("AI Framework", "Ultralytics YOLO + ONNX Runtime"),
            ("UI", "CustomTkinter"),
            ("Tracking", "ByteTrack (built-in)"),
            ("Video", "OpenCV + FFmpeg (PyAV)"),
        ]

        frame = ctk.CTkFrame(self, fg_color="#131F35", corner_radius=10)
        frame.pack(padx=24, pady=0, fill="x")

        for label, value in info:
            row = ctk.CTkFrame(frame, fg_color="transparent")
            row.pack(fill="x", padx=16, pady=3)
            ctk.CTkLabel(row, text=label, font=("Segoe UI", 11), text_color="#64748B",
                         width=130, anchor="w").pack(side="left")
            ctk.CTkLabel(row, text=value, font=("Segoe UI", 11, "bold"),
                         text_color="#CBD5E1", anchor="w").pack(side="left")

        ctk.CTkLabel(self, text="High-FPS Dual RTSP AI Object Detection",
                     font=("Segoe UI", 10), text_color="#334155").pack(pady=(16, 4))

        ctk.CTkButton(self, text="Close", command=self.destroy,
                      fg_color="#2563EB", hover_color="#1D4ED8",
                      width=100).pack(pady=10)
