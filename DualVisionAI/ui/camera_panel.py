import customtkinter as ctk
from PIL import Image, ImageTk
import numpy as np
import cv2
import threading
import time


class CameraPanel(ctk.CTkFrame):
    """Single camera display panel with zoom/pan and double-click fullscreen."""

    def __init__(self, parent, title: str = "Camera", **kwargs):
        super().__init__(parent, fg_color="#0A0F1E",
                         corner_radius=10, border_width=1,
                         border_color="#1E3A5F", **kwargs)
        self.title = title

        self._current_frame = None
        self._lock = threading.Lock()
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._drag_start = None
        self._fullscreen_win = None

        self._build()
        self._bind_events()

    def _build(self):
        header = ctk.CTkFrame(self, fg_color="#0D1626", corner_radius=0, height=32)
        header.pack(fill="x")
        header.pack_propagate(False)

        self._title_label = ctk.CTkLabel(
            header, text=self.title,
            font=("Segoe UI", 11, "bold"),
            text_color="#CBD5E1"
        )
        self._title_label.pack(side="left", padx=10)

        self._status_dot = ctk.CTkLabel(header, text="●", font=("Segoe UI", 10),
                                         text_color="#EF4444")
        self._status_dot.pack(side="right", padx=6)
        self._status_label = ctk.CTkLabel(header, text="Disconnected",
                                           font=("Segoe UI", 10), text_color="#94A3B8")
        self._status_label.pack(side="right")

        self._fps_label = ctk.CTkLabel(header, text="0 FPS",
                                        font=("Segoe UI", 10), text_color="#64748B")
        self._fps_label.pack(side="right", padx=10)

        self._canvas = ctk.CTkCanvas(self, bg="#050A14", highlightthickness=0)
        self._canvas.pack(fill="both", expand=True, padx=1, pady=(0, 1))

        self._canvas.create_text(
            10, 10, anchor="nw",
            text="Waiting for stream...",
            fill="#334155", font=("Segoe UI", 14),
            tags="placeholder"
        )

        self._info_label = ctk.CTkLabel(
            self, text="",
            font=("Consolas", 10),
            text_color="#2563EB",
            fg_color="transparent"
        )
        self._info_label.place(x=6, y=36)

    def _bind_events(self):
        self._canvas.bind("<MouseWheel>", self._on_scroll)
        self._canvas.bind("<Button-4>", self._on_scroll)
        self._canvas.bind("<Button-5>", self._on_scroll)
        self._canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self._canvas.bind("<B1-Motion>", self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_drag_end)
        self._canvas.bind("<Double-Button-1>", self._on_double_click)

    def update_frame(self, frame_bgr, fps: float = 0.0,
                     det_count: int = 0, inf_ms: float = 0.0):
        with self._lock:
            self._current_frame = frame_bgr
        self._render_frame(frame_bgr, fps, det_count, inf_ms)

    def set_status(self, connected: bool, status_text: str = ""):
        color = "#22C55E" if connected else "#EF4444"
        self._status_dot.configure(text_color=color)
        self._status_label.configure(text=status_text or ("Connected" if connected else "Disconnected"))

    def _render_frame(self, frame, fps, det_count, inf_ms):
        if frame is None:
            return
        try:
            cw = self._canvas.winfo_width()
            ch = self._canvas.winfo_height()
            if cw < 10 or ch < 10:
                return

            h, w = frame.shape[:2]
            scale = min(cw / w, ch / h) * self._zoom
            nw = int(w * scale)
            nh = int(h * scale)

            ox = int((cw - nw) / 2 + self._pan_x)
            oy = int((ch - nh) / 2 + self._pan_y)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            pil_img = pil_img.resize((nw, nh), Image.BILINEAR)

            self._photo = ImageTk.PhotoImage(pil_img)
            self._canvas.delete("all")
            self._canvas.create_image(ox, oy, anchor="nw", image=self._photo)

            overlay = f"FPS:{fps:4.1f}  Det:{det_count}  Inf:{inf_ms:.0f}ms"
            self._canvas.create_text(6, 6, anchor="nw", text=overlay,
                                     fill="#2563EB", font=("Consolas", 9))
            self._fps_label.configure(text=f"{fps:.0f} FPS")
        except Exception:
            pass

    def get_current_frame(self):
        with self._lock:
            return self._current_frame.copy() if self._current_frame is not None else None

    def _on_scroll(self, event):
        delta = 0
        if hasattr(event, "delta") and event.delta:
            delta = event.delta / 120
        elif event.num == 4:
            delta = 1
        elif event.num == 5:
            delta = -1
        self._zoom = max(0.5, min(6.0, self._zoom + delta * 0.15))

    def _on_drag_start(self, event):
        self._drag_start = (event.x, event.y)

    def _on_drag(self, event):
        if self._drag_start:
            dx = event.x - self._drag_start[0]
            dy = event.y - self._drag_start[1]
            self._pan_x += dx
            self._pan_y += dy
            self._drag_start = (event.x, event.y)

    def _on_drag_end(self, event):
        self._drag_start = None

    def _on_double_click(self, event):
        if self._fullscreen_win and self._fullscreen_win.winfo_exists():
            self._fullscreen_win.destroy()
            self._fullscreen_win = None
            return

        frame = self.get_current_frame()
        if frame is None:
            return

        win = ctk.CTkToplevel(self)
        win.title(self.title)
        win.attributes("-fullscreen", True)
        win.configure(fg_color="black")
        win.bind("<Escape>", lambda e: win.destroy())
        win.bind("<Double-Button-1>", lambda e: win.destroy())
        self._fullscreen_win = win

        canvas = ctk.CTkCanvas(win, bg="black", highlightthickness=0)
        canvas.pack(fill="both", expand=True)

        def _render():
            while win.winfo_exists():
                with self._lock:
                    f = self._current_frame
                if f is not None:
                    try:
                        cw = canvas.winfo_width()
                        ch = canvas.winfo_height()
                        if cw > 10 and ch > 10:
                            h, w = f.shape[:2]
                            scale = min(cw / w, ch / h)
                            nw, nh = int(w * scale), int(h * scale)
                            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                            pil = Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR)
                            photo = ImageTk.PhotoImage(pil)
                            ox = (cw - nw) // 2
                            oy = (ch - nh) // 2
                            canvas.delete("all")
                            canvas.create_image(ox, oy, anchor="nw", image=photo)
                            canvas._photo = photo
                    except Exception:
                        pass
                time.sleep(0.033)

        import threading as _t
        _t.Thread(target=_render, daemon=True).start()
