"""
Main toolbar.
Layout rule: pack RIGHT-side buttons FIRST so they always get their space.
Then pack the LEFT-side content — it fills whatever remains.
"""
import customtkinter as ctk
from pathlib import Path


_LOGO_IMG = None          # cached CTkImage — module-level to survive GC


def _load_logo(size: int = 32):
    """Return a CTkImage of the logo at `size`×`size`, or None on failure."""
    global _LOGO_IMG
    if _LOGO_IMG is not None:
        return _LOGO_IMG
    try:
        from PIL import Image as PILImage
        import customtkinter as _ctk

        assets = Path(__file__).parent.parent / "assets"
        logo_png = assets / "logo.png"

        # Auto-generate if missing
        if not logo_png.exists():
            from assets.make_icon import generate
            generate(assets)

        if logo_png.exists():
            pil_img = PILImage.open(str(logo_png)).resize((size, size))
            _LOGO_IMG = _ctk.CTkImage(light_image=pil_img,
                                      dark_image=pil_img,
                                      size=(size, size))
            return _LOGO_IMG
    except Exception:
        pass
    return None


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
        # ── RIGHT side packed FIRST so it is never clipped ──────────────
        right = ctk.CTkFrame(self, fg_color="transparent")
        right.pack(side="right", padx=6, pady=6)

        self._btn(right, "CSV",   "#1E3A5F", "#334155",
                  lambda: self._call("export_csv"))
        self._btn(right, "JSON",  "#1E3A5F", "#334155",
                  lambda: self._call("export_json"))
        _sep(right)
        self._btn(right, "About", "#1E3A5F", "#334155",
                  lambda: self._call("about"))
        self._btn(right, "✕ Exit", "#7F1D1D", "#991B1B",
                  lambda: self._call("exit"), width=72)

        # ── LEFT side packed after right ────────────────────────────────
        left = ctk.CTkFrame(self, fg_color="transparent")
        left.pack(side="left", padx=6, pady=6, fill="x", expand=True)

        # Logo image (or fallback text)
        logo_img = _load_logo(30)
        if logo_img:
            ctk.CTkLabel(left, image=logo_img, text="",
                         width=34).pack(side="left", padx=(0, 4))
        logo_lbl = ctk.CTkLabel(left, text="DualVision AI",
                                font=("Segoe UI", 12, "bold"),
                                text_color="#2563EB")
        logo_lbl.pack(side="left", padx=(0, 14))

        self._btn_start = self._btn(left, "▶ Start",  "#22C55E", "#16A34A",
                                    self._on_start)
        self._btn_stop  = self._btn(left, "■ Stop",   "#EF4444", "#DC2626",
                                    self._on_stop,  state="disabled")
        self._btn_pause = self._btn(left, "⏸ Pause",  "#F59E0B", "#D97706",
                                    self._on_pause, state="disabled")

        _sep(left)

        self._btn(left, "📷 Shot",   "#1E3A5F", "#334155",
                  lambda: self._call("screenshot"))
        self._btn_rec = self._btn(left, "⏺ Record", "#1E3A5F", "#334155",
                                  self._on_record)

        _sep(left)

        self._btn(left, "⚙ Settings", "#1E3A5F", "#334155",
                  lambda: self._call("settings"))

    # ── helpers ─────────────────────────────────────────────────────────
    def _btn(self, parent, text, fg, hover, cmd, state="normal", width=None):
        kw = dict(text=text, command=cmd,
                  fg_color=fg, hover_color=hover,
                  font=("Segoe UI", 10, "bold"),
                  height=32, corner_radius=6, state=state)
        if width:
            kw["width"] = width
        b = ctk.CTkButton(parent, **kw)
        b.pack(side="left", padx=3)
        return b

    def _call(self, name, *args):
        cb = self._callbacks.get(name)
        if cb:
            cb(*args)

    # ── button handlers ──────────────────────────────────────────────────
    def _on_start(self):
        self._is_detecting = True
        self._is_paused    = False
        self._btn_start.configure(state="disabled")
        self._btn_stop.configure(state="normal")
        self._btn_pause.configure(state="normal")
        self._call("start")

    def _on_stop(self):
        self._is_detecting  = False
        self._is_paused     = False
        self._is_recording  = False
        self._btn_start.configure(state="normal")
        self._btn_stop.configure(state="disabled")
        self._btn_pause.configure(state="disabled")
        self._btn_rec.configure(text="⏺ Record", fg_color="#1E3A5F")
        self._call("stop")

    def _on_pause(self):
        if not self._is_paused:
            self._is_paused = True
            self._btn_pause.configure(text="▶ Resume")
            self._call("pause")
        else:
            self._is_paused = False
            self._btn_pause.configure(text="⏸ Pause")
            self._call("resume")

    def _on_record(self):
        if not self._is_recording:
            self._is_recording = True
            self._btn_rec.configure(text="⏹ Stop Rec", fg_color="#DC2626")
            self._call("record_start")
        else:
            self._is_recording = False
            self._btn_rec.configure(text="⏺ Record", fg_color="#1E3A5F")
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


def _sep(parent):
    ctk.CTkFrame(parent, width=1, fg_color="#1E3A5F").pack(
        side="left", fill="y", padx=8)
