"""
Model Manager Dialog — DualVision AI v1.3 Stable CPU Edition.
Dedicated window for managing all YOLO26 model variants.

PART 1  — Model list with full status per variant
PART 2  — Per-model download (official Ultralytics only)
PART 3  — Import local .pt / .onnx file
PART 4  — Automatic ONNX export (export-once policy)
PART 5  — Load model without application restart
PART 6  — Export progress: step, progress bar, elapsed, estimated remaining
PART 7  — Detailed model information panel
PART 8  — Expected CPU performance reference table (informational)
PART 9  — models/ folder management
PART 10 — File validation before loading
PART 11 — Last-selected model remembered via Settings
PART 12 — All operations logged to logs/model_manager.log
PART 13 — Regression protection (no camera / ONNX pipeline modifications)
"""

import datetime
import threading
import time
from pathlib import Path

import customtkinter as ctk
import tkinter.messagebox as mb
import tkinter.filedialog as fd

from ai.model_manager import ModelManager, MODEL_VARIANTS
from config.settings   import Settings


# ── Informational performance table (PART 8) ─────────────────────────────────
# Values are representative ranges only — never faked measured numbers.
_PERF_TABLE = [
    ("YOLO26n", "Nano",   "15–35 FPS", "Fastest",  "Lowest",   "#22C55E"),
    ("YOLO26s", "Small",  "8–18 FPS",  "Fast",     "Better",   "#86EFAC"),
    ("YOLO26m", "Medium", "4–10 FPS",  "Medium",   "High",     "#FCD34D"),
    ("YOLO26l", "Large",  "2–6 FPS",   "Slow",     "Very High","#FB923C"),
    ("YOLO26x", "XLarge", "1–3 FPS",   "Slowest",  "Highest",  "#F87171"),
]

# Export duration guide for remaining-time estimate (seconds, rough)
_EXPORT_EXPECTED_S = {
    "yolo26n": 45,
    "yolo26s": 75,
    "yolo26m": 130,
    "yolo26l": 200,
    "yolo26x": 300,
}

_VARIANT_KEYS = list(MODEL_VARIANTS.keys())


class ModelManagerDialog(ctk.CTkToplevel):
    """
    Full-featured Model Manager window.
    Opened via toolbar "🗂 Models" button.
    """

    def __init__(self, parent, settings: Settings, on_load_model=None):
        super().__init__(parent)
        self.title("Model Manager — DualVision AI")
        self.resizable(True, True)
        self.configure(fg_color="#0D1626")
        self.grab_set()

        self._settings      = settings
        self._on_load_model = on_load_model   # callback(variant_key: str)
        self._model_mgr     = ModelManager(model_dir="models")

        self._export_thread: threading.Thread | None = None
        self._import_thread: threading.Thread | None = None
        self._export_timer  = None
        self._export_start  = 0.0
        self._active_export = None           # variant being exported

        self._model_rows: dict = {}          # variant → widget refs
        self._row_frames: dict = {}

        w, h = 840, 800
        pw = parent.winfo_rootx() + parent.winfo_width()  // 2
        ph = parent.winfo_rooty() + parent.winfo_height() // 2
        self.geometry(f"{w}x{h}+{pw - w//2}+{ph - h//2}")
        self.minsize(700, 600)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build()
        self.after(120, self._refresh_all)

    # ─────────────────────────────────────────────────────────────────────────
    # Layout
    # ─────────────────────────────────────────────────────────────────────────

    def _build(self):
        # Header bar
        hdr = ctk.CTkFrame(self, fg_color="#080E1C", corner_radius=0)
        hdr.pack(fill="x")
        ctk.CTkLabel(hdr, text="🗂  Model Manager",
                     font=("Segoe UI", 16, "bold"),
                     text_color="#2563EB").pack(side="left", padx=16, pady=10)
        ctk.CTkLabel(hdr,
                     text="Download • Import • Export • Load — YOLO26 variants",
                     font=("Segoe UI", 10), text_color="#475569").pack(
                     side="left", padx=4, pady=10)

        # Tabs
        self._tabs = ctk.CTkTabview(
            self, fg_color="#0D1626",
            segmented_button_fg_color="#131F35",
            segmented_button_selected_color="#2563EB",
            segmented_button_unselected_color="#1E293B",
            segmented_button_selected_hover_color="#1D4ED8")
        self._tabs.pack(fill="both", expand=True, padx=12, pady=(4, 0))

        for name in ("Models", "Import", "Performance"):
            self._tabs.add(name)

        self._build_models_tab(self._tabs.tab("Models"))
        self._build_import_tab(self._tabs.tab("Import"))
        self._build_perf_tab(self._tabs.tab("Performance"))

        # Footer
        footer = ctk.CTkFrame(self, fg_color="#080E1C", corner_radius=0)
        footer.pack(fill="x")
        ctk.CTkButton(footer, text="⟳ Refresh",
                      fg_color="#1E293B", hover_color="#334155",
                      height=30, command=self._refresh_all).pack(
                      side="left", padx=10, pady=8)
        ctk.CTkButton(footer, text="📁 Open Models Folder",
                      fg_color="#1E293B", hover_color="#334155",
                      height=30, command=self._open_folder).pack(
                      side="left", padx=4, pady=8)
        ctk.CTkButton(footer, text="✕ Close",
                      fg_color="#7F1D1D", hover_color="#991B1B",
                      height=30, command=self._on_close).pack(
                      side="right", padx=10, pady=8)

    # ── Models tab ────────────────────────────────────────────────────────────

    def _build_models_tab(self, tab):
        # Column header row
        hdr = ctk.CTkFrame(tab, fg_color="#080E1C", corner_radius=4)
        hdr.pack(fill="x", pady=(0, 2))
        for txt, w in [("Model", 182), (".pt Status", 122), ("ONNX Status", 122),
                       ("Sizes", 102), ("Modified", 96), ("Actions", 184)]:
            ctk.CTkLabel(hdr, text=txt.upper(),
                         font=("Segoe UI", 9, "bold"),
                         text_color="#2563EB", width=w, anchor="w").pack(
                         side="left", padx=(8, 0), pady=6)

        # Scrollable model list (PART 1)
        self._model_scroll = ctk.CTkScrollableFrame(
            tab, fg_color="#0A0F1E", corner_radius=6, height=200)
        self._model_scroll.pack(fill="x", pady=(0, 6))

        for variant, meta in MODEL_VARIANTS.items():
            self._build_model_row(variant, meta)

        # ── Export progress panel (PART 6) ────────────────────────────────
        exp_card = ctk.CTkFrame(tab, fg_color="#0A0F1E", corner_radius=8)
        exp_card.pack(fill="x", pady=(0, 6))

        ctk.CTkLabel(exp_card, text="EXPORT PROGRESS",
                     font=("Segoe UI", 9, "bold"),
                     text_color="#2563EB").pack(anchor="w", padx=12, pady=(8, 2))

        self._exp_model_lbl = ctk.CTkLabel(
            exp_card, text="No export in progress",
            font=("Segoe UI", 10, "bold"), text_color="#475569")
        self._exp_model_lbl.pack(anchor="w", padx=12, pady=2)

        self._exp_step_lbl = ctk.CTkLabel(
            exp_card, text="",
            font=("Segoe UI", 10), text_color="#94A3B8")
        self._exp_step_lbl.pack(anchor="w", padx=12)

        bar_row = ctk.CTkFrame(exp_card, fg_color="transparent")
        bar_row.pack(fill="x", padx=12, pady=(4, 2))
        self._exp_bar = ctk.CTkProgressBar(
            bar_row, fg_color="#131F35", progress_color="#2563EB")
        self._exp_bar.set(0)
        self._exp_bar.pack(side="left", fill="x", expand=True)
        self._exp_pct_lbl = ctk.CTkLabel(
            bar_row, text="0%",
            font=("Segoe UI", 10), text_color="#CBD5E1", width=45)
        self._exp_pct_lbl.pack(side="right", padx=(8, 0))

        time_row = ctk.CTkFrame(exp_card, fg_color="transparent")
        time_row.pack(fill="x", padx=12, pady=(0, 8))
        self._exp_elapsed_lbl = ctk.CTkLabel(
            time_row, text="Elapsed: —",
            font=("Segoe UI", 9), text_color="#64748B")
        self._exp_elapsed_lbl.pack(side="left")
        self._exp_remain_lbl = ctk.CTkLabel(
            time_row, text="Est. remaining: —",
            font=("Segoe UI", 9), text_color="#64748B")
        self._exp_remain_lbl.pack(side="left", padx=16)
        self._exp_status_lbl = ctk.CTkLabel(
            time_row, text="",
            font=("Segoe UI", 9), text_color="#94A3B8")
        self._exp_status_lbl.pack(side="right")

        # ── Model information panel (PART 7) ──────────────────────────────
        info_card = ctk.CTkFrame(tab, fg_color="#0A0F1E", corner_radius=8)
        info_card.pack(fill="x", pady=(0, 0))

        ctk.CTkLabel(info_card, text="MODEL INFORMATION",
                     font=("Segoe UI", 9, "bold"),
                     text_color="#2563EB").pack(anchor="w", padx=12, pady=(8, 4))

        info_grid = ctk.CTkFrame(info_card, fg_color="transparent")
        info_grid.pack(fill="x", padx=12, pady=(0, 10))

        self._info_vars: dict = {}
        fields = [
            ("Model",            "name"),
            ("Input Resolution", "input_res"),
            ("Backend",          "backend"),
            ("Device",           "device"),
            ("Classes",          "classes"),
            ("Parameters",       "params"),
            ("PT File",          "pt_path"),
            ("PT Size",          "pt_size"),
            ("ONNX File",        "onnx_path"),
            ("ONNX Size",        "onnx_size"),
            ("Export Date",      "export_date"),
            ("Variant Key",      "variant_key"),
        ]
        for i, (lbl, key) in enumerate(fields):
            col   = i % 2
            row_i = i // 2
            cell  = ctk.CTkFrame(info_grid, fg_color="transparent")
            cell.grid(row=row_i, column=col, sticky="w", padx=(0, 20), pady=1)
            ctk.CTkLabel(cell, text=f"{lbl}:",
                         font=("Segoe UI", 9), text_color="#64748B",
                         width=120, anchor="w").pack(side="left")
            var = ctk.StringVar(value="—")
            self._info_vars[key] = var
            ctk.CTkLabel(cell, textvariable=var,
                         font=("Segoe UI", 9, "bold"),
                         text_color="#CBD5E1", anchor="w").pack(side="left")

        self.after(250, lambda: self._update_info_panel(
            self._settings.get("inference", "active_model", "yolo26n")))

    def _build_model_row(self, variant: str, meta: dict):
        """One row in the model table (PART 1 + PART 2)."""
        bg = "#0D1626" if _VARIANT_KEYS.index(variant) % 2 == 0 else "#0A0F1E"
        row = ctk.CTkFrame(self._model_scroll, fg_color=bg, corner_radius=4)
        row.pack(fill="x", pady=1)
        self._row_frames[variant] = row

        # Name cell
        nc = ctk.CTkFrame(row, fg_color="transparent", width=182)
        nc.pack(side="left", padx=(8, 0), pady=6)
        nc.pack_propagate(False)
        short  = meta["label"].split("(")[0].strip()
        detail = meta["label"].split("(")[1].rstrip(")") if "(" in meta["label"] else ""
        ctk.CTkLabel(nc, text=short,
                     font=("Segoe UI", 11, "bold"),
                     text_color="#E2E8F0", anchor="w").pack(anchor="w")
        ctk.CTkLabel(nc, text=detail,
                     font=("Segoe UI", 9),
                     text_color="#475569", anchor="w").pack(anchor="w")

        # PT status
        pt_var = ctk.StringVar(value="…")
        pt_cell = ctk.CTkFrame(row, fg_color="transparent", width=122)
        pt_cell.pack(side="left", pady=6)
        pt_cell.pack_propagate(False)
        pt_lbl = ctk.CTkLabel(pt_cell, textvariable=pt_var,
                              font=("Segoe UI", 10, "bold"),
                              text_color="#94A3B8", anchor="w")
        pt_lbl.pack(anchor="w")

        # ONNX status
        onnx_var = ctk.StringVar(value="…")
        onnx_cell = ctk.CTkFrame(row, fg_color="transparent", width=122)
        onnx_cell.pack(side="left", pady=6)
        onnx_cell.pack_propagate(False)
        onnx_lbl = ctk.CTkLabel(onnx_cell, textvariable=onnx_var,
                                font=("Segoe UI", 10, "bold"),
                                text_color="#94A3B8", anchor="w")
        onnx_lbl.pack(anchor="w")

        # Sizes
        sz_var  = ctk.StringVar(value="—")
        sz_cell = ctk.CTkFrame(row, fg_color="transparent", width=102)
        sz_cell.pack(side="left", pady=6)
        sz_cell.pack_propagate(False)
        ctk.CTkLabel(sz_cell, textvariable=sz_var,
                     font=("Segoe UI", 9), text_color="#64748B",
                     anchor="w").pack(anchor="w")

        # Last modified
        mod_var  = ctk.StringVar(value="—")
        mod_cell = ctk.CTkFrame(row, fg_color="transparent", width=96)
        mod_cell.pack(side="left", pady=6)
        mod_cell.pack_propagate(False)
        ctk.CTkLabel(mod_cell, textvariable=mod_var,
                     font=("Segoe UI", 9), text_color="#64748B",
                     anchor="w").pack(anchor="w")

        # Action buttons (PART 2: Download, PART 4: Export, PART 5: Load)
        act_cell = ctk.CTkFrame(row, fg_color="transparent", width=184)
        act_cell.pack(side="left", padx=(4, 8), pady=4)
        act_cell.pack_propagate(False)

        dl_btn = ctk.CTkButton(
            act_cell, text="⬇ Download",
            fg_color="#1E3A5F", hover_color="#2563EB",
            font=("Segoe UI", 9), height=26, width=88,
            command=lambda v=variant: self._on_download(v))
        dl_btn.pack(side="left", padx=(0, 3))

        exp_btn = ctk.CTkButton(
            act_cell, text="Export",
            fg_color="#1E293B", hover_color="#334155",
            font=("Segoe UI", 9), height=26, width=54,
            command=lambda v=variant: self._on_export(v))
        exp_btn.pack(side="left", padx=(0, 3))

        load_btn = ctk.CTkButton(
            act_cell, text="Load",
            fg_color="#064E3B", hover_color="#065F46",
            font=("Segoe UI", 9, "bold"), height=26, width=38,
            command=lambda v=variant: self._on_load(v))
        load_btn.pack(side="left")

        self._model_rows[variant] = {
            "pt_var": pt_var, "pt_lbl": pt_lbl,
            "onnx_var": onnx_var, "onnx_lbl": onnx_lbl,
            "sz_var": sz_var, "mod_var": mod_var,
            "dl_btn": dl_btn, "exp_btn": exp_btn, "load_btn": load_btn,
            "row": row,
        }

    # ── Import tab (PART 3) ───────────────────────────────────────────────────

    def _build_import_tab(self, tab):
        scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        self._section(scroll, "Import Local Model File")
        ctk.CTkLabel(scroll,
            text="Select a local .pt (PyTorch weights) or .onnx file to import "
                 "into models/.\n"
                 "The filename must identify a YOLO26 variant:\n"
                 "  yolo26n, yolo26s, yolo26m, yolo26l, yolo26x\n\n"
                 "Examples:  my_yolo26s.pt  •  exported_yolo26m.onnx  •  yolo26l.pt",
            font=("Segoe UI", 10), text_color="#64748B",
            justify="left", wraplength=700).pack(anchor="w", pady=(0, 10))

        btn_row = ctk.CTkFrame(scroll, fg_color="transparent")
        btn_row.pack(anchor="w", pady=(0, 6))
        ctk.CTkButton(btn_row, text="📂  Browse .pt / .onnx File",
                      fg_color="#1E3A5F", hover_color="#2563EB",
                      height=36, width=230,
                      command=self._on_import_browse).pack(side="left", padx=(0, 12))
        self._import_status_lbl = ctk.CTkLabel(
            btn_row, text="",
            font=("Segoe UI", 10), text_color="#94A3B8", anchor="w")
        self._import_status_lbl.pack(side="left")

        self._import_bar = ctk.CTkProgressBar(
            scroll, fg_color="#131F35", progress_color="#2563EB")
        self._import_bar.set(0)
        self._import_bar.pack(fill="x", pady=(4, 2))

        self._import_detail_lbl = ctk.CTkLabel(
            scroll, text="",
            font=("Segoe UI", 10), text_color="#94A3B8",
            anchor="w", wraplength=680)
        self._import_detail_lbl.pack(anchor="w", pady=(2, 12))

        self._section(scroll, "Validation Rules (PART 10)")
        rules = [
            ("✓", "#94A3B8", "Official Ultralytics YOLO26 .pt files are accepted"),
            ("✓", "#94A3B8", "ONNX files exported from YOLO26 .pt are accepted"),
            ("✓", "#94A3B8", "ONNX files are validated via ONNX Runtime before copying"),
            ("✓", "#94A3B8", "Files must be ≥ 100 KB (corrupted files are rejected)"),
            ("✗", "#64748B", "Files from unrecognised variants are rejected"),
            ("✗", "#64748B", "Non-YOLO26 architectures (wrong input shape) are rejected"),
            ("✗", "#64748B", "Files < 1 KB are always rejected as corrupted"),
        ]
        for sym, clr, txt in rules:
            r = ctk.CTkFrame(scroll, fg_color="transparent")
            r.pack(anchor="w", pady=1)
            ctk.CTkLabel(r, text=sym, font=("Segoe UI", 10, "bold"),
                         text_color="#22C55E" if sym == "✓" else "#EF4444",
                         width=18).pack(side="left")
            ctk.CTkLabel(r, text=txt, font=("Segoe UI", 10),
                         text_color=clr, anchor="w").pack(side="left")

        self._section(scroll, "Models Folder")
        models_path = str(Path("models").resolve())
        pf = ctk.CTkFrame(scroll, fg_color="#0A0F1E", corner_radius=6)
        pf.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(pf, text=models_path,
                     font=("Courier New", 9), text_color="#A78BFA",
                     anchor="w").pack(side="left", padx=10, pady=8)
        ctk.CTkButton(pf, text="Open",
                      fg_color="#1E293B", hover_color="#334155",
                      height=26, width=60,
                      command=self._open_folder).pack(side="right", padx=8, pady=4)

    # ── Performance tab (PART 8) ──────────────────────────────────────────────

    def _build_perf_tab(self, tab):
        scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        self._section(scroll, "Expected CPU Performance  (Informational — not measured)")
        ctk.CTkLabel(scroll,
            text="Estimated ranges on Intel i7 / AMD Ryzen 7, CPU-only, 640 × 640 input.\n"
                 "Run  python tools/benchmark.py  for real measured values on your hardware.",
            font=("Segoe UI", 10), text_color="#64748B",
            justify="left", wraplength=700).pack(anchor="w", pady=(0, 10))

        # Header
        ph = ctk.CTkFrame(scroll, fg_color="#080E1C", corner_radius=4)
        ph.pack(fill="x", pady=(0, 2))
        for txt, w in [("Model", 120), ("Variant", 80),
                       ("Est. FPS (640px)", 150), ("Speed", 110), ("Accuracy", 110)]:
            ctk.CTkLabel(ph, text=txt.upper(),
                         font=("Segoe UI", 9, "bold"),
                         text_color="#2563EB", width=w, anchor="w").pack(
                         side="left", padx=(10, 0), pady=6)

        for i, (name, variant, fps, speed, acc, clr) in enumerate(_PERF_TABLE):
            bg = "#0A0F1E" if i % 2 == 0 else "#0D1626"
            r  = ctk.CTkFrame(scroll, fg_color=bg, corner_radius=4)
            r.pack(fill="x", pady=1)
            for txt, w, tc in [
                (name,  120, "#CBD5E1"),
                (variant, 80, "#94A3B8"),
                (fps,   150, "#CBD5E1"),
                (speed, 110, clr),
                (acc,   110, clr),
            ]:
                ctk.CTkLabel(r, text=txt, font=("Segoe UI", 10),
                             text_color=tc, width=w, anchor="w").pack(
                             side="left", padx=(10, 0), pady=6)

        self._section(scroll, "Use-Case Recommendations")
        recs = [
            ("Real-time detection, limited CPU (1–4 cores)",     "YOLO26n"),
            ("Balanced accuracy + speed, modern 8-core CPU",     "YOLO26s"),
            ("High accuracy, can tolerate 4–10 FPS",             "YOLO26m"),
            ("Best accuracy, offline / batch analysis",          "YOLO26l or YOLO26x"),
            ("Search & Rescue, long-duration, CPU-only",
             "YOLO26s + Smoother ON + Persistence 5–8 frames"),
        ]
        for scenario, rec in recs:
            rr = ctk.CTkFrame(scroll, fg_color="transparent")
            rr.pack(fill="x", pady=2)
            ctk.CTkLabel(rr, text=f"• {scenario}:",
                         font=("Segoe UI", 10), text_color="#94A3B8",
                         width=390, anchor="w").pack(side="left")
            ctk.CTkLabel(rr, text=rec,
                         font=("Segoe UI", 10, "bold"),
                         text_color="#22C55E", anchor="w").pack(side="left")

        self._section(scroll, "Approximate ONNX File Sizes")
        sf = ctk.CTkFrame(scroll, fg_color="#0A0F1E", corner_radius=6)
        sf.pack(fill="x", pady=(0, 10))
        for name, sz in [("YOLO26n", "~6 MB"), ("YOLO26s", "~22 MB"),
                         ("YOLO26m", "~52 MB"), ("YOLO26l", "~87 MB"),
                         ("YOLO26x", "~136 MB")]:
            col = ctk.CTkFrame(sf, fg_color="transparent")
            col.pack(side="left", padx=14, pady=8)
            ctk.CTkLabel(col, text=name,
                         font=("Segoe UI", 10, "bold"),
                         text_color="#A78BFA").pack()
            ctk.CTkLabel(col, text=sz,
                         font=("Segoe UI", 9),
                         text_color="#64748B").pack()

        self._section(scroll, "Run Benchmark")
        ctk.CTkLabel(scroll,
            text="To generate a MODEL_BENCHMARK.md with real numbers for your hardware:\n\n"
                 "  python tools/benchmark.py\n"
                 "  python tools/benchmark.py --video clip.mp4 --frames 300",
            font=("Courier New", 10), text_color="#94A3B8",
            justify="left").pack(anchor="w", pady=(0, 10))

    # ─────────────────────────────────────────────────────────────────────────
    # Refresh (PART 1: status per variant)
    # ─────────────────────────────────────────────────────────────────────────

    def _refresh_all(self):
        """Reload all model statuses from disk and update every row."""
        active   = self._settings.get("inference", "active_model", "yolo26n")
        statuses = self._model_mgr.get_all_model_status()

        for st in statuses:
            v    = st["variant"]
            refs = self._model_rows.get(v)
            if not refs:
                continue

            pt_ok   = st["pt_ready"]
            onnx_ok = st["onnx_ready"]

            # ── .pt status ───────────────────────────────────────────────
            if pt_ok:
                refs["pt_var"].set("Downloaded ✓")
                refs["pt_lbl"].configure(text_color="#22C55E")
                refs["dl_btn"].configure(state="disabled",
                                          text="✓ Downloaded",
                                          fg_color="#064E3B")
            else:
                refs["pt_var"].set("Not Downloaded")
                refs["pt_lbl"].configure(text_color="#F59E0B")
                refs["dl_btn"].configure(state="normal",
                                          text="⬇ Download",
                                          fg_color="#1E3A5F")

            # ── ONNX status ──────────────────────────────────────────────
            if onnx_ok:
                refs["onnx_var"].set("ONNX Ready ✓")
                refs["onnx_lbl"].configure(text_color="#22C55E")
                refs["exp_btn"].configure(state="disabled",
                                           text="✓ Exported",
                                           fg_color="#064E3B")
            elif pt_ok:
                refs["onnx_var"].set("Export Required")
                refs["onnx_lbl"].configure(text_color="#F59E0B")
                refs["exp_btn"].configure(state="normal",
                                           text="Export",
                                           fg_color="#1E3A5F")
            else:
                refs["onnx_var"].set("Download First")
                refs["onnx_lbl"].configure(text_color="#EF4444")
                refs["exp_btn"].configure(state="disabled",
                                           text="Export",
                                           fg_color="#1E293B")

            # ── Sizes ─────────────────────────────────────────────────────
            parts = []
            if st["pt_mb"]   > 0: parts.append(f".pt {st['pt_mb']}MB")
            if st["onnx_mb"] > 0: parts.append(f".onnx {st['onnx_mb']}MB")
            refs["sz_var"].set("\n".join(parts) if parts else "—")

            # ── Last modified ─────────────────────────────────────────────
            refs["mod_var"].set(self._last_modified(v))

            # ── Load / Active button (PART 5, 11) ────────────────────────
            row_bg = "#0D1626" if _VARIANT_KEYS.index(v) % 2 == 0 else "#0A0F1E"
            if v == active:
                refs["load_btn"].configure(text="✓ Active",
                                            fg_color="#065F46",
                                            state="disabled")
                refs["row"].configure(fg_color="#0B2040")
            else:
                refs["row"].configure(fg_color=row_bg)
                if onnx_ok:
                    refs["load_btn"].configure(text="Load",
                                               fg_color="#064E3B",
                                               state="normal")
                else:
                    refs["load_btn"].configure(text="Load",
                                               fg_color="#1E293B",
                                               state="disabled")

    def _last_modified(self, variant: str) -> str:
        """Return most-recent file modification date for a variant."""
        paths = [self._model_mgr.get_onnx_path(variant),
                 self._model_mgr.get_pt_path(variant)]
        ts = None
        for p in paths:
            if p.exists():
                t = p.stat().st_mtime
                if ts is None or t > ts:
                    ts = t
        if ts:
            return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        return "—"

    def _update_info_panel(self, variant: str):
        """Fill the Model Information panel (PART 7)."""
        try:
            info = self._model_mgr.get_model_info(variant)
            onnx_p = self._model_mgr.get_onnx_path(variant)
            export_date = "—"
            if onnx_p.exists():
                ts = onnx_p.stat().st_mtime
                export_date = datetime.datetime.fromtimestamp(ts).strftime(
                    "%Y-%m-%d  %H:%M")

            param_guide = {
                "yolo26n": "~3.2 M", "yolo26s": "~11 M",
                "yolo26m": "~25 M",  "yolo26l": "~43 M", "yolo26x": "~68 M",
            }

            self._info_vars["name"].set(info.get("label", variant))
            self._info_vars["variant_key"].set(variant)
            self._info_vars["input_res"].set("640 × 640")
            self._info_vars["backend"].set("ONNX Runtime")
            self._info_vars["device"].set("CPU  (CPUExecutionProvider)")
            self._info_vars["classes"].set("80  (COCO)")
            self._info_vars["params"].set(
                param_guide.get(variant, "—") + "  (approx.)")
            self._info_vars["pt_path"].set(
                Path(info["pt_path"]).name if info.get("pt_ready") else "Not downloaded")
            self._info_vars["pt_size"].set(
                f"{info['pt_mb']} MB" if info["pt_mb"] else "—")
            onnx_ready = info.get("onnx_ready") and info.get("onnx_path") != "Not exported"
            self._info_vars["onnx_path"].set(
                Path(info["onnx_path"]).name if onnx_ready else "Not exported")
            self._info_vars["onnx_size"].set(
                f"{info['onnx_mb']} MB" if info["onnx_mb"] else "—")
            self._info_vars["export_date"].set(export_date)
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # Download (PART 2)
    # ─────────────────────────────────────────────────────────────────────────

    def _on_download(self, variant: str):
        refs = self._model_rows[variant]
        refs["dl_btn"].configure(state="disabled", text="Downloading …")
        refs["pt_var"].set("Downloading …")
        refs["pt_lbl"].configure(text_color="#F59E0B")

        def _do():
            self._model_mgr.set_callbacks(
                progress=lambda p: self._dl_progress(variant, p),
                status=lambda m: self._dl_status(variant, m))
            self._model_mgr.ensure_pt(blocking=True, variant=variant)
            try:
                self.after(0, self._refresh_all)
                self.after(0, lambda: self._update_info_panel(variant))
            except Exception:
                pass

        threading.Thread(target=_do, daemon=True,
                         name=f"DL-{variant}").start()

    def _dl_progress(self, variant: str, pct: float):
        refs = self._model_rows.get(variant)
        if not refs:
            return
        try:
            self.after(0, lambda p=pct:
                       refs["pt_var"].set(f"Downloading … {p:.0f}%"))
        except Exception:
            pass

    def _dl_status(self, variant: str, msg: str):
        refs = self._model_rows.get(variant)
        if not refs:
            return
        try:
            short = msg[:48]
            self.after(0, lambda m=short: refs["pt_var"].set(m))
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # Export (PART 4 + PART 6)
    # ─────────────────────────────────────────────────────────────────────────

    def _on_export(self, variant: str):
        if self._export_thread and self._export_thread.is_alive():
            mb.showwarning("Export Running",
                           "An export is already in progress. Please wait.",
                           parent=self)
            return

        refs = self._model_rows[variant]
        refs["exp_btn"].configure(state="disabled", text="Exporting …")
        refs["onnx_var"].set("Exporting …")
        refs["onnx_lbl"].configure(text_color="#F59E0B")

        self._tabs.set("Models")
        self._active_export = variant
        self._export_start  = time.perf_counter()

        label = MODEL_VARIANTS[variant]["label"]
        self._exp_model_lbl.configure(text=f"Exporting  {label}",
                                       text_color="#A78BFA")
        self._exp_step_lbl.configure(text="Initialising …", text_color="#CBD5E1")
        self._exp_bar.set(0)
        self._exp_pct_lbl.configure(text="0%")
        self._exp_elapsed_lbl.configure(text="Elapsed: 0s")
        self._exp_remain_lbl.configure(text="Est. remaining: …")
        self._exp_status_lbl.configure(text="")

        self._start_export_timer(variant)

        def _do():
            try:
                self._model_mgr.set_variant(variant)
                result = self._model_mgr.export_onnx(variant=variant)
                self.after(0, lambda: self._export_done(variant, result, None))
            except Exception as exc:
                self.after(0, lambda e=exc: self._export_done(variant, None, e))

        self._export_thread = threading.Thread(
            target=_do, daemon=True, name=f"Export-{variant}")
        self._export_thread.start()

    def _start_export_timer(self, variant: str):
        if self._export_timer:
            try: self.after_cancel(self._export_timer)
            except Exception: pass
        self._export_timer = self.after(1000, self._tick_timer, variant)

    def _tick_timer(self, variant: str):
        """Update elapsed / remaining / step / progress bar every second."""
        if not (self._export_thread and self._export_thread.is_alive()):
            return
        elapsed  = time.perf_counter() - self._export_start
        expected = _EXPORT_EXPECTED_S.get(variant, 60)
        frac     = min(0.99, elapsed / expected)
        remaining = max(0.0, expected - elapsed)

        # Simulated step names based on elapsed fraction (PART 6)
        if elapsed < 5:
            step = "Loading .pt model …"
        elif frac < 0.50:
            step = "Exporting to ONNX (may take 30–60 s) …"
        elif frac < 0.80:
            step = "Optimising ONNX graph …"
        else:
            step = "Saving .onnx file …"

        try:
            self._exp_step_lbl.configure(text=f"Step:  {step}")
            self._exp_elapsed_lbl.configure(text=f"Elapsed: {int(elapsed)}s")
            self._exp_remain_lbl.configure(
                text=f"Est. remaining: {int(remaining)}s"
                     if elapsed < expected else "Finalising …")
            self._exp_bar.set(frac)
            self._exp_pct_lbl.configure(text=f"{int(frac * 100)}%")
            self._export_timer = self.after(1000, self._tick_timer, variant)
        except Exception:
            pass

    def _export_done(self, variant: str, result, error):
        if self._export_timer:
            try: self.after_cancel(self._export_timer)
            except Exception: pass
        self._export_timer = None
        self._active_export = None

        refs    = self._model_rows.get(variant, {})
        elapsed = int(time.perf_counter() - self._export_start)

        if error:
            self._exp_model_lbl.configure(
                text=f"Export FAILED — {MODEL_VARIANTS[variant]['label']}",
                text_color="#EF4444")
            self._exp_step_lbl.configure(
                text=str(error)[:120], text_color="#EF4444")
            self._exp_bar.set(0)
            self._exp_pct_lbl.configure(text="Error")
            self._exp_status_lbl.configure(
                text="See logs/model_manager.log",
                text_color="#EF4444")
            if refs:
                refs["exp_btn"].configure(state="normal", text="Export")
                refs["onnx_var"].set("Export Failed")
                refs["onnx_lbl"].configure(text_color="#EF4444")
            mb.showerror(
                "Export Failed",
                f"Could not export {variant}:\n\n{error}\n\n"
                "See logs/model_manager.log for the full traceback.",
                parent=self)
        else:
            self._exp_model_lbl.configure(
                text=f"Export complete — {MODEL_VARIANTS[variant]['label']}",
                text_color="#22C55E")
            self._exp_step_lbl.configure(text="Completed ✓", text_color="#22C55E")
            self._exp_bar.set(1.0)
            self._exp_pct_lbl.configure(text="100%")
            self._exp_status_lbl.configure(
                text=f"Finished in {elapsed}s", text_color="#22C55E")

        self._refresh_all()
        self._update_info_panel(variant)

    # ─────────────────────────────────────────────────────────────────────────
    # Load / Hot-swap (PART 5)
    # ─────────────────────────────────────────────────────────────────────────

    def _on_load(self, variant: str):
        """Validate then set this model as active (PART 5, 10, 11)."""
        # Validate files before loading (PART 10)
        ok, vmsg = self._model_mgr.validate_model(variant)
        if not ok:
            mb.showerror(
                "Validation Failed",
                f"Model validation failed for {variant}:\n\n{vmsg}\n\n"
                "Re-download or re-export the model, then try again.",
                parent=self)
            return

        # Persist selection (PART 11)
        self._settings.set("inference", "active_model", variant)
        self._settings.save()

        self._refresh_all()
        self._update_info_panel(variant)

        # Hot-swap via main window callback (PART 5)
        if self._on_load_model:
            try:
                self._on_load_model(variant)
            except Exception as exc:
                mb.showwarning(
                    "Load Model",
                    f"Active model saved as {variant}.\n\n"
                    f"Hot-swap notification error: {exc}\n\n"
                    "Stop and Start detection to apply the new model.",
                    parent=self)

    # ─────────────────────────────────────────────────────────────────────────
    # Import (PART 3)
    # ─────────────────────────────────────────────────────────────────────────

    def _on_import_browse(self):
        path = fd.askopenfilename(
            parent=self,
            title="Select YOLO26 .pt or .onnx model file",
            filetypes=[
                ("YOLO26 model files", "*.pt *.onnx"),
                ("PyTorch weights",    "*.pt"),
                ("ONNX model",         "*.onnx"),
                ("All files",          "*.*"),
            ])
        if not path:
            return

        self._import_status_lbl.configure(
            text=f"Importing  {Path(path).name} …", text_color="#F59E0B")
        self._import_bar.set(0.25)
        self._import_detail_lbl.configure(text="Validating …", text_color="#94A3B8")

        def _status(msg):
            try:
                self.after(0, lambda m=msg: (
                    self._import_status_lbl.configure(text=m[:60]),
                    self._import_detail_lbl.configure(text=m),
                ))
            except Exception:
                pass

        def _do():
            ok, msg = self._model_mgr.import_local(path, status_cb=_status)
            try:
                self.after(0, lambda s=ok, m=msg: self._import_done(s, m))
            except Exception:
                pass

        self._import_thread = threading.Thread(
            target=_do, daemon=True, name="Import-local")
        self._import_thread.start()

    def _import_done(self, success: bool, msg: str):
        if success:
            self._import_bar.set(1.0)
            self._import_status_lbl.configure(
                text="Import successful ✓", text_color="#22C55E")
            self._import_detail_lbl.configure(text=msg, text_color="#22C55E")
            self._refresh_all()
        else:
            self._import_bar.set(0)
            self._import_status_lbl.configure(
                text="Import failed", text_color="#EF4444")
            self._import_detail_lbl.configure(text=msg, text_color="#EF4444")

    # ─────────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────────

    def _section(self, parent, text: str):
        ctk.CTkLabel(parent, text=text.upper(),
                     font=("Segoe UI", 9, "bold"),
                     text_color="#2563EB").pack(anchor="w", pady=(12, 2))
        ctk.CTkFrame(parent, height=1,
                     fg_color="#1E3A5F").pack(fill="x", pady=(0, 6))

    def _open_folder(self):
        """Open the models/ directory in the system file manager."""
        import subprocess, platform
        folder = str(Path("models").resolve())
        try:
            if platform.system() == "Windows":
                subprocess.Popen(["explorer", folder])
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", folder])
            else:
                subprocess.Popen(["xdg-open", folder])
        except Exception as exc:
            mb.showinfo("Models Folder",
                        f"Path:\n{folder}\n\n(Could not open explorer: {exc})",
                        parent=self)

    def _on_close(self):
        if self._export_timer:
            try: self.after_cancel(self._export_timer)
            except Exception: pass
        self.destroy()
