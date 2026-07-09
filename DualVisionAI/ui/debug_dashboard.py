"""
Debug Dashboard — DualVision AI v1.3 Stable CPU Edition.

Floating window (CTkToplevel) displaying real-time detection and tracking
stability metrics — Part 10 of Phase 3 Optimization.

Sections:
  • Detection Stability  — per-stream aggregate continuity metrics
  • Track Stability      — ID changes, ghost/recovered counts
  • Confidence Trends    — EMA-smoothed confidence, min/max/avg per class
  • Per-Object Table     — one row per confirmed track with full lifecycle
  • System              — CPU, RAM, inference latency
"""

import time
import threading
import customtkinter as ctk
from collections import defaultdict, deque


class DebugDashboard(ctk.CTkToplevel):
    """
    Real-time debugging overlay.  Call update_data() from the main loop.
    Thread-safe: all Tk mutations are dispatched via after(0).
    """

    _REFRESH_MS = 500   # UI update interval

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.title("DualVision AI — Debug Dashboard")
        self.geometry("780x620")
        self.configure(fg_color="#080E1C")
        self.resizable(True, True)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._lock   = threading.Lock()
        self._data   = {}      # latest payload from update_data()
        self._closed = False

        self._build()
        self._schedule_refresh()

    # ── Build ──────────────────────────────────────────────────────────────────
    def _build(self):
        hdr = ctk.CTkLabel(
            self, text="🔍  Detection & Tracking Debug Dashboard",
            font=("Segoe UI", 13, "bold"), text_color="#2563EB")
        hdr.pack(fill="x", padx=14, pady=(10, 4))

        self._tabs = ctk.CTkTabview(self, fg_color="#0D1626",
                                    segmented_button_fg_color="#0D1626",
                                    segmented_button_selected_color="#2563EB",
                                    segmented_button_selected_hover_color="#1D4ED8")
        self._tabs.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        t1 = self._tabs.add("Stability")
        t2 = self._tabs.add("Confidence")
        t3 = self._tabs.add("Per-Object")
        t4 = self._tabs.add("System")

        self._build_stability(t1)
        self._build_confidence(t2)
        self._build_per_object(t3)
        self._build_system(t4)

    # ── Tab 1: Stability ───────────────────────────────────────────────────────
    def _build_stability(self, parent):
        scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        self._section(scroll, "Detection Stability")
        df = self._card(scroll)
        self._d_stab_pct  = self._row(df, "Detection Continuity %", "#22C55E")
        self._d_det_total = self._row(df, "Frames Detected",         "#3B82F6")
        self._d_det_miss  = self._row(df, "Frames Missing",          "#EF4444")
        self._d_drop_cnt  = self._row(df, "Detection Drops",         "#F59E0B")
        self._d_avg_conf  = self._row(df, "Avg Confidence",          "#A78BFA")

        self._section(scroll, "Track Stability")
        tf = self._card(scroll)
        self._t_stab_pct  = self._row(tf, "Track Stability %",       "#22C55E")
        self._t_id_chg    = self._row(tf, "ID Changes",               "#EF4444")
        self._t_lost      = self._row(tf, "Lost Frames",              "#F59E0B")
        self._t_recovered = self._row(tf, "Recovered Frames",         "#A78BFA")
        self._t_ghost_cnt = self._row(tf, "Ghost Detections Active",  "#64748B")
        self._t_conf_avg  = self._row(tf, "Avg Smooth Confidence",    "#3B82F6")
        self._t_lifetime  = self._row(tf, "Avg Detection Lifetime",   "#818CF8")

    # ── Tab 2: Confidence ──────────────────────────────────────────────────────
    def _build_confidence(self, parent):
        scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        self._section(scroll, "Confidence Distribution (last 50 frames)")
        cf = self._card(scroll)
        self._c_cur   = self._row(cf, "Current Frame Avg",  "#22C55E")
        self._c_avg   = self._row(cf, "Rolling Avg (50f)",  "#3B82F6")
        self._c_min   = self._row(cf, "Rolling Min",        "#EF4444")
        self._c_max   = self._row(cf, "Rolling Max",        "#16A34A")
        self._c_drops = self._row(cf, "Conf Drop Events",   "#F59E0B")

        self._section(scroll, "Smoother Stats")
        sf = self._card(scroll)
        self._s_active = self._row(sf, "Active Smoother Entries", "#22C55E")
        self._s_ghost  = self._row(sf, "Ghost Entries",            "#F59E0B")
        self._s_avg_sc = self._row(sf, "Avg Smoothed Conf",        "#A78BFA")

        self._section(scroll, "Confidence Trend (last 20 values)")
        self._conf_trend_label = ctk.CTkLabel(
            scroll, text="—",
            font=("Consolas", 9), text_color="#64748B",
            justify="left", anchor="w", wraplength=700)
        self._conf_trend_label.pack(fill="x", padx=14, pady=(2, 8))

    # ── Tab 3: Per-Object ──────────────────────────────────────────────────────
    def _build_per_object(self, parent):
        hdr_row = ctk.CTkFrame(parent, fg_color="#0D1626")
        hdr_row.pack(fill="x", padx=4, pady=(6, 0))
        for col, w in [
            ("#ID", 40), ("Class", 90), ("Age(s)", 55), ("Hits", 45),
            ("Lost", 40), ("Recov", 45), ("Conf", 55), ("Dir", 80),
        ]:
            ctk.CTkLabel(hdr_row, text=col,
                         font=("Segoe UI", 9, "bold"),
                         text_color="#2563EB",
                         width=w, anchor="w").pack(side="left", padx=2)

        self._obj_scroll = ctk.CTkScrollableFrame(
            parent, fg_color="#050A14", corner_radius=8)
        self._obj_scroll.pack(fill="both", expand=True, padx=4, pady=4)

        self._obj_rows: dict = {}

    # ── Tab 4: System ─────────────────────────────────────────────────────────
    def _build_system(self, parent):
        scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        self._section(scroll, "Inference Performance")
        pf = self._card(scroll)
        self._p_fps     = self._row(pf, "Inference FPS",   "#22C55E")
        self._p_avg_fps = self._row(pf, "Avg FPS",         "#16A34A")
        self._p_inf_ms  = self._row(pf, "Inference ms",    "#F59E0B")
        self._p_pre_ms  = self._row(pf, "Preprocess ms",   "#64748B")
        self._p_post_ms = self._row(pf, "Postprocess ms",  "#64748B")
        self._p_drops   = self._row(pf, "Frame Drops",     "#EF4444")

        self._section(scroll, "System Resources")
        rf = self._card(scroll)
        self._r_cpu = self._row(rf, "CPU Usage",  "#F59E0B")
        self._r_ram = self._row(rf, "RAM Usage",  "#3B82F6")

        self._section(scroll, "Model")
        mf = self._card(scroll)
        self._m_name    = self._row(mf, "Active Model",  "#A78BFA")
        self._m_size    = self._row(mf, "Input Size",    "#64748B")
        self._m_classes = self._row(mf, "Classes",       "#64748B")

    # ── Widget helpers ─────────────────────────────────────────────────────────
    def _section(self, parent, text):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.pack(fill="x", padx=10, pady=(8, 2))
        ctk.CTkLabel(f, text=text,
                     font=("Segoe UI", 9, "bold"),
                     text_color="#2563EB").pack(side="left")
        ctk.CTkFrame(f, height=1, fg_color="#1E3A5F").pack(
            side="left", fill="x", expand=True, padx=4)

    def _card(self, parent):
        f = ctk.CTkFrame(parent, fg_color="#0A0F1E", corner_radius=8)
        f.pack(fill="x", padx=10, pady=(0, 6))
        return f

    def _row(self, parent, label, color="#CBD5E1"):
        var = ctk.StringVar(value="—")
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=1)
        ctk.CTkLabel(row, text=label,
                     font=("Segoe UI", 10), text_color="#64748B",
                     width=160, anchor="w").pack(side="left")
        ctk.CTkLabel(row, textvariable=var,
                     font=("Segoe UI", 10, "bold"),
                     text_color=color, anchor="e").pack(side="right")
        return var

    # ── Public update API ──────────────────────────────────────────────────────
    def update_data(self, payload: dict):
        """
        Thread-safe. Call from any thread.

        Expected keys (all optional — missing values show '—'):
          fps_inf, avg_fps, inf_ms, pre_ms, post_ms, frame_drops
          cpu_pct, ram_mb
          model_name, input_size, num_classes
          det_frames_detected, det_frames_missing, det_drops, avg_conf
          trk_id_changes, trk_lost, trk_recovered, trk_ghost, trk_lifetime_avg
          smoother_active, smoother_ghost, smoother_avg_conf
          conf_current, conf_history      (list of floats, recent confidences)
          tracked_objects                 (list of dicts from tracker.update())
          tracker_stats                   (dict from ByteTracker.get_stats())
        """
        with self._lock:
            self._data = payload

    # ── Refresh loop ───────────────────────────────────────────────────────────
    def _schedule_refresh(self):
        if not self._closed:
            try:
                self.after(self._REFRESH_MS, self._refresh)
            except Exception:
                pass

    def _refresh(self):
        if self._closed:
            return
        with self._lock:
            d = dict(self._data)
        self._apply(d)
        self._schedule_refresh()

    def _apply(self, d: dict):
        try:
            self._apply_stability(d)
            self._apply_confidence(d)
            self._apply_per_object(d)
            self._apply_system(d)
        except Exception:
            pass

    def _apply_stability(self, d: dict):
        total   = d.get("det_frames_detected", 0)
        missing = d.get("det_frames_missing",  0)
        drops   = d.get("det_drops",           0)
        det_all = total + missing
        det_pct = (total / det_all * 100) if det_all > 0 else 0.0

        ts = d.get("tracker_stats", {})
        active = ts.get("active_tracks", 0)
        lost   = ts.get("lost_tracks",   0)
        trk_pct = (active / (active + lost) * 100) if (active + lost) > 0 else 0.0

        self._d_stab_pct.set(f"{det_pct:.1f}%")
        self._d_det_total.set(str(total))
        self._d_det_miss.set(str(missing))
        self._d_drop_cnt.set(str(drops))
        self._d_avg_conf.set(f"{d.get('avg_conf', 0.0):.3f}")

        self._t_stab_pct.set(f"{trk_pct:.1f}%")
        self._t_id_chg.set(str(d.get("trk_id_changes", 0)))
        self._t_lost.set(str(d.get("trk_lost",      0)))
        self._t_recovered.set(str(ts.get("recovered_total", 0)))
        self._t_ghost_cnt.set(str(d.get("trk_ghost", 0)))
        self._t_conf_avg.set(f"{d.get('smoother_avg_conf', 0.0):.3f}")
        self._t_lifetime.set(f"{d.get('trk_lifetime_avg', 0.0):.1f} s")

    def _apply_confidence(self, d: dict):
        hist = d.get("conf_history", [])
        if hist:
            self._c_cur.set(f"{d.get('conf_current', 0.0):.3f}")
            self._c_avg.set(f"{sum(hist)/len(hist):.3f}")
            self._c_min.set(f"{min(hist):.3f}")
            self._c_max.set(f"{max(hist):.3f}")
            trend = "  ".join(f"{v:.2f}" for v in hist[-20:])
            self._conf_trend_label.configure(text=trend)

        self._c_drops.set(str(d.get("conf_drop_events", 0)))
        sm = d.get("smoother_stats", {})
        self._s_active.set(str(sm.get("active_entries", 0)))
        self._s_ghost.set(str(sm.get("ghost_entries",   0)))
        self._s_avg_sc.set(f"{sm.get('avg_smooth_conf', 0.0):.3f}")

    def _apply_per_object(self, d: dict):
        tracked = d.get("tracked_objects", [])
        names   = d.get("class_names",     {})

        existing_ids = set()
        for obj in tracked:
            tid = obj.get("track_id", 0)
            existing_ids.add(tid)
            cls_name = names.get(obj.get("class_id", 0), f"cls{obj.get('class_id',0)}")
            age     = f"{obj.get('age_sec', 0.0):.1f}"
            hits    = str(obj.get("hits",            0))
            lost    = str(obj.get("lost_count",      0))
            recov   = str(obj.get("recovered_count", 0))
            conf    = f"{obj.get('confidence', 0.0):.2f}"
            dirn    = obj.get("direction", "—")
            ghost   = obj.get("ghost", False)
            color   = "#64748B" if ghost else "#CBD5E1"

            if tid in self._obj_rows:
                lbls = self._obj_rows[tid]
                vals = [str(tid), cls_name, age, hits, lost, recov, conf, dirn]
                for lbl, val in zip(lbls, vals):
                    lbl.configure(text=val, text_color=color)
            else:
                row = ctk.CTkFrame(self._obj_scroll, fg_color="transparent")
                row.pack(fill="x", padx=2, pady=1)
                vals  = [str(tid), cls_name, age, hits, lost, recov, conf, dirn]
                widths = [40, 90, 55, 45, 40, 45, 55, 80]
                lbls  = []
                for val, w in zip(vals, widths):
                    lbl = ctk.CTkLabel(row, text=val,
                                       font=("Consolas", 9),
                                       text_color=color, width=w, anchor="w")
                    lbl.pack(side="left", padx=2)
                    lbls.append(lbl)
                self._obj_rows[tid] = lbls

        # Remove rows for tracks that are gone
        gone = set(self._obj_rows.keys()) - existing_ids
        for tid in gone:
            del self._obj_rows[tid]
        # Refresh visible rows in scroll (CTk doesn't auto-update)

    def _apply_system(self, d: dict):
        self._p_fps.set(f"{d.get('fps_inf',  0.0):.1f}")
        self._p_avg_fps.set(f"{d.get('avg_fps', 0.0):.1f}")
        self._p_inf_ms.set(f"{d.get('inf_ms',  0.0):.1f} ms")
        self._p_pre_ms.set(f"{d.get('pre_ms',  0.0):.1f} ms")
        self._p_post_ms.set(f"{d.get('post_ms', 0.0):.1f} ms")
        self._p_drops.set(str(d.get("frame_drops", 0)))
        self._r_cpu.set(f"{d.get('cpu_pct', 0.0):.0f}%")
        self._r_ram.set(f"{d.get('ram_mb',  0):.0f} MB")
        self._m_name.set(d.get("model_name",  "—"))
        self._m_size.set(str(d.get("input_size", "—")))
        self._m_classes.set(str(d.get("num_classes", "—")))

    def _on_close(self):
        self._closed = True
        self.destroy()
