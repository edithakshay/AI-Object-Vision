"""
Mission Manager Dialog — DualVision AI Phase 3
Full-featured mission management window with tabs:
  1. Setup     — Start/Stop a mission
  2. Evidence  — Auto-captured evidence list
  3. Review    — Operator verify / annotate detections
  4. Timeline  — Automatic mission log
  5. Statistics — Live mission dashboard
  6. Filters   — Live class filter
  7. History   — Past missions database
"""

import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, simpledialog
from typing import Optional

import customtkinter as ctk

from mission.mission_state import (
    MissionState, MissionStatus, MissionType,
    get_priority, priority_label
)
from mission.evidence_manager import EvidenceManager, Evidence
from mission.alert_system import AlertSystem, Alert


# ── Priority colours ──────────────────────────────────────────────────────────
_PCOL = {"high": "#EF4444", "medium": "#F59E0B", "low": "#22C55E"}


class MissionDialog(ctk.CTkToplevel):
    """Floating mission manager window."""

    def __init__(self, parent,
                 mission_state: MissionState,
                 evidence_manager: EvidenceManager,
                 alert_system: AlertSystem):
        super().__init__(parent)
        self._ms  = mission_state
        self._em  = evidence_manager
        self._als = alert_system

        self.title("Mission Manager — DualVision AI")
        self.geometry("860x680")
        self.minsize(800, 600)
        self.configure(fg_color="#0D1626")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Callbacks wired into state objects
        self._ms.set_callbacks(
            on_timeline=self._on_timeline_event,
            on_stats=self._on_stats_update)
        self._em.set_callback(self._on_evidence_captured)
        self._als.set_callbacks(
            on_alert=self._on_alert,
            bell_fn=parent.bell)

        # Active class filter set (empty = show all)
        self._filter_vars: dict = {}   # class_name → BooleanVar
        self._active_filters: set = set()   # classes to SHOW (empty=all)

        self._alert_queue: list = []
        self._pending_timeline: list = []
        self._pending_evidence: list = []
        self._pending_stats: Optional[dict] = None
        self._queue_lock = threading.Lock()

        self._build()
        self._refresh_history()
        self._start_ticker()

    # ── Build ─────────────────────────────────────────────────────────────────
    def _build(self):
        # Header
        hdr = ctk.CTkFrame(self, fg_color="#080E1C", height=48,
                           corner_radius=0)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        ctk.CTkLabel(hdr, text="🎯  Mission Manager",
                     font=("Segoe UI", 15, "bold"),
                     text_color="#2563EB").pack(side="left", padx=16, pady=10)
        self._status_lbl = ctk.CTkLabel(
            hdr, text="● Idle",
            font=("Segoe UI", 11, "bold"),
            text_color="#6B7280")
        self._status_lbl.pack(side="right", padx=16)

        # Tabs
        self._tabs = ctk.CTkTabview(
            self, fg_color="#0D1626",
            segmented_button_fg_color="#131F35",
            segmented_button_selected_color="#2563EB",
            segmented_button_unselected_color="#1E293B",
            segmented_button_selected_hover_color="#1D4ED8",
            text_color="#CBD5E1",
        )
        self._tabs.pack(fill="both", expand=True, padx=10, pady=6)

        for t in ("Setup", "Evidence", "Review", "Timeline",
                  "Statistics", "Filters", "History"):
            self._tabs.add(t)

        self._build_setup()
        self._build_evidence()
        self._build_review()
        self._build_timeline()
        self._build_statistics()
        self._build_filters()
        self._build_history()

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1 — SETUP
    # ─────────────────────────────────────────────────────────────────────────
    def _build_setup(self):
        tab = self._tabs.tab("Setup")

        scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        def row(label, widget_fn):
            fr = ctk.CTkFrame(scroll, fg_color="transparent")
            fr.pack(fill="x", pady=4, padx=8)
            ctk.CTkLabel(fr, text=label, width=140, anchor="w",
                         text_color="#94A3B8").pack(side="left")
            w = widget_fn(fr)
            w.pack(side="left", fill="x", expand=True, padx=(8, 0))
            return w

        self._e_name     = row("Mission Name",  lambda p: ctk.CTkEntry(p, placeholder_text="e.g. Highland Search 01"))
        self._e_operator = row("Operator Name", lambda p: ctk.CTkEntry(p, placeholder_text="Operator"))
        self._e_drone    = row("Drone Name",    lambda p: ctk.CTkEntry(p, placeholder_text="e.g. DJI Matrice 30T"))
        self._e_area     = row("Search Area",   lambda p: ctk.CTkEntry(p, placeholder_text="Grid / GPS region"))

        # Mission type
        fr = ctk.CTkFrame(scroll, fg_color="transparent")
        fr.pack(fill="x", pady=4, padx=8)
        ctk.CTkLabel(fr, text="Mission Type", width=140, anchor="w",
                     text_color="#94A3B8").pack(side="left")
        self._type_var = ctk.StringVar(value=MissionType.SEARCH_RESCUE.value)
        ctk.CTkOptionMenu(
            fr,
            values=[m.value for m in MissionType],
            variable=self._type_var,
            fg_color="#1E3A5F", button_color="#2563EB",
            button_hover_color="#1D4ED8",
        ).pack(side="left", padx=(8, 0))

        # Auto ID display
        self._id_lbl = ctk.CTkLabel(scroll, text="Mission ID:  —",
                                    font=("Segoe UI", 10),
                                    text_color="#475569")
        self._id_lbl.pack(anchor="w", padx=16, pady=(4, 0))

        self._folder_lbl = ctk.CTkLabel(scroll, text="Folder:  —",
                                        font=("Segoe UI", 10),
                                        text_color="#475569", wraplength=600,
                                        justify="left")
        self._folder_lbl.pack(anchor="w", padx=16, pady=(2, 12))

        # Buttons
        btn_row = ctk.CTkFrame(scroll, fg_color="transparent")
        btn_row.pack(pady=10)
        self._btn_start = ctk.CTkButton(
            btn_row, text="▶  Start Mission",
            fg_color="#22C55E", hover_color="#16A34A",
            font=("Segoe UI", 12, "bold"), width=160,
            command=self._on_start_mission)
        self._btn_start.pack(side="left", padx=8)
        self._btn_pause_msn = ctk.CTkButton(
            btn_row, text="⏸  Pause Mission",
            fg_color="#F59E0B", hover_color="#D97706",
            font=("Segoe UI", 12, "bold"), width=160, state="disabled",
            command=self._on_pause_resume_mission)
        self._btn_pause_msn.pack(side="left", padx=8)
        self._btn_finish = ctk.CTkButton(
            btn_row, text="⏹  Finish Mission",
            fg_color="#EF4444", hover_color="#DC2626",
            font=("Segoe UI", 12, "bold"), width=160, state="disabled",
            command=self._on_finish_mission)
        self._btn_finish.pack(side="left", padx=8)

        # Evidence capture settings
        sep = ctk.CTkFrame(scroll, height=1, fg_color="#1E3A5F")
        sep.pack(fill="x", padx=16, pady=12)
        ctk.CTkLabel(scroll, text="  Automatic Evidence Capture",
                     font=("Segoe UI", 11, "bold"),
                     text_color="#CBD5E1").pack(anchor="w", padx=16)

        self._chk_new_track = self._switch(scroll, "Screenshot every new track",
                                           default=True,
                                           cmd=lambda v: setattr(self._em, "screenshot_every_new_track", v))
        self._chk_high_only = self._switch(scroll, "Screenshot High Priority only",
                                           default=False,
                                           cmd=lambda v: setattr(self._em, "screenshot_high_priority_only", v))

        fr2 = ctk.CTkFrame(scroll, fg_color="transparent")
        fr2.pack(fill="x", padx=16, pady=4)
        ctk.CTkLabel(fr2, text="Min Confidence", width=160,
                     anchor="w", text_color="#94A3B8").pack(side="left")
        self._conf_slider = ctk.CTkSlider(fr2, from_=0, to=1, number_of_steps=20,
                                          command=self._on_conf_slider)
        self._conf_slider.set(0.0)
        self._conf_slider.pack(side="left", fill="x", expand=True)
        self._conf_val = ctk.CTkLabel(fr2, text="0.00", width=48,
                                      text_color="#94A3B8")
        self._conf_val.pack(side="left", padx=6)

        # Alert settings
        sep2 = ctk.CTkFrame(scroll, height=1, fg_color="#1E3A5F")
        sep2.pack(fill="x", padx=16, pady=12)
        ctk.CTkLabel(scroll, text="  Alert Settings",
                     font=("Segoe UI", 11, "bold"),
                     text_color="#CBD5E1").pack(anchor="w", padx=16)
        self._switch(scroll, "Enable alerts",
                     default=True,
                     cmd=lambda v: self._als.configure(enabled=v))
        self._switch(scroll, "High Priority alerts only",
                     default=False,
                     cmd=lambda v: self._als.configure(high_only=v))
        self._switch(scroll, "Sound on alert",
                     default=True,
                     cmd=lambda v: self._als.configure(sound=v))

    def _switch(self, parent, label: str, default: bool, cmd=None):
        fr = ctk.CTkFrame(parent, fg_color="transparent")
        fr.pack(fill="x", padx=16, pady=2)
        var = ctk.BooleanVar(value=default)
        def _toggle():
            if cmd:
                cmd(var.get())
        ctk.CTkCheckBox(fr, text=label, variable=var, command=_toggle,
                        text_color="#CBD5E1",
                        checkmark_color="#2563EB",
                        fg_color="#2563EB").pack(side="left")
        return var

    def _on_conf_slider(self, val):
        self._conf_val.configure(text=f"{val:.2f}")
        self._em.min_confidence = val

    # ── Mission control ───────────────────────────────────────────────────────
    def _on_start_mission(self):
        name = self._e_name.get().strip()
        if not name:
            messagebox.showwarning("Mission", "Enter a Mission Name.", parent=self)
            return
        operator = self._e_operator.get().strip() or "Unknown"
        drone    = self._e_drone.get().strip()    or "Unknown"
        area     = self._e_area.get().strip()     or "Unknown"
        mtype    = MissionType(self._type_var.get())

        # ── Order matters:
        # 1. reset()    — clears all in-memory state from the previous mission
        # 2. start()    — creates the NEW unique folder, returns its Path
        # 3. pin_folder() — locks EvidenceManager to that exact folder
        # This sequence guarantees Mission N never inherits Mission N-1 data.
        self._em.reset()
        self._als.reset()
        folder = self._ms.start(name, operator, drone, area, mtype)
        self._em.pin_folder(folder)   # pin AFTER start() so folder is fresh

        self._id_lbl.configure(text=f"Mission ID:  {self._ms.mission_id}")
        self._folder_lbl.configure(text=f"Folder:  {folder}")
        self._btn_start.configure(state="disabled")
        self._btn_pause_msn.configure(state="normal")
        self._btn_finish.configure(state="normal")
        self._update_status_label()
        self._refresh_evidence_list()

    def _on_pause_resume_mission(self):
        if self._ms.status == MissionStatus.ACTIVE:
            self._ms.pause()
            self._btn_pause_msn.configure(text="▶  Resume Mission")
        else:
            self._ms.resume()
            self._btn_pause_msn.configure(text="⏸  Pause Mission")
        self._update_status_label()

    def _on_finish_mission(self):
        if not messagebox.askyesno("Finish Mission",
                                   "Finish and save this mission?",
                                   parent=self):
            return
        # Flush evidence JSON before closing the mission so the final
        # detections.json is complete.  finish() then writes the report
        # and mission.log, and clears mission_dir.
        self._em.flush_final()
        self._ms.finish()
        self._btn_start.configure(state="normal")
        self._btn_pause_msn.configure(state="disabled", text="⏸  Pause Mission")
        self._btn_finish.configure(state="disabled")
        self._update_status_label()
        self._refresh_history()

    def _update_status_label(self):
        st  = self._ms.status
        col = {"Idle": "#6B7280", "Active": "#22C55E",
               "Paused": "#F59E0B", "Finished": "#3B82F6"}.get(st.value, "#6B7280")
        self._status_lbl.configure(text=f"● {st.value}", text_color=col)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2 — EVIDENCE
    # ─────────────────────────────────────────────────────────────────────────
    def _build_evidence(self):
        tab = self._tabs.tab("Evidence")
        hdr = ctk.CTkFrame(tab, fg_color="transparent")
        hdr.pack(fill="x", padx=6, pady=4)
        ctk.CTkLabel(hdr, text="Auto-Captured Evidence",
                     font=("Segoe UI", 12, "bold"),
                     text_color="#E2E8F0").pack(side="left")
        ctk.CTkButton(hdr, text="Open Folder", width=100,
                      fg_color="#1E3A5F", hover_color="#334155",
                      command=self._open_evidence_folder).pack(side="right")

        self._ev_list = ctk.CTkScrollableFrame(tab, fg_color="#0A1628")
        self._ev_list.pack(fill="both", expand=True, padx=6, pady=4)
        self._ev_rows: dict = {}   # evidence_id → CTkFrame

    def _refresh_evidence_list(self):
        for w in list(self._ev_rows.values()):
            try:
                w.destroy()
            except Exception:
                pass
        self._ev_rows.clear()
        for ev in self._em.get_all():
            self._add_evidence_row(ev)

    def _add_evidence_row(self, ev: Evidence):
        if ev.evidence_id in self._ev_rows:
            return
        pcol = _PCOL.get(ev.priority, "#22C55E")
        fr = ctk.CTkFrame(self._ev_list, fg_color="#131F35",
                          corner_radius=6, border_width=1,
                          border_color=pcol)
        fr.pack(fill="x", padx=4, pady=2)

        # Priority strip
        ctk.CTkFrame(fr, width=4, fg_color=pcol,
                     corner_radius=2).pack(side="left", fill="y", padx=(4, 6))

        info = ctk.CTkFrame(fr, fg_color="transparent")
        info.pack(side="left", fill="x", expand=True, pady=4)
        ctk.CTkLabel(info,
                     text=f"{priority_label(ev.priority)}  {ev.class_name.title()}",
                     font=("Segoe UI", 10, "bold"),
                     text_color="#E2E8F0").pack(anchor="w")
        ctk.CTkLabel(info,
                     text=(f"Conf: {ev.confidence:.2f}  |  Track#{ev.track_id}"
                           f"  |  {ev.camera}  |  {ev.timestamp.strftime('%H:%M:%S')}"),
                     font=("Segoe UI", 9),
                     text_color="#94A3B8").pack(anchor="w")

        acts = ctk.CTkFrame(fr, fg_color="transparent")
        acts.pack(side="right", padx=6)
        if ev.image_path and os.path.exists(ev.image_path):
            ctk.CTkButton(acts, text="View", width=50,
                          fg_color="#1E3A5F", hover_color="#334155",
                          font=("Segoe UI", 9),
                          command=lambda p=ev.image_path: self._open_image(p)
                          ).pack(side="left", padx=2)

        self._ev_rows[ev.evidence_id] = fr

    def _open_evidence_folder(self):
        if self._ms.mission_dir:
            path = str(self._ms.mission_dir / "evidence")
        else:
            path = "missions"
        try:
            if sys.platform == "win32":
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception:
            pass

    def _open_image(self, path: str):
        try:
            import tkinter as tk
            from PIL import Image, ImageTk
            top = ctk.CTkToplevel(self)
            top.title(os.path.basename(path))
            img  = Image.open(path)
            img.thumbnail((800, 600))
            photo = ImageTk.PhotoImage(img)
            lbl = ctk.CTkLabel(top, image=photo, text="")
            lbl.image = photo
            lbl.pack(padx=10, pady=10)
        except Exception as exc:
            messagebox.showinfo("Image", str(exc), parent=self)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3 — REVIEW
    # ─────────────────────────────────────────────────────────────────────────
    def _build_review(self):
        tab = self._tabs.tab("Review")
        hdr = ctk.CTkFrame(tab, fg_color="transparent")
        hdr.pack(fill="x", padx=6, pady=4)
        ctk.CTkLabel(hdr, text="Detection Review",
                     font=("Segoe UI", 12, "bold"),
                     text_color="#E2E8F0").pack(side="left")
        ctk.CTkButton(hdr, text="↻ Refresh", width=90,
                      fg_color="#1E3A5F", hover_color="#334155",
                      font=("Segoe UI", 9),
                      command=self._refresh_review).pack(side="right")

        self._rev_list = ctk.CTkScrollableFrame(tab, fg_color="#0A1628")
        self._rev_list.pack(fill="both", expand=True, padx=6, pady=4)

    def _refresh_review(self):
        for w in self._rev_list.winfo_children():
            try:
                w.destroy()
            except Exception:
                pass

        items = self._em.get_all()
        if not items:
            ctk.CTkLabel(self._rev_list, text="No detections yet.",
                         text_color="#475569").pack(pady=20)
            return

        for ev in items:
            self._add_review_row(ev)

    def _add_review_row(self, ev: Evidence):
        pcol = _PCOL.get(ev.priority, "#22C55E")
        fr = ctk.CTkFrame(self._rev_list, fg_color="#131F35",
                          corner_radius=6, border_width=1,
                          border_color=pcol if not ev.verified else "#22C55E")
        fr.pack(fill="x", padx=4, pady=3)

        top_row = ctk.CTkFrame(fr, fg_color="transparent")
        top_row.pack(fill="x", padx=8, pady=(6, 2))

        ctk.CTkLabel(top_row,
                     text=f"{priority_label(ev.priority)}  {ev.class_name.title()}  "
                          f"| conf={ev.confidence:.2f}  | Track#{ev.track_id}  "
                          f"| {ev.camera}  | {ev.timestamp.strftime('%H:%M:%S')}",
                     font=("Segoe UI", 10), text_color="#E2E8F0").pack(side="left")

        btn_row = ctk.CTkFrame(fr, fg_color="transparent")
        btn_row.pack(fill="x", padx=8, pady=(0, 6))

        verify_text = "✓ Verified" if ev.verified else "Verify"
        ctk.CTkButton(btn_row, text=verify_text, width=80,
                      fg_color="#22C55E" if ev.verified else "#1E3A5F",
                      hover_color="#16A34A",
                      font=("Segoe UI", 9),
                      command=lambda e=ev: self._do_verify(e)).pack(side="left", padx=2)
        ctk.CTkButton(btn_row, text="📝 Note", width=80,
                      fg_color="#1E3A5F", hover_color="#334155",
                      font=("Segoe UI", 9),
                      command=lambda e=ev: self._do_note(e)).pack(side="left", padx=2)
        ctk.CTkButton(btn_row, text="🗑 Delete", width=80,
                      fg_color="#7F1D1D", hover_color="#991B1B",
                      font=("Segoe UI", 9),
                      command=lambda e=ev, f=fr: self._do_delete(e, f)
                      ).pack(side="left", padx=2)
        if ev.image_path and os.path.exists(ev.image_path):
            ctk.CTkButton(btn_row, text="🔍 View", width=80,
                          fg_color="#1E3A5F", hover_color="#334155",
                          font=("Segoe UI", 9),
                          command=lambda p=ev.image_path: self._open_image(p)
                          ).pack(side="left", padx=2)

        if ev.notes:
            ctk.CTkLabel(fr, text=f"Note: {ev.notes}",
                         font=("Segoe UI", 9), text_color="#94A3B8",
                         justify="left").pack(anchor="w", padx=16, pady=(0, 4))

    def _do_verify(self, ev: Evidence):
        ev.verified = not ev.verified
        self._em.verify(ev.evidence_id, ev.verified, ev.notes)
        self._refresh_review()

    def _do_note(self, ev: Evidence):
        note = simpledialog.askstring("Add Note",
                                      f"Note for {ev.class_name} #{ev.evidence_id}:",
                                      parent=self)
        if note is not None:
            ev.notes = note
            self._em.verify(ev.evidence_id, ev.verified, note)
            self._refresh_review()

    def _do_delete(self, ev: Evidence, frame):
        if messagebox.askyesno("Delete", f"Delete evidence {ev.evidence_id}?",
                                parent=self):
            self._em.delete(ev.evidence_id)
            try:
                frame.destroy()
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 4 — TIMELINE
    # ─────────────────────────────────────────────────────────────────────────
    def _build_timeline(self):
        tab = self._tabs.tab("Timeline")
        hdr = ctk.CTkFrame(tab, fg_color="transparent")
        hdr.pack(fill="x", padx=6, pady=4)
        ctk.CTkLabel(hdr, text="Mission Timeline",
                     font=("Segoe UI", 12, "bold"),
                     text_color="#E2E8F0").pack(side="left")
        ctk.CTkButton(hdr, text="Clear", width=70,
                      fg_color="#1E3A5F", hover_color="#334155",
                      font=("Segoe UI", 9),
                      command=self._clear_timeline).pack(side="right")

        self._timeline_box = ctk.CTkTextbox(
            tab, fg_color="#080E1C", text_color="#CBD5E1",
            font=("Consolas", 10), wrap="word", state="disabled")
        self._timeline_box.pack(fill="both", expand=True, padx=6, pady=4)

        self._timeline_box.tag_config("info",      foreground="#CBD5E1")
        self._timeline_box.tag_config("warning",   foreground="#F59E0B")
        self._timeline_box.tag_config("detection", foreground="#22C55E")
        self._timeline_box.tag_config("alert",     foreground="#EF4444")

    def _append_timeline_ui(self, entry):
        ts  = entry.timestamp.strftime("%H:%M:%S")
        tag = entry.level if entry.level in ("info", "warning", "detection", "alert") else "info"
        self._timeline_box.configure(state="normal")
        self._timeline_box.insert("end", f"{ts}  {entry.message}\n", tag)
        self._timeline_box.configure(state="disabled")
        self._timeline_box.see("end")

    def _clear_timeline(self):
        self._timeline_box.configure(state="normal")
        self._timeline_box.delete("1.0", "end")
        self._timeline_box.configure(state="disabled")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 5 — STATISTICS
    # ─────────────────────────────────────────────────────────────────────────
    def _build_statistics(self):
        tab = self._tabs.tab("Statistics")
        ctk.CTkLabel(tab, text="Mission Statistics",
                     font=("Segoe UI", 12, "bold"),
                     text_color="#E2E8F0").pack(anchor="w", padx=12, pady=(10, 6))

        grid = ctk.CTkFrame(tab, fg_color="transparent")
        grid.pack(fill="both", expand=True, padx=10, pady=4)
        grid.columnconfigure((0, 1, 2), weight=1)

        self._stat_widgets: dict = {}

        stat_defs = [
            ("Mission Time",        "time",              "#3B82F6"),
            ("Total Detections",    "total_detections",  "#22C55E"),
            ("Persons Found",       "persons",           "#EF4444"),
            ("Vehicles Found",      "vehicles",          "#F59E0B"),
            ("Animals Found",       "animals",           "#8B5CF6"),
            ("Fire/Smoke Events",   "fire_smoke",        "#F97316"),
            ("Screenshots Saved",   "screenshots",       "#06B6D4"),
            ("Avg Confidence",      "avg_confidence",    "#10B981"),
            ("Detection Rate/min",  "detection_rate",    "#6366F1"),
        ]

        for i, (label, key, colour) in enumerate(stat_defs):
            row, col = divmod(i, 3)
            card = ctk.CTkFrame(grid, fg_color="#131F35", corner_radius=8)
            card.grid(row=row, column=col, padx=6, pady=6, sticky="nsew")
            ctk.CTkLabel(card, text=label, font=("Segoe UI", 9),
                         text_color="#94A3B8").pack(pady=(10, 0))
            val_lbl = ctk.CTkLabel(card, text="—",
                                   font=("Segoe UI", 20, "bold"),
                                   text_color=colour)
            val_lbl.pack(pady=(2, 10))
            self._stat_widgets[key] = val_lbl
            grid.rowconfigure(row, weight=1)

    def _update_statistics_ui(self, stats: dict):
        self._stat_widgets["time"].configure(text=self._ms.elapsed_str)
        for key, w in self._stat_widgets.items():
            if key == "time":
                continue
            val = stats.get(key, 0)
            if key == "avg_confidence":
                w.configure(text=f"{val:.2f}")
            elif key == "detection_rate":
                w.configure(text=f"{val*60:.1f}")
            else:
                w.configure(text=str(val))

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 6 — FILTERS
    # ─────────────────────────────────────────────────────────────────────────
    _FILTER_CLASSES = [
        ("Person", "person"), ("Vehicle", "car"), ("Animal", "dog"),
        ("Fire", "fire"), ("Smoke", "smoke"), ("Boat", "boat"),
        ("Backpack", "backpack"), ("Chair", "chair"), ("Bottle", "bottle"),
        ("Truck", "truck"), ("Bus", "bus"), ("Bicycle", "bicycle"),
    ]

    def _build_filters(self):
        tab = self._tabs.tab("Filters")
        ctk.CTkLabel(tab, text="Live Detection Filters",
                     font=("Segoe UI", 12, "bold"),
                     text_color="#E2E8F0").pack(anchor="w", padx=12, pady=(10, 4))
        ctk.CTkLabel(tab,
                     text="Checked classes will be shown in the Evidence list.\n"
                          "Uncheck to hide from the live feed filter.",
                     font=("Segoe UI", 9), text_color="#94A3B8",
                     justify="left").pack(anchor="w", padx=12, pady=(0, 10))

        frame = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        frame.pack(fill="both", expand=True, padx=12)

        for label, cls in self._FILTER_CLASSES:
            var = ctk.BooleanVar(value=True)
            self._filter_vars[cls] = var

            def _on_toggle(c=cls, v=var):
                if v.get():
                    self._active_filters.discard(c)
                else:
                    self._active_filters.add(c)

            ctk.CTkCheckBox(frame, text=label, variable=var, command=_on_toggle,
                            text_color="#CBD5E1", fg_color="#2563EB",
                            checkmark_color="white").pack(anchor="w", pady=4)

        btn_row = ctk.CTkFrame(tab, fg_color="transparent")
        btn_row.pack(pady=8)
        ctk.CTkButton(btn_row, text="Select All", width=110,
                      fg_color="#1E3A5F", hover_color="#334155",
                      command=self._filters_all).pack(side="left", padx=4)
        ctk.CTkButton(btn_row, text="Clear All", width=110,
                      fg_color="#7F1D1D", hover_color="#991B1B",
                      command=self._filters_none).pack(side="left", padx=4)

    def _filters_all(self):
        for var in self._filter_vars.values():
            var.set(True)
        self._active_filters.clear()

    def _filters_none(self):
        for cls, var in self._filter_vars.items():
            var.set(False)
            self._active_filters.add(cls)

    def is_class_filtered(self, class_name: str) -> bool:
        """Return True if this class is hidden (filtered out)."""
        return class_name.lower() in self._active_filters

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 7 — HISTORY
    # ─────────────────────────────────────────────────────────────────────────
    def _build_history(self):
        tab = self._tabs.tab("History")
        hdr = ctk.CTkFrame(tab, fg_color="transparent")
        hdr.pack(fill="x", padx=6, pady=4)
        ctk.CTkLabel(hdr, text="Mission Database",
                     font=("Segoe UI", 12, "bold"),
                     text_color="#E2E8F0").pack(side="left")
        ctk.CTkButton(hdr, text="↻ Refresh", width=90,
                      fg_color="#1E3A5F", hover_color="#334155",
                      font=("Segoe UI", 9),
                      command=self._refresh_history).pack(side="right")

        self._hist_frame = ctk.CTkScrollableFrame(tab, fg_color="#0A1628")
        self._hist_frame.pack(fill="both", expand=True, padx=6, pady=4)

    def _refresh_history(self):
        for w in self._hist_frame.winfo_children():
            try:
                w.destroy()
            except Exception:
                pass

        missions = MissionState.list_missions()
        if not missions:
            ctk.CTkLabel(self._hist_frame, text="No past missions found.",
                         text_color="#475569").pack(pady=20)
            return

        for m in missions:
            self._add_history_row(m)

    def _add_history_row(self, m: dict):
        folder = m.get("_folder", "")
        status = m.get("status", "?")
        scol   = {"Active": "#22C55E", "Paused": "#F59E0B",
                  "Finished": "#3B82F6", "Idle": "#6B7280"}.get(status, "#6B7280")
        fr = ctk.CTkFrame(self._hist_frame, fg_color="#131F35",
                          corner_radius=6)
        fr.pack(fill="x", padx=4, pady=3)

        info = ctk.CTkFrame(fr, fg_color="transparent")
        info.pack(side="left", fill="x", expand=True, padx=10, pady=6)
        ctk.CTkLabel(info,
                     text=f"{m.get('mission_name', '?')}  [{m.get('mission_id', '?')}]",
                     font=("Segoe UI", 10, "bold"),
                     text_color="#E2E8F0").pack(anchor="w")
        s = m.get("stats", {})
        ctk.CTkLabel(info,
                     text=(f"Type: {m.get('mission_type','?')}  |  "
                           f"Operator: {m.get('operator','?')}  |  "
                           f"Detections: {s.get('total_detections',0)}  |  "
                           f"Persons: {s.get('persons',0)}"),
                     font=("Segoe UI", 9), text_color="#94A3B8").pack(anchor="w")
        ctk.CTkLabel(info,
                     text=f"Start: {m.get('start_time','—')[:19]}",
                     font=("Segoe UI", 9), text_color="#475569").pack(anchor="w")

        act = ctk.CTkFrame(fr, fg_color="transparent")
        act.pack(side="right", padx=8)
        ctk.CTkLabel(act, text=f"● {status}",
                     font=("Segoe UI", 9, "bold"),
                     text_color=scol).pack(pady=2)
        if folder:
            ctk.CTkButton(act, text="Open Folder", width=96,
                          fg_color="#1E3A5F", hover_color="#334155",
                          font=("Segoe UI", 9),
                          command=lambda f=folder: self._open_folder(f)
                          ).pack(pady=2)

    def _open_folder(self, path: str):
        try:
            if sys.platform == "win32":
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # ALERT POPUP
    # ─────────────────────────────────────────────────────────────────────────
    def _show_alert_popup(self, alert: Alert):
        """Show a non-blocking alert popup that auto-closes in 4 s."""
        try:
            popup = ctk.CTkToplevel(self)
            popup.title("🚨 Detection Alert")
            popup.geometry("340x160")
            popup.resizable(False, False)
            popup.configure(fg_color="#1A0A0A")
            popup.attributes("-topmost", True)
            pcol = _PCOL.get(alert.priority, "#22C55E")
            ctk.CTkLabel(popup, text="🚨  DETECTION ALERT",
                         font=("Segoe UI", 13, "bold"),
                         text_color=pcol).pack(pady=(18, 4))
            ctk.CTkLabel(popup,
                         text=(f"{priority_label(alert.priority)}\n"
                               f"{alert.class_name.title()}  |  "
                               f"conf={alert.confidence:.2f}\n"
                               f"Track#{alert.track_id}  |  {alert.camera}  "
                               f"|  {alert.timestamp}"),
                         font=("Segoe UI", 11),
                         text_color="#E2E8F0",
                         justify="center").pack()
            ctk.CTkButton(popup, text="Dismiss", width=100,
                          fg_color="#1E3A5F", hover_color="#334155",
                          command=popup.destroy).pack(pady=10)
            popup.after(4000, lambda: popup.destroy() if popup.winfo_exists() else None)
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # Thread-safe callbacks (called from worker / mission threads)
    # ─────────────────────────────────────────────────────────────────────────
    def _on_timeline_event(self, entry):
        with self._queue_lock:
            self._pending_timeline.append(entry)

    def _on_stats_update(self, stats: dict):
        with self._queue_lock:
            self._pending_stats = stats

    def _on_evidence_captured(self, ev: Evidence):
        with self._queue_lock:
            self._pending_evidence.append(ev)

    def _on_alert(self, alert: Alert):
        with self._queue_lock:
            self._alert_queue.append(alert)
        self._ms.log_event(
            f"🚨 ALERT — {alert.class_name.title()} "
            f"{priority_label(alert.priority)}  conf={alert.confidence:.2f}",
            level="alert")

    # ─────────────────────────────────────────────────────────────────────────
    # UI TICKER — runs on main thread via after()
    # ─────────────────────────────────────────────────────────────────────────
    def _start_ticker(self):
        self._ticker_running = True
        self._tick()

    def _tick(self):
        if not self._ticker_running:
            return
        try:
            with self._queue_lock:
                tl  = list(self._pending_timeline);  self._pending_timeline.clear()
                ev  = list(self._pending_evidence);  self._pending_evidence.clear()
                sts = self._pending_stats;           self._pending_stats = None
                al  = list(self._alert_queue);       self._alert_queue.clear()

            for entry in tl:
                self._append_timeline_ui(entry)
            for e in ev:
                self._add_evidence_row(e)
            if sts:
                self._update_statistics_ui(sts)
            # Always update elapsed time when active
            if self._ms.is_active:
                self._stat_widgets["time"].configure(text=self._ms.elapsed_str)
                self._update_status_label()
            for a in al:
                self._show_alert_popup(a)

        except Exception:
            pass
        self.after(500, self._tick)

    # ─────────────────────────────────────────────────────────────────────────
    def _on_close(self):
        self._ticker_running = False
        self.destroy()
