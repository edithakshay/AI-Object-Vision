---
name: DualVisionAI Tracker Phase 3
description: Phase 3 Mission Manager architecture decisions, file layout, and wiring conventions.
---

## Phase 3 — Mission Manager (Search & Rescue Core)

**New package:** `DualVisionAI/mission/`
- `mission_state.py` — MissionState (lifecycle, folder creation, stats, timeline, DB listing)
- `evidence_manager.py` — EvidenceManager (auto-screenshot per new track ID → evidence/folder)
- `alert_system.py` — AlertSystem (per-track, non-blocking beep, popup via callback)

**New UI:** `DualVisionAI/ui/mission_dialog.py` — MissionDialog (7 tabs: Setup, Evidence, Review, Timeline, Statistics, Filters, History)

**Toolbar:** Mission button (🎯 Mission) added AFTER Settings button; both coexist permanently per Rule 1/3.

**Why separate mission package:** Keeps file-system + state concerns out of main_window; main_window only calls `_evidence_mgr.capture()` and `_alert_system.check()` inside the STEP 3 detection loop.

**Evidence capture wiring:** Called inside `_process_result` for every detection, after `_det_log.log()`. Uses a copy of `_rgb_display_frame` / `_thermal_display_frame` — the already-annotated frame. Guard: `try/except pass` so mission failures never break detection.

**Alert wiring:** Same location — `_alert_system.check()` called per detection; alerts fire at most once per unique track_id (deduped inside AlertSystem). Callbacks go to MissionDialog via `set_callbacks()`.

**Mission folder auto-created on Start:** `missions/Mission_<date>_<name>/` with sub-dirs: recordings, screenshots, evidence, logs, report. Also creates empty `detections.csv` (with header) and `detections.json` (`[]`).

**Thread-safety:** MissionDialog collects UI updates via `_pending_*` queues guarded by `_queue_lock`; drains them on the main thread via `after(500, _tick)` so no Tkinter calls happen from worker threads.

**Why:** Tkinter is not thread-safe; all widget updates must be on the main thread.

**Priority system:**
- 🔴 High: person, fire, smoke, boat
- 🟡 Medium: car, truck, bus, motorcycle, bicycle, vehicle, animal, cat, dog, horse, backpack
- 🟢 Low: everything else

**Mission history:** `MissionState.list_missions()` scans `missions/*/mission.json` — works without a DB.

**VS Code packaging:** Phase 3 ships as `DualVisionAI_Phase3.tar.gz` at workspace root. VSCODE_SETUP.md updated with full Phase 3 feature documentation.
