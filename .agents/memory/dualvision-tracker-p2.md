---
name: DualVisionAI Tracker Phase 2
description: Phase 2 tracking architecture decisions, constraints, and non-obvious choices for DualVision AI Detector v1.3.
---

## Key decisions

**Tracker factory pattern:**
`_make_tracker()` in `MainWindow` reads `settings.section("tracking")` and
constructs `ByteTracker` with all params. Called at startup AND on every Start
press so settings changes take effect without a full app restart.

**Why:** Avoids stale tracker params when user changes Tracking tab settings.

**Empty-frame handling (critical fix):**
On `result.is_empty()`, call `tracker.update([])` instead of `tracker.reset()`.
This ages tracks naturally so lost-track re-identification works across brief
occlusion gaps (≤ max_age frames).

**Why:** `reset()` was destroying all track history on any single empty frame,
preventing re-identification. `update([])` just increments missed counters.

**Trail drawing location:**
`draw_trails()` is called in `_process_one_frame` AFTER `draw_detections()`,
so bounding boxes always appear on top of trails. Trails are only drawn when
`tracking.enable_trails=True` AND `detection.enable_tracking=True`.

**Why:** Additive — zero risk to existing detection pipeline; both flags checked
so disable tracking also disables trails.

**Event callback pattern:**
`ByteTracker.set_event_callback(fn)` registers a single callable
`fn(event_type, track)`. Emitter wraps call in try/except so tracker never
crashes from a logging error. Events: created / lost / recovered / removed.

**Why:** Keeps file-system concerns out of the tracker; caller decides what to
do with events (log file, UI, etc.).

**Tracking logger:**
Separate `logging.FileHandler` → `logs/tracking.log`, propagate=False.
Guard `if not self._trk_logger.handlers:` prevents duplicate handlers if
`_init_services()` is ever called twice.

**Settings dialog:**
`_add_switch()` returns `BooleanVar` (not the widget). Use `.set(False)` to
reset, NOT `.deselect()` — the widget has no such method.
`_add_slider()` sets its own command; overriding it replaces the built-in value
label update — avoid calling `.configure(command=...)` on returned sliders.

**Control panel stat keys (Phase 2 additions):**
`trk_confirmed` → `confirmed_tracks` stat
`trk_longest`   → `longest_active_sec` stat
Both are keyword args with default=0 so existing callers without them still work.

**Phase 1 fix structure MUST be preserved:**
`_process_result` STEP 1 / STEP 2 / STEP 3 structure is the regression fix from
Phase 1. STEP 3 (`self._rgb_draw_result = result`) must always execute.
The tracking `try/except` in STEP 2 must never be removed.
