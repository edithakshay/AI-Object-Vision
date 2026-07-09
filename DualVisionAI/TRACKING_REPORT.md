# DualVision AI Detector — Phase 2 Tracking Report
## Version 1.3 Stable CPU Edition

---

## Overview

Phase 2 adds stable object tracking with full trail-line visualisation, motion
estimation, lifecycle event logging, expanded statistics, and a dedicated
Tracking settings tab — all with **zero regression** to existing features.

---

## Changes by File

### `config/settings.py`
Added `"tracking"` section to `DEFAULT_SETTINGS`:

| Key                         | Default | Description                            |
|-----------------------------|---------|----------------------------------------|
| `enable_trails`             | `False` | Draw trail polylines on the video feed |
| `max_trail_length`          | `30`    | Max centre-point history per track     |
| `track_timeout`             | `5`     | Missed frames before track is removed  |
| `min_confirmation_hits`     | `1`     | Min detections before track is emitted |
| `tracking_confidence`       | `0.45`  | Confidence split (high vs low dets)    |
| `association_threshold`     | `0.35`  | Stage-1 IoU threshold                  |
| `low_association_threshold` | `0.20`  | Stage-2 / lost-track IoU threshold     |

---

### `tracking/tracker.py`
**`_Track` class — new per-track fields:**
- `first_seen` / `last_seen` — wall-clock timestamps
- `velocity` — `(vx, vy)` pixels/frame, computed on every `update()`
- `direction` — string label: stationary / left / right / up / down / diagonal
- `center_history` — `deque(maxlen=max_trail_length)` of `(cx, cy)` centre points

**`ByteTracker` — new constructor parameters:**
- `max_trail_length` (default 30) — passed to each `_Track`

**`ByteTracker` — new public API:**
- `set_event_callback(fn)` — register `fn(event_type, track)` for lifecycle events
- `get_trails() -> dict` — returns `{track_id: [(cx,cy), ...]}` for all active tracks
- `confirmed_tracks` property — count of active tracks that have met `min_hits`
- `longest_active_track_sec` property — age of the oldest active track
- `get_stats()` — now includes `confirmed_tracks` and `longest_active_sec`

**Event types emitted:**
| Event       | When                                                |
|-------------|-----------------------------------------------------|
| `created`   | New `_Track` spawned                                |
| `lost`      | Active track missed for the first time in a frame   |
| `recovered` | Lost track re-matched to a new detection            |
| `removed`   | Track expired (missed > max_age)                    |

**`draw_trails(frame, trails, line_color, thickness)` — new standalone function:**
- Draws polylines connecting historic centre-points
- Brightness fades old→new to indicate motion direction
- Bright dot rendered at the current tip position

**Empty-detection fix:**
- `update([])` on empty frames ages tracks naturally (missed count increments)
- Replaces the prior `reset()` call, enabling re-identification after brief occlusion

---

### `ui/control_panel.py`
Added two new rows to the **TRACKING** section:

| Row             | Colour    | Source stat             |
|-----------------|-----------|-------------------------|
| Confirmed       | `#16A34A` | `confirmed_tracks`      |
| Longest Active  | `#818CF8` | `longest_active_sec`    |

Updated `update_stats()` and `update_tracking_stats()` signatures to accept
the two new keyword arguments (`trk_confirmed`, `trk_longest`).

---

### `ui/settings_dialog.py`
Added **"Tracking"** as a 4th tab. The tab contains:

**Trail Lines section:**
- Enable Trail Lines (switch)
- Max Trail Length (slider, 5–80 frames, live frame count display)

**Track Parameters section:**
- Track Timeout (slider, 1–30 frames) — takes effect on next Start
- Min Confirmation Frames (option menu, 1–5)
- Association IoU Threshold (slider, 0.10–0.80)
- Confidence Split (slider, 0.10–0.95)
- _Restore Tracking Defaults_ button

**Tracking Log section:**
- Informational text pointing users to `logs/tracking.log`

All values are loaded from `settings.tracking.*` on dialog open and persisted
on Save.

---

### `ui/main_window.py`
**New `_make_tracker()` method:**
- Reads `settings.tracking.*` and constructs a `ByteTracker` with correct params
- Called at startup and on every Start press (picks up changed settings)

**New `_on_tracking_event(event_type, track)` method:**
- Called by `ByteTracker` via event callback
- Writes one line per event to `logs/tracking.log`
- Line format: `EVENT  Track#NNN  class=…  hits=…  age=…s  spd=…px  dir=…  box=(…)`

**Trail drawing in `_process_one_frame()`:**
- After `draw_detections()`, if `tracking.enable_trails=True`, calls
  `draw_trails(display, tracker.get_trails())`
- Trail drawing is skipped when tracking is disabled

**`_do_ui_tick()` — extended stats push:**
- `trk_confirmed` and `trk_longest` added to `update_stats()` call

**Tracking logger setup in `_init_services()`:**
- Creates `logs/tracking.log` file handler with `propagate=False`
- Handler is only added once (guard against duplicate handlers on re-entry)

---

## Non-Regression Guarantees

| Preserved feature                   | Mechanism                                           |
|-------------------------------------|-----------------------------------------------------|
| Bounding boxes always visible       | STEP 1 / STEP 2 / STEP 3 structure unchanged        |
| Exception-safe tracking             | `try/except` wrapper in STEP 2 unchanged            |
| RGB / Thermal camera switching      | `_make_tracker()` called per camera stream          |
| CSV / JSON export                   | `DetectionLog` unchanged                            |
| Screenshot / Recording              | Display frame pipeline unchanged                    |
| ONNX Runtime CPU-only               | Detector untouched                                  |
| Settings persist on save            | `_save()` extended, existing keys preserved         |
| Dashboard / control panel existing  | Only additive changes (new rows, new params)        |

---

## Log Files

| File                   | Contents                                          |
|------------------------|---------------------------------------------------|
| `logs/tracking.log`    | Track lifecycle events (created / lost / …)       |
| `logs/startup.log`     | Unchanged                                         |
| `logs/inference.log`   | Unchanged                                         |
| `logs/camera.log`      | Unchanged                                         |
| `logs/debug.log`       | Unchanged                                         |
| `logs/fps_debug.log`   | Unchanged                                         |
