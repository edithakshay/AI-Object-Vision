# DualVision AI Detector — Phase 2 Tracking Test Results
## Version 1.3 Stable CPU Edition

All tests are static-analysis / logic tests performed by code inspection.
Runtime tests require a live RTSP stream or simulated frames.

---

## 1. Syntax Validation

| File                            | Result | Notes                              |
|---------------------------------|--------|------------------------------------|
| `tracking/tracker.py`           | PASS   | AST-clean; cv2/np/deque all imported |
| `ui/main_window.py`             | PASS   | `draw_trails` import added          |
| `ui/control_panel.py`           | PASS   | 2 new StringVar rows added          |
| `ui/settings_dialog.py`         | PASS   | 4th tab added; `_save()` extended   |
| `config/settings.py`            | PASS   | `"tracking"` section appended       |

---

## 2. Non-Regression Checklist

| Existing feature                        | Regression risk | Status |
|-----------------------------------------|-----------------|--------|
| STEP 3 always reached in `_process_result` | None — structure unchanged | PASS |
| Tracking exception-safe wrapper (STEP 2)  | None — wrapper unchanged   | PASS |
| RGB / Thermal camera switch              | Trackers re-created by `_make_tracker()` on switch (existing `reset()` calls preserved) | PASS |
| `_on_stop()` resets trackers             | `.reset()` still called; event callbacks survive reset (callbacks are on the instance, not cleared by reset) | PASS |
| Bounding boxes always drawn              | `draw_detections()` call unchanged; trail drawing is additive after it | PASS |
| CSV / JSON export                        | `DetectionLog` not touched | PASS |
| Screenshot / Recording                  | Display frame pipeline: trail overlay happens before recorder write | PASS |
| Existing 7 TRACKING stat rows            | All preserved; 2 new rows appended below them | PASS |
| Settings: General / ONNX / Dashboard tabs| Unchanged — only `_build_tracking()` added | PASS |
| `_save()` existing keys                  | New tracking block wrapped in separate `try/except`; failure leaves existing saves intact | PASS |

---

## 3. New Feature Verification

### 3.1 Track History / Motion

| Scenario                               | Expected behaviour                        | Status |
|----------------------------------------|-------------------------------------------|--------|
| First detection of a track             | `center_history` seeded with 1 point      | PASS (code review) |
| Subsequent detections                  | Each `_Track.update()` appends `(cx,cy)`  | PASS (code review) |
| Trail length capped                    | `deque(maxlen=max_trail_length)` auto-trims | PASS (code review) |
| Velocity computed                      | `vx = new_cx − prev_cx` per frame        | PASS (code review) |
| Direction label                        | `_compute_direction()` returns one of 8 strings | PASS (code review) |

### 3.2 Trail Drawing

| Scenario                               | Expected behaviour                        | Status |
|----------------------------------------|-------------------------------------------|--------|
| `enable_trails=False` (default)        | `draw_trails()` never called; zero overhead | PASS (code review) |
| `enable_trails=True`, 1 track, 5 pts  | Polyline drawn with fading brightness     | PASS (code review) |
| `enable_trails=True`, empty trails     | Early return — original frame returned    | PASS (code review) |
| Tracking disabled entirely             | `enable_trails` is ANDed with `track`; trails silently skipped | PASS (code review) |

### 3.3 Event Logging

| Scenario                               | Expected behaviour                        | Status |
|----------------------------------------|-------------------------------------------|--------|
| New object detected                    | `CREATED` line in `logs/tracking.log`     | PASS (code review) |
| Object disappears 1 frame              | `LOST` line written                       | PASS (code review) |
| Object reappears                       | `RECOVERED` line written                  | PASS (code review) |
| Track exceeds `max_age`                | `REMOVED` line written                    | PASS (code review) |
| Logger exception in callback           | Swallowed by `try/except` in `_emit()`    | PASS (code review) |
| Duplicate handler guard                | `if not self._trk_logger.handlers:` prevents double-logging on re-entry | PASS (code review) |

### 3.4 Confirmed Tracks & Longest Active Stats

| Scenario                               | Expected                                  | Status |
|----------------------------------------|-------------------------------------------|--------|
| `min_hits=1`, track hits=1             | Counted as confirmed                      | PASS (code review) |
| `min_hits=3`, track hits=2             | Not confirmed yet                         | PASS (code review) |
| Longest active track stat              | `max(track_age_sec)` across active tracks | PASS (code review) |
| Control panel shows both new rows      | `Confirmed` + `Longest Active` rows visible | PASS (code review) |

### 3.5 Tracking Settings Tab

| Scenario                               | Expected                                  | Status |
|----------------------------------------|-------------------------------------------|--------|
| Dialog opens                           | 4 tabs visible: General / ONNX CPU / Dashboard / Tracking | PASS (code review) |
| Slider values loaded from settings     | Each control reads `settings.get("tracking", ...)` | PASS (code review) |
| Save button pressed                    | All tracking keys written to settings and saved to disk | PASS (code review) |
| Restore Defaults pressed               | `BooleanVar.set(False)` + sliders reset to defaults | PASS (fixed — was incorrectly calling `.deselect()`) |
| Invalid slider value                   | Wrapped in `try/except` in `_save()` | PASS (code review) |

### 3.6 Empty-Frame Re-identification Fix

| Scenario                               | Expected                                  | Status |
|----------------------------------------|-------------------------------------------|--------|
| Object absent for 1 frame              | `update([])` ages track to `missed=1`; track marked `lost` | PASS (code review) |
| Object returns within `max_age`        | Lost-track stage matches it; `RECOVERED` event emitted | PASS (code review) |
| Object absent > `max_age` frames       | Track removed; fresh track created on return | PASS (code review) |
| Prior behaviour: `reset()` on empty    | Replaced by `update([])` — no data loss from single-frame gap | FIXED |

---

## 4. Known Limitations

- Trail lines are drawn on the CPU using OpenCV; on very slow machines (< 4
  cores) with many tracks (> 30) and a long trail (> 60 pts), trail drawing
  may add 1–3 ms per frame. Keep `max_trail_length ≤ 40` on low-power hardware.
- `draw_trails()` does not anti-alias the bright tip dot; on high-DPI displays
  the dot may appear slightly pixelated.
- Tracking events are written to `tracking.log` from the worker thread; on
  extremely high-throughput sessions the log file may grow quickly. Rotate
  or clear it periodically.

---

## 5. Upgrade / Rollback Notes

- No schema changes to `config/app_config.json`; new `tracking` keys are
  merged in from `DEFAULT_SETTINGS` on load if absent.
- Downgrading to Phase 1 requires reverting `tracker.py`, `main_window.py`,
  `control_panel.py`, and `settings_dialog.py`. `config/settings.py` changes
  are backward-compatible (extra keys are ignored by Phase 1 code).
