# DualVisionAI v1.3 — Phase 2 Final Report

**Date:** 2025-07-09
**Edition:** Stable CPU (ONNX Runtime)

---

## Summary

Phase 2 adds two targeted fixes to an otherwise stable codebase:

| # | Issue | Status |
|---|-------|--------|
| 1 | Stable Tracking IDs — same object changes ID while moving | **Fixed** |
| 2 | Recording Bug — 1 KB corrupted file when Detection is OFF | **Fixed** |

---

## Issue 1 — Stable Tracking IDs

### Root Causes Identified

1. **`max_age = 5` (old default)** — Lost tracks expired after only 5 frames (~330 ms at 15 fps). Any single missed detection permanently killed the track, forcing a new ID on the next match.

2. **`iou_threshold = 0.35` (old default)** — Stage 1a matching rejected legitimate correspondences for fast-moving objects whose Kalman-predicted box lagged behind the real position. Rejection meant the track became "lost" for one frame, then got re-matched as a *new* track.

3. **Stage 2 gap — only high-conf detections were tried against lost tracks.** If an object re-appeared at low confidence (common near track boundaries), Stage 2 skipped it. The low-conf detection fell through to Step 3 and spawned a fresh track with a new ID. This is the primary cause of the ID 5→8→12 pattern reported.

4. **`min_hits = 1` (old default)** — Every single-frame false detection instantly produced a confirmed track, inflating the ID counter and causing noisy ID assignments for real objects.

### Parameters Tuned

| Parameter | Old Value | New Value | Reason |
|-----------|-----------|-----------|--------|
| `track_timeout` (`max_age`) | 5 | **30** | Keep lost tracks alive ~2 s; allows re-identification across brief occlusions |
| `min_confirmation_hits` | 1 | **3** | Require 3 consecutive frames before confirming a track; stops single-frame noise IDs |
| `association_threshold` (`iou_threshold`) | 0.35 | **0.20** | More lenient Stage 1a; accommodates fast motion where Kalman prediction lags |
| `low_association_threshold` (`low_iou`) | 0.20 | **0.10** | More lenient Stage 2 recovery; catches objects that re-appear slightly off position |
| `tracking_confidence` (`high_conf`) | 0.45 | **0.40** | Slightly lower split keeps more detections in the high-conf pool for better Stage 1a matching |

### Code Fix — Stage 2b (New)

`tracking/tracker.py` — added **Stage 2b** immediately after Stage 2a:

```
Stage 2a  — unmatched HIGH-conf dets  vs  lost tracks   (existing)
Stage 2b  — unmatched LOW-conf dets   vs  still-unmatched lost tracks   (NEW)
```

When an object re-appears at low detector confidence (common near edges, partial occlusion, or at distance), it is now matched against existing lost tracks before a new track is created. This closes the primary ID-churn gap.

### Per-Track Lifecycle Stats (New)

Each track now carries:
- `lost_count` — how many times the track transitioned to "lost" state
- `recovered_count` — how many times it was successfully re-matched after being lost

Both fields are emitted in the `update()` result list and in the `_on_tracking_event` log.

---

## Issue 2 — Recording Bug

### Root Cause

`_process_one_frame()` contained an early-return branch for the non-detecting path:

```python
if not self._detecting or self._paused or self._detector is None:
    # store frame for UI display
    return                        # ← write() was NEVER called
```

`recorder.start()` still created the MP4 file and opened the `VideoWriter`, but `write()` was never called. On `stop()`, the writer flushed a file containing only the MP4/H.264 container headers — approximately 1 KB — which is not a valid playable video.

### Fix

**`ui/main_window.py` — non-detecting branch:**

```python
if not self._detecting or ...:
    if frame is not None:
        # store for UI
        ...
        # NEW: feed recorder even when detection is off
        rec_label = "RGB" if mode == "rgb" else "Thermal"
        if self._recorder.is_recording(rec_label):
            self._recorder.write(rec_label, frame)
    return
```

**`ui/main_window.py` — detecting branch (new Recording Mode logic):**

```python
rec_mode = self._settings.get("recording", "recording_mode", "overlay")
...
if self._recorder.is_recording("RGB"):
    rec_frame = frame if (rec_mode == "raw" and frame is not None) else display
    if rec_frame is not None:
        self._recorder.write("RGB", rec_frame)
```

### Recording Modes

| Mode | Detection OFF | Detection ON |
|------|---------------|--------------|
| **overlay** (default) | raw camera frame | frame with bounding boxes + trails |
| **raw** | raw camera frame | raw camera frame (no overlay) |

The mode is user-selectable in **Settings → General → Recording Mode**.

---

## Regression Test Results

All previously working features were verified unchanged:

| Feature | Status |
|---------|--------|
| RGB stream display | ✅ Unchanged |
| Thermal stream display | ✅ Unchanged |
| Camera switching (RGB ↔ Thermal) | ✅ Unchanged |
| ONNX Runtime CPU inference pipeline | ✅ Unchanged |
| Bounding box overlay | ✅ Unchanged |
| Trail lines | ✅ Unchanged |
| Dashboard stats | ✅ Unchanged |
| CSV export | ✅ Unchanged |
| JSON export | ✅ Unchanged |
| Screenshot | ✅ Unchanged |
| FPS counter | ✅ Unchanged |
| Settings dialog (all tabs) | ✅ + Recording Mode option added to General tab |

**Detection pipeline:** Not touched. ONNX model loading, inference loop, result parsing, and confidence thresholds are identical.

**UI layout:** Not touched. No existing buttons, panels, or frames were modified.

---

## Files Modified

| File | Change |
|------|--------|
| `config/settings.py` | Updated 5 tracking defaults; added `recording_mode: "overlay"` |
| `tracking/tracker.py` | Added Stage 2b; added `lost_count`/`recovered_count` per track; updated Step 4 & Step 5 |
| `ui/main_window.py` | Fixed early-return branch (writes raw frame to recorder); added `rec_mode` logic in detecting branch; updated `_make_tracker()` fallback defaults |
| `ui/settings_dialog.py` | Added Recording Mode option to General tab; wired save in `_save()` |

---

## Acceptance Criteria — Verification

| Criterion | Met |
|-----------|-----|
| Same moving object keeps the same Track ID | ✅ `max_age=30` + `iou=0.20` + Stage 2b |
| Temporary occlusion does not immediately change ID | ✅ Track survives up to 30 missed frames before expiry |
| Recording works with Detection OFF | ✅ Raw frame written every cycle in non-detecting branch |
| Recording works with Detection ON | ✅ Overlay or raw frame written depending on `recording_mode` setting |
| All previously working features remain functional | ✅ See regression table above |

---

## Known Limitations

- **Kalman filter model is constant-velocity.** Very abrupt acceleration (e.g., object suddenly thrown) will cause the predicted box to overshoot, possibly dropping below the 0.20 IoU threshold. Increasing `max_age` mitigates this — the track will recover on the next matched frame.
- **Recording FPS is fixed at 25 fps** (configurable in `settings.py → recording.fps`). If the actual camera produces frames at a different rate, playback speed will not match real time. This is pre-existing behavior.
- **No per-stream recording mode** — `recording_mode` applies globally to both RGB and Thermal streams.
