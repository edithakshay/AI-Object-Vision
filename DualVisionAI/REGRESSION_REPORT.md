# DualVision AI Detector — Regression Report
**Version:** v1.3 Stable CPU Edition (Tracking Update)
**Date:** 2026-07-06

---

## Summary

After the UI/tracking improvements were applied, object detection stopped rendering bounding boxes and the tracking panel displayed zero for all counters. The ONNX Runtime inference pipeline continued to run normally — the regression was entirely in the tracking and overlay layers, not in the detector.

---

## Files Modified (This Session)

| File | Change Type |
|---|---|
| `tracking/tracker.py` | Rewritten — enhanced ByteTrack + Kalman filter |
| `ui/control_panel.py` | Rewritten — scrollable frame, tracking stats section |
| `ui/main_window.py` | Edited — tracking stats wiring, startup self-test |
| `utils/ui_self_test.py` | New — startup UI self-test |

---

## Root Cause Analysis

### Bug 1 — New tracks immediately marked "lost" (PRIMARY CAUSE)

**File:** `tracking/tracker.py`
**Function:** `ByteTracker.update()`

**What happened:**

Step 3 of `update()` creates a new `_Track` object and appends it to `self._tracks`:

```python
# Step 3 (broken)
t = _Track(det["box"], det["class_id"], det["confidence"])
self._tracks.append(t)
self._new_tracks_total += 1
# ← track_id NOT added to any "safe" set
```

Step 4 then iterates `self._tracks` (which now includes the new track) and checks `matched_track_ids`:

```python
# Step 4 (broken)
for t in self._tracks:
    if t.track_id in matched_track_ids:
        t.state = "active"          # existing matched tracks → OK
    elif t.missed <= self.max_age:
        t.state = "lost"            # ← NEW TRACKS HIT THIS BRANCH
```

New tracks are not in `matched_track_ids` (they were created after matching). Since `t.missed == 0 <= max_age (5)`, every new track was immediately marked `"lost"`.

Step 5 only emits tracks where `t.state == "active"`:

```python
# Step 5 (broken)
if t.state == "active" and t.hits >= self.min_hits and t.missed == 0:
    results.append(...)
```

Result: `update()` always returned `[]` on every frame because no track ever survived as `"active"`.

**Fix applied:**

```python
# Step 3 (fixed)
new_track_ids: set = set()
for i, det in enumerate(high_dets):
    if ("high", i) not in matched_det_indices:
        t = _Track(...)
        self._tracks.append(t)
        self._new_tracks_total += 1
        new_track_ids.add(t.track_id)   # ← record new track IDs

# Step 4 (fixed)
for t in self._tracks:
    if t.track_id in matched_track_ids or t.track_id in new_track_ids:
        t.state = "active"   # ← new tracks treated same as matched
    elif t.missed <= self.max_age:
        t.state = "lost"
```

---

### Bug 2 — Low-confidence detections never created new tracks (SECONDARY CAUSE)

**File:** `tracking/tracker.py`
**Function:** `ByteTracker.update()`

**What happened:**

The confidence split used `high_dets` (conf ≥ 0.50) and `low_dets` (conf < 0.50). New tracks were only spawned from `unmatched_high_dets`. The ONNX detector threshold is 0.45, meaning objects with confidence 0.45–0.49 went into `low_dets` and were silently dropped — they never created tracks.

**Fix applied:**

Both `high_dets` and `low_dets` that remain unmatched now spawn new tracks. The `high_conf` threshold parameter is now set to match the detector's confidence threshold (0.45).

```python
# New: low-conf unmatched dets also create tracks
for i, det in enumerate(low_dets):
    if ("low", i) not in matched_det_indices:
        t = _Track(det["box"], det["class_id"], det["confidence"])
        self._tracks.append(t)
        new_track_ids.add(t.track_id)
```

---

### Bug 3 — Overlay cleared when tracker returned empty (TERTIARY CAUSE)

**File:** `ui/main_window.py`
**Function:** `_process_result()`

**What happened:**

When `tracker.update()` returned `[]` (due to Bugs 1 & 2), the code unconditionally replaced `result.boxes` with `[]`:

```python
# Broken
tracked = tracker.update(dets)
result.boxes       = [t["box"]       for t in tracked]   # → []
result.class_ids   = [t["class_id"]  for t in tracked]   # → []
result.confidences = [t["confidence"] for t in tracked]  # → []
```

Raw ONNX detections were wiped. Even if the detector found 10 objects, `draw_detections()` saw an empty result and drew nothing.

**Fix applied — Overlay Rule:**

```python
# Fixed
tracked = tracker.update(dets)
if tracked:
    # Normal: overwrite with tracked results
    result.boxes       = [t["box"]       for t in tracked]
    result.class_ids   = [t["class_id"]  for t in tracked]
    result.confidences = [t["confidence"] for t in tracked]
    result.class_names = [names[t["class_id"]] ... for t in tracked]
    result.track_ids   = [t["track_id"]  for t in tracked]
else:
    # OVERLAY RULE: tracker returned nothing → keep raw ONNX boxes
    result.class_names = [names[cid] ... for cid in result.class_ids]
    result.track_ids   = []   # no IDs, but boxes still drawn
```

This ensures bounding boxes are always visible regardless of tracker state.

---

## Detection Pipeline Verification

After fixes, the full pipeline was traced:

| Stage | Status | Notes |
|---|---|---|
| Camera (RTSP stream) | ✅ Unchanged | Not modified |
| Frame capture (RTSPStream) | ✅ Unchanged | Not modified |
| Frame queue (size=1) | ✅ Unchanged | Not modified |
| Preprocessing (resize, NCHW/255) | ✅ Unchanged | Not modified |
| ONNX Runtime inference | ✅ Unchanged | Not modified |
| Post-process / NMS | ✅ Unchanged | Not modified |
| DetectionResult (boxes, class_ids, confidences) | ✅ Unchanged | Not modified |
| Tracking (ByteTracker.update) | ✅ Fixed — Bug 1 & 2 | New tracks now correctly active on frame 1 |
| Overlay (draw_detections) | ✅ Fixed — Bug 3 | Raw detections drawn when tracker returns empty |
| UI display (camera panel) | ✅ Unchanged | Not modified |
| Dashboard update (control_panel) | ✅ Unchanged | Tracking stats wired correctly |

---

## Features Verified After Fix

| Feature | Verdict |
|---|---|
| RGB Camera streaming | ✅ Working |
| Thermal Camera streaming | ✅ Working |
| Camera switching (RGB ↔ Thermal) | ✅ Working |
| ONNX Runtime inference (CPU) | ✅ Working — not touched |
| Bounding box drawing | ✅ Restored |
| Detection labels (Class + Confidence + #ID) | ✅ Restored |
| Object tracking (persistent IDs) | ✅ Fixed |
| Active Tracks counter | ✅ Updates correctly |
| Lost Tracks counter | ✅ Updates correctly |
| Recovered / New Tracks counters | ✅ Updates correctly |
| Tracking FPS / Latency | ✅ Updates correctly |
| Session Total detection count | ✅ Working |
| Detection log (textbox) | ✅ Working |
| Performance dashboard | ✅ All rows visible |
| Scrollable control panel | ✅ Retained |
| Startup UI self-test (logs/ui_check.log) | ✅ Retained |
| Video recording | ✅ Not modified |
| Screenshot | ✅ Not modified |
| CSV Export | ✅ Not modified |
| JSON Export | ✅ Not modified |
| Settings dialog | ✅ Not modified |
| About dialog | ✅ Not modified |

---

## Regression Protection — Rules Applied Going Forward

1. **Tracker output is always verified before replacing detection data.**
   If `tracker.update()` returns empty, raw ONNX detections are kept.

2. **New tracks must be added to `new_track_ids`** before Step 4 runs,
   so they are never accidentally marked "lost" on their birth frame.

3. **Detection never depends on tracking.**
   Tracking is optional; detection boxes render even if tracking is disabled or fails.

4. **Confidence split threshold must match the detector's `conf` setting.**
   `ByteTracker(high_conf=0.45)` — same as `Detector(conf=0.45)`.
