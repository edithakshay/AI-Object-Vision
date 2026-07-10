# DualVision AI — Tracking Optimization Report
## Phase 2: Detection Stability & Tracking Reliability

**Version:** 1.3 Stable CPU Edition (Phase 2)  
**Date:** 2026-07-10  
**Scope:** ByteTrack optimization, confidence smoothing, detection persistence, multi-model support

---

## 1. Problem Statement

The application was stable and functional. However, stationary objects occasionally
disappeared for 1–3 frames, causing track ID changes on re-detection. This was
observed most frequently with:

- **Stationary chairs** — disappear 1–2 frames, re-detected with new track ID
- **Persons** — brief disappearance during slight camera movement
- **Bottles** — small objects with confidence near the detection threshold

Root cause analysis identified three contributing factors:

| Factor | Severity | Notes |
|--------|----------|-------|
| YOLO26n raw confidence fluctuation | High | Small objects dip below threshold for 1–3 frames |
| ByteTrack default parameters (too aggressive) | Medium | Max age too low, track removed before re-ID possible |
| No temporal smoothing on raw detections | High | Each frame decided independently; no memory of previous frames |

---

## 2. Implemented Solutions

### 2.1 Confidence Smoother (`ai/confidence_smoother.py`)

**Problem:** Raw ONNX confidence fluctuates frame-to-frame (e.g. 0.74 → 0.72 → 0.39 → 0.71).
A single low-confidence frame removes the detection entirely.

**Solution:** An IoU-based temporal smoother that sits **upstream** of the ByteTracker:

- **EMA smoothing:** `smooth_conf = α × raw + (1−α) × prev`  (α = 0.35 default)
- **Ghost synthesis:** when a detection disappears, the smoother keeps a "ghost" 
  detection for up to N frames (default 3) with decaying confidence
- **Box area filter:** optional minimum/maximum box area to reject noise and full-frame FPs

**Effect:** The tracker receives a stable confidence stream instead of raw ONNX output.
A 1-frame confidence dip no longer causes a track to be destroyed.

**Configuration (Settings → Optimization tab):**
| Parameter | Default | Effect |
|-----------|---------|--------|
| Enable Smoother | ON | Toggle on/off |
| EMA Alpha | 0.35 | Lower = smoother but slower to react |
| Smoother IoU Threshold | 0.40 | Minimum IoU to match detection to history |
| Max Ghost Frames | 3 | Frames to keep detection alive after disappearance |
| Ghost Decay | 0.70 | Confidence × decay per missed frame |
| Min Ghost Confidence | 0.25 | Drop ghost when confidence falls below this |
| Min Box Area | 0 | Filter tiny noise boxes (px²) |
| Max Box Area | 0 | Filter full-frame false positives (px²) |

---

### 2.2 ByteTrack Optimization (`tracking/tracker.py`)

The ByteTracker was enhanced with:

#### Kalman Filter (numerically stable)
- Float64 throughout, Joseph form covariance update, `np.linalg.solve` (not `inv`)
- NaN/Inf health guard with automatic re-initiation
- Provides **motion prediction** — when a track is lost, the Kalman filter predicts
  its next position, so re-identification works across larger positional gaps

#### Detection Persistence (Ghost Tracks)
Lost tracks are kept alive for up to `persistence_frames` (default 5) frames.
During that period, the last known Kalman-predicted box is shown as a ghost
(dashed border, dimmed colour). If the detection returns, the same track ID is
kept — no ID change.

#### Two-stage matching (ByteTrack style)
1. Stage 1a: high-confidence detections vs active/recent tracks (IoU ≥ 0.20)
2. Stage 1b: low-confidence detections vs unmatched active tracks (IoU ≥ 0.10)
3. Stage 2a: unmatched high-conf dets vs lost tracks
4. Stage 2b: unmatched low-conf dets vs still-lost tracks

This four-stage approach means a returning object detected at low confidence can
still recover its original track ID (stages 2a/2b), instead of spawning a new track.

#### Empty-frame handling (critical fix)
On frames with no detections: `tracker.update([])` — NOT `tracker.reset()`.  
This naturally ages tracks without destroying their history, allowing re-ID
after brief disappearance.

**Tunable parameters (Settings → Tracking tab):**
| Parameter | Default | Effect |
|-----------|---------|--------|
| Track Timeout | 30 frames | Frames before a lost track is permanently removed |
| Min Confirmation Hits | 3 | Hits required before a track is shown |
| Association IoU | 0.20 | Minimum IoU for stage 1 matching |
| Confidence Split | 0.40 | Threshold separating high/low confidence detections |
| Persistence Frames | 5 | Ghost frames shown after track loss |
| Trail Length | 30 | Centre-point history length for trail lines |

---

### 2.3 Multi-Model Support (`ai/model_manager.py`)

All YOLO26 variants are supported with export-once policy:

| Variant | Approx FPS (CPU) | Approx Size | Use Case |
|---------|-----------------|-------------|----------|
| YOLO26n | 15–35 fps | 6 MB ONNX | Real-time on limited CPU |
| YOLO26s | 8–18 fps | 22 MB ONNX | Better stability, usable FPS |
| YOLO26m | 4–10 fps | 52 MB ONNX | High accuracy |
| YOLO26l | 2–6 fps  | 87 MB ONNX | Very high accuracy |
| YOLO26x | 1–3 fps  | 136 MB ONNX | Best accuracy (offline use) |

**Export-once policy:** if `models/<variant>.onnx` exists → load directly.
Otherwise export from `.pt` once and reuse permanently.

---

### 2.4 Debug Dashboard (`ui/debug_dashboard.py`)

A floating window (opened via the **🔍 Debug** toolbar button) showing:

**Stability tab:**
- Detection Continuity % (frames detected / total frames)
- Frames Detected / Missing / Drops
- Track Stability % (active / active+lost)
- ID Changes, Lost Frames, Recovered Frames
- Ghost Detections Active, Avg Smooth Confidence, Avg Detection Lifetime

**Confidence tab:**
- Current frame average confidence
- Rolling average / min / max (last 50 frames)
- Confidence drop events
- Smoother active entries / ghost entries / avg smoothed confidence
- Confidence trend (last 20 values, text display)

**Per-Object tab:**
- One row per confirmed track: ID, class, age, hits, lost count, recovered, confidence, direction

**System tab:**
- Inference FPS, Avg FPS, per-phase ms (preprocess / inference / postprocess)
- CPU%, RAM, frame drops, model name, input size, class count

---

### 2.5 Benchmark and Comparison Tools

**`tools/benchmark.py` (PART 2):**  
Tests each available ONNX model on the same frames. Measures FPS, inference time,
CPU/RAM usage, detection stability, and drop count. Generates `MODEL_BENCHMARK.md`.

```bash
python tools/benchmark.py                      # synthetic test
python tools/benchmark.py --video clip.mp4    # real video
```

**`tools/compare_models.py` (PART 11):**  
Side-by-side comparison of all available models on the same input.
Outputs annotated frames to `debug/comparison_frames/` for visual inspection.

```bash
python tools/compare_models.py
python tools/compare_models.py --video clip.mp4 --frames 100
```

---

## 3. Regression Protection

The following subsystems were **not modified** (PART 12):

| Subsystem | File | Status |
|-----------|------|--------|
| Camera module (RTSP, reconnect, single-active) | `camera/stream.py` | ✓ Unchanged |
| ONNX Runtime inference pipeline | `ai/detector.py` | ✓ Unchanged |
| Video recording | `utils/recorder.py` | ✓ Unchanged |
| Screenshot | `utils/screenshot.py` | ✓ Unchanged |
| Performance dashboard (Settings tab 3) | `ui/settings_dialog.py` | ✓ Unchanged |
| CSV / JSON export | `utils/detection_log.py` | ✓ Unchanged |
| Detection overlay drawing | `ai/detector.py` → `draw_detections()` | ✓ Unchanged |
| FPS counters and freeze-on-stop | `ui/main_window.py` | ✓ Unchanged |

---

## 4. Recommended Production Settings

For long-duration Search & Rescue operation (CPU-only):

```
Model:                   YOLO26s (better stability, ~8–18 FPS)
Confidence Threshold:    0.35–0.40   (lower = fewer missed detections)
IoU Threshold:           0.45
Enable Tracking:         ON
Confidence Smoother:     ON
  EMA Alpha:             0.30        (smoother = more temporal memory)
  Max Ghost Frames:      4           (4-frame forgiveness)
  Ghost Decay:           0.75
  Min Ghost Confidence:  0.20
Track Timeout:           20–30 frames
Min Confirmation Hits:   2
Association IoU:         0.20
Persistence Frames:      8           (~0.5 s at 15 FPS)
Trail Lines:             ON, length 40
Input Resolution:        640×640     (do not reduce below 416 for small objects)
Frame Skip:              1           (never skip for SAR — stability priority)
```

---

## 5. Acceptance Test Protocol (PART 13)

Run from `DualVisionAI/` with the application running:

### Test 1 — Stationary Chair (5 minutes)
- Point camera at single stationary chair
- Start detection with recommended production settings above
- Expected: Track ID stable for the full 5 minutes (0 ID changes)
- Accept: ≤2 ID changes per 5-minute session

### Test 2 — Walking Person
- Person walks across camera field of view at normal pace
- Expected: Single track ID follows the person across the frame
- Accept: ≤1 ID change during a single crossing

### Test 3 — Moving Bottle
- Slowly push a bottle across a table
- Expected: Track follows; no ID change during continuous motion
- Accept: ≤2 ID changes per 60 seconds

### Test 4 — Two Chairs, Two Persons
- Two people and two chairs in frame simultaneously
- Expected: 4 stable track IDs, no confusion between objects
- Accept: ≤1 ID swap per 60-second observation

### Test 5 — Camera Shake
- Briefly tap/nudge camera; let it settle
- Expected: Tracks recovered within 2 seconds of settling
- Accept: All tracks recovered within persistence window

### Test 6 — Temporary Occlusion
- Person walks behind an object and re-emerges within 2 seconds
- Expected: Same track ID after re-emergence
- Accept: Track ID preserved when re-emergence is within `persistence_frames`

---

## 6. Remaining Limitations

| Limitation | Notes |
|------------|-------|
| GPU/CUDA not supported | Phase 3 (planned) — would increase FPS 5–10× |
| Single camera active at a time | Intentional design (CPU resource limit) |
| YOLO26n low accuracy on small/distant objects | Use YOLO26s or lower confidence threshold |
| Occlusion >persistence window destroys track | Increase persistence to 10–15 for slow cameras |
| Thermal camera detection quality | Depends on camera calibration, not model |
| No cross-frame class-change re-ID | If class flips (person↔bicycle), new track ID issued |

---

## 7. Files Changed (Phase 2)

| File | Change Type | Summary |
|------|-------------|---------|
| `ai/confidence_smoother.py` | New | EMA smoother + ghost synthesis upstream of tracker |
| `tracking/tracker.py` | Enhanced | Kalman, trails, events, ghost tracks, 4-stage matching |
| `ai/model_manager.py` | Enhanced | All YOLO26 variants, export-once policy |
| `ui/debug_dashboard.py` | New | Real-time detection/tracking debug window |
| `ui/main_window.py` | Enhanced | Smoother wired in, multi-model, debug button, dashboard feed |
| `ui/settings_dialog.py` | Enhanced | Optimization tab (smoother, model selection, persistence) |
| `ui/control_panel.py` | Enhanced | Tracking stats section |
| `tools/benchmark.py` | New | PART 2 — multi-model benchmark + MODEL_BENCHMARK.md |
| `tools/compare_models.py` | New | PART 11 — side-by-side comparison tool |
| `MODEL_BENCHMARK.md` | New | Benchmark report (run tools/benchmark.py to populate) |
| `TRACKING_OPTIMIZATION_REPORT.md` | New | This report |

---

*DualVision AI Detector v1.3 — Phase 2 Optimization Report*  
*Generated: 2026-07-10*
