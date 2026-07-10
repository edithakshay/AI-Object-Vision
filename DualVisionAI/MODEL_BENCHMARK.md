# DualVision AI — Model Benchmark Report

> **This file is a template.**  
> Run `python tools/benchmark.py` from the `DualVisionAI/` directory to generate  
> a real benchmark on your hardware.  The script tests every available ONNX model  
> and overwrites this file with measured results.

```
python tools/benchmark.py                         # synthetic frames
python tools/benchmark.py --video clip.mp4        # real video
python tools/benchmark.py --video clip.mp4 --frames 500
```

---

## Expected Performance Ranges (CPU-only, Intel i7 / Ryzen 7)

| Model | Approx FPS | Avg Inference | Detection Stability | Use Case |
|-------|-----------|---------------|---------------------|----------|
| **YOLO26n** (Nano)   | 15–35 fps | 30–70 ms  | Good   | Real-time, limited CPU |
| **YOLO26s** (Small)  | 8–18 fps  | 55–130 ms | Better | Balanced accuracy/speed |
| **YOLO26m** (Medium) | 4–10 fps  | 100–250 ms| Great  | High accuracy required |
| **YOLO26l** (Large)  | 2–6 fps   | 170–500 ms| Excellent | Best CPU accuracy |
| **YOLO26x** (XLarge) | 1–3 fps   | 330–1000 ms| Best  | Research / offline |

*Actual performance depends on CPU cores, clock speed, and input resolution.*

---

## How to Run the Benchmark

### Prerequisites
- At least one YOLO26 `.onnx` model in `models/`
- Models can be exported via **Settings → ONNX/CPU → Export to ONNX**

### Basic Run (synthetic frames)
```bash
cd DualVisionAI
python tools/benchmark.py
```

### With a Real Video
```bash
python tools/benchmark.py --video recordings/my_clip.mp4 --frames 300
```

### Options
| Flag | Default | Description |
|------|---------|-------------|
| `--video` | None | Path to video file (synthetic if omitted) |
| `--frames` | 200 | Frames per model |
| `--size` | 640 | ONNX input resolution |
| `--conf` | 0.45 | Confidence threshold |

---

## Phase 2 Recommendation (without running benchmark)

For **Search & Rescue, CPU-only, long-duration operation**:

| Priority | Recommended Setting |
|----------|---------------------|
| **Model** | YOLO26s (better stability than Nano, still usable FPS) |
| **Confidence Threshold** | 0.35–0.40 (lower = fewer missed detections) |
| **Confidence Smoother** | ON, EMA α = 0.35, Ghost Frames = 3 |
| **Persistence Frames** | 5–8 (keep ghost track for up to ~0.5 s) |
| **Track Timeout** | 15–30 frames (don't kill tracks too quickly) |
| **Association IoU** | 0.20 (allows re-ID after brief occlusion) |

---

*Run `python tools/benchmark.py` to populate this file with your hardware's real numbers.*
