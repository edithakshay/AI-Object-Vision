# DualVision AI Detector

**High-FPS Dual RTSP AI Object Detection — Windows Desktop Software**

A professional, production-ready Windows desktop application for real-time AI object detection from two simultaneous RTSP camera streams (RGB + Thermal).  
Built with Python 3.12, CustomTkinter, OpenCV, and **YOLO26 via ONNX Runtime** for maximum CPU performance.

---

## What's New — YOLO26 Migration

| | Before | Now |
|---|---|---|
| **AI Model** | YOLOv8 / YOLO11 | **YOLO26 only** |
| **Inference Backend** | PyTorch / Ultralytics | **ONNX Runtime (primary) + PyTorch fallback** |
| **CPU Performance** | Baseline | YOLO26 + ONNX = maximum CPU FPS |
| **Post-processing** | Manual NMS | Native end-to-end (NMS-free) where supported |
| **Settings** | Single page | **3-tab dialog: General / YOLO26 / Dashboard** |
| **Performance Panel** | Basic FPS | **Full dashboard: FPS, Avg FPS, Inference ms, CPU, RAM, Threads, Queue, Drops** |

---

## Features

- **Dual RTSP Streams** — RGB and Thermal camera streams simultaneously
- **YOLO26 Only** — Latest Ultralytics model, auto-downloaded once, 100% offline after
- **ONNX Runtime** — Primary CPU inference engine (fastest); PyTorch fallback automatic
- **High FPS** — CPU-optimised: parallel inference threads, frame queue, zero-copy where possible
- **Object Tracking** — Built-in ByteTrack with persistent IDs (no extra deps)
- **Performance Dashboard** — Live FPS, Avg FPS, Inference ms, CPU%, RAM, Active Threads, Queue Size, Frame Drops
- **YOLO26 Settings Tab** — Model info, ONNX status, device info, one-click ONNX export
- **Modern Dark UI** — CustomTkinter professional dark theme
- **Zoom & Pan** — Mouse wheel zoom, click-drag pan per camera panel
- **Fullscreen** — Double-click any camera panel
- **Screenshot** — PNG capture with timestamp filename
- **Video Recording** — MP4 recording of both streams
- **Detection Log** — CSV/JSON export, on-screen scrolling log
- **Auto-reconnect** — Streams reconnect automatically on disconnect
- **Future-ready** — Architecture stubs for Segmentation, Pose, OBB, Classification, Open-Vocab

---

## Camera URLs (Default)

| Camera | RTSP URL |
|--------|----------|
| RGB | `rtsp://192.168.144.108:554/stream=1` |
| Thermal | `rtsp://192.168.144.108:555/stream=2` |

Edit in **Settings → General → RTSP Streams**, or in `config/app_config.json`.

---

## Quick Start

```bash
# 1. Open the DualVisionAI folder in VS Code
#    File → Open Folder → select DualVisionAI/

# 2. Create virtual environment
python -m venv venv

# 3. Activate (Windows)
venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Download YOLO26 model (internet required ONCE — ~6 MB)
python setup.py

# 6. Run the application
python main.py
```

After `setup.py` completes, the model is stored in `models/yolo26n.pt`.  
The app works **completely offline** from that point forward.

### Run from VS Code

Open the **Run and Debug** panel (`Ctrl+Shift+D`) and select:

| Configuration | Purpose |
|---|---|
| `▶ Run DualVision AI` | Launch the main application |
| `⚙ Setup (First-Time Model Download)` | Run setup.py to download the model |
| `⬇ Download All YOLO26 Models` | Pre-download all YOLO26 variants |
| `🖼 Generate Icons` | Regenerate app icons |

---

## Project Structure

```
DualVisionAI/
├── main.py                  ← Entry point
├── setup.py                 ← First-time setup (downloads yolo26n.pt)
├── requirements.txt         ← CPU dependencies
├── requirements_gpu.txt     ← GPU dependencies (optional)
├── download_all_models.py   ← Pre-download all YOLO26 models
├── generate_icons.py        ← Icon/logo generator
├── build.bat                ← One-click EXE builder (PyInstaller)
├── .vscode/
│   ├── launch.json          ← VS Code run configurations
│   └── settings.json        ← VS Code Python settings
├── config/
│   ├── settings.py          ← JSON config manager
│   └── app_config.json      ← Auto-created on first run
├── ai/
│   ├── detector.py          ← YOLO26 ONNX inference engine (threaded)
│   ├── model_manager.py     ← Model download + ONNX export (YOLO26 only)
│   └── tasks.py             ← Task registry (Detect active; Seg/Pose/OBB stubs)
├── camera/
│   └── stream.py            ← RTSP capture with auto-reconnect
├── tracking/
│   └── tracker.py           ← ByteTrack (pure NumPy, no extra deps)
├── ui/
│   ├── main_window.py       ← Main application window
│   ├── camera_panel.py      ← Video display with zoom/pan/fullscreen
│   ├── control_panel.py     ← Right panel: system, model, performance, log
│   ├── toolbar.py           ← Top toolbar buttons
│   ├── statusbar.py         ← Bottom status bar
│   ├── settings_dialog.py   ← 3-tab settings: General / YOLO26 / Dashboard
│   ├── splash_screen.py     ← Startup splash
│   └── about_dialog.py      ← About popup
├── utils/
│   ├── logger.py            ← Rotating file logger
│   ├── screenshot.py        ← PNG screenshot utility
│   ├── recorder.py          ← MP4 video recorder
│   └── detection_log.py     ← CSV/JSON detection logger
├── assets/                  ← Logo and splash banner (auto-generated)
├── models/                  ← YOLO26 models (auto-created by setup.py)
├── logs/                    ← App + detection logs (auto-created)
├── recordings/              ← Video recordings (auto-created)
├── screenshots/             ← Screenshots (auto-created)
└── docs/
    ├── installation_guide.md
    └── troubleshooting.md
```

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Pause / Resume |
| `Ctrl+S` | Screenshot |
| `Ctrl+R` | Start Recording |
| `Esc` | Exit |

## Mouse Controls (Camera Panels)

| Action | Effect |
|--------|--------|
| Scroll wheel | Zoom in/out |
| Click + drag | Pan |
| Double-click | Fullscreen / Exit fullscreen |

---

## YOLO26 Models

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `yolo26n.pt` | ~6 MB | Fastest | Good |
| `yolo26s.pt` | ~20 MB | Fast | Better |
| `yolo26m.pt` | ~50 MB | Medium | Great |
| `yolo26l.pt` | ~85 MB | Slower | Excellent |
| `yolo26x.pt` | ~125 MB | Slowest | Best |

Switch models in **Settings → General → YOLO26 Model**. Click **Download Model** then restart detection.

---

## Performance Tips

- Keep `yolo26n.pt` (Nano) for maximum FPS on CPU
- Enable **ONNX Runtime** in Settings (on by default) — fastest CPU path
- Increase **Frame Skip** (2–3) if CPU is saturated
- Reduce **Input Resolution** to 320 or 416 for more FPS
- Set **CPU Threads** to match your core count (0 = auto)

---

## Architecture — Future Tasks (Stubs Ready)

The codebase is structured for easy expansion. To enable a new task:

1. Set `enabled = True` in `ai/tasks.py` registry
2. Implement `run()` in the corresponding task class
3. Export the correct model variant (e.g. `yolo26n-seg.pt` for segmentation)

| Task | Model Suffix | Status |
|------|------|--------|
| Object Detection | *(none)* | **Active** |
| Instance Segmentation | `-seg` | Stub ready |
| Pose Estimation | `-pose` | Stub ready |
| Image Classification | `-cls` | Stub ready |
| Oriented Bounding Boxes | `-obb` | Stub ready |
| Open-Vocabulary (YOLOE-26) | *(special)* | Stub ready |

---

## Build Windows EXE

```bash
venv\Scripts\activate
pip install pyinstaller
build.bat
```

Output: `dist\DualVisionAI\DualVisionAI.exe`

---

## Changelog

### v1.1.0 — YOLO26 Migration
- **Removed** YOLOv8 and YOLO11 completely (models, code, imports, configs)
- **Added** YOLO26 as the sole detection engine
- **Added** ONNX Runtime as primary CPU inference backend
- **Added** 3-tab Settings dialog (General / YOLO26 / Dashboard)
- **Added** full Performance Dashboard (FPS, Avg FPS, Inference ms, CPU, RAM, Threads, Queue, Drops)
- **Added** YOLO26 Settings tab with model info, ONNX status, one-click export
- **Fixed** default model name fallback (`yolov8n.pt` → `yolo26n.pt`)
- **Fixed** Settings dialog blocking YOLO26 model downloads
- **Updated** setup.py to download `yolo26n.pt` directly
- **Updated** VS Code launch configurations with all run targets
