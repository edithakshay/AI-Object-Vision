# DualVision AI Detector

**High-FPS Dual RTSP AI Object Detection — Windows Desktop Software**

A professional, production-ready Windows desktop application for real-time AI object detection from two simultaneous RTSP camera streams (RGB + Thermal). Built with Python 3.12, CustomTkinter, OpenCV, and Ultralytics YOLO.

---

## Features

- **Dual RTSP Streams** — RGB and Thermal camera streams simultaneously
- **High FPS** — 30–60 FPS target (GPU); maximum possible on CPU
- **AI Object Detection** — YOLOv8n (fastest model, auto-downloaded on first run)
- **Object Tracking** — Built-in ByteTrack with persistent IDs
- **Modern Dark UI** — CustomTkinter with professional dark theme
- **Zoom & Pan** — Mouse wheel zoom, click-drag pan on each camera view
- **Fullscreen** — Double-click any camera panel for fullscreen view
- **Screenshot** — PNG capture with timestamp filename
- **Video Recording** — MP4 recording of both streams
- **Detection Log** — CSV/JSON/TXT export of all detections
- **Settings** — Persistent JSON config, adjustable confidence/IOU/FPS/resolution
- **Keyboard Shortcuts** — Space (pause), Ctrl+S (screenshot), Ctrl+R (record), Esc (exit)
- **Auto-reconnect** — Streams reconnect automatically if disconnected
- **Offline after first run** — Model cached locally, no internet required afterwards

---

## Camera URLs

| Camera | RTSP URL |
|--------|----------|
| RGB | `rtsp://192.168.144.108:554/stream=1` |
| Thermal | `rtsp://192.168.144.108:554/stream=2` |

Edit these in **Settings** or directly in `config/app_config.json`.

---

## Quick Start

```bash
# 1. Clone / unzip the project folder
# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate icons (first time only)
python generate_icons.py

# 5. Run the application
python main.py
```

The first run will automatically download the YOLO model (~6 MB for yolov8n). After that, the app works completely offline.

---

## Project Structure

```
DualVisionAI/
├── main.py                  ← Entry point
├── requirements.txt         ← CPU dependencies
├── requirements_gpu.txt     ← GPU dependencies
├── generate_icons.py        ← Icon/logo generator
├── build.bat                ← One-click EXE builder
├── DualVisionAI.spec        ← PyInstaller spec
├── version_info.txt         ← Windows version metadata
├── config/
│   ├── settings.py          ← JSON config manager
│   └── app_config.json      ← Auto-created on first run
├── ui/
│   ├── main_window.py       ← Main application window
│   ├── camera_panel.py      ← Video display with zoom/pan
│   ├── control_panel.py     ← Right panel: stats + log
│   ├── toolbar.py           ← Top toolbar buttons
│   ├── statusbar.py         ← Bottom status bar
│   ├── settings_dialog.py   ← Settings popup
│   ├── splash_screen.py     ← Startup splash
│   └── about_dialog.py      ← About popup
├── ai/
│   ├── detector.py          ← YOLO inference engine (threaded)
│   └── model_manager.py     ← Model download + ONNX export
├── camera/
│   └── stream.py            ← RTSP capture with auto-reconnect
├── tracking/
│   └── tracker.py           ← ByteTrack (pure NumPy, no deps)
├── utils/
│   ├── logger.py            ← Rotating file logger
│   ├── screenshot.py        ← PNG screenshot utility
│   ├── recorder.py          ← MP4 video recorder
│   └── detection_log.py     ← CSV/JSON/TXT detection logger
├── icons/                   ← App icons (generated)
├── assets/                  ← Logo and splash banner (generated)
├── models/                  ← Downloaded YOLO models (auto-created)
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

## Build Windows EXE

```bash
# Activate venv first
venv\Scripts\activate

# Run the build script
build.bat

# OR manually:
pyinstaller DualVisionAI.spec
```

Output: `dist\DualVisionAI\DualVisionAI.exe`

See `docs/installation_guide.md` for full instructions.
