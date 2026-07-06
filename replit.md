# DualVision AI Detector — v1.3 Stable CPU Edition

## Project overview
Windows desktop application for real-time AI object detection from dual RTSP camera streams (RGB + Thermal). Built with Python 3.12, CustomTkinter, OpenCV, and YOLO26 via ONNX Runtime.

This project lives entirely in the `DualVisionAI/` subdirectory. It is designed to be **opened in VS Code on Windows** and run manually — it is a desktop GUI app (Tkinter) and cannot display visually in Replit's browser preview.

## How to run (in VS Code on Windows)
1. Open the `DualVisionAI/` folder in VS Code
2. Create a virtual environment: `python -m venv venv`
3. Activate: `venv\Scripts\activate`
4. Install deps: `pip install -r requirements.txt`
5. First-time setup (downloads YOLO26n model ~6 MB): `python setup.py`
6. Run: `python main.py`

VS Code launch configurations are in `.vscode/launch.json`.

## Architecture
- `main.py` — entry point, splash screen
- `ai/detector.py` — ONNX Runtime CPU inference engine, threaded
- `ai/backend_manager.py` — CPU backend config, ORT thread tuning
- `ai/model_manager.py` — YOLO26n .pt download + .onnx export
- `ai/tasks.py` — task registry (Detection active; Seg/Pose/OBB stubs)
- `camera/stream.py` — RTSP capture with auto-reconnect
- `tracking/tracker.py` — lightweight IoU ByteTracker
- `ui/main_window.py` — main window, single-camera mode logic
- `ui/control_panel.py` — right panel: camera selector, backend info, stats
- `ui/settings_dialog.py` — 3-tab settings (General / ONNX-CPU / Dashboard)
- `config/settings.py` — JSON config manager
- `tools/diagnose_model.py` — ONNX vs PyTorch diagnostic tool

## Key design rules
- **Single Active Camera** — only one RTSP stream runs at a time; inactive stream = 0 CPU/RAM
- **ONNX Runtime CPUExecutionProvider only** — no GPU, no CUDA, no PyTorch at inference time
- **Fixed model**: YOLO26n → yolo26n.onnx
- **No silent failures** — all exceptions logged with full traceback and shown in UI popups

## User preferences
- Deliver a ready-to-run folder for VS Code (Windows); do not attempt to run it on Replit server
- Continue development from the v1.3 Stable CPU Edition as the master branch
- v1.4 will add GPU/CUDA — do not add GPU code to this branch
