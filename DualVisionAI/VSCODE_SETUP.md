# DualVision AI Detector v1.3 — VS Code Setup Guide

## Requirements

| Software | Minimum Version |
|---|---|
| Python | 3.10+ (3.11 or 3.12 recommended) |
| VS Code | 1.88+ |
| VS Code Python Extension | ms-python.python |

---

## Step 1 — Open the project in VS Code

```bash
# Open only the DualVisionAI folder as the workspace root
code DualVisionAI
```

The `.vscode/` folder already contains `launch.json` and `settings.json`
with pre-configured run/debug profiles.

---

## Step 2 — Create a virtual environment

```bash
# Inside the DualVisionAI folder
python -m venv venv
```

### Activate it

| Platform | Command |
|---|---|
| Windows (CMD) | `venv\Scripts\activate.bat` |
| Windows (PowerShell) | `venv\Scripts\Activate.ps1` |
| macOS / Linux | `source venv/bin/activate` |

---

## Step 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- `customtkinter` — dark-mode UI framework
- `opencv-python` — camera capture and drawing
- `numpy` — array operations
- `onnxruntime` — CPU inference engine
- `ultralytics` — model export (YOLO → ONNX)
- `psutil` — CPU/RAM monitoring
- `Pillow` — image utilities
- `av` — video recording

> **Note:** `ultralytics` is only used once to export the `.pt` model to `.onnx`.
> After the first export, it is never needed again.

---

## Step 4 — Select Python interpreter in VS Code

1. Press `Ctrl+Shift+P` → "Python: Select Interpreter"
2. Choose the one inside your venv:
   - Windows: `.\venv\Scripts\python.exe`
   - macOS/Linux: `./venv/bin/python`

---

## Step 5 — Run the application

### Option A: F5 in VS Code
- Press `F5` to launch the **"▶ Run DualVision AI"** configuration.
- The integrated terminal will appear and the splash screen will open.

### Option B: Command line
```bash
# From the DualVisionAI folder with venv active
python main.py
```

### Option C: Additional tools
- **"⚙ Setup"** — downloads and exports the YOLO26n model (run this first if no internet later).
- **"⬇ Download All Models"** — pre-downloads all YOLO26 variants.
- **"🖼 Generate Icons"** — regenerates app icons.

---

## Step 6 — First launch checklist

On the very first launch the application will:

1. Show the splash screen
2. Load your saved configuration (or use defaults)
3. Start the RGB RTSP stream (configures as offline if no URL set)
4. Write startup logs to `logs/`
5. Run the UI self-test and write `logs/ui_check.log`

When you click **Start**:
- If `models/yolo26n.pt` is missing → downloads from Ultralytics (~6 MB, internet required once)
- If `models/yolo26n.onnx` is missing → exports from the `.pt` file automatically
- Detection begins on the active camera stream

---

## Configuration

Click **Settings** (⚙) to configure:

| Setting | Description |
|---|---|
| RGB URL | RTSP stream URL for the RGB camera (e.g. `rtsp://192.168.1.100/stream1`) |
| Thermal URL | RTSP stream URL for the thermal camera |
| Confidence | Detection confidence threshold (default 0.45) |
| IoU | NMS IoU threshold (default 0.45) |
| Frame Skip | Process every Nth frame (1 = every frame) |
| Input Width | ONNX model input size (default 640) |
| Enable Tracking | Toggle ByteTrack object tracking |
| CPU Threads | Number of ORT intra-op threads (0 = auto) |

Settings are saved to `config/settings.json` automatically.

---

## Logs

| File | Contents |
|---|---|
| `logs/startup.log` | App init, model loading, ONNX export |
| `logs/inference.log` | Per-inference timings, detection counts |
| `logs/camera.log` | RTSP stream events |
| `logs/fps_debug.log` | Per-second FPS diagnostics |
| `logs/debug.log` | Everything (verbose) |
| `logs/ui_check.log` | Startup UI self-test results |

---

## Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| Space | Pause / Resume detection |
| Ctrl+S | Screenshot |
| Ctrl+R | Start recording |
| Escape | Exit (with confirmation) |

---

## Project Structure

```
DualVisionAI/
├── main.py                     ← Entry point
├── requirements.txt            ← Python dependencies
├── setup.py                    ← First-time model download/export
├── UI_COMPONENT_LIST.md        ← Complete UI inventory
├── UI_VERIFICATION_CHECKLIST.md← Manual verification guide
├── VSCODE_SETUP.md             ← This file
│
├── ai/
│   ├── detector.py             ← ONNX Runtime inference engine
│   ├── backend_manager.py      ← ORT session configuration
│   └── model_manager.py        ← .pt → .onnx export
│
├── camera/
│   └── stream.py               ← RTSP stream with auto-reconnect
│
├── config/
│   └── settings.py             ← JSON-backed configuration
│
├── tracking/
│   └── tracker.py              ← Enhanced ByteTrack + Kalman filter
│
├── ui/
│   ├── main_window.py          ← Main application window
│   ├── toolbar.py              ← Top toolbar
│   ├── camera_panel.py         ← Video feed display
│   ├── control_panel.py        ← Right dashboard (scrollable)
│   ├── statusbar.py            ← Bottom status bar
│   ├── settings_dialog.py      ← Settings modal
│   ├── about_dialog.py         ← About modal
│   └── splash_screen.py        ← Startup splash
│
├── utils/
│   ├── app_logger.py           ← Logging setup
│   ├── recorder.py             ← Video recording
│   ├── screenshot.py           ← Screenshot capture
│   ├── detection_log.py        ← CSV/JSON export log
│   └── ui_self_test.py         ← Startup UI self-test
│
├── models/                     ← Auto-created on first Start
│   ├── yolo26n.pt
│   └── yolo26n.onnx
│
├── logs/                       ← Auto-created on startup
├── recordings/                 ← Auto-created on first recording
├── screenshots/                ← Auto-created on first screenshot
│
└── .vscode/
    ├── launch.json             ← Run/Debug profiles
    └── settings.json           ← Editor settings
```

---

## Troubleshooting

### "Missing dependencies" error on startup
```bash
pip install -r requirements.txt
```

### Camera shows black / "Disconnected"
- The app works without a live camera. Configure the RTSP URL in Settings.
- For local webcam testing, use `rtsp://` or modify stream.py to use `cv2.VideoCapture(0)`.

### ONNX export fails
- Ensure `ultralytics` is installed: `pip install ultralytics`
- Run `setup.py` manually for detailed output

### UI appears on wrong monitor or off-screen
- Delete `config/settings.json` and restart to reset window position

### Detection starts but no boxes appear
- Check confidence threshold in Settings (try lowering to 0.30)
- Check `logs/inference.log` for output shape diagnostics
- Check `logs/debug.log` for full error trace
