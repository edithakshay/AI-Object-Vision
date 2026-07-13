# DualVision AI Detector v1.3 — VS Code Setup Guide
## Phase 3: Search & Rescue Mission Core

---

## Requirements

| Software | Minimum Version |
|---|---|
| Python | 3.10+ (3.11 or 3.12 recommended) |
| VS Code | 1.88+ |
| VS Code Python Extension | ms-python.python |

---

## Step 1 — Get the project

Download `DualVisionAI_Phase3.zip` and extract it.  You should have a folder called `DualVisionAI/`.

Open it in VS Code:
```bash
code DualVisionAI
```

The `.vscode/` folder already contains `launch.json` and `settings.json` with pre-configured run/debug profiles.

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

### Option B: Command line
```bash
# From the DualVisionAI folder with venv active
python main.py
```

---

## Step 6 — First launch checklist

On the very first launch the application will:
1. Show the splash screen
2. Load your saved configuration (or use defaults)
3. Start the RGB RTSP stream
4. Write startup logs to `logs/`
5. Run the UI self-test and write `logs/ui_check.log`

When you click **Start**:
- If `models/yolo26n.pt` is missing → downloads from Ultralytics (~6 MB)
- If `models/yolo26n.onnx` is missing → exports automatically
- Detection begins

---

## Toolbar (Phase 3)

```
▶ Start | ■ Stop | ⏸ Pause | 📷 Shot | ⏺ Record | ⚙ Settings | 🎯 Mission | 🗂 Models | 🔍 Debug | CSV | JSON | About | ✕ Exit
```

| Button | Function |
|---|---|
| ▶ Start | Begin detection |
| ■ Stop | Stop detection |
| ⏸ Pause | Pause / Resume |
| 📷 Shot | Screenshot |
| ⏺ Record | Start/stop recording |
| ⚙ Settings | Open Settings window |
| 🎯 Mission | Open Mission Manager |
| 🗂 Models | Model Manager |
| 🔍 Debug | Debug Dashboard |
| CSV / JSON | Export detection log |
| About | App info |
| ✕ Exit | Exit with confirmation |

---

## Mission Manager (Phase 3)

Click **🎯 Mission** to open the Mission Manager.  It has 7 tabs:

### Setup
Fill in:
- **Mission Name** (required)
- Operator Name, Drone Name, Search Area
- **Mission Type**: Search & Rescue / Disaster Assessment / Fire Monitoring / Vehicle Search / Wildlife Monitoring

Click **▶ Start Mission** — a folder is auto-created:
```
missions/
└── Mission_2026-07-13_MyMission/
    ├── recordings/
    ├── screenshots/
    ├── evidence/
    ├── detections.csv
    ├── detections.json
    ├── mission.json
    ├── logs/
    └── report/
```

### Evidence
Auto-captured screenshots of every new detection.  Each item shows:
- Priority (🔴 High / 🟡 Medium / 🟢 Low)
- Class name, confidence, track ID, camera, timestamp
- **View** button to see the image

### Review
Operator manually reviews detections:
- **Verify** — mark as confirmed
- **Note** — add a text annotation
- **Delete** — remove from list
- **View** — zoom into the image

### Timeline
Automatic chronological log of all mission events:
- Mission started / paused / finished
- Detections with priority colour-coding
- Alerts in red

### Statistics
Live dashboard:
- Mission Time (HH:MM:SS)
- Total Detections, Persons, Vehicles, Animals, Fire/Smoke
- Screenshots Saved, Avg Confidence, Detection Rate/min

### Filters
Toggle which object classes appear in Evidence:
- Person, Vehicle, Animal, Fire, Smoke, Boat, Backpack, Chair, Bottle, …

### History
Database of all past missions.  Click **Open Folder** to browse any mission's saved files.

---

## Detection Priority

| Priority | Classes |
|---|---|
| 🔴 High | Person, Fire, Smoke, Boat |
| 🟡 Medium | Car, Truck, Bus, Bicycle, Animal, Backpack |
| 🟢 Low | Chair, Bottle, TV, Cup, etc. |

High-priority detections trigger:
- Alert popup (auto-closes in 4 s)
- Sound beep (Windows only)
- Red timeline entry

---

## Evidence Capture Settings (in Setup tab)

| Setting | Description |
|---|---|
| Screenshot every new track | Capture once per unique track ID |
| Screenshot High Priority only | Only capture 🔴 High detections |
| Min Confidence | Skip detections below this threshold |

---

## Configuration

Click **⚙ Settings** to configure RTSP, detection, tracking, ONNX options.

Settings are saved to `config/settings.json` automatically.

---

## Logs

| File | Contents |
|---|---|
| `logs/startup.log` | App init, model loading |
| `logs/inference.log` | Per-inference timings |
| `logs/camera.log` | RTSP stream events |
| `logs/tracking.log` | Track lifecycle events |
| `logs/fps_debug.log` | Per-second FPS |
| `logs/debug.log` | Everything (verbose) |
| `logs/ui_check.log` | Startup UI self-test |
| `missions/*/mission.json` | Full mission record |
| `missions/*/detections.csv` | Evidence CSV |
| `missions/*/detections.json` | Evidence JSON |

---

## Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| Space | Pause / Resume detection |
| Ctrl+S | Screenshot |
| Ctrl+R | Start recording |
| Escape | Exit (with confirmation) |

---

## Project Structure (Phase 3)

```
DualVisionAI/
├── main.py
├── requirements.txt
│
├── ai/            ← ONNX inference, model manager, backend
├── camera/        ← RTSP stream with auto-reconnect
├── config/        ← JSON-backed settings
├── tracking/      ← ByteTrack + Kalman filter
│
├── mission/                     ← NEW Phase 3
│   ├── __init__.py
│   ├── mission_state.py         ← Mission lifecycle, folder, stats, DB
│   ├── evidence_manager.py      ← Auto evidence capture + CSV/JSON
│   └── alert_system.py          ← Detection alerts + sound
│
├── ui/
│   ├── main_window.py           ← Main window (wires mission)
│   ├── toolbar.py               ← Toolbar (Settings + Mission buttons)
│   ├── mission_dialog.py        ← Mission Manager (7 tabs)  NEW
│   ├── settings_dialog.py       ← Settings (4 tabs)  unchanged
│   ├── camera_panel.py
│   ├── control_panel.py
│   ├── debug_dashboard.py
│   ├── model_manager_dialog.py
│   ├── statusbar.py
│   ├── about_dialog.py
│   └── splash_screen.py
│
├── utils/         ← logging, recorder, screenshot, detection_log
├── models/        ← auto-created
├── logs/          ← auto-created
├── recordings/    ← auto-created
├── screenshots/   ← auto-created
└── missions/      ← auto-created on first mission start
```

---

## Troubleshooting

### "Missing dependencies" error on startup
```bash
pip install -r requirements.txt
```

### Camera shows black / "Disconnected"
Configure the RTSP URL in Settings (⚙).

### ONNX export fails
```bash
pip install ultralytics
python setup.py
```

### Mission folder not created
Make sure you fill in **Mission Name** before clicking Start Mission.

### Alert sound not working
Sound (winsound) is Windows-only.  On macOS/Linux the popup and bell still fire; the beep is silenced automatically.
