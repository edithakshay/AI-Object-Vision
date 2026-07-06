# DualVision AI Detector v1.3 — UI Component Inventory

Generated automatically. Every visible UI element is listed below.

---

## Main Window
- Application title bar: "DualVision AI Detector v1.3 — Stable CPU Edition"
- Window icon (icon.ico / logo.png)

---

## Top Toolbar
- **DualVision AI** branding label (left)
- **Start** button (green) — begins detection
- **Stop** button (red) — stops detection
- **Pause** button (orange) — pauses detection
- **Shot** button — captures screenshot
- **Record** button — toggles video recording
- **Settings** button — opens Settings dialog
- **CSV** button — exports detection log as CSV
- **JSON** button — exports detection log as JSON
- **About** button — opens About dialog
- **Exit** button (red) — closes application

---

## RGB Camera Window (left panel, active when RGB selected)
- Camera feed canvas (live video)
- Title label: "RGB Camera"
- FPS overlay (top-right of canvas)
- Connection status indicator
- Active detection count overlay
- Inference time overlay
- Zoom controls (mouse wheel)
- Pan controls (click + drag)
- Fullscreen toggle (double-click)

## Thermal Camera Window (left panel, active when Thermal selected)
- Camera feed canvas (live video)
- Title label: "Thermal Camera"
- FPS overlay (top-right of canvas)
- Connection status indicator
- Active detection count overlay
- Inference time overlay
- Zoom controls (mouse wheel)
- Pan controls (click + drag)
- Fullscreen toggle (double-click)

---

## Right Dashboard (Control Panel — scrollable)

### CAMERA MODE section
- RGB Camera radio button
- Thermal Camera radio button
- Active camera status label (e.g. "Active: RGB Camera")

### BACKEND section
- Backend label + value (e.g. "ONNX Runtime CPU")
- Provider label + value (e.g. "CPUExecutionProvider")
- Model label + value (e.g. "YOLO26n")
- Device label + value (e.g. "CPU")
- ORT Ver label + value (e.g. "1.27.0")
- CPU Thds label + value (e.g. "14 intra / 7 inter")

### SYSTEM section
- CPU Usage label + value (e.g. "75%")
- RAM Usage label + value (e.g. "6572 MB")

### PERFORMANCE section
- Infer FPS label + value
- Avg FPS label + value
- Capture FPS label + value
- Display FPS label + value
- Total ms label + value
- Preprocess label + value (ms)
- Infer ms label + value
- Postproc label + value (ms)
- Threads label + value
- Frame Queue label + value
- Drops label + value

### TRACKING section
- Active Tracks label + value
- Lost Tracks label + value
- Recovered label + value (total recovered since session start)
- New Tracks label + value (total new since session start)
- Avg Track Age label + value (seconds)
- Tracking FPS label + value
- Track Latency label + value (ms)

### DETECTION section
- Active Dets label + value (objects detected in current frame)
- Session Total label + value (cumulative detections this session)
- Camera label + value (active camera name)
- Track IDs label + value (current visible track IDs)

### DETECTION LOG section
- Detection log textbox (scrollable, read-only)
  - Per-detection rows: [HH:MM:SS] Camera  ClassName  Confidence  #TrackID
- Clear button (clears log textbox)

### RECORDING section
- Recording status label (● Not recording / ● Recording …)
- Recording path label (path to current recording file)

---

## Bottom Status Bar
- FPS indicator (left)
- Model indicator
- Device indicator
- Resolution indicator
- Detection Status indicator (Detecting / Stopped)
- RGB status indicator (Connected / Disconnected / Inactive)
- Thermal status indicator (Connected / Disconnected / Inactive)
- Clock / timestamp (right) — HH:MM:SS

---

## Settings Dialog (modal window)
- Title: "Settings"
- **RTSP** section:
  - RGB URL input field
  - Thermal URL input field
  - Reconnect Delay slider/field
  - Timeout slider/field
- **DETECTION** section:
  - Confidence threshold slider
  - IoU threshold slider
  - Frame skip slider
  - Input Width dropdown/field
  - Enable Tracking toggle
- **INFERENCE** section:
  - CPU Threads slider/field
- **RECORDING** section:
  - Output directory field
  - FPS field
- **SCREENSHOTS** section:
  - Output directory field
- **LOGGING** section:
  - Output directory field
  - Max log entries field
- **UI** section:
  - Window Width field
  - Window Height field
  - Maximized toggle
- Save button
- Cancel button

---

## About Dialog (modal window)
- Application name label
- Version label
- Edition label
- Description text
- Backend/provider info
- Close button

---

## Splash Screen (startup only)
- Application name label
- Version label
- Loading progress bar
- Status message label

---

## Bounding Box Overlays (drawn on camera feed)
- Coloured rectangle per detected object
- Label text: `ClassName  Confidence  #TrackID`
  - Example: `chair 0.92 #110`
- Per-class colour from COCO palette

---

## File Outputs (not UI widgets, but user-accessible)
- `logs/startup.log`
- `logs/inference.log`
- `logs/camera.log`
- `logs/fps_debug.log`
- `logs/debug.log`
- `logs/ui_check.log` (startup self-test)
- `recordings/` (video files)
- `screenshots/` (PNG captures)
- CSV export (user-selected path)
- JSON export (user-selected path)
