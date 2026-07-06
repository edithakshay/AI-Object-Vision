# DualVision AI Detector v1.3 — UI Verification Checklist

Use this checklist to manually verify the complete application after any change.
Mark each item ☑ when confirmed working.

---

## Application Startup
- ☐ Application launches without errors
- ☐ Splash screen appears during startup
- ☐ Splash screen progress bar animates
- ☐ Splash screen closes automatically
- ☐ Main window appears
- ☐ UI self-test log created at `logs/ui_check.log`
- ☐ All startup log files created in `logs/`
- ☐ Window title shows: "DualVision AI Detector  v1.3 — Stable CPU Edition"
- ☐ Window is at least 900×600 px
- ☐ Application icon visible in taskbar

---

## Top Toolbar
- ☐ DualVision AI label visible (left side)
- ☐ Start button visible and clickable
- ☐ Stop button visible and clickable
- ☐ Pause button visible and clickable
- ☐ Shot (screenshot) button visible and clickable
- ☐ Record button visible and clickable
- ☐ Settings button visible and clickable
- ☐ CSV button visible and clickable
- ☐ JSON button visible and clickable
- ☐ About button visible and clickable
- ☐ Exit button visible and clickable
- ☐ No toolbar buttons overlap each other
- ☐ All toolbar text labels are readable

---

## Camera Panel
- ☐ RGB Camera panel visible on startup
- ☐ "RGB Camera" title label visible in panel header
- ☐ Camera feed area is visible (blank/black before stream connects)
- ☐ FPS counter visible in camera panel (top-right corner)
- ☐ Camera panel fills available space correctly
- ☐ No camera panel content clipped by window edge

---

## Right Control Panel (Dashboard)
- ☐ Control panel visible on the right side
- ☐ Control panel is scrollable (mouse wheel works)
- ☐ No content hidden below visible area (scroll to verify)
- ☐ All section headers visible (CAMERA MODE, BACKEND, SYSTEM, PERFORMANCE, TRACKING, DETECTION, DETECTION LOG, RECORDING)

### CAMERA MODE section
- ☐ "RGB Camera" radio button visible and selectable
- ☐ "Thermal Camera" radio button visible and selectable
- ☐ Active camera status label shows correct camera name
- ☐ Switching camera updates status label

### BACKEND section
- ☐ "Backend" row shows "ONNX Runtime CPU"
- ☐ "Provider" row shows "CPUExecutionProvider"
- ☐ "Model" row shows "YOLO26n"
- ☐ "Device" row shows "CPU"
- ☐ "ORT Ver" row shows version number
- ☐ "CPU Thds" row shows thread counts

### SYSTEM section
- ☐ CPU Usage updates every ~2 seconds
- ☐ RAM Usage updates every ~2 seconds
- ☐ Values are not stuck at 0% / 0 MB

### PERFORMANCE section
- ☐ Infer FPS updates during detection
- ☐ Avg FPS updates during detection
- ☐ Capture FPS shows stream FPS during detection
- ☐ Capture FPS freezes (does not reset to 0) after Stop
- ☐ Display FPS updates during detection
- ☐ Total ms shows inference time in ms
- ☐ Preprocess shows preprocess time
- ☐ Infer ms shows inference time
- ☐ Postproc shows postprocess time
- ☐ Threads shows active thread count
- ☐ Frame Queue shows queue depth
- ☐ Drops shows dropped frame count

### TRACKING section
- ☐ Active Tracks updates during detection
- ☐ Lost Tracks updates when objects disappear
- ☐ Recovered count increments when objects reappear
- ☐ New Tracks count increments for new objects
- ☐ Avg Track Age shows average age in seconds
- ☐ Tracking FPS shows tracking speed
- ☐ Track Latency shows latency in ms

### DETECTION section
- ☐ Active Dets shows current frame object count
- ☐ Session Total increments during detection
- ☐ Camera shows current active camera name
- ☐ Track IDs shows current visible track IDs (e.g. "#1, #2, #3")
- ☐ Track IDs shows "—" when no objects detected

### DETECTION LOG section
- ☐ Log textbox visible
- ☐ Detection entries appear during detection
- ☐ Each entry shows: [HH:MM:SS] Camera ClassName Confidence #TrackID
- ☐ Log scrolls automatically to latest entry
- ☐ "Clear" button clears the log
- ☐ Log textbox does not overflow outside its container

### RECORDING section
- ☐ "● Not recording" label visible when not recording
- ☐ "● Recording …" label shown when recording active
- ☐ Recording path shown when recording active
- ☐ Status resets to "● Not recording" after Stop

---

## Bottom Status Bar
- ☐ Status bar visible at bottom of window
- ☐ FPS value updates during detection
- ☐ "Model: YOLO26n" label visible
- ☐ "Device: CPU" label visible
- ☐ "Res: WxH" label visible after stream connects
- ☐ Detection status shows "Detecting" during detection
- ☐ Detection status shows "Stopped" when stopped
- ☐ RGB status updates (Connected / Disconnected / Inactive)
- ☐ Thermal status updates (Connected / Disconnected / Inactive)
- ☐ Clock (HH:MM:SS) ticks every second
- ☐ No status bar content clipped or hidden

---

## Detection Workflow
- ☐ Click Start → detection begins (no error popup)
- ☐ Model downloads automatically if not present
- ☐ ONNX model exports automatically if not present
- ☐ Bounding boxes drawn on detected objects
- ☐ Label format: "ClassName  Confidence  #TrackID" (e.g. "chair 0.92 #110")
- ☐ Each object keeps same ID across frames (tracking stable)
- ☐ Object ID does not randomly change every frame
- ☐ Object briefly leaving frame → same ID when it returns
- ☐ Multiple objects detected simultaneously with different IDs
- ☐ Click Stop → detection stops, bounding boxes cleared
- ☐ FPS values freeze (not reset) after Stop
- ☐ Click Pause → detection pauses, last frame held
- ☐ After Pause → click Start or Resume → detection resumes

---

## Camera Switching
- ☐ Select "Thermal Camera" radio → switch occurs without crash
- ☐ Camera status label updates to "Active: Thermal Camera"
- ☐ Thermal panel becomes visible, RGB panel hidden
- ☐ Tracking resets cleanly on camera switch
- ☐ Select "RGB Camera" radio → switch back works correctly
- ☐ Camera buttons are disabled during switch, re-enabled after
- ☐ Recording stops automatically on camera switch

---

## Screenshot Feature
- ☐ Click Shot button → file saved to screenshots/ directory
- ☐ Screenshot file named with timestamp
- ☐ No error popup on screenshot
- ☐ Screenshot captured with detection overlays visible

---

## Recording Feature
- ☐ Click Record button → recording starts
- ☐ Recording status label shows "● Recording …"
- ☐ File path shown in recording status area
- ☐ Click Record again (or Stop) → recording stops
- ☐ Video file saved to recordings/ directory
- ☐ Recorded video contains detection bounding boxes

---

## Settings Dialog
- ☐ Click Settings → dialog opens without error
- ☐ All settings sections visible (RTSP, DETECTION, INFERENCE, RECORDING, SCREENSHOTS, LOGGING, UI)
- ☐ RGB URL field shows current value
- ☐ Thermal URL field shows current value
- ☐ Confidence threshold adjustable
- ☐ IoU threshold adjustable
- ☐ Frame skip adjustable
- ☐ Input Width adjustable
- ☐ Enable Tracking toggle works
- ☐ CPU Threads adjustable
- ☐ Save button saves settings and closes dialog
- ☐ Cancel button discards changes and closes dialog
- ☐ Settings persist after application restart

---

## CSV Export
- ☐ Click CSV button → file-save dialog opens
- ☐ CSV file saved to chosen path
- ☐ CSV contains all detection log entries
- ☐ CSV format is valid (columns: timestamp, camera, class, confidence, track_id, box)

---

## JSON Export
- ☐ Click JSON button → file-save dialog opens
- ☐ JSON file saved to chosen path
- ☐ JSON contains all detection log entries
- ☐ JSON is valid (parseable)

---

## About Dialog
- ☐ Click About → dialog opens
- ☐ Application name displayed
- ☐ Version number displayed
- ☐ Edition displayed
- ☐ Close button works

---

## Responsive Layout — Window Resizing
- ☐ Resize to 1920×1080 → no clipped content
- ☐ Resize to 1600×900 → no clipped content
- ☐ Resize to 1536×864 → no clipped content
- ☐ Resize to 1366×768 → no clipped content
- ☐ Resize to 900×600 (minimum) → scroll in control panel works
- ☐ Control panel scrollbar appears when content exceeds height
- ☐ Camera panel scales with window size
- ☐ Status bar always visible at bottom

---

## Log Files
- ☐ `logs/startup.log` created on startup
- ☐ `logs/inference.log` created after first detection
- ☐ `logs/camera.log` created after stream connects
- ☐ `logs/fps_debug.log` created (one line per second during detection)
- ☐ `logs/debug.log` created (verbose — everything)
- ☐ `logs/ui_check.log` created on startup with self-test results

---

## Keyboard Shortcuts
- ☐ Space → Pause / Resume
- ☐ Escape → Exit (or confirm dialog)
- ☐ F5 → Start
- ☐ F6 → Stop
- ☐ Ctrl+S → Screenshot
- ☐ Ctrl+R → Toggle Recording

---

## No Layout Errors
- ☐ No widget overlaps another widget
- ☐ No widget disappears when window is resized
- ☐ No text is clipped or truncated
- ☐ No scrollbar required in the top toolbar (all buttons fit)
- ☐ No hidden controls anywhere in the UI
- ☐ Detection panel shows ALL rows (Active Dets, Session Total, Camera, Track IDs)
- ☐ Tracking section shows ALL rows (Active, Lost, Recovered, New, Avg Age, FPS, Latency)
