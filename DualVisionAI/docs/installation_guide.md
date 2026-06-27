# DualVision AI Detector — Installation Guide

## System Requirements

- **OS:** Windows 10 / Windows 11 (64-bit)
- **Python:** 3.12 (recommended) or 3.10+
- **RAM:** 4 GB minimum, 8 GB recommended
- **GPU:** Optional — NVIDIA GPU with CUDA for best performance
- **Network:** Internet once (for first model download); offline afterwards

---

## Step 1 — Install Python 3.12

1. Go to https://www.python.org/downloads/
2. Download **Python 3.12.x** (Windows installer, 64-bit)
3. Run the installer
   - ✅ Check **"Add Python to PATH"**
   - ✅ Check **"Install for all users"** (optional)
4. Click **Install Now**
5. Verify installation:
   ```cmd
   python --version
   ```
   Should print: `Python 3.12.x`

---

## Step 2 — Open VS Code (Recommended)

1. Download VS Code from https://code.visualstudio.com/
2. Install the **Python extension** (by Microsoft)
3. Open the project folder: **File → Open Folder** → select `DualVisionAI/`

---

## Step 3 — Create a Virtual Environment

Open the integrated terminal in VS Code (`Ctrl+\``) or a regular Command Prompt:

```cmd
cd DualVisionAI
python -m venv venv
```

Activate the virtual environment:

```cmd
venv\Scripts\activate
```

You should see `(venv)` at the start of the prompt.

---

## Step 4 — Install Dependencies

### CPU Only (Default)

```cmd
pip install -r requirements.txt
```

### GPU (NVIDIA CUDA) — For Maximum FPS

```cmd
# Step 4a: Install PyTorch with CUDA (visit https://pytorch.org for the exact command for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Step 4b: Install ONNX Runtime GPU
pip install onnxruntime-gpu

# Step 4c: Install everything else
pip install customtkinter opencv-python Pillow numpy ultralytics psutil av pyinstaller
```

---

## Step 5 — Generate Icons

Run once to create all app icons and the logo:

```cmd
python generate_icons.py
```

This creates `icons/app.ico`, `icons/app_256.png`, and `assets/logo.png`.

---

## Step 6 — Run the Application

```cmd
python main.py
```

**First run:** The app will automatically download the YOLOv8n model (~6 MB).  
**Subsequent runs:** Fully offline — no internet required.

---

## Step 7 — Configure Camera URLs

1. Click **Settings** in the toolbar
2. Update the RTSP URLs under **RTSP Streams**:
   - RGB Stream URL: `rtsp://192.168.144.108:554/stream=1`
   - Thermal Stream URL: `rtsp://192.168.144.108:554/stream=2`
3. Click **Save Settings**
4. Click **Start** to begin detection

Or edit `config/app_config.json` directly.

---

## Build Windows EXE

To create a standalone `.exe` that doesn't require Python installed:

```cmd
# Make sure venv is active
venv\Scripts\activate

# Run the build script
build.bat
```

Or manually:

```cmd
pyinstaller DualVisionAI.spec
```

Output is in `dist\DualVisionAI\`. Copy the entire `dist\DualVisionAI\` folder to any Windows machine — it runs without Python.

---

## VS Code Configuration

Create `.vscode/settings.json` in the project root:

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/Scripts/python.exe",
    "python.terminal.activateEnvironment": true,
    "editor.formatOnSave": true
}
```

Create `.vscode/launch.json` to run with F5:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Run DualVision AI",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/main.py",
            "cwd": "${workspaceFolder}",
            "console": "integratedTerminal"
        }
    ]
}
```

---

## Folder Permissions

Make sure the following folders are writable (they are auto-created):

- `models/` — YOLO model files
- `logs/` — Application and detection logs
- `recordings/` — MP4 video files
- `screenshots/` — PNG screenshots
- `config/` — Settings JSON

---

## Choosing a Model

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `yolov8n.pt` | ~6 MB | Fastest | Good |
| `yolov8s.pt` | ~22 MB | Fast | Better |
| `yolo11n.pt` | ~5 MB | Fastest | Good |

Change the model in **Settings → Model**.
