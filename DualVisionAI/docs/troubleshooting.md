# DualVision AI Detector — Troubleshooting Guide

---

## Application Won't Start

**Error: `ModuleNotFoundError: No module named 'customtkinter'`**
```cmd
pip install -r requirements.txt
```

**Error: `python is not recognized`**
- Reinstall Python and check "Add Python to PATH"
- Or use full path: `C:\Python312\python.exe main.py`

**Black window / nothing shows**
- Make sure your virtual environment is activated: `venv\Scripts\activate`
- Try running from the project root directory, not a subfolder

---

## Camera / RTSP Issues

**Streams show "Disconnected"**
- Verify the camera is powered on and connected to the same network
- Test the URL in VLC: Media → Open Network Stream → paste RTSP URL
- Check firewall — allow port 554 inbound
- Try adjusting Reconnect Delay in Settings (default: 3 seconds)

**Stream connects but video is black / corrupted**
- The camera may use a different codec — try `cv2.CAP_ANY` by editing `stream.py`
- Install FFmpeg and ensure it's on PATH
- Try a different RTSP transport: add `?tcp` to the URL if UDP drops packets

**High latency / lag**
- Reduce Input Resolution in Settings (try 416 or 320)
- Increase Frame Skip (try 2 or 3)
- Disable Tracking if not needed
- Switch to GPU inference if you have an NVIDIA card

---

## Model Issues

**Model download fails**
- Check internet connection
- Manually download the model:
  - YOLOv8n: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
  - Place in `models/yolov8n.pt`

**Model loads but detects nothing**
- Lower the Confidence Threshold in Settings (try 0.25–0.35)
- Make sure the camera stream is actually showing video content
- YOLO is trained on COCO objects — it detects: people, cars, trucks, etc.

**Inference is slow**
- Enable GPU in Settings (requires NVIDIA GPU + CUDA)
- Use a smaller input resolution (320 or 416)
- Use `yolov8n` (nano) — it's the fastest model
- Increase Frame Skip to reduce inference calls per second

---

## GPU Issues

**GPU not detected**
- Install PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
- Install ONNX Runtime GPU: `pip install onnxruntime-gpu`
- Verify CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

**CUDA out of memory**
- Reduce Input Resolution in Settings
- Close other GPU-heavy applications
- Use CPU mode as a fallback (Settings → Enable GPU: OFF)

---

## Recording Issues

**Recording button does nothing**
- Start detection first (click Start), then try recording
- Ensure `recordings/` folder is writable
- Check available disk space

**Recorded video is empty / won't open**
- Make sure the stream is connected before starting recording
- Try VLC to open the MP4 file
- The recording writes at 25 FPS — if inference is slower, video may be sparse

---

## Export Issues

**CSV/JSON export fails**
- Check that the `logs/` directory is writable
- Make sure detection has been running long enough to log entries

---

## Build / PyInstaller Issues

**`pyinstaller` command not found**
```cmd
pip install pyinstaller
```

**Build fails with missing module**
- Add the missing module to `hiddenimports` in `DualVisionAI.spec`

**EXE crashes on launch**
- Try with `console=True` in the spec file to see error output
- Check that all model files are included in `datas`

**Icons missing in EXE**
- Run `generate_icons.py` before building
- Verify `icons/app.ico` exists

---

## Performance Tips

| Setting | Recommended Value for High FPS |
|---------|-------------------------------|
| Model | yolov8n.pt |
| Input Resolution | 320 or 416 |
| Frame Skip | 2 or 3 |
| GPU | Enabled (NVIDIA) |
| Tracking | Optional (disabling saves CPU) |
| Max FPS | 60 |

---

## Log Files

Application logs are saved to `logs/app_YYYYMMDD_HHMMSS.log`.  
Detection logs are saved to `logs/detections_YYYYMMDD_HHMMSS.csv`.

Share these files when reporting issues.
