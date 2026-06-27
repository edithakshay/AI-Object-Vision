import os
import logging
import shutil
import threading
from pathlib import Path

logger = logging.getLogger("DualVisionAI.model")

SUPPORTED_MODELS = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolo11n.pt",
    "yolo11s.pt",
]


class ModelManager:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._progress_callback = None
        self._status_callback = None

    def set_callbacks(self, progress=None, status=None):
        self._progress_callback = progress
        self._status_callback = status

    def _notify_status(self, msg: str):
        logger.info(msg)
        if self._status_callback:
            try:
                self._status_callback(msg)
            except Exception:
                pass

    def _notify_progress(self, pct: float):
        if self._progress_callback:
            try:
                self._progress_callback(pct)
            except Exception:
                pass

    def get_model_path(self, model_name: str) -> Path:
        return self.model_dir / model_name

    def is_downloaded(self, model_name: str) -> bool:
        p = self.get_model_path(model_name)
        return p.exists() and p.stat().st_size > 100_000

    def ensure_model(self, model_name: str, blocking: bool = True) -> Path | None:
        dest = self.get_model_path(model_name)
        if self.is_downloaded(model_name):
            self._notify_status(f"Model already cached: {model_name}")
            return dest
        if blocking:
            return self._download(model_name)
        else:
            threading.Thread(target=self._download, args=(model_name,), daemon=True).start()
            return None

    def _download(self, model_name: str) -> Path | None:
        dest = self.get_model_path(model_name)
        self._notify_status(f"Downloading {model_name} via Ultralytics ...")
        self._notify_progress(0.0)
        try:
            from ultralytics import YOLO
            # Ultralytics will download to its own cache automatically
            model = YOLO(model_name)
            self._notify_progress(80.0)

            # After loading, find where Ultralytics put the file and copy it
            found = self._find_downloaded_file(model_name)
            if found and found != dest:
                shutil.copy2(str(found), str(dest))
                self._notify_status(f"Model copied to: {dest}")
            elif not dest.exists():
                # Ultralytics manages the file internally — record a marker
                # so we know it's available via YOLO(model_name) next time
                dest.write_text(f"managed:{model_name}")

            self._notify_progress(100.0)
            self._notify_status(f"Model ready: {model_name}")
            return dest
        except Exception as e:
            logger.error(f"Download failed for {model_name}: {e}")
            self._notify_status(f"Download failed: {e}")
            return None

    def _find_downloaded_file(self, model_name: str) -> Path | None:
        """Search common Ultralytics cache locations for the downloaded model."""
        candidates = [
            Path(model_name),                                         # current dir
            Path.home() / ".ultralytics" / "assets" / model_name,
            Path.home() / ".cache" / "ultralytics" / model_name,
            Path.home() / "AppData" / "Roaming" / "ultralytics" / model_name,
            Path.home() / "AppData" / "Local" / "ultralytics" / model_name,
            Path("runs") / model_name,
        ]
        for p in candidates:
            if p.exists() and p.stat().st_size > 100_000:
                return p
        return None

    def list_available(self) -> list[str]:
        return [f.name for f in self.model_dir.iterdir()
                if f.suffix in (".pt", ".onnx")]
