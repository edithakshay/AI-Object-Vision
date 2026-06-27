import os
import logging
import threading
from pathlib import Path

logger = logging.getLogger("DualVisionAI.model")

MODEL_URLS = {
    "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
    "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt",
    "yolo11n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
}


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
            self._status_callback(msg)

    def _notify_progress(self, pct: float):
        if self._progress_callback:
            self._progress_callback(pct)

    def get_model_path(self, model_name: str) -> Path:
        return self.model_dir / model_name

    def is_downloaded(self, model_name: str) -> bool:
        p = self.get_model_path(model_name)
        return p.exists() and p.stat().st_size > 1_000_000

    def ensure_model(self, model_name: str, blocking: bool = True) -> Path | None:
        if self.is_downloaded(model_name):
            self._notify_status(f"Model already cached: {model_name}")
            return self.get_model_path(model_name)

        if blocking:
            return self._download(model_name)
        else:
            t = threading.Thread(target=self._download, args=(model_name,),
                                 daemon=True)
            t.start()
            return None

    def _download(self, model_name: str) -> Path | None:
        self._notify_status(f"Downloading model: {model_name} ...")
        self._notify_progress(0.0)
        dest = self.get_model_path(model_name)
        try:
            from ultralytics import YOLO
            self._notify_status("Using Ultralytics auto-download ...")
            model = YOLO(model_name)
            pt_path = Path(model_name)
            if pt_path.exists():
                import shutil
                shutil.move(str(pt_path), str(dest))
            self._notify_progress(100.0)
            self._notify_status(f"Model ready: {dest}")
            return dest
        except Exception as e:
            logger.warning(f"Ultralytics download failed: {e}, trying urllib ...")
            return self._urllib_download(model_name, dest)

    def _urllib_download(self, model_name: str, dest: Path) -> Path | None:
        import urllib.request
        url = MODEL_URLS.get(model_name)
        if not url:
            self._notify_status(f"No URL for model: {model_name}")
            return None
        try:
            def reporthook(count, block_size, total_size):
                if total_size > 0:
                    pct = min(100.0, count * block_size / total_size * 100)
                    self._notify_progress(pct)
            urllib.request.urlretrieve(url, dest, reporthook)
            self._notify_progress(100.0)
            self._notify_status(f"Downloaded: {dest}")
            return dest
        except Exception as e:
            logger.error(f"Download failed: {e}")
            self._notify_status(f"Download failed: {e}")
            return None

    def export_to_onnx(self, model_path: Path) -> Path | None:
        onnx_path = model_path.with_suffix(".onnx")
        if onnx_path.exists():
            logger.info(f"ONNX already exists: {onnx_path}")
            return onnx_path
        try:
            self._notify_status("Exporting to ONNX ...")
            from ultralytics import YOLO
            model = YOLO(str(model_path))
            model.export(format="onnx", dynamic=False, simplify=True,
                         imgsz=640, opset=12)
            exported = model_path.with_suffix(".onnx")
            if not exported.exists():
                pt_stem = model_path.stem
                exported = Path(f"{pt_stem}.onnx")
            if exported.exists() and exported != onnx_path:
                import shutil
                shutil.move(str(exported), str(onnx_path))
            self._notify_status(f"ONNX ready: {onnx_path}")
            return onnx_path
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            self._notify_status(f"ONNX export failed: {e}")
            return None

    def list_available(self) -> list[str]:
        return [f.name for f in self.model_dir.iterdir()
                if f.suffix in (".pt", ".onnx")]
