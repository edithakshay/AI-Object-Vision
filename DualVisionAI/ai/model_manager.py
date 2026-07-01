"""
Model Manager — YOLO26 only.
Downloads .pt once, exports to .onnx for fast CPU inference, then runs 100% offline.
"""
import os
import logging
import threading
from pathlib import Path

logger = logging.getLogger("DualVisionAI.model")

# ── YOLO26 models ─────────────────────────────────────────────────────────────
SUPPORTED_MODELS = [
    "yolo26n.pt",
    "yolo26s.pt",
    "yolo26m.pt",
    "yolo26l.pt",
    "yolo26x.pt",
]

# All YOLO26 models are downloadable
_DOWNLOADABLE = SUPPORTED_MODELS

_BASE_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0"
_URLS: dict[str, str] = {m: f"{_BASE_URL}/{m}" for m in SUPPORTED_MODELS}

# Approximate model sizes (MB) for UI display
MODEL_SIZES_MB: dict[str, int] = {
    "yolo26n.pt":   6,
    "yolo26s.pt":  20,
    "yolo26m.pt":  50,
    "yolo26l.pt":  85,
    "yolo26x.pt": 125,
}

MODEL_VERSION = "YOLO26 (Ultralytics v8.4.0)"


class ModelManager:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._progress_cb = None
        self._status_cb   = None

    # ── callbacks ─────────────────────────────────────────────────────────────
    def set_callbacks(self, progress=None, status=None):
        self._progress_cb = progress
        self._status_cb   = status

    def _progress(self, pct: float):
        if self._progress_cb:
            try: self._progress_cb(pct)
            except Exception: pass

    def _status(self, msg: str):
        logger.info(msg)
        if self._status_cb:
            try: self._status_cb(msg)
            except Exception: pass

    # ── public API ────────────────────────────────────────────────────────────
    def get_model_path(self, name: str) -> Path:
        return self.model_dir / name

    def get_onnx_path(self, name: str) -> Path:
        """Return path of the exported ONNX version of a .pt model."""
        return self.model_dir / name.replace(".pt", ".onnx")

    def is_downloaded(self, name: str) -> bool:
        p = self.get_model_path(name)
        return p.exists() and p.stat().st_size > 100_000

    def is_onnx_ready(self, name: str) -> bool:
        p = self.get_onnx_path(name)
        return p.exists() and p.stat().st_size > 100_000

    def get_model_size_mb(self, name: str) -> float:
        p = self.get_model_path(name)
        if p.exists():
            return p.stat().st_size / 1_048_576
        return MODEL_SIZES_MB.get(name, 0)

    def get_model_info(self, name: str) -> dict:
        return {
            "name":    name,
            "version": MODEL_VERSION,
            "size_mb": round(self.get_model_size_mb(name), 1),
            "path":    str(self.get_model_path(name)),
            "onnx":    str(self.get_onnx_path(name)) if self.is_onnx_ready(name) else "Not exported",
            "cache":   str(self.model_dir.resolve()),
        }

    def ensure_model(self, name: str, blocking: bool = True) -> Path | None:
        """Ensure .pt is available; download if missing. Returns local path."""
        if name not in SUPPORTED_MODELS:
            self._status(f"{name} is not a supported YOLO26 model.")
            return None
        if self.is_downloaded(name):
            self._status(f"Model ready: {name}")
            return self.get_model_path(name)
        if blocking:
            return self._download(name)
        threading.Thread(target=self._download, args=(name,),
                         daemon=True, name=f"Download-{name}").start()
        return None

    def export_onnx(self, name: str, imgsz: int = 640,
                    simplify: bool = True) -> Path | None:
        """Export .pt → .onnx using ultralytics. Returns .onnx path or None."""
        onnx_path = self.get_onnx_path(name)
        if self.is_onnx_ready(name):
            self._status(f"ONNX already exported: {onnx_path.name}")
            return onnx_path

        pt_path = self.get_model_path(name)
        if not self.is_downloaded(name):
            self._status(f"Cannot export — .pt not downloaded: {name}")
            return None

        try:
            self._status(f"Exporting {name} → ONNX (imgsz={imgsz}) …")
            from ultralytics import YOLO
            model = YOLO(str(pt_path))
            exported = model.export(
                format="onnx",
                imgsz=imgsz,
                simplify=simplify,
                half=False,
                dynamic=False,
                opset=17,
            )
            exported_p = Path(exported) if exported else None
            if exported_p and exported_p.exists() and exported_p != onnx_path:
                exported_p.rename(onnx_path)

            if self.is_onnx_ready(name):
                mb = onnx_path.stat().st_size / 1_048_576
                self._status(f"ONNX exported: {onnx_path.name} ({mb:.1f} MB)")
                return onnx_path
        except Exception as e:
            logger.error(f"ONNX export failed for {name}: {e}")
            self._status(f"ONNX export failed: {e}")
        return None

    def list_available(self) -> list[str]:
        return [m for m in SUPPORTED_MODELS if self.is_downloaded(m)]

    # ── download ─────────────────────────────────────────────────────────────
    def _download(self, name: str) -> Path | None:
        from urllib.request import urlretrieve
        from urllib.error import URLError, HTTPError

        dest = self.get_model_path(name)
        url  = _URLS.get(name)
        if not url:
            self._status(f"No download URL for {name}.")
            return None

        self._status(f"Downloading {name} …")
        self._progress(0.0)
        tmp = dest.with_suffix(".tmp")

        def _hook(blocks, block_size, total):
            if total > 0:
                pct = min(100.0, blocks * block_size * 100.0 / total)
                self._progress(pct)

        try:
            urlretrieve(url, str(tmp), reporthook=_hook)
            tmp.rename(dest)
            mb = dest.stat().st_size / 1_048_576
            self._progress(100.0)
            self._status(f"Saved: {name} ({mb:.1f} MB)")
            return dest
        except (HTTPError, URLError) as e:
            logger.error(f"Download failed for {name}: {e}")
            self._status(f"Download failed: {e}")
            self._progress(0.0)
            if tmp.exists(): tmp.unlink()
            return None
        except Exception as e:
            logger.error(f"Download error for {name}: {e}")
            self._status(f"Error: {e}")
            self._progress(0.0)
            if tmp.exists(): tmp.unlink()
            return None
