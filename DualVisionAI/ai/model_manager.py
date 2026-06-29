"""
Model manager — downloads YOLO models to local models/ folder.
After download the app works 100% offline.

YOLO26 is listed as a future placeholder; download is skipped until
Ultralytics releases it.
"""
import os
import logging
import shutil
import threading
import time
from pathlib import Path

logger = logging.getLogger("DualVisionAI.model")

# ── Available models ──────────────────────────────────────────────────────────
SUPPORTED_MODELS = [
    # YOLOv8 family
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
    # YOLO11 family
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolo11l.pt",
    "yolo11x.pt",
    # YOLO26 family (placeholder — not yet released by Ultralytics)
    "yolo26n.pt",
    "yolo26s.pt",
    "yolo26m.pt",
    "yolo26l.pt",
    "yolo26x.pt",
]

# Models that can actually be downloaded from Ultralytics right now
_DOWNLOADABLE = {m for m in SUPPORTED_MODELS if not m.startswith("yolo26")}

# Ultralytics cache search paths (Windows + Linux)
_CACHE_ROOTS = [
    Path.home() / ".ultralytics" / "assets",
    Path.home() / ".cache" / "ultralytics",
    Path.home() / "AppData" / "Roaming" / "ultralytics",
    Path.home() / "AppData" / "Local"   / "ultralytics",
    Path.home() / "AppData" / "Local"   / "Temp",
]


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

    def is_downloaded(self, name: str) -> bool:
        p = self.get_model_path(name)
        if not p.exists():
            return False
        # Marker file written when Ultralytics manages the cache itself
        if p.stat().st_size < 100_000:
            content = p.read_text(errors="ignore")
            return content.startswith("managed:")
        return True

    def ensure_model(self, name: str, blocking: bool = True) -> Path | None:
        """Ensure model is available; download if needed. Returns local path."""
        if self.is_downloaded(name):
            self._status(f"Model ready (cached): {name}")
            return self.get_model_path(name)

        if name not in _DOWNLOADABLE:
            self._status(
                f"{name} is not yet released by Ultralytics — cannot download.")
            return None

        if blocking:
            return self._download(name)
        threading.Thread(target=self._download, args=(name,),
                         daemon=True, name=f"Download-{name}").start()
        return None

    def list_available(self) -> list[str]:
        out = []
        for f in self.model_dir.iterdir():
            if f.suffix not in (".pt", ".onnx"):
                continue
            if self.is_downloaded(f.name):
                out.append(f.name)
        return out

    # ── download ─────────────────────────────────────────────────────────────
    def _download(self, name: str) -> Path | None:
        dest = self.get_model_path(name)
        self._status(f"Downloading {name} …")
        self._progress(0.0)

        try:
            from ultralytics import YOLO
            self._progress(10.0)
            _ = YOLO(name)          # triggers Ultralytics auto-download
            self._progress(80.0)

            # 1) Check cwd (Ultralytics sometimes drops file here)
            cwd_file = Path(name)
            if cwd_file.exists() and cwd_file.stat().st_size > 100_000:
                shutil.copy2(str(cwd_file), str(dest))
                self._progress(100.0)
                self._status(f"Model saved: {dest}")
                return dest

            # 2) Search Ultralytics cache directories
            found = self._find_in_cache(name)
            if found:
                shutil.copy2(str(found), str(dest))
                self._progress(100.0)
                self._status(f"Model saved: {dest}")
                return dest

            # 3) Ultralytics manages the file internally — write marker
            dest.write_text(f"managed:{name}")
            self._progress(100.0)
            self._status(f"Model ready via Ultralytics cache: {name}")
            return dest

        except Exception as e:
            logger.error(f"Download failed for {name}: {e}")
            self._status(f"Download failed: {e}")
            self._progress(0.0)
            return None

    def _find_in_cache(self, name: str) -> Path | None:
        for root in _CACHE_ROOTS:
            p = root / name
            if p.exists() and p.stat().st_size > 100_000:
                return p
        return None
