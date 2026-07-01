"""
Model Manager — DualVision AI v1.3 Stable CPU Edition.

Fixed model: YOLO26n only.
  • Downloads yolo26n.pt once (internet required only the first time).
  • Exports yolo26n.onnx once via ultralytics (PyTorch is used only here).
  • Every subsequent run loads the .onnx directly — no PyTorch at runtime.

No model switching. No model list. One model. One backend.
"""

import logging
import threading
import traceback
from pathlib import Path

logger = logging.getLogger("DualVisionAI.model")

# ── Fixed model constants ──────────────────────────────────────────────────────
MODEL_NAME    = "yolo26n.pt"
MODEL_VERSION = "YOLO26n (Ultralytics v8.4.0)"
MODEL_SIZE_MB = 6        # approximate download size
ONNX_OPSET   = 17
ONNX_IMGSZ   = 640

_BASE_URL  = "https://github.com/ultralytics/assets/releases/download/v8.4.0"
_MODEL_URL = f"{_BASE_URL}/{MODEL_NAME}"


class ModelManager:
    """
    Manages the single YOLO26n model file.
    Responsibilities: download .pt, export .onnx, report status.
    """

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._progress_cb = None
        self._status_cb   = None

    # ── Callbacks ─────────────────────────────────────────────────────────────
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

    # ── Paths ─────────────────────────────────────────────────────────────────
    def get_pt_path(self) -> Path:
        return self.model_dir / MODEL_NAME

    def get_onnx_path(self) -> Path:
        return self.model_dir / MODEL_NAME.replace(".pt", ".onnx")

    # Kept for compatibility with code that calls get_model_path(name)
    def get_model_path(self, name: str = MODEL_NAME) -> Path:
        return self.model_dir / name

    # ── Status checks ─────────────────────────────────────────────────────────
    def is_pt_ready(self) -> bool:
        p = self.get_pt_path()
        return p.exists() and p.stat().st_size > 100_000

    def is_onnx_ready(self) -> bool:
        p = self.get_onnx_path()
        return p.exists() and p.stat().st_size > 100_000

    def is_downloaded(self, name: str = MODEL_NAME) -> bool:
        """Compatibility alias."""
        return self.is_pt_ready()

    # ── Info ──────────────────────────────────────────────────────────────────
    def get_model_info(self) -> dict:
        pt_path   = self.get_pt_path()
        onnx_path = self.get_onnx_path()
        pt_mb = (pt_path.stat().st_size / 1_048_576
                 if pt_path.exists() else 0.0)
        onnx_mb = (onnx_path.stat().st_size / 1_048_576
                   if onnx_path.exists() else 0.0)
        return {
            "name":       MODEL_NAME,
            "version":    MODEL_VERSION,
            "pt_path":    str(pt_path),
            "onnx_path":  str(onnx_path) if self.is_onnx_ready() else "Not exported",
            "pt_mb":      round(pt_mb, 1),
            "onnx_mb":    round(onnx_mb, 1),
            "cache_dir":  str(self.model_dir.resolve()),
            "pt_ready":   self.is_pt_ready(),
            "onnx_ready": self.is_onnx_ready(),
        }

    # ── Download ──────────────────────────────────────────────────────────────
    def ensure_pt(self, blocking: bool = True) -> Path | None:
        """
        Ensure yolo26n.pt is present locally.
        Downloads from ultralytics if missing.
        Returns local path or None on failure.
        """
        if self.is_pt_ready():
            self._status(f"Model ready: {MODEL_NAME}")
            return self.get_pt_path()

        if blocking:
            return self._download_pt()
        threading.Thread(target=self._download_pt, daemon=True,
                         name="Download-YOLO26n").start()
        return None

    def _download_pt(self) -> Path | None:
        from urllib.request import urlretrieve
        from urllib.error import URLError, HTTPError

        dest = self.get_pt_path()
        self._status(f"Downloading {MODEL_NAME} (~{MODEL_SIZE_MB} MB) …")
        self._progress(0.0)
        tmp = dest.with_suffix(".tmp")

        def _hook(blocks, block_size, total):
            if total > 0:
                pct = min(100.0, blocks * block_size * 100.0 / total)
                self._progress(pct)

        try:
            urlretrieve(_MODEL_URL, str(tmp), reporthook=_hook)
            tmp.rename(dest)
            mb = dest.stat().st_size / 1_048_576
            self._progress(100.0)
            self._status(f"Downloaded: {MODEL_NAME} ({mb:.1f} MB)")
            logger.info(f"Model downloaded: {dest}")
            return dest
        except (HTTPError, URLError) as exc:
            logger.error(f"Download failed: {exc}")
            self._status(f"Download failed: {exc}")
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error(f"Download error:\n{tb}")
            self._status(f"Download error: {exc}")
        finally:
            self._progress(0.0)
            if tmp.exists():
                try: tmp.unlink()
                except Exception: pass
        return None

    # ── ONNX Export ───────────────────────────────────────────────────────────
    def export_onnx(self) -> Path | None:
        """
        Export yolo26n.pt → yolo26n.onnx using ultralytics (PyTorch).
        PyTorch is used ONLY here — never during inference.
        Returns .onnx Path on success, None on failure.
        Raises on unexpected errors so the caller can show a popup.
        """
        onnx_path = self.get_onnx_path()

        if self.is_onnx_ready():
            self._status(f"ONNX already exported: {onnx_path.name}")
            return onnx_path

        if not self.is_pt_ready():
            raise FileNotFoundError(
                f"{MODEL_NAME} not found in {self.model_dir}.\n"
                "Click 'Download Model' first.")

        pt_path = self.get_pt_path()
        self._status(f"Exporting {MODEL_NAME} → ONNX (opset={ONNX_OPSET}, "
                     f"imgsz={ONNX_IMGSZ}) …")
        logger.info(f"Starting ONNX export: {pt_path.name}")

        try:
            from ultralytics import YOLO
            model    = YOLO(str(pt_path))
            exported = model.export(
                format="onnx",
                imgsz=ONNX_IMGSZ,
                simplify=True,
                half=False,       # CPU inference — FP32 always
                dynamic=False,
                opset=ONNX_OPSET,
            )
            exported_p = Path(exported) if exported else None

            # Ultralytics may save alongside .pt — move to models/
            if exported_p and exported_p.exists() and exported_p != onnx_path:
                exported_p.rename(onnx_path)

            if self.is_onnx_ready():
                mb = onnx_path.stat().st_size / 1_048_576
                self._status(f"ONNX exported: {onnx_path.name} ({mb:.1f} MB)")
                logger.info(f"ONNX export complete: {onnx_path} ({mb:.1f} MB)")
                return onnx_path

            raise RuntimeError(
                f"ONNX export ran but file not found at {onnx_path}.\n"
                "Check ultralytics logs for errors.")
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error(f"ONNX export failed:\n{tb}")
            raise RuntimeError(
                f"ONNX export failed: {exc}\n\n{tb}") from exc
