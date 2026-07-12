"""
Model Manager — DualVision AI v1.3 Stable CPU Edition.

Multi-model support: YOLO26n, YOLO26s, YOLO26m, YOLO26l, YOLO26x.

Export-once policy:
  • If models/<variant>.onnx already exists  → load directly.
  • Otherwise export from <variant>.pt once  → reuse permanently.
  • Never re-export on every startup.

Backward compatibility:
  • All v1.2 callers that relied on the single-model API still work.
  • get_onnx_path()  → returns path for the currently selected variant.
  • is_onnx_ready()  → checks the currently selected variant.
"""

import logging
import shutil
import threading
import traceback
from pathlib import Path

logger = logging.getLogger("DualVisionAI.model")


def _setup_model_log():
    """Attach a dedicated file handler to logs/model_manager.log (PART 12)."""
    try:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "model_manager.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"))
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
        if not logger.level or logger.level > logging.DEBUG:
            logger.setLevel(logging.DEBUG)
    except Exception:
        pass


_setup_model_log()

# ── Model registry ─────────────────────────────────────────────────────────────
# Keys are the variant suffix (also the user-visible display name).
MODEL_VARIANTS = {
    "yolo26n": {
        "pt_name":    "yolo26n.pt",
        "onnx_name":  "yolo26n.onnx",
        "size_mb":    6,
        "label":      "YOLO26n  (Nano  — fastest,  ~6 MB)",
        "url_path":   "yolo26n.pt",
    },
    "yolo26s": {
        "pt_name":    "yolo26s.pt",
        "onnx_name":  "yolo26s.onnx",
        "size_mb":    22,
        "label":      "YOLO26s  (Small — balanced, ~22 MB)",
        "url_path":   "yolo26s.pt",
    },
    "yolo26m": {
        "pt_name":    "yolo26m.pt",
        "onnx_name":  "yolo26m.onnx",
        "size_mb":    52,
        "label":      "YOLO26m  (Medium — accurate, ~52 MB)",
        "url_path":   "yolo26m.pt",
    },
    "yolo26l": {
        "pt_name":    "yolo26l.pt",
        "onnx_name":  "yolo26l.onnx",
        "size_mb":    87,
        "label":      "YOLO26l  (Large  — high accuracy, ~87 MB)",
        "url_path":   "yolo26l.pt",
    },
    "yolo26x": {
        "pt_name":    "yolo26x.pt",
        "onnx_name":  "yolo26x.onnx",
        "size_mb":    136,
        "label":      "YOLO26x  (XLarge — best accuracy, ~136 MB)",
        "url_path":   "yolo26x.pt",
    },
}

# Default variant (kept for legacy callers)
DEFAULT_VARIANT = "yolo26n"
MODEL_NAME      = MODEL_VARIANTS[DEFAULT_VARIANT]["pt_name"]   # legacy alias
MODEL_VERSION   = "YOLO26n (Ultralytics v8.4.0)"               # legacy alias
MODEL_SIZE_MB   = 6                                             # legacy alias
ONNX_OPSET      = 17
ONNX_IMGSZ      = 640
_BASE_URL       = "https://github.com/ultralytics/assets/releases/download/v8.4.0"


class ModelManager:
    """
    Manages any YOLO26 variant.
    Default variant: yolo26n (backward compatible with v1.2).
    """

    def __init__(self, model_dir: str = "models", variant: str = DEFAULT_VARIANT):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._variant      = variant if variant in MODEL_VARIANTS else DEFAULT_VARIANT
        self._progress_cb  = None
        self._status_cb    = None

    # ── Variant selection ─────────────────────────────────────────────────────
    @property
    def variant(self) -> str:
        return self._variant

    def set_variant(self, variant: str):
        if variant not in MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Valid: {list(MODEL_VARIANTS.keys())}")
        self._variant = variant
        logger.info(f"Model variant set to: {variant}")

    @staticmethod
    def list_variants() -> list:
        """Return list of (key, label) pairs for all supported models."""
        return [(k, v["label"]) for k, v in MODEL_VARIANTS.items()]

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
    def _meta(self, variant: str | None = None) -> dict:
        v = variant or self._variant
        return MODEL_VARIANTS[v]

    def get_pt_path(self, variant: str | None = None) -> Path:
        return self.model_dir / self._meta(variant)["pt_name"]

    def get_onnx_path(self, variant: str | None = None) -> Path:
        return self.model_dir / self._meta(variant)["onnx_name"]

    # Legacy alias — returns path for active variant
    def get_model_path(self, name: str | None = None) -> Path:
        if name and name != self._meta()["pt_name"]:
            return self.model_dir / name
        return self.get_pt_path()

    # ── Status checks ─────────────────────────────────────────────────────────
    def is_pt_ready(self, variant: str | None = None) -> bool:
        p = self.get_pt_path(variant)
        return p.exists() and p.stat().st_size > 100_000

    def is_onnx_ready(self, variant: str | None = None) -> bool:
        p = self.get_onnx_path(variant)
        return p.exists() and p.stat().st_size > 100_000

    def is_downloaded(self, name: str | None = None) -> bool:
        """Legacy alias — checks whether the active variant's .pt is present."""
        return self.is_pt_ready()

    def get_all_model_status(self) -> list:
        """Return status for every variant (for benchmark / comparison UI)."""
        result = []
        for key, meta in MODEL_VARIANTS.items():
            result.append({
                "variant":    key,
                "label":      meta["label"],
                "pt_ready":   self.is_pt_ready(key),
                "onnx_ready": self.is_onnx_ready(key),
                "pt_mb":      self._file_mb(self.get_pt_path(key)),
                "onnx_mb":    self._file_mb(self.get_onnx_path(key)),
            })
        return result

    def _file_mb(self, p: Path) -> float:
        return round(p.stat().st_size / 1_048_576, 1) if p.exists() else 0.0

    # ── Info ──────────────────────────────────────────────────────────────────
    def get_model_info(self, variant: str | None = None) -> dict:
        v     = variant or self._variant
        meta  = self._meta(v)
        pt_p  = self.get_pt_path(v)
        on_p  = self.get_onnx_path(v)
        return {
            "name":       meta["pt_name"],
            "variant":    v,
            "label":      meta["label"],
            "version":    f"YOLO26{v[-1].upper()} (Ultralytics v8.4.0)",
            "pt_path":    str(pt_p),
            "onnx_path":  str(on_p) if self.is_onnx_ready(v) else "Not exported",
            "pt_mb":      self._file_mb(pt_p),
            "onnx_mb":    self._file_mb(on_p),
            "cache_dir":  str(self.model_dir.resolve()),
            "pt_ready":   self.is_pt_ready(v),
            "onnx_ready": self.is_onnx_ready(v),
        }

    # ── Import local file (PART 3) ────────────────────────────────────────────
    def import_local(self, src_path: str,
                     status_cb=None) -> tuple:
        """
        Copy a local .pt or .onnx file into models/.
        Returns (success: bool, message: str).

        Validation rules (PART 10):
          - File must exist and be readable
          - Extension must be .pt or .onnx
          - Filename must identify a known YOLO26 variant
          - ONNX files are validated via ONNX Runtime before import
        """
        src = Path(src_path)

        def _status(msg):
            logger.info(f"[IMPORT] {msg}")
            if status_cb:
                try: status_cb(msg)
                except Exception: pass

        # ── Basic checks ──────────────────────────────────────────────────────
        if not src.exists():
            msg = f"File not found: {src}"
            logger.error(f"[IMPORT] {msg}")
            return False, msg

        try:
            sz = src.stat().st_size
        except OSError as exc:
            msg = f"Cannot read file: {exc}"
            logger.error(f"[IMPORT] {msg}")
            return False, msg

        if sz < 1_000:
            msg = f"File too small ({sz} bytes) — possibly corrupted: {src.name}"
            logger.error(f"[IMPORT] {msg}")
            return False, msg

        ext = src.suffix.lower()
        if ext not in (".pt", ".onnx"):
            msg = (f"Unsupported format '{ext}'.  "
                   f"Accepted: .pt (PyTorch) or .onnx")
            logger.error(f"[IMPORT] {msg}")
            return False, msg

        # ── Identify variant from filename ────────────────────────────────────
        stem = src.stem.lower()
        variant = None
        for k in MODEL_VARIANTS:
            if k in stem or stem in (k, k + "_onnx"):
                variant = k
                break
        if variant is None:
            msg = (f"Cannot identify YOLO26 variant from '{src.name}'.\n"
                   f"Rename the file to include the variant name, e.g.:\n"
                   f"  yolo26n.pt  yolo26s.onnx  yolo26m.pt  …")
            logger.error(f"[IMPORT] {msg}")
            return False, msg

        # ── ONNX architecture validation (PART 10) ────────────────────────────
        if ext == ".onnx":
            _status(f"Validating ONNX architecture for {src.name} …")
            ok, vmsg = self._validate_onnx_file(src)
            if not ok:
                msg = f"ONNX validation failed: {vmsg}"
                logger.error(f"[IMPORT] {msg}")
                return False, msg

        # ── Copy ──────────────────────────────────────────────────────────────
        meta = MODEL_VARIANTS[variant]
        dest = self.model_dir / (meta["pt_name"] if ext == ".pt"
                                 else meta["onnx_name"])
        _status(f"Copying {src.name} → models/{dest.name} …")
        try:
            shutil.copy2(str(src), str(dest))
        except Exception as exc:
            msg = f"Copy failed: {exc}"
            logger.error(f"[IMPORT] {msg}")
            return False, msg

        mb = dest.stat().st_size / 1_048_576
        msg = f"Imported {dest.name}  ({mb:.1f} MB)  variant={variant}"
        _status(msg)
        return True, msg

    # ── Validation (PART 10) ─────────────────────────────────────────────────
    def validate_model(self, variant: str | None = None) -> tuple:
        """
        Validate installed files for a given variant.
        Returns (ok: bool, message: str).
        Checks: file exists, readable, size, ONNX architecture.
        """
        v = variant or self._variant
        issues = []

        onnx_p = self.get_onnx_path(v)
        pt_p   = self.get_pt_path(v)

        # PT check
        if pt_p.exists():
            if pt_p.stat().st_size < 100_000:
                issues.append(
                    f".pt file too small ({pt_p.stat().st_size} bytes) — may be corrupted")
        # ONNX check
        if onnx_p.exists():
            if onnx_p.stat().st_size < 100_000:
                issues.append(
                    f"ONNX file too small ({onnx_p.stat().st_size} bytes) — may be corrupted")
            else:
                ok, vmsg = self._validate_onnx_file(onnx_p)
                if not ok:
                    issues.append(f"ONNX invalid: {vmsg}")

        if not pt_p.exists() and not onnx_p.exists():
            issues.append("Neither .pt nor .onnx file found in models/")

        if issues:
            msg = "; ".join(issues)
            logger.warning(f"[VALIDATE] {v}: {msg}")
            return False, msg

        logger.info(f"[VALIDATE] {v}: OK")
        return True, "OK"

    def _validate_onnx_file(self, path: Path) -> tuple:
        """
        Attempt to load an ONNX file with ORT and verify input shape.
        Returns (ok: bool, message: str).
        """
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"])
            inp   = sess.get_inputs()[0]
            shape = inp.shape
            if len(shape) != 4:
                return (False,
                        f"Unexpected input rank {len(shape)} (expected 4 for NCHW)")
            if shape[1] not in (3, "C"):
                return (False,
                        f"Expected 3-channel input, got channels={shape[1]}")
            return True, "OK"
        except ImportError:
            # ORT not available — accept without full validation
            return True, "OK (ORT not available for deep validation)"
        except Exception as exc:
            return False, str(exc)

    # ── Switch model + log (PART 5, 12) ───────────────────────────────────────
    def switch_variant(self, variant: str):
        """Switch active variant and log the change (PART 5 / PART 12)."""
        old = self._variant
        self.set_variant(variant)
        logger.info(f"[SWITCH] {old} → {variant}")

    # ── Download ──────────────────────────────────────────────────────────────
    def ensure_pt(self, blocking: bool = True,
                  variant: str | None = None) -> Path | None:
        v = variant or self._variant
        if self.is_pt_ready(v):
            self._status(f"Model ready: {self._meta(v)['pt_name']}")
            return self.get_pt_path(v)
        if blocking:
            return self._download_pt(v)
        threading.Thread(target=self._download_pt, args=(v,),
                         daemon=True,
                         name=f"Download-{v}").start()
        return None

    def _download_pt(self, variant: str) -> Path | None:
        from urllib.request import urlretrieve
        from urllib.error   import URLError, HTTPError

        meta = self._meta(variant)
        url  = f"{_BASE_URL}/{meta['url_path']}"
        dest = self.get_pt_path(variant)
        self._status(f"Downloading {meta['pt_name']} "
                     f"(~{meta['size_mb']} MB) …")
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
            self._status(f"Downloaded: {meta['pt_name']} ({mb:.1f} MB)")
            logger.info(f"Model downloaded: {dest}")
            return dest
        except (HTTPError, URLError) as exc:
            logger.error(f"Download failed: {exc}")
            self._status(f"Download failed: {exc}")
        except Exception as exc:
            logger.error(f"Download error:\n{traceback.format_exc()}")
            self._status(f"Download error: {exc}")
        finally:
            self._progress(0.0)
            if tmp.exists():
                try: tmp.unlink()
                except Exception: pass
        return None

    # ── ONNX Export ───────────────────────────────────────────────────────────
    def export_onnx(self, variant: str | None = None) -> Path | None:
        """
        Export <variant>.pt → <variant>.onnx once; skip if already done.
        PyTorch used only here — never during inference.
        """
        v         = variant or self._variant
        onnx_path = self.get_onnx_path(v)

        if self.is_onnx_ready(v):
            self._status(f"ONNX ready: {onnx_path.name}")
            return onnx_path

        if not self.is_pt_ready(v):
            raise FileNotFoundError(
                f"{self._meta(v)['pt_name']} not found in {self.model_dir}.\n"
                "Click 'Download Model' first.")

        pt_path = self.get_pt_path(v)
        self._status(f"Exporting {pt_path.name} → ONNX "
                     f"(opset={ONNX_OPSET}, imgsz={ONNX_IMGSZ}) …")
        logger.info(f"ONNX export start: {pt_path.name}")

        try:
            from ultralytics import YOLO
            model    = YOLO(str(pt_path))
            exported = model.export(
                format="onnx",
                imgsz=ONNX_IMGSZ,
                simplify=True,
                half=False,
                dynamic=False,
                opset=ONNX_OPSET,
            )
            exported_p = Path(exported) if exported else None

            if exported_p and exported_p.exists() and exported_p != onnx_path:
                exported_p.rename(onnx_path)

            if self.is_onnx_ready(v):
                mb = onnx_path.stat().st_size / 1_048_576
                self._status(f"ONNX exported: {onnx_path.name} ({mb:.1f} MB)")
                logger.info(f"Export complete: {onnx_path} ({mb:.1f} MB)")
                return onnx_path

            raise RuntimeError(
                f"Export ran but ONNX not found at {onnx_path}. "
                "Check ultralytics logs.")
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error(f"ONNX export failed:\n{tb}")
            raise RuntimeError(f"ONNX export failed: {exc}\n\n{tb}") from exc
