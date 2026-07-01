"""
Backend Manager — DualVision AI Detector v1.3 Stable CPU Edition.

Fixed backend: ONNX Runtime + CPUExecutionProvider.
No GPU. No CUDA. No TensorRT. No PyTorch inference.

Responsibilities:
  • Configure ORT SessionOptions for maximum CPU performance.
  • Detect optimal thread count for Intel CPUs.
  • Build ORT InferenceSession — raises on failure (no silent fallback).
  • Write logs/startup.log with full diagnostics.
  • Expose diagnostics dict for the dashboard.
"""

import logging
import datetime
import platform
from pathlib import Path

logger = logging.getLogger("DualVisionAI.backend")

BACKEND_LABEL   = "ONNX Runtime CPU"
PROVIDER_LABEL  = "CPUExecutionProvider"
DEVICE_LABEL    = "CPU"
MODEL_LABEL     = "YOLO26n"


class BackendManager:
    """
    CPU-only inference backend.
    Builds ONNX Runtime sessions with CPUExecutionProvider and
    graph/thread optimisation.
    """

    def __init__(self, num_threads: int = 0, inter_threads: int = 0):
        """
        Args:
            num_threads:   intra-op thread count (0 = auto-detect)
            inter_threads: inter-op thread count (0 = half of intra)
        """
        # ── Configuration ─────────────────────────────────────────────────────
        self._req_intra  = num_threads
        self._req_inter  = inter_threads

        # ── Resolved at detection time ────────────────────────────────────────
        self.num_threads:    int   = 0
        self.inter_threads:  int   = 0
        self.ort_version:    str   = "N/A"
        self.cpu_name:       str   = "Unknown CPU"
        self.cpu_count:      int   = 0
        self.cpu_count_phys: int   = 0
        self.platform:       str   = platform.platform()
        self.python_version: str   = platform.python_version()

        # Fixed labels (CPU edition — never changes)
        self.backend          = BACKEND_LABEL
        self.provider         = PROVIDER_LABEL
        self.device           = DEVICE_LABEL
        self.inference_device = DEVICE_LABEL

        self._detect()

    # ── Detection ─────────────────────────────────────────────────────────────
    def _detect(self):
        self._detect_ort()
        self._detect_cpu()
        self._resolve_threads()
        logger.info(
            f"BackendManager ready — backend={self.backend}  "
            f"provider={self.provider}  "
            f"threads(intra/inter)={self.num_threads}/{self.inter_threads}  "
            f"cpu='{self.cpu_name}'")

    def _detect_ort(self):
        try:
            import onnxruntime as ort
            self.ort_version = ort.__version__
            providers = ort.get_available_providers()
            if PROVIDER_LABEL not in providers:
                raise RuntimeError(
                    f"CPUExecutionProvider not found in ONNX Runtime. "
                    f"Available: {providers}. "
                    f"Reinstall onnxruntime: pip install onnxruntime>=1.18.0")
            logger.info(f"ONNX Runtime {self.ort_version} — providers: {providers}")
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is not installed. "
                "Run: pip install onnxruntime>=1.18.0") from exc

    def _detect_cpu(self):
        try:
            import psutil
            self.cpu_count      = psutil.cpu_count(logical=True)  or 1
            self.cpu_count_phys = psutil.cpu_count(logical=False) or 1
        except Exception:
            import os
            self.cpu_count      = os.cpu_count() or 1
            self.cpu_count_phys = self.cpu_count

        try:
            import cpuinfo  # optional: py-cpuinfo
            info = cpuinfo.get_cpu_info()
            self.cpu_name = info.get("brand_raw", "Unknown CPU")
        except Exception:
            self.cpu_name = platform.processor() or "Unknown CPU"

    def _resolve_threads(self):
        # For inference: use physical cores (avoids hyperthreading contention)
        if self._req_intra > 0:
            self.num_threads = self._req_intra
        else:
            self.num_threads = max(1, self.cpu_count_phys)

        if self._req_inter > 0:
            self.inter_threads = self._req_inter
        else:
            self.inter_threads = max(1, self.num_threads // 2)

    # ── Session builder ───────────────────────────────────────────────────────
    def build_ort_session(self, onnx_path: str):
        """
        Build an ORT InferenceSession with full CPU optimisation.
        Raises RuntimeError with a clear message if anything fails.
        Never falls back silently.
        """
        import onnxruntime as ort

        path = Path(onnx_path)
        if not path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        if path.stat().st_size < 100_000:
            raise ValueError(
                f"ONNX file appears truncated ({path.stat().st_size} bytes): "
                f"{onnx_path}")

        opts = ort.SessionOptions()

        # Maximum graph-level optimisation
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Thread configuration
        opts.intra_op_num_threads = self.num_threads
        opts.inter_op_num_threads = self.inter_threads

        # Memory: use pre-allocated arena
        opts.enable_cpu_mem_arena = True
        opts.enable_mem_pattern   = True
        opts.enable_mem_reuse     = True

        # Execution mode: sequential (single model, no parallel ops needed)
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        logger.info(
            f"Building ORT session: {path.name}  "
            f"provider={PROVIDER_LABEL}  "
            f"intra={self.num_threads}  inter={self.inter_threads}")

        try:
            session = ort.InferenceSession(
                str(path),
                sess_options=opts,
                providers=[PROVIDER_LABEL])
        except Exception as exc:
            raise RuntimeError(
                f"ONNX Runtime failed to load model '{path.name}': {exc}\n"
                f"Check that the ONNX file is valid and not corrupted.\n"
                f"Re-export with: python -c \"from ultralytics import YOLO; "
                f"YOLO('models/yolo26n.pt').export(format='onnx')\"") from exc

        # Verify provider
        active = session.get_providers()
        if PROVIDER_LABEL not in active:
            raise RuntimeError(
                f"CPUExecutionProvider is not active. Active: {active}")

        logger.info(f"ORT session ready — active providers: {active}")
        return session

    # ── Diagnostics ───────────────────────────────────────────────────────────
    def get_diagnostics(self) -> dict:
        return {
            "backend":          self.backend,
            "provider":         self.provider,
            "device":           self.device,
            "model":            MODEL_LABEL,
            "ort_version":      self.ort_version,
            "cpu_name":         self.cpu_name,
            "cpu_logical":      self.cpu_count,
            "cpu_physical":     self.cpu_count_phys,
            "intra_threads":    self.num_threads,
            "inter_threads":    self.inter_threads,
            "platform":         self.platform,
            "python_version":   self.python_version,
        }

    # ── Startup log ───────────────────────────────────────────────────────────
    def write_startup_log(self, log_dir: str = "logs"):
        """Write startup.log — called once at application launch."""
        try:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            d = self.get_diagnostics()
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            lines = [
                "=" * 60,
                "  DualVision AI Detector — v1.3 Stable CPU Edition",
                "  Startup Diagnostics",
                "=" * 60,
                f"  Timestamp      : {ts}",
                "",
                "  INFERENCE BACKEND",
                f"  Backend        : {d['backend']}",
                f"  Provider       : {d['provider']}",
                f"  Device         : {d['device']}",
                f"  Model          : {d['model']}",
                "",
                "  ONNX RUNTIME",
                f"  ORT Version    : {d['ort_version']}",
                "",
                "  CPU",
                f"  CPU Name       : {d['cpu_name']}",
                f"  Logical Cores  : {d['cpu_logical']}",
                f"  Physical Cores : {d['cpu_physical']}",
                f"  Intra Threads  : {d['intra_threads']}",
                f"  Inter Threads  : {d['inter_threads']}",
                "",
                "  SYSTEM",
                f"  Platform       : {d['platform']}",
                f"  Python         : {d['python_version']}",
                "=" * 60,
            ]
            log_path = Path(log_dir) / "startup.log"
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            logger.info(f"Startup log written: {log_path}")
        except Exception as e:
            logger.warning(f"Could not write startup log: {e}")
