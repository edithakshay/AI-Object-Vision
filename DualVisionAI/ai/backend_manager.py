"""
Backend Manager — GPU/CUDA detection and inference backend selection.

Priority order:
  1. ONNX Runtime + CUDAExecutionProvider  (fastest on RTX)
  2. PyTorch CUDA                           (if ORT CUDA unavailable)
  3. ONNX Runtime + CPUExecutionProvider   (ONNX, CPU only)
  4. PyTorch CPU                            (last resort fallback)

Future TensorRT backend slot is already reserved as BACKEND_TENSORRT.
"""

import logging
import datetime
from pathlib import Path

logger = logging.getLogger("DualVisionAI.backend")

# ── Backend constants ─────────────────────────────────────────────────────────
BACKEND_ONNX_CUDA  = "ONNX Runtime CUDA"
BACKEND_ONNX_CPU   = "ONNX Runtime CPU"
BACKEND_PT_CUDA    = "PyTorch CUDA"
BACKEND_PT_CPU     = "PyTorch CPU"
BACKEND_TENSORRT   = "TensorRT"        # reserved — not yet implemented


class BackendManager:
    """
    Detects available hardware/drivers, selects the best inference backend,
    builds ONNX Runtime sessions with the correct Execution Provider,
    and exposes live GPU diagnostics for the dashboard.
    """

    def __init__(self, use_gpu: bool = True, use_fp16: bool = False,
                 max_vram_gb: float = 0.0, force_cpu: bool = False,
                 force_onnx: bool = True):
        self.use_gpu     = use_gpu and not force_cpu
        self.use_fp16    = use_fp16
        self.max_vram_gb = max_vram_gb
        self.force_onnx  = force_onnx

        # ── Detected hardware ─────────────────────────────────────────────────
        self.cuda_available:    bool  = False
        self.gpu_name:          str   = "N/A"
        self.gpu_vram_mb:       int   = 0
        self.cuda_version:      str   = "N/A"
        self.cudnn_version:     str   = "N/A"
        self.torch_version:     str   = "N/A"
        self.torch_cuda:        bool  = False
        self.ort_version:       str   = "N/A"
        self.ort_cuda_available:bool  = False
        self.driver_version:    str   = "N/A"

        # ── Selected backend ──────────────────────────────────────────────────
        self.backend:          str = BACKEND_PT_CPU
        self.device:           str = "cpu"          # "cpu" | "0" | "cuda:0"
        self.inference_device: str = "CPU"
        self.ort_active_provider: str = "CPUExecutionProvider"

        # ── CUDA error detail (shown in UI if CUDA fails) ────────────────────
        self.cuda_error: str = ""

        self._detect()

    # ── hardware detection ────────────────────────────────────────────────────
    def _detect(self):
        self._detect_torch()
        self._detect_ort()
        self._detect_driver()
        self._select_backend()

    def _detect_torch(self):
        try:
            import torch
            self.torch_version = torch.__version__
            self.torch_cuda    = torch.cuda.is_available()
            if self.torch_cuda:
                self.cuda_available = True
                self.gpu_name       = torch.cuda.get_device_name(0)
                props               = torch.cuda.get_device_properties(0)
                self.gpu_vram_mb    = props.total_memory // (1024 * 1024)
                self.cuda_version   = torch.version.cuda or "N/A"
                try:
                    self.cudnn_version = str(torch.backends.cudnn.version())
                except Exception:
                    pass
                logger.info(f"GPU detected: {self.gpu_name}  "
                            f"VRAM={self.gpu_vram_mb} MB  CUDA={self.cuda_version}")
            else:
                logger.info("torch.cuda.is_available() = False — using CPU.")
        except ImportError:
            logger.warning("PyTorch not installed — GPU detection skipped.")

    def _detect_ort(self):
        try:
            import onnxruntime as ort
            self.ort_version        = ort.__version__
            available               = ort.get_available_providers()
            self.ort_cuda_available = "CUDAExecutionProvider" in available
            logger.info(f"ONNX Runtime {self.ort_version} — "
                        f"providers: {available}")
            if not self.ort_cuda_available and self.cuda_available:
                logger.warning(
                    "CUDAExecutionProvider not found in ONNX Runtime. "
                    "Install onnxruntime-gpu (not onnxruntime) for GPU inference.")
        except ImportError:
            logger.warning("onnxruntime not installed.")

    def _detect_driver(self):
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                self.driver_version = str(gpus[0].driver)
        except Exception:
            pass

    def _select_backend(self):
        if not self.use_gpu:
            self.backend          = BACKEND_ONNX_CPU
            self.device           = "cpu"
            self.inference_device = "CPU (forced)"
            logger.info("Backend: ONNX CPU (GPU disabled by user)")
            return

        if self.cuda_available and self.ort_cuda_available:
            self.backend          = BACKEND_ONNX_CUDA
            self.device           = "0"
            self.inference_device = f"GPU — {self.gpu_name}"
            logger.info(f"Backend: {BACKEND_ONNX_CUDA}")
        elif self.cuda_available:
            # onnxruntime-gpu not installed — fall back to PyTorch CUDA
            self.backend          = BACKEND_PT_CUDA
            self.device           = "0"
            self.inference_device = f"GPU — {self.gpu_name}"
            logger.info(f"Backend: {BACKEND_PT_CUDA} "
                        "(install onnxruntime-gpu for better GPU performance)")
        else:
            self.backend          = BACKEND_ONNX_CPU
            self.device           = "cpu"
            self.inference_device = "CPU"
            if self.use_gpu:
                logger.warning("No CUDA GPU found — falling back to CPU.")
            logger.info(f"Backend: {BACKEND_ONNX_CPU}")

    # ── ORT session builder ───────────────────────────────────────────────────
    def build_ort_session(self, onnx_path: str, num_threads: int = 0):
        """
        Build an ONNX Runtime InferenceSession with the best available provider.
        Tries CUDAExecutionProvider first; logs exact error if it fails.
        Never silently hides why CUDA was skipped.
        """
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if num_threads > 0:
            opts.intra_op_num_threads = num_threads
            opts.inter_op_num_threads = max(1, num_threads // 2)

        providers = []
        cuda_provider_error = ""

        if self.use_gpu and self.cuda_available and self.ort_cuda_available:
            cuda_opts = {
                "device_id":              0,
                "arena_extend_strategy":  "kNextPowerOfTwo",
                "cudnn_conv_algo_search": "DEFAULT",
                "do_copy_in_default_stream": True,
            }
            if self.max_vram_gb > 0:
                cuda_opts["gpu_mem_limit"] = int(self.max_vram_gb * 1024 ** 3)
            if self.use_fp16:
                cuda_opts["enable_cuda_graph"] = True
            providers.append(("CUDAExecutionProvider", cuda_opts))

        providers.append("CPUExecutionProvider")

        try:
            session = ort.InferenceSession(
                onnx_path, sess_options=opts, providers=providers)
        except Exception as e:
            cuda_provider_error = str(e)
            logger.error(f"ORT session creation failed: {e}")
            # Last-resort: CPU only
            session = ort.InferenceSession(
                onnx_path, sess_options=opts,
                providers=["CPUExecutionProvider"])

        # Report which provider is actually active
        active = session.get_providers()[0] if session.get_providers() else "Unknown"
        self.ort_active_provider = active

        if active == "CUDAExecutionProvider":
            logger.info("ONNX Runtime active provider: CUDAExecutionProvider ✓")
        else:
            if self.use_gpu and self.cuda_available and self.ort_cuda_available:
                msg = (f"CUDAExecutionProvider requested but not active. "
                       f"Active: {active}. "
                       f"Error: {cuda_provider_error or 'unknown'}")
                self.cuda_error = msg
                logger.warning(msg)
            logger.info(f"ONNX Runtime active provider: {active}")

        return session

    # ── live GPU metrics ──────────────────────────────────────────────────────
    def get_live_gpu_metrics(self) -> dict:
        """Poll real-time GPU metrics via GPUtil. Returns dict."""
        metrics = {
            "gpu_load_pct":  0.0,
            "gpu_mem_used":  0,
            "gpu_mem_free":  0,
            "gpu_mem_total": self.gpu_vram_mb,
            "gpu_temp":      0.0,
            "gpu_power":     "N/A",
        }
        if not self.cuda_available:
            return metrics
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                g = gpus[0]
                metrics["gpu_load_pct"]  = round(g.load * 100, 1)
                metrics["gpu_mem_used"]  = int(g.memoryUsed)
                metrics["gpu_mem_free"]  = int(g.memoryFree)
                metrics["gpu_mem_total"] = int(g.memoryTotal)
                metrics["gpu_temp"]      = round(g.temperature, 1)
        except Exception:
            pass
        return metrics

    # ── diagnostics snapshot ──────────────────────────────────────────────────
    def get_diagnostics(self) -> dict:
        return {
            "gpu_name":         self.gpu_name,
            "gpu_vram_mb":      self.gpu_vram_mb,
            "cuda_available":   self.cuda_available,
            "cuda_version":     self.cuda_version,
            "cudnn_version":    self.cudnn_version,
            "torch_version":    self.torch_version,
            "torch_cuda":       self.torch_cuda,
            "ort_version":      self.ort_version,
            "ort_cuda":         self.ort_cuda_available,
            "ort_provider":     self.ort_active_provider,
            "backend":          self.backend,
            "device":           self.device,
            "inference_device": self.inference_device,
            "driver_version":   self.driver_version,
            "cuda_error":       self.cuda_error,
            "use_fp16":         self.use_fp16,
        }

    # ── startup log ───────────────────────────────────────────────────────────
    def write_startup_log(self, log_dir: str = "logs"):
        """Write GPU / backend diagnostics to logs/startup.log."""
        try:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            d = self.get_diagnostics()
            lines = [
                "=" * 58,
                "  DualVision AI Detector — Startup Diagnostics",
                "=" * 58,
                f"  Timestamp     : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "  HARDWARE",
                f"  GPU Name      : {d['gpu_name']}",
                f"  GPU VRAM      : {d['gpu_vram_mb']} MB",
                f"  Driver        : {d['driver_version']}",
                "",
                "  CUDA / cuDNN",
                f"  CUDA Available: {'YES' if d['cuda_available'] else 'NO'}",
                f"  CUDA Version  : {d['cuda_version']}",
                f"  cuDNN Version : {d['cudnn_version']}",
                "",
                "  PYTORCH",
                f"  Torch Version : {d['torch_version']}",
                f"  Torch CUDA    : {'YES' if d['torch_cuda'] else 'NO'}",
                "",
                "  ONNX RUNTIME",
                f"  ORT Version   : {d['ort_version']}",
                f"  ORT CUDA EP   : {'YES' if d['ort_cuda'] else 'NO'}",
                f"  ORT Provider  : {d['ort_provider']}",
                "",
                "  INFERENCE BACKEND",
                f"  Backend       : {d['backend']}",
                f"  Device        : {d['inference_device']}",
                f"  FP16          : {'YES' if d['use_fp16'] else 'NO'}",
            ]
            if d["cuda_error"]:
                lines += ["", f"  CUDA ERROR    : {d['cuda_error']}"]
            lines.append("=" * 58)

            log_path = Path(log_dir) / "startup.log"
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            logger.info(f"Startup log written: {log_path}")
        except Exception as e:
            logger.warning(f"Could not write startup log: {e}")
