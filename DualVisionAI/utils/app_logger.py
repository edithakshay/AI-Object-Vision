"""
Centralised logging setup — DualVision AI Detector v1.3 Stable CPU Edition.

Creates four log files:
  logs/startup.log   — app startup, model loading, ONNX export
  logs/inference.log — per-inference timings and detection counts
  logs/camera.log    — stream connect / disconnect / reconnect events
  logs/debug.log     — everything (DEBUG level, full verbose trace)

Console output shows INFO and above.
Exceptions are never suppressed.
"""

import logging
import logging.handlers
import sys
from pathlib import Path


# ── module-level reference so callers can check if setup has run ──────────────
_configured = False


def setup_logging(log_dir: str = "logs", console_level=logging.INFO) -> None:
    global _configured
    if _configured:
        return

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fmt_verbose = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)-30s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")
    fmt_simple = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S")

    # ── debug.log — everything ────────────────────────────────────────────────
    _add_file_handler(root, f"{log_dir}/debug.log",   logging.DEBUG,   fmt_verbose)

    # ── startup.log — INFO+ filtered to startup logger ────────────────────────
    _add_file_handler_filtered(
        root, f"{log_dir}/startup.log", logging.INFO, fmt_verbose,
        prefixes=("DualVisionAI.main", "DualVisionAI.model",
                  "DualVisionAI.backend", "root"))

    # ── inference.log — INFO+ filtered to detector ────────────────────────────
    _add_file_handler_filtered(
        root, f"{log_dir}/inference.log", logging.INFO, fmt_simple,
        prefixes=("DualVisionAI.detector",))

    # ── camera.log — INFO+ filtered to camera ─────────────────────────────────
    _add_file_handler_filtered(
        root, f"{log_dir}/camera.log", logging.INFO, fmt_simple,
        prefixes=("DualVisionAI.camera",))

    # ── console — INFO+ ───────────────────────────────────────────────────────
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(fmt_simple)
    root.addHandler(ch)

    _configured = True
    logging.getLogger("DualVisionAI").info(
        "Logging initialised — writing to %s/", log_dir)


# ── helpers ───────────────────────────────────────────────────────────────────

def _add_file_handler(logger, path: str, level: int, formatter):
    h = logging.handlers.RotatingFileHandler(
        path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    h.setLevel(level)
    h.setFormatter(formatter)
    logger.addHandler(h)


def _add_file_handler_filtered(logger, path: str, level: int,
                                formatter, prefixes: tuple):
    h = logging.handlers.RotatingFileHandler(
        path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    h.setLevel(level)
    h.setFormatter(formatter)
    h.addFilter(_PrefixFilter(prefixes))
    logger.addHandler(h)


class _PrefixFilter(logging.Filter):
    def __init__(self, prefixes: tuple):
        super().__init__()
        self._prefixes = prefixes

    def filter(self, record: logging.LogRecord) -> bool:
        return any(record.name.startswith(p) for p in self._prefixes)
