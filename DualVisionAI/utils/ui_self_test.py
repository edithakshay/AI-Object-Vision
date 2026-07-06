"""
DualVision AI Detector — Startup UI Self-Test
v1.3 Stable CPU Edition

Verifies all UI widgets were created, all panels are visible, no layout errors.
Writes results to logs/ui_check.log.
"""

import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("DualVisionAI.ui_selftest")


class UIResult:
    def __init__(self, name: str):
        self.name    = name
        self.passed  = 0
        self.failed  = 0
        self.entries = []

    def check(self, label: str, condition: bool, detail: str = ""):
        status = "PASS" if condition else "FAIL"
        msg    = f"  [{status}] {label}"
        if detail:
            msg += f" — {detail}"
        self.entries.append(msg)
        if condition:
            self.passed += 1
        else:
            self.failed += 1
        if not condition:
            logger.warning(f"UI self-test FAIL: {label} {detail}")

    @property
    def ok(self) -> bool:
        return self.failed == 0


def _widget_exists(widget) -> bool:
    try:
        return widget is not None and widget.winfo_exists()
    except Exception:
        return False


def _widget_visible(widget) -> bool:
    try:
        return widget is not None and widget.winfo_viewable()
    except Exception:
        return False


def run_ui_selftest(main_window) -> UIResult:
    """
    Run the startup self-test against the fully constructed MainWindow.
    Returns a UIResult with pass/fail counts.
    Writes everything to logs/ui_check.log.
    """
    result = UIResult("DualVision AI UI Self-Test")

    mw = main_window   # alias

    # ── 1. Top-level window ───────────────────────────────────────────────────
    try:
        result.check("Main window exists",   _widget_exists(mw))
        result.check("Main window visible",  _widget_visible(mw))
        result.check("Window title set",
                     bool(mw.title()),
                     mw.title())
        result.check("Window width >= 900",
                     mw.winfo_width() >= 900 or mw.winfo_reqwidth() >= 900)
        result.check("Window height >= 600",
                     mw.winfo_height() >= 600 or mw.winfo_reqheight() >= 600)
    except Exception as e:
        result.check("Main window check", False, str(e))

    # ── 2. Toolbar ────────────────────────────────────────────────────────────
    try:
        tb = mw._toolbar
        result.check("Toolbar exists",        _widget_exists(tb))
        result.check("Start button exists",   _widget_exists(getattr(tb, "_btn_start", None)))
        result.check("Stop button exists",    _widget_exists(getattr(tb, "_btn_stop",  None)))
        result.check("Pause button exists",   _widget_exists(getattr(tb, "_btn_pause", None)))
        result.check("Record button exists",  _widget_exists(getattr(tb, "_btn_rec",   None)))
        result.check("Toolbar is packed",     _widget_visible(tb))
    except Exception as e:
        result.check("Toolbar check", False, str(e))

    # ── 3. Camera panels ──────────────────────────────────────────────────────
    try:
        rgb = mw._rgb_panel
        th  = mw._thermal_panel
        result.check("RGB camera panel exists",     _widget_exists(rgb))
        result.check("Thermal camera panel exists", _widget_exists(th))
    except Exception as e:
        result.check("Camera panel check", False, str(e))

    # ── 4. Control panel ──────────────────────────────────────────────────────
    try:
        cp = mw._control_panel
        result.check("Control panel exists",          _widget_exists(cp))
        result.check("RGB radio button exists",        _widget_exists(getattr(cp, "_rb_rgb",           None)))
        result.check("Thermal radio button exists",    _widget_exists(getattr(cp, "_rb_thermal",       None)))
        result.check("Camera status label exists",    _widget_exists(getattr(cp, "_cam_status_label",  None)))
        result.check("CPU usage var exists",           getattr(cp, "_cpu_var",          None) is not None)
        result.check("RAM usage var exists",           getattr(cp, "_ram_var",          None) is not None)
        result.check("Infer FPS var exists",           getattr(cp, "_fps_var",          None) is not None)
        result.check("Avg FPS var exists",             getattr(cp, "_avg_fps_var",      None) is not None)
        result.check("Capture FPS var exists",         getattr(cp, "_cap_fps_var",      None) is not None)
        result.check("Display FPS var exists",         getattr(cp, "_disp_fps_var",     None) is not None)
        result.check("Total ms var exists",            getattr(cp, "_inf_ms_var",       None) is not None)
        result.check("Preprocess ms var exists",       getattr(cp, "_pre_ms_var",       None) is not None)
        result.check("Infer ms var exists",            getattr(cp, "_infer_ms_var",     None) is not None)
        result.check("Postproc ms var exists",         getattr(cp, "_post_ms_var",      None) is not None)
        result.check("Threads var exists",             getattr(cp, "_thr_count_var",    None) is not None)
        result.check("Frame Queue var exists",         getattr(cp, "_q_size_var",       None) is not None)
        result.check("Drops var exists",               getattr(cp, "_drops_var",        None) is not None)
        result.check("Active Dets var exists",         getattr(cp, "_active_det_var",   None) is not None)
        result.check("Session Total var exists",       getattr(cp, "_session_var",      None) is not None)
        result.check("Camera var exists",              getattr(cp, "_camera_var",       None) is not None)
        result.check("Track IDs var exists",           getattr(cp, "_track_ids_var",    None) is not None)
        result.check("Active Tracks var exists",       getattr(cp, "_trk_active_var",   None) is not None)
        result.check("Lost Tracks var exists",         getattr(cp, "_trk_lost_var",     None) is not None)
        result.check("Recovered Tracks var exists",    getattr(cp, "_trk_recovered_var",None) is not None)
        result.check("New Tracks var exists",          getattr(cp, "_trk_new_var",      None) is not None)
        result.check("Avg Track Age var exists",       getattr(cp, "_trk_age_var",      None) is not None)
        result.check("Tracking FPS var exists",        getattr(cp, "_trk_fps_var",      None) is not None)
        result.check("Track Latency var exists",       getattr(cp, "_trk_latency_var",  None) is not None)
        result.check("Detection log textbox exists",   _widget_exists(getattr(cp, "_log_text",          None)))
        result.check("Scrollable frame exists",        _widget_exists(getattr(cp, "_scroll",            None)))
        result.check("Recording status label exists",  _widget_exists(getattr(cp, "_rec_status",        None)))
        result.check("Recording path label exists",    _widget_exists(getattr(cp, "_rec_path",          None)))
    except Exception as e:
        result.check("Control panel check", False, str(e))

    # ── 5. Status bar ─────────────────────────────────────────────────────────
    try:
        sb = mw._statusbar
        result.check("Status bar exists",         _widget_exists(sb))
        result.check("FPS label var exists",       "fps"            in getattr(sb, "_vars", {}))
        result.check("Model label var exists",     "model"          in getattr(sb, "_vars", {}))
        result.check("Device label var exists",    "device"         in getattr(sb, "_vars", {}))
        result.check("Det status var exists",      "det_status"     in getattr(sb, "_vars", {}))
        result.check("RGB status var exists",      "rgb_status"     in getattr(sb, "_vars", {}))
        result.check("Thermal status var exists",  "thermal_status" in getattr(sb, "_vars", {}))
        result.check("Clock var exists",           "time"           in getattr(sb, "_vars", {}))
        result.check("Resolution var exists",      "resolution"     in getattr(sb, "_vars", {}))
    except Exception as e:
        result.check("Status bar check", False, str(e))

    # ── 6. Services / trackers ────────────────────────────────────────────────
    try:
        result.check("RGB stream exists",       mw._rgb_stream    is not None)
        result.check("Thermal stream exists",   mw._thermal_stream is not None)
        result.check("RGB tracker exists",      mw._rgb_tracker   is not None)
        result.check("Thermal tracker exists",  mw._thermal_tracker is not None)
        result.check("Screenshot util exists",  mw._screenshot_util is not None)
        result.check("Video recorder exists",   mw._recorder      is not None)
        result.check("Settings exists",         mw._settings      is not None)
        result.check("Backend manager exists",  mw._backend_manager is not None)
    except Exception as e:
        result.check("Services check", False, str(e))

    # ── 7. Logs directory ─────────────────────────────────────────────────────
    try:
        log_dir = Path(mw._settings.get("logging", "output_dir", "logs"))
        result.check("Logs directory exists", log_dir.exists(),
                     str(log_dir.resolve()))
    except Exception as e:
        result.check("Logs directory check", False, str(e))

    _write_log(result)
    return result


def _write_log(result: UIResult):
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "ui_check.log"

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 70,
        f"DualVision AI — UI Self-Test Report",
        f"Timestamp : {now}",
        f"Result    : {'PASSED' if result.ok else 'FAILED'}",
        f"Checks    : {result.passed + result.failed}  "
        f"(PASS={result.passed}  FAIL={result.failed})",
        "=" * 70,
        "",
    ] + result.entries + [
        "",
        f"{'All checks passed.' if result.ok else f'{result.failed} check(s) FAILED — review above.'}",
        "",
    ]

    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        logger.info(
            f"UI self-test complete — "
            f"PASS={result.passed}  FAIL={result.failed}  "
            f"log={path}")
    except Exception as e:
        logger.warning(f"Could not write ui_check.log: {e}")
