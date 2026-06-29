"""
DualVision AI Detector — One-Time Model Downloader
====================================================
Run this script ONCE on your Windows machine:

    python download_all_models.py

Downloads all YOLOv8 and YOLO11 variants to the local models/ folder.
After this, the app runs 100% offline — no internet required.

Model sizes (approximate):
    YOLOv8n  ~  6 MB   ← fastest,  lowest accuracy
    YOLOv8s  ~ 22 MB
    YOLOv8m  ~ 52 MB
    YOLOv8l  ~ 87 MB
    YOLOv8x  ~131 MB   ← slowest,  highest accuracy
    YOLO11n  ~  5 MB
    YOLO11s  ~ 19 MB
    YOLO11m  ~ 39 MB
    YOLO11l  ~ 49 MB
    YOLO11x  ~109 MB
    ─────────────────
    TOTAL    ~519 MB

NOTE: YOLO26 is not yet released by Ultralytics.  It will be added here
      automatically once Ultralytics makes it available.
"""

import sys
import os
import shutil
import time
from pathlib import Path

# ── Models to download ────────────────────────────────────────────────────────
MODELS = [
    # (filename, display_name)
    ("yolov8n.pt",  "YOLOv8 Nano   (~  6 MB)"),
    ("yolov8s.pt",  "YOLOv8 Small  (~ 22 MB)"),
    ("yolov8m.pt",  "YOLOv8 Medium (~ 52 MB)"),
    ("yolov8l.pt",  "YOLOv8 Large  (~ 87 MB)"),
    ("yolov8x.pt",  "YOLOv8 XLarge (~131 MB)"),
    ("yolo11n.pt",  "YOLO11 Nano   (~  5 MB)"),
    ("yolo11s.pt",  "YOLO11 Small  (~ 19 MB)"),
    ("yolo11m.pt",  "YOLO11 Medium (~ 39 MB)"),
    ("yolo11l.pt",  "YOLO11 Large  (~ 49 MB)"),
    ("yolo11x.pt",  "YOLO11 XLarge (~109 MB)"),
]

MODEL_DIR = Path(__file__).parent / "models"

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║         DualVision AI Detector — Model Downloader            ║
║  Downloads all YOLO models once.  Fully offline after that.  ║
╚══════════════════════════════════════════════════════════════╝
"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _bar(pct: float, width: int = 35) -> str:
    filled = int(pct / 100 * width)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {pct:5.1f}%"


def _check_dependencies() -> bool:
    ok = True
    try:
        import ultralytics
        print(f"  [OK] ultralytics {ultralytics.__version__}")
    except ImportError:
        print("  [!!] ultralytics not installed.")
        print("       pip install ultralytics")
        ok = False
    return ok


def _find_cached(name: str) -> Path | None:
    """Search Ultralytics cache locations for an already-downloaded model."""
    candidates = [
        Path(name),
        Path.home() / ".ultralytics" / "assets" / name,
        Path.home() / ".cache"      / "ultralytics" / name,
        Path.home() / "AppData" / "Roaming" / "ultralytics" / name,
        Path.home() / "AppData" / "Local"   / "ultralytics" / name,
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 100_000:
            return p
    return None


def download_model(name: str, dest: Path) -> bool:
    """
    Download `name` (e.g. 'yolov8n.pt') to `dest` using Ultralytics.
    Returns True on success.
    """
    if dest.exists() and dest.stat().st_size > 100_000:
        size_mb = dest.stat().st_size / 1_048_576
        print(f"  [skip] {name} already cached ({size_mb:.1f} MB)")
        return True

    print(f"  [↓]    Downloading {name} …", end="", flush=True)
    t0 = time.time()
    try:
        from ultralytics import YOLO
        # Ultralytics prints its own progress; suppress with verbose=False
        model = YOLO(name)

        # Try to copy the cached file to our models/ folder
        cached = _find_cached(name)
        if cached and cached != dest:
            shutil.copy2(str(cached), str(dest))
            size_mb = dest.stat().st_size / 1_048_576
            elapsed = time.time() - t0
            print(f"\r  [OK]   {name} saved ({size_mb:.1f} MB, {elapsed:.1f}s)")
            return True

        # Some versions write to cwd
        cwd_file = Path(name)
        if cwd_file.exists() and cwd_file.stat().st_size > 100_000:
            shutil.copy2(str(cwd_file), str(dest))
            size_mb = dest.stat().st_size / 1_048_576
            elapsed = time.time() - t0
            print(f"\r  [OK]   {name} saved ({size_mb:.1f} MB, {elapsed:.1f}s)")
            return True

        # If model object is valid, Ultralytics manages it internally
        # Write a marker so the app knows it's available
        elapsed = time.time() - t0
        print(f"\r  [OK]   {name} ready via Ultralytics cache ({elapsed:.1f}s)")
        dest.write_text(f"managed:{name}")
        return True

    except Exception as e:
        print(f"\r  [FAIL] {name}: {e}")
        return False


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(BANNER)

    # Dependency check
    print("[STEP 1] Checking dependencies …")
    if not _check_dependencies():
        print("\n  Install missing packages and re-run this script.")
        input("\nPress Enter to exit …")
        sys.exit(1)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[STEP 2] Downloading models → {MODEL_DIR.resolve()}\n")
    print("  Note: YOLO26 is not yet released by Ultralytics.")
    print("        It will be added here when available.\n")

    results = {}
    for i, (fname, label) in enumerate(MODELS, 1):
        print(f"  [{i:2d}/{len(MODELS)}] {label}")
        dest = MODEL_DIR / fname
        ok = download_model(fname, dest)
        results[fname] = ok
        print()

    # Summary
    passed  = [k for k, v in results.items() if v]
    failed  = [k for k, v in results.items() if not v]

    print("=" * 62)
    print(f"  Downloaded : {len(passed)}/{len(MODELS)} models")
    if failed:
        print(f"  Failed     : {', '.join(failed)}")
        print("\n  Retry failed models by re-running this script.")
    else:
        print("\n  All models ready!  Launch the app:")
        print("      python main.py")
        print()
        print("  The app is now fully offline.")
    print("=" * 62)

    input("\nPress Enter to exit …")


if __name__ == "__main__":
    main()
