"""
DualVision AI Detector — One-Time Model Downloader
====================================================
Run this script ONCE on your Windows machine:

    python download_all_models.py

Downloads all YOLOv8 and YOLO11 model files directly from GitHub
using Python's built-in urllib — NO torch / ultralytics import needed.
After this, the app runs 100% offline.

Model sizes (approximate):
    YOLOv8n  ~  6 MB    YOLO11n  ~  5 MB
    YOLOv8s  ~ 22 MB    YOLO11s  ~ 19 MB
    YOLOv8m  ~ 52 MB    YOLO11m  ~ 39 MB
    YOLOv8l  ~ 87 MB    YOLO11l  ~ 49 MB
    YOLOv8x  ~131 MB    YOLO11x  ~109 MB
    YOLO26n  ~  6 MB
    YOLO26s  ~ 20 MB
    YOLO26m  ~ 50 MB
    YOLO26l  ~ 85 MB
    YOLO26x  ~125 MB
    ─────────────────────────────────────
    TOTAL    ~~759 MB
"""

import sys
import os
import time
import shutil
from pathlib import Path
from urllib.request import urlretrieve, urlopen
from urllib.error import URLError, HTTPError

# ── Download URLs ─────────────────────────────────────────────────────────────
# Each entry: (filename, size_mb, full_url)
_V830 = "https://github.com/ultralytics/assets/releases/download/v8.3.0"
_V840 = "https://github.com/ultralytics/assets/releases/download/v8.4.0"

MODELS = [
    # (filename,       approx MB,  direct URL)
    ("yolov8n.pt",   6,  f"{_V830}/yolov8n.pt"),
    ("yolov8s.pt",  22,  f"{_V830}/yolov8s.pt"),
    ("yolov8m.pt",  52,  f"{_V830}/yolov8m.pt"),
    ("yolov8l.pt",  87,  f"{_V830}/yolov8l.pt"),
    ("yolov8x.pt", 131,  f"{_V830}/yolov8x.pt"),
    ("yolo11n.pt",   5,  f"{_V830}/yolo11n.pt"),
    ("yolo11s.pt",  19,  f"{_V830}/yolo11s.pt"),
    ("yolo11m.pt",  39,  f"{_V830}/yolo11m.pt"),
    ("yolo11l.pt",  49,  f"{_V830}/yolo11l.pt"),
    ("yolo11x.pt", 109,  f"{_V830}/yolo11x.pt"),
    ("yolo26n.pt",   6,  f"{_V840}/yolo26n.pt"),
    ("yolo26s.pt",  20,  f"{_V840}/yolo26s.pt"),
    ("yolo26m.pt",  50,  f"{_V840}/yolo26m.pt"),
    ("yolo26l.pt",  85,  f"{_V840}/yolo26l.pt"),
    ("yolo26x.pt", 125,  f"{_V840}/yolo26x.pt"),
]

MODEL_DIR = Path(__file__).parent / "models"

BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║         DualVision AI Detector — Model Downloader            ║
║  Direct download — no PyTorch import required.               ║
╚══════════════════════════════════════════════════════════════╝
"""


# ── Progress callback ─────────────────────────────────────────────────────────
_last_pct = [-1]

def _reporthook(block_num, block_size, total_size):
    if total_size <= 0:
        print(f"\r  {block_num * block_size // 1024} KB downloaded …",
              end="", flush=True)
        return
    downloaded = block_num * block_size
    pct = min(100, int(downloaded * 100 / total_size))
    if pct != _last_pct[0]:
        _last_pct[0] = pct
        bar_w = 30
        filled = pct * bar_w // 100
        bar = "█" * filled + "░" * (bar_w - filled)
        mb_done  = downloaded   / 1_048_576
        mb_total = total_size   / 1_048_576
        print(f"\r  [{bar}] {pct:3d}%  {mb_done:.1f}/{mb_total:.1f} MB",
              end="", flush=True)


# ── Downloader ────────────────────────────────────────────────────────────────
def download_model(name: str, size_mb: int, url: str) -> bool:
    dest = MODEL_DIR / name

    if dest.exists() and dest.stat().st_size > 100_000:
        mb = dest.stat().st_size / 1_048_576
        print(f"  [skip] {name:<14}  already cached  ({mb:.1f} MB)")
        return True
    print(f"  [↓] {name:<14}  ~{size_mb} MB")
    _last_pct[0] = -1
    t0 = time.time()

    try:
        tmp = dest.with_suffix(".tmp")
        urlretrieve(url, str(tmp), reporthook=_reporthook)
        tmp.rename(dest)
        elapsed = time.time() - t0
        mb = dest.stat().st_size / 1_048_576
        print(f"\r  [OK] {name:<14}  {mb:.1f} MB  in {elapsed:.0f}s" + " " * 20)
        return True

    except HTTPError as e:
        print(f"\r  [!!] HTTP {e.code} for {name}: {e.reason}")
        if tmp.exists():
            tmp.unlink()
        return False

    except URLError as e:
        print(f"\r  [!!] Network error for {name}: {e.reason}")
        if tmp.exists():
            tmp.unlink()
        return False

    except KeyboardInterrupt:
        print(f"\r  [--] Download cancelled by user.")
        tmp_path = dest.with_suffix(".tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    except Exception as e:
        print(f"\r  [!!] {name}: {e}")
        tmp_path = dest.with_suffix(".tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        return False


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(BANNER)
    print("[INFO] Downloads use Python's built-in urllib — no PyTorch needed.")
    print(f"[INFO] Saving models to: {MODEL_DIR.resolve()}\n")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Check internet connectivity
    print("[STEP 1] Checking internet connectivity …", end="", flush=True)
    try:
        urlopen("https://github.com", timeout=8)
        print(" OK\n")
    except Exception:
        print(" FAILED")
        print("\n  Cannot reach github.com — check your network connection.")
        input("\nPress Enter to exit …")
        sys.exit(1)

    # Download each model
    print(f"[STEP 2] Downloading {len(MODELS)} models …\n")
    results: dict[str, bool] = {}
    for i, (name, size_mb, url) in enumerate(MODELS, 1):
        print(f"  ({i}/{len(MODELS)})", end="  ", flush=True)
        try:
            results[name] = download_model(name, size_mb, url)
        except KeyboardInterrupt:
            print("\n\n  Download interrupted by user.")
            break

    # Summary
    passed = [k for k, v in results.items() if v]
    failed = [k for k, v in results.items() if not v]
    skipped = len(MODELS) - len(results)

    print("\n" + "=" * 62)
    print(f"  Downloaded / cached : {len(passed)}")
    if failed:
        print(f"  Failed              : {len(failed)}")
        for f in failed:
            print(f"    - {f}")
        print("\n  Re-run this script to retry failed models.")
    if skipped:
        print(f"  Interrupted (skipped): {skipped} model(s)")

    if not failed and not skipped:
        print("\n  All models ready!  You can now launch the app:")
        print("      python main.py")
        print("\n  The app is fully offline — no internet needed.")

    print("=" * 62)
    input("\nPress Enter to exit …")


if __name__ == "__main__":
    main()
