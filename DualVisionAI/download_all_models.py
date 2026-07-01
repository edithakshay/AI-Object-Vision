"""
DualVision AI Detector — YOLO26 Model Downloader
=================================================
Run ONCE on your Windows machine to download YOLO26 models:

    python download_all_models.py

Uses Python's built-in urllib — NO PyTorch or ultralytics import needed.
After downloading, the app runs 100% offline.

Model sizes (approximate):
    yolo26n  ~  6 MB   (fastest, recommended for CPU)
    yolo26s  ~ 20 MB
    yolo26m  ~ 50 MB
    yolo26l  ~ 85 MB
    yolo26x  ~125 MB   (most accurate)
"""

import sys
import time
from pathlib import Path
from urllib.request import urlretrieve, urlopen
from urllib.error import URLError, HTTPError

_BASE = "https://github.com/ultralytics/assets/releases/download/v8.4.0"

MODELS = [
    ("yolo26n.pt",   6, f"{_BASE}/yolo26n.pt"),
    ("yolo26s.pt",  20, f"{_BASE}/yolo26s.pt"),
    ("yolo26m.pt",  50, f"{_BASE}/yolo26m.pt"),
    ("yolo26l.pt",  85, f"{_BASE}/yolo26l.pt"),
    ("yolo26x.pt", 125, f"{_BASE}/yolo26x.pt"),
]

MODEL_DIR = Path(__file__).parent / "models"

BANNER = r"""
╔═══════════════════════════════════════════════════════════════╗
║      DualVision AI Detector — YOLO26 Model Downloader         ║
║   Powered by Ultralytics YOLO26  |  CPU-Optimized Edition     ║
╚═══════════════════════════════════════════════════════════════╝
"""

_last_pct = [-1]


def _reporthook(block_num, block_size, total_size):
    if total_size <= 0:
        print(f"\r  {block_num * block_size // 1024} KB …", end="", flush=True)
        return
    downloaded = block_num * block_size
    pct = min(100, int(downloaded * 100 / total_size))
    if pct != _last_pct[0]:
        _last_pct[0] = pct
        bar_w  = 30
        filled = pct * bar_w // 100
        bar    = "█" * filled + "░" * (bar_w - filled)
        done   = downloaded   / 1_048_576
        total  = total_size   / 1_048_576
        print(f"\r  [{bar}] {pct:3d}%  {done:.1f}/{total:.1f} MB",
              end="", flush=True)


def download_model(name: str, size_mb: int, url: str) -> bool:
    dest = MODEL_DIR / name
    if dest.exists() and dest.stat().st_size > 100_000:
        mb = dest.stat().st_size / 1_048_576
        print(f"  [skip] {name:<14}  already cached  ({mb:.1f} MB)")
        return True

    print(f"  [↓] {name:<14}  ~{size_mb} MB")
    _last_pct[0] = -1
    t0  = time.time()
    tmp = dest.with_suffix(".tmp")

    try:
        urlretrieve(url, str(tmp), reporthook=_reporthook)
        tmp.rename(dest)
        elapsed = time.time() - t0
        mb = dest.stat().st_size / 1_048_576
        print(f"\r  [OK] {name:<14}  {mb:.1f} MB  in {elapsed:.0f}s" + " " * 20)
        return True

    except HTTPError as e:
        print(f"\r  [!!] HTTP {e.code} — {e.reason}")
        if tmp.exists(): tmp.unlink()
        return False

    except URLError as e:
        print(f"\r  [!!] Network error: {e.reason}")
        if tmp.exists(): tmp.unlink()
        return False

    except KeyboardInterrupt:
        print("\r  [--] Cancelled by user.")
        if tmp.exists(): tmp.unlink()
        raise

    except Exception as e:
        print(f"\r  [!!] {name}: {e}")
        if tmp.exists(): tmp.unlink()
        return False


def main():
    print(BANNER)
    print("[INFO] Uses Python's built-in urllib — no PyTorch import needed.")
    print(f"[INFO] Saving to: {MODEL_DIR.resolve()}\n")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("[STEP 1] Checking internet …", end="", flush=True)
    try:
        urlopen("https://github.com", timeout=8)
        print(" OK\n")
    except Exception:
        print(" FAILED")
        print("\n  Cannot reach github.com — check your network connection.")
        input("\nPress Enter to exit …")
        sys.exit(1)

    print(f"[STEP 2] Downloading {len(MODELS)} YOLO26 models …\n")
    results: dict[str, bool] = {}
    for i, (name, size_mb, url) in enumerate(MODELS, 1):
        print(f"  ({i}/{len(MODELS)})", end="  ", flush=True)
        try:
            results[name] = download_model(name, size_mb, url)
        except KeyboardInterrupt:
            print("\n\n  Interrupted.")
            break

    passed  = [k for k, v in results.items() if v]
    failed  = [k for k, v in results.items() if not v]
    skipped = len(MODELS) - len(results)

    print("\n" + "=" * 63)
    print(f"  Downloaded / cached : {len(passed)}")
    if failed:
        print(f"  Failed              : {len(failed)}")
        for f in failed:
            print(f"    - {f}")
        print("\n  Re-run to retry failed models.")
    if skipped:
        print(f"  Interrupted (skipped): {skipped} model(s)")

    if not failed and not skipped:
        print("\n  All YOLO26 models ready!  Launch the app:")
        print("      python main.py")
        print("\n  Fully offline after this — no internet needed.")
        print("\n  TIP: On first Start Detection, the app exports each model")
        print("  to ONNX for maximum CPU performance (one-time, ~30 seconds).")

    print("=" * 63)
    input("\nPress Enter to exit …")


if __name__ == "__main__":
    main()
