"""
DualVision AI Detector — First-Time Setup Script
Run this ONCE before launching main.py:

    python setup.py

This downloads yolo26n.pt into the models/ folder.
After this, the app runs completely offline forever.
"""
import sys
import os
from pathlib import Path

MODEL_DIR  = Path("models")
MODEL_NAME = "yolo26n.pt"
MODEL_DEST = MODEL_DIR / MODEL_NAME

BANNER = """
╔══════════════════════════════════════════════════════╗
║        DualVision AI Detector — Setup Script         ║
║     YOLO26 — CPU-Optimised Dual RTSP Detection       ║
║  Downloads the AI model once. Offline forever after. ║
╚══════════════════════════════════════════════════════╝
"""

_BASE_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0"
_DL_URL   = f"{_BASE_URL}/{MODEL_NAME}"


def check_python():
    if sys.version_info < (3, 10):
        print("[ERROR] Python 3.10 or higher is required.")
        sys.exit(1)
    print(f"[OK] Python {sys.version_info.major}.{sys.version_info.minor}")


def check_dependencies():
    ok = True
    for pkg, imp in [
        ("customtkinter", "customtkinter"),
        ("ultralytics",   "ultralytics"),
        ("opencv-python", "cv2"),
        ("Pillow",        "PIL"),
        ("numpy",         "numpy"),
        ("onnxruntime",   "onnxruntime"),
        ("psutil",        "psutil"),
    ]:
        try:
            __import__(imp)
            print(f"  [OK] {pkg}")
        except ImportError:
            print(f"  [MISSING] {pkg}  →  pip install {pkg}")
            ok = False
    if not ok:
        print("\n  Install all missing packages:")
        print("      pip install -r requirements.txt\n")
    return ok


def download_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_DEST.exists() and MODEL_DEST.stat().st_size > 100_000:
        print(f"[OK] Model already downloaded: {MODEL_DEST}")
        return True

    print(f"\n[DOWNLOADING] {MODEL_NAME} → {MODEL_DEST}")
    print("     (≈ 6 MB — internet required only this once)\n")

    import urllib.request

    def _progress(count, block, total):
        if total > 0:
            pct = min(100, count * block / total * 100)
            bar = "█" * int(pct / 4) + "░" * (25 - int(pct / 4))
            print(f"\r     [{bar}] {pct:.1f}%", end="", flush=True)

    try:
        tmp = MODEL_DEST.with_suffix(".tmp")
        urllib.request.urlretrieve(_DL_URL, str(tmp), _progress)
        tmp.rename(MODEL_DEST)
        print()
        return True
    except Exception as e:
        print(f"\n[ERROR] Direct download failed: {e}")

    # Fallback: Ultralytics auto-download
    print("[INFO] Trying Ultralytics auto-download fallback …")
    try:
        from ultralytics import YOLO
        YOLO(MODEL_NAME)
        import shutil
        candidates = [
            Path(MODEL_NAME),
            Path.home() / ".ultralytics" / "assets" / MODEL_NAME,
            Path.home() / ".cache"       / "ultralytics" / MODEL_NAME,
            Path.home() / "AppData" / "Roaming" / "ultralytics" / MODEL_NAME,
            Path.home() / "AppData" / "Local"   / "ultralytics" / MODEL_NAME,
        ]
        for src in candidates:
            if src.exists() and src.stat().st_size > 100_000:
                if src != MODEL_DEST:
                    shutil.copy2(str(src), str(MODEL_DEST))
                print(f"[OK] Model saved to: {MODEL_DEST}")
                return True
    except Exception as e2:
        print(f"[ERROR] Ultralytics fallback also failed: {e2}")

    print("\nManual download:")
    print(f"  1. Visit: {_BASE_URL}")
    print(f"  2. Download: {MODEL_NAME}")
    print(f"  3. Place it in: {MODEL_DEST.resolve()}")
    return False


def verify_model():
    if not MODEL_DEST.exists():
        return False
    size_mb = MODEL_DEST.stat().st_size / 1_048_576
    print(f"[OK] Model verified: {MODEL_DEST} ({size_mb:.1f} MB)")
    return True


def generate_icons():
    print("[ICON GENERATION]")
    try:
        from PIL import Image
        assets_dir = Path(__file__).parent / "assets"
        sys.path.insert(0, str(Path(__file__).parent))
        from assets.make_icon import generate
        generate(assets_dir)
        print("  [OK] Icons generated.")
    except ImportError:
        print("  [!] Pillow not installed — skipping icon generation.")
    except Exception as e:
        print(f"  [!] Icon generation skipped (non-fatal): {e}")


def check_gpu():
    print("[GPU CHECK]")
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
            print(f"  [OK] CUDA GPU: {name}  ({vram} MB VRAM)")
            print("       GPU will be used when 'Enable GPU' is ON in Settings.")
        else:
            print("  [INFO] No CUDA GPU detected — CPU inference mode.")
            print("         YOLO26 is highly optimised for CPU via ONNX Runtime.")
    except ImportError:
        print("  [INFO] PyTorch not installed — GPU check skipped.")
        print("         CPU inference via ONNX Runtime is fully supported.")


def main():
    print(BANNER)
    check_python()
    print()
    print("[DEPENDENCY CHECK]")
    check_dependencies()
    print()

    success = download_model()
    if not success or not verify_model():
        print("\n[FAILED] Setup did not complete successfully.")
        print("See manual instructions above.")
        input("\nPress Enter to exit …")
        sys.exit(1)

    print()
    print("-" * 54)
    generate_icons()
    print("-" * 54)
    print()
    print("-" * 54)
    check_gpu()
    print("-" * 54)

    print("\n" + "=" * 54)
    print("  Setup complete! You can now run the app:")
    print()
    print("      python main.py")
    print()
    print("  The app is fully offline from this point on.")
    print("=" * 54 + "\n")
    input("Press Enter to exit …")


if __name__ == "__main__":
    main()
