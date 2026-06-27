"""
DualVision AI Detector — First-Time Setup Script
Run this ONCE before launching main.py:

    python setup.py

This downloads yolov8n.pt into the models/ folder.
After this, the app runs completely offline forever.
"""
import sys
import os
from pathlib import Path

MODEL_DIR = Path("models")
MODEL_NAME = "yolov8n.pt"
MODEL_DEST = MODEL_DIR / MODEL_NAME

BANNER = """
╔══════════════════════════════════════════════════════╗
║        DualVision AI Detector — Setup Script         ║
║  Downloads the AI model once. Offline forever after. ║
╚══════════════════════════════════════════════════════╝
"""


def check_python():
    if sys.version_info < (3, 10):
        print("[ERROR] Python 3.10 or higher is required.")
        sys.exit(1)
    print(f"[OK] Python {sys.version_info.major}.{sys.version_info.minor}")


def check_ultralytics():
    try:
        import ultralytics
        print(f"[OK] ultralytics {ultralytics.__version__}")
        return True
    except ImportError:
        print("[ERROR] ultralytics not installed.")
        print("  Run: pip install -r requirements.txt")
        return False


def download_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_DEST.exists() and MODEL_DEST.stat().st_size > 100_000:
        print(f"[OK] Model already downloaded: {MODEL_DEST}")
        return True

    print(f"\n[DOWNLOADING] {MODEL_NAME} → {MODEL_DEST}")
    print("     (≈ 6 MB — internet required only this once)\n")

    try:
        from ultralytics import YOLO

        # YOLO auto-downloads if not found locally
        model = YOLO(MODEL_NAME)
        print()

        # Find where Ultralytics saved it and copy to our models/ folder
        candidates = [
            Path(MODEL_NAME),
            Path.home() / ".ultralytics" / "assets" / MODEL_NAME,
            Path.home() / ".cache" / "ultralytics" / MODEL_NAME,
            Path.home() / "AppData" / "Roaming" / "ultralytics" / MODEL_NAME,
            Path.home() / "AppData" / "Local" / "ultralytics" / MODEL_NAME,
        ]

        import shutil
        for src in candidates:
            if src.exists() and src.stat().st_size > 100_000:
                if src != MODEL_DEST:
                    shutil.copy2(str(src), str(MODEL_DEST))
                    print(f"[OK] Model saved to: {MODEL_DEST}")
                else:
                    print(f"[OK] Model already at: {MODEL_DEST}")
                return True

        # If we couldn't find the copied file, it may be in Ultralytics cache
        # The Detector will still find it via YOLO(model_name) on next run
        if not MODEL_DEST.exists():
            # Try direct urllib download as a final fallback
            print("[INFO] Trying direct download fallback ...")
            url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
            import urllib.request

            def progress(count, block, total):
                pct = min(100, count * block / total * 100)
                bar = "█" * int(pct / 4) + "░" * (25 - int(pct / 4))
                print(f"\r     [{bar}] {pct:.1f}%", end="", flush=True)

            urllib.request.urlretrieve(url, str(MODEL_DEST), progress)
            print(f"\n[OK] Downloaded: {MODEL_DEST}")

        return MODEL_DEST.exists()

    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print("\nManual download instructions:")
        print("  1. Visit: https://github.com/ultralytics/assets/releases/tag/v8.3.0")
        print("  2. Download: yolov8n.pt")
        print(f"  3. Place it in: {MODEL_DEST.resolve()}")
        return False


def verify_model():
    if not MODEL_DEST.exists():
        return False
    size_mb = MODEL_DEST.stat().st_size / 1_048_576
    print(f"[OK] Model verified: {MODEL_DEST} ({size_mb:.1f} MB)")
    return True


def main():
    print(BANNER)
    check_python()

    if not check_ultralytics():
        sys.exit(1)

    success = download_model()
    if not success or not verify_model():
        print("\n[FAILED] Setup did not complete successfully.")
        print("See manual instructions above.")
        input("\nPress Enter to exit ...")
        sys.exit(1)

    print("\n" + "=" * 54)
    print("  Setup complete! You can now run the app:")
    print()
    print("      python main.py")
    print()
    print("  The app is fully offline from this point on.")
    print("=" * 54 + "\n")
    input("Press Enter to exit ...")


if __name__ == "__main__":
    main()
