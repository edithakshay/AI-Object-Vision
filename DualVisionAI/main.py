"""
DualVision AI Detector — main entry point.
High-FPS Dual RTSP AI Object Detection
"""
import sys
import os
import time
import threading

# Ensure script directory is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logger import setup_logger
logger = setup_logger()

import customtkinter as ctk
from config.settings import Settings


def check_dependencies() -> list[str]:
    missing = []
    for pkg, import_name in [
        ("customtkinter", "customtkinter"),
        ("cv2", "cv2"),
        ("PIL", "PIL"),
        ("numpy", "numpy"),
        ("ultralytics", "ultralytics"),
        ("psutil", "psutil"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)
    return missing


def main():
    logger.info("=" * 60)
    logger.info("DualVision AI Detector v1.0.0 starting ...")
    logger.info("=" * 60)

    missing = check_dependencies()
    if missing:
        print("\n[ERROR] Missing dependencies:")
        for pkg in missing:
            print(f"  pip install {pkg}")
        print("\nRun: pip install -r requirements.txt\n")
        input("Press Enter to exit ...")
        sys.exit(1)

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    settings = Settings()

    # Splash screen + init sequence
    splash_root = ctk.CTk()
    splash_root.withdraw()

    from ui.splash_screen import SplashScreen
    splash = SplashScreen(splash_root)

    init_done = threading.Event()

    def _initialize():
        steps = [
            ("Loading configuration ...", 0.15),
            ("Initializing camera modules ...", 0.35),
            ("Preparing AI engine ...", 0.60),
            ("Loading tracking system ...", 0.80),
            ("Starting user interface ...", 0.95),
        ]
        for text, progress in steps:
            splash.set_status(text, progress)
            time.sleep(0.4)
        init_done.set()

    init_thread = threading.Thread(target=_initialize, daemon=True)
    init_thread.start()

    def _check_done():
        if init_done.is_set():
            splash.close()
            splash_root.destroy()
        else:
            splash_root.after(100, _check_done)

    splash_root.after(100, _check_done)
    splash_root.mainloop()

    logger.info("Launching main window ...")
    from ui.main_window import MainWindow
    app = MainWindow(settings)
    app.mainloop()

    logger.info("Application closed.")


if __name__ == "__main__":
    main()
