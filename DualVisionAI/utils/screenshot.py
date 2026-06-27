import cv2
import os
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("DualVisionAI.screenshot")


class ScreenshotUtil:
    def __init__(self, output_dir: str = "screenshots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, frame, camera_name: str = "camera") -> str | None:
        if frame is None:
            return None
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:20]
        filename = self.output_dir / f"screenshot_{camera_name}_{ts}.png"
        try:
            cv2.imwrite(str(filename), frame)
            logger.info(f"Screenshot saved: {filename}")
            return str(filename)
        except Exception as e:
            logger.error(f"Screenshot save failed: {e}")
            return None

    def save_both(self, rgb_frame, thermal_frame) -> list[str]:
        paths = []
        if rgb_frame is not None:
            p = self.save(rgb_frame, "RGB")
            if p:
                paths.append(p)
        if thermal_frame is not None:
            p = self.save(thermal_frame, "Thermal")
            if p:
                paths.append(p)
        return paths
