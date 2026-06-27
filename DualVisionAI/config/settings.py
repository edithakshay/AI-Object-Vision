import json
import os
from pathlib import Path

DEFAULT_SETTINGS = {
    "rtsp": {
        "rgb_url": "rtsp://192.168.144.108:554/stream=1",
        "thermal_url": "rtsp://192.168.144.108:555/stream=2",
        "reconnect_delay": 3,
        "buffer_size": 2,
        "timeout": 10
    },
    "detection": {
        "confidence": 0.45,
        "iou": 0.45,
        "input_width": 640,
        "input_height": 640,
        "frame_skip": 1,
        "max_fps": 60,
        "enable_tracking": True
    },
    "inference": {
        "use_gpu": True,
        "use_fp16": True,
        "model_name": "yolov8n.pt",
        "model_path": "models/yolov8n.onnx"
    },
    "ui": {
        "theme": "dark",
        "color_accent": "#2563EB",
        "window_width": 1600,
        "window_height": 950,
        "window_x": 100,
        "window_y": 100,
        "maximized": False
    },
    "recording": {
        "output_dir": "recordings",
        "codec": "mp4v",
        "fps": 25
    },
    "screenshots": {
        "output_dir": "screenshots",
        "format": "png"
    },
    "logging": {
        "output_dir": "logs",
        "csv_enabled": True,
        "max_log_entries": 100000
    }
}

CONFIG_PATH = Path("config/app_config.json")


class Settings:
    def __init__(self):
        self._data = {}
        self.load()

    def load(self):
        if CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                self._data = self._merge(DEFAULT_SETTINGS, loaded)
            except Exception:
                self._data = dict(DEFAULT_SETTINGS)
        else:
            self._data = dict(DEFAULT_SETTINGS)

    def save(self):
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    def get(self, section: str, key: str, fallback=None):
        return self._data.get(section, {}).get(key, fallback)

    def set(self, section: str, key: str, value):
        if section not in self._data:
            self._data[section] = {}
        self._data[section][key] = value

    def section(self, name: str) -> dict:
        return self._data.get(name, {})

    def _merge(self, default: dict, override: dict) -> dict:
        result = dict(default)
        for k, v in override.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = self._merge(result[k], v)
            else:
                result[k] = v
        return result
