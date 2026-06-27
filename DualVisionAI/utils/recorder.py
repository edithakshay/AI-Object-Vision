import cv2
import threading
import queue
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("DualVisionAI.recorder")


class VideoRecorder:
    def __init__(self, output_dir: str = "recordings", fps: float = 25.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self._writers: dict[str, cv2.VideoWriter] = {}
        self._queues: dict[str, queue.Queue] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._active: dict[str, bool] = {}
        self._lock = threading.Lock()

    def start(self, name: str, width: int, height: int) -> str:
        with self._lock:
            if name in self._active and self._active[name]:
                return ""
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.output_dir / f"recording_{name}_{ts}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(filepath), fourcc, self.fps, (width, height))
            if not writer.isOpened():
                logger.error(f"VideoWriter failed for {name}")
                return ""
            self._writers[name] = writer
            self._queues[name] = queue.Queue(maxsize=60)
            self._active[name] = True
            t = threading.Thread(target=self._write_loop, args=(name,),
                                 daemon=True, name=f"Recorder-{name}")
            self._threads[name] = t
            t.start()
            logger.info(f"[Recorder] Started: {filepath}")
            return str(filepath)

    def write(self, name: str, frame):
        if not self._active.get(name):
            return
        try:
            self._queues[name].put_nowait(frame)
        except queue.Full:
            pass

    def stop(self, name: str):
        with self._lock:
            if not self._active.get(name):
                return
            self._active[name] = False
        if name in self._queues:
            self._queues[name].put(None)
        if name in self._threads:
            self._threads[name].join(timeout=5)
        with self._lock:
            if name in self._writers:
                self._writers[name].release()
                del self._writers[name]
        logger.info(f"[Recorder] Stopped: {name}")

    def stop_all(self):
        for name in list(self._active.keys()):
            self.stop(name)

    def is_recording(self, name: str) -> bool:
        return self._active.get(name, False)

    def _write_loop(self, name: str):
        q = self._queues[name]
        while True:
            frame = q.get()
            if frame is None:
                break
            with self._lock:
                if name in self._writers and self._writers[name].isOpened():
                    self._writers[name].write(frame)
