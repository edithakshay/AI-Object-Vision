import cv2
import threading
import time
import queue
import logging
from enum import Enum

logger = logging.getLogger("DualVisionAI.camera")


class StreamStatus(Enum):
    CONNECTING = "Connecting"
    CONNECTED = "Connected"
    DISCONNECTED = "Disconnected"
    RECONNECTING = "Reconnecting"
    STOPPED = "Stopped"


class RTSPStream:
    def __init__(self, name: str, url: str, buffer_size: int = 2,
                 reconnect_delay: float = 3.0, timeout: int = 10):
        self.name = name
        self.url = url
        self.buffer_size = buffer_size
        self.reconnect_delay = reconnect_delay
        self.timeout = timeout

        self._cap = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._thread: threading.Thread | None = None
        self._running = False
        self._paused = False
        self._status = StreamStatus.DISCONNECTED
        self._lock = threading.Lock()

        self.fps_actual = 0.0
        self._frame_count = 0
        self._fps_timer = time.time()
        self._status_callback = None
        self._reconnect_callback = None   # called each time stream (re)connects

    def set_status_callback(self, callback):
        self._status_callback = callback

    def set_reconnect_callback(self, callback):
        """Register a callable(name: str) fired on every successful (re)connect.
        Used by MainWindow to reset FPS counters on reconnect.
        """
        self._reconnect_callback = callback

    def reset_fps(self):
        """Zero Capture FPS — call after camera switch or detection restart."""
        self.fps_actual   = 0.0
        self._frame_count = 0
        self._fps_timer   = time.time()

    def start(self):
        if self._running:
            return
        self.reset_fps()     # clean slate every time the stream starts
        self._running = True
        self._paused = False
        self._thread = threading.Thread(target=self._capture_loop,
                                        daemon=True, name=f"Stream-{self.name}")
        self._thread.start()
        logger.info(f"[{self.name}] Stream thread started: {self.url}")

    def stop(self):
        self._running = False
        self._set_status(StreamStatus.STOPPED)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._release_cap()
        with self._frame_queue.mutex:
            self._frame_queue.queue.clear()
        logger.info(f"[{self.name}] Stream stopped.")

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def read(self):
        try:
            return self._frame_queue.get_nowait()
        except queue.Empty:
            return None

    @property
    def status(self) -> StreamStatus:
        return self._status

    @property
    def is_connected(self) -> bool:
        return self._status == StreamStatus.CONNECTED

    def update_url(self, url: str):
        self.url = url

    def _set_status(self, status: StreamStatus):
        if self._status != status:
            self._status = status
            if self._status_callback:
                try:
                    self._status_callback(self.name, status)
                except Exception:
                    pass

    def _open_cap(self) -> bool:
        self._release_cap()
        self._set_status(StreamStatus.CONNECTING)

        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.timeout * 1000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.timeout * 1000)

        if cap.isOpened():
            self._cap = cap
            self._set_status(StreamStatus.CONNECTED)
            # Reset capture FPS so stale pre-reconnect values don't bleed in
            self.reset_fps()
            logger.info(f"[{self.name}] Connected to {self.url}")
            if self._reconnect_callback:
                try:
                    self._reconnect_callback(self.name)
                except Exception:
                    pass
            return True
        else:
            cap.release()
            self._set_status(StreamStatus.DISCONNECTED)
            logger.warning(f"[{self.name}] Failed to open {self.url}")
            return False

    def _release_cap(self):
        with self._lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None

    def _capture_loop(self):
        fail_count = 0
        while self._running:
            if self._paused:
                time.sleep(0.05)
                continue

            if self._cap is None or not self._cap.isOpened():
                self._set_status(StreamStatus.RECONNECTING)
                logger.info(f"[{self.name}] Reconnecting in {self.reconnect_delay}s ...")
                time.sleep(self.reconnect_delay)
                if not self._running:
                    break
                if not self._open_cap():
                    continue

            ret, frame = self._read_frame()
            if not ret or frame is None:
                fail_count += 1
                if fail_count > 10:
                    logger.warning(f"[{self.name}] Too many read failures, reconnecting.")
                    self._release_cap()
                    self._set_status(StreamStatus.RECONNECTING)
                    fail_count = 0
                time.sleep(0.05)
                continue

            fail_count = 0
            self._update_fps()

            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self._frame_queue.put_nowait(frame)
            except queue.Full:
                pass

        self._set_status(StreamStatus.STOPPED)

    def _read_frame(self):
        with self._lock:
            if self._cap is None:
                return False, None
            try:
                ret, frame = self._cap.read()
                return ret, frame
            except Exception as e:
                logger.error(f"[{self.name}] Read error: {e}")
                return False, None

    def _update_fps(self):
        self._frame_count += 1
        now = time.time()
        elapsed = now - self._fps_timer
        if elapsed >= 1.0:
            self.fps_actual = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_timer = now
