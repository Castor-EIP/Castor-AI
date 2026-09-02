import threading
import time

import cv2

from .benchmark import BENCHMARK, Timer
from .constants import READ_RETRY_DELAY_SEC


class LatestFrameCapture:
    def __init__(self, source: str, name: str):
        self.source = source
        self.name = name
        self.cap = None
        self.thread = None
        self.running = False

        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_ts = 0.0
        self.read_failures = 0
        self.total_frames = 0

    def open(self):
        open_start = time.perf_counter_ns()
        self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        open_ms = (time.perf_counter_ns() - open_start) / 1_000_000.0
        BENCHMARK.record(f"capture.{self.name}.open", open_ms, warmup=False)

        if not self.cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la source {self.name}: {self.source}")

        with Timer(f"capture.{self.name}.set_buffer", warmup=False):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def start(self):
        start_ns = time.perf_counter_ns()
        self.open()
        self.running = True
        self.thread = threading.Thread(
            target=self._reader_loop,
            name=f"CaptureThread-{self.name}",
            daemon=True,
        )
        self.thread.start()
        BENCHMARK.record(
            f"capture.{self.name}.start_total",
            (time.perf_counter_ns() - start_ns) / 1_000_000.0,
            warmup=False,
        )

    def _reader_loop(self):
        while self.running:
            read_start = time.perf_counter_ns()
            ret, frame = self.cap.read()
            BENCHMARK.record(
                f"capture.{self.name}.read",
                (time.perf_counter_ns() - read_start) / 1_000_000.0,
            )

            if not ret or frame is None or frame.size == 0:
                self.read_failures += 1
                time.sleep(READ_RETRY_DELAY_SEC)
                continue

            self.read_failures = 0
            self.total_frames += 1

            with Timer(f"capture.{self.name}.lock_write"):
                with self.lock:
                    self.latest_frame = frame
                    self.latest_ts = time.monotonic()

    def get_latest(self):
        start_ns = time.perf_counter_ns()
        with self.lock:
            if self.latest_frame is None:
                BENCHMARK.record(
                    f"capture.{self.name}.get_latest",
                    (time.perf_counter_ns() - start_ns) / 1_000_000.0,
                )
                return None, 0.0
            frame = self.latest_frame.copy()
            timestamp = self.latest_ts
        BENCHMARK.record(
            f"capture.{self.name}.get_latest",
            (time.perf_counter_ns() - start_ns) / 1_000_000.0,
        )
        return frame, timestamp

    def stop(self):
        stop_start = time.perf_counter_ns()
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
        BENCHMARK.record(
            f"capture.{self.name}.stop",
            (time.perf_counter_ns() - stop_start) / 1_000_000.0,
            warmup=False,
        )
