import time
from collections.abc import Sequence

from .benchmark import BENCHMARK, Timer
from .capture import LatestFrameCapture
from .constants import READ_RETRY_DELAY_SEC
from .detector import BallDetector


class FootballAnalyzer:
    """Analyze one or more football video streams with a shared ball detector."""

    def __init__(self, stream_urls: Sequence[str], frameskip: int = 0):
        if not stream_urls:
            raise ValueError("FootballAnalyzer requires at least one stream")

        init_start = time.perf_counter_ns()
        self.detector = BallDetector()
        self.stream_readers = [
            LatestFrameCapture(url, f"STREAM_{index + 1}")
            for index, url in enumerate(stream_urls)
        ]
        self.frameskip = frameskip
        self.frame_count = 0
        self.last_focus: str | None = None
        BENCHMARK.set_metadata(stream_count=len(self.stream_readers), frameskip=frameskip)
        BENCHMARK.record(
            "startup.analyzer_construct",
            (time.perf_counter_ns() - init_start) / 1_000_000.0,
            warmup=False,
        )

    @property
    def stream_count(self) -> int:
        return len(self.stream_readers)

    def wait_until_ready(self, timeout_sec: float = 5.0) -> bool:
        start_ns = time.perf_counter_ns()
        while (time.perf_counter_ns() - start_ns) / 1_000_000_000.0 < timeout_sec:
            frames = [reader.get_latest()[0] for reader in self.stream_readers]
            if all(frame is not None for frame in frames):
                BENCHMARK.record(
                    "startup.wait_streams_ready",
                    (time.perf_counter_ns() - start_ns) / 1_000_000.0,
                    warmup=False,
                )
                return True
            time.sleep(READ_RETRY_DELAY_SEC)
        BENCHMARK.record(
            "startup.wait_streams_ready",
            (time.perf_counter_ns() - start_ns) / 1_000_000.0,
            warmup=False,
        )
        return False

    def start(self) -> None:
        with Timer("startup.streams_start_total", warmup=False):
            for reader in self.stream_readers:
                reader.start()

    def stop(self) -> None:
        with Timer("shutdown.streams_stop_total", warmup=False):
            for reader in self.stream_readers:
                reader.stop()

    def analyze_once(self) -> str | None:
        total_start = time.perf_counter_ns()

        frames = []
        timestamps = []
        copy_all_start = time.perf_counter_ns()
        for reader in self.stream_readers:
            frame, timestamp = reader.get_latest()
            frames.append(frame)
            timestamps.append(timestamp)
        BENCHMARK.record(
            "analyzer.copy_all_frames",
            (time.perf_counter_ns() - copy_all_start) / 1_000_000.0,
        )

        if any(frame is None for frame in frames):
            time.sleep(READ_RETRY_DELAY_SEC)
            BENCHMARK.record(
                "analyzer.total_waiting",
                (time.perf_counter_ns() - total_start) / 1_000_000.0,
            )
            return None

        if self.frame_count % (self.frameskip + 1) != 0:
            self.frame_count += 1
            BENCHMARK.record(
                "analyzer.total_skipped",
                (time.perf_counter_ns() - total_start) / 1_000_000.0,
            )
            return self.last_focus

        detection_all_start = time.perf_counter_ns()
        found = []
        inference_count = 0
        for index, frame in enumerate(frames):
            stream_name = f"STREAM_{index + 1}"
            stream_detect_start = time.perf_counter_ns()
            _, ball_found = self.detector.detect(
                frame, draw=False, stream_name=stream_name
            )
            stream_detect_ms = (time.perf_counter_ns() - stream_detect_start) / 1_000_000.0
            BENCHMARK.record(f"analyzer.{stream_name}.detect", stream_detect_ms)
            found.append(ball_found)
            inference_count += 1

        detect_all_ms = (time.perf_counter_ns() - detection_all_start) / 1_000_000.0
        BENCHMARK.record("analyzer.detect_all_streams", detect_all_ms)
        BENCHMARK.record(
            f"analyzer.detect_cycle_{inference_count}_inferences",
            detect_all_ms,
        )

        decision_start = time.perf_counter_ns()
        previous_focus = self.last_focus
        detected_indices = [index for index, ball_found in enumerate(found) if ball_found]
        if len(detected_indices) == 1:
            self.last_focus = f"STREAM_{detected_indices[0] + 1}"
        # If none or several streams contain the ball, preserve previous focus.
        _ = previous_focus
        BENCHMARK.record(
            "analyzer.focus_decision",
            (time.perf_counter_ns() - decision_start) / 1_000_000.0,
        )

        self.frame_count += 1
        total_ms = (time.perf_counter_ns() - total_start) / 1_000_000.0
        BENCHMARK.record("analyzer.total", total_ms)
        BENCHMARK.record(
            f"analyzer.total_cycle_{inference_count}_inferences",
            total_ms,
        )
        return self.last_focus
