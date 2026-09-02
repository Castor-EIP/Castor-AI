# modules/foot_ai/detector.py
import time

import cv2
import torch
from ultralytics import YOLO

from .benchmark import BENCHMARK, Timer
from .constants import (
    BALL_CONFIDENCE,
    DETECTION_IMAGE_SIZE,
    FOOT_BALL_CLASS_ID,
    MODEL_PATH,
    PREDICTION_CONFIDENCE,
)


class BallDetector:
    def __init__(self):
        init_start = time.perf_counter_ns()

        with Timer("startup.cuda_stream", warmup=False):
            self.stream = torch.cuda.Stream()

        with Timer("startup.model_load", warmup=False):
            self.model = YOLO(str(MODEL_PATH))

        model_cuda_start = time.perf_counter_ns()
        self.model.to("cuda")
        torch.cuda.synchronize()
        BENCHMARK.record(
            "startup.model_to_cuda",
            (time.perf_counter_ns() - model_cuda_start) / 1_000_000.0,
            warmup=False,
        )

        BENCHMARK.record(
            "startup.detector_total",
            (time.perf_counter_ns() - init_start) / 1_000_000.0,
            warmup=False,
        )

    def detect(self, frame, draw=True, stream_name: str | None = None):
        detector_start = time.perf_counter_ns()
        stream_prefix = f"detector.{stream_name}" if stream_name else None
        try:
            resize_start = time.perf_counter_ns()
            img = cv2.resize(frame, (DETECTION_IMAGE_SIZE, DETECTION_IMAGE_SIZE))
            resize_ms = (time.perf_counter_ns() - resize_start) / 1_000_000.0
            BENCHMARK.record("detector.resize", resize_ms)
            if stream_prefix:
                BENCHMARK.record(f"{stream_prefix}.resize", resize_ms)

            rgb_start = time.perf_counter_ns()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb_ms = (time.perf_counter_ns() - rgb_start) / 1_000_000.0
            BENCHMARK.record("detector.bgr_to_rgb", rgb_ms)
            if stream_prefix:
                BENCHMARK.record(f"{stream_prefix}.bgr_to_rgb", rgb_ms)

            # Synchronization is deliberate: it makes the measured inference
            # latency include the actual CUDA execution, which is what the C++
            # benchmark must reproduce.
            torch.cuda.synchronize()
            inference_start = time.perf_counter_ns()
            with torch.cuda.stream(self.stream):
                results = self.model.predict(
                    img,
                    conf=PREDICTION_CONFIDENCE,
                    imgsz=DETECTION_IMAGE_SIZE,
                    device=0,
                    half=True,
                    verbose=False,
                    show=False,
                )[0]
            self.stream.synchronize()
            inference_ms = (time.perf_counter_ns() - inference_start) / 1_000_000.0
            BENCHMARK.record("detector.inference", inference_ms)
            if stream_prefix:
                BENCHMARK.record(f"{stream_prefix}.inference", inference_ms)

            post_start = time.perf_counter_ns()
            scale_x = frame.shape[1] / DETECTION_IMAGE_SIZE
            scale_y = frame.shape[0] / DETECTION_IMAGE_SIZE
            ball_found = False

            for box in results.boxes:
                cls = int(box.cls)
                conf = float(box.conf)
                if cls == FOOT_BALL_CLASS_ID and conf >= BALL_CONFIDENCE:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)
                    if draw:
                        self._draw_detection(frame, x1, y1, x2, y2, conf)
                    ball_found = True

            post_ms = (time.perf_counter_ns() - post_start) / 1_000_000.0
            BENCHMARK.record("detector.postprocess", post_ms)
            if stream_prefix:
                BENCHMARK.record(f"{stream_prefix}.postprocess", post_ms)

            total_ms = (time.perf_counter_ns() - detector_start) / 1_000_000.0
            BENCHMARK.record("detector.total", total_ms)
            if stream_prefix:
                BENCHMARK.record(f"{stream_prefix}.total", total_ms)
            return frame, ball_found

        except Exception:
            import traceback
            print("\n========== DETECTOR ERROR ==========")
            traceback.print_exc()
            print("====================================\n")
            raise

    @staticmethod
    def _draw_detection(frame, x1, y1, x2, y2, conf):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Ball {conf:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
