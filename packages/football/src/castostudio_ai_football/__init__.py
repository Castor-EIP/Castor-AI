from __future__ import annotations

import os
import platform
import sys
import time
from collections.abc import Sequence

import cv2
import torch
from castostudio_ai_core import AiModule, SceneDecision, SessionContext, Source

from .analyzer import FootballAnalyzer
from .benchmark import BENCHMARK


class FootballModule(AiModule):
    def __init__(self) -> None:
        self._analyzer: FootballAnalyzer | None = None
        self._confidence = 1.0
        self._frameskip = 0
        self._scene_ids_by_stream: dict[str, str] = {}
        self._last_sent_scene_id: str | None = None

    async def start(self, context: SessionContext) -> None:
        # A Castor process may start/stop several AI sessions without being restarted.
        # Never mix measurements from two sessions in the same benchmark summary.
        BENCHMARK.reset()

        self._confidence = float(context.config.get("default_confidence", 1.0))
        self._frameskip = int(context.config.get("frameskip", 0))
        BENCHMARK.set_metadata(
            language="Python",
            python=sys.version.split()[0],
            platform=platform.platform(),
            opencv=cv2.__version__,
            pytorch=torch.__version__,
            cuda=torch.version.cuda,
            device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            detection_image_size=416,
        )

    async def analyze_sources(self, sources: Sequence[Source]) -> SceneDecision | None:
        module_start = time.perf_counter_ns()

        if len(sources) < 1:
            BENCHMARK.record(
                "module.analyze_sources_total",
                (time.perf_counter_ns() - module_start) / 1_000_000.0,
            )
            return None

        if self._analyzer is None:
            self._scene_ids_by_stream = {
                f"STREAM_{index + 1}": source.scene_id
                for index, source in enumerate(sources)
            }
            init_start = time.perf_counter_ns()
            self._analyzer = FootballAnalyzer(
                stream_urls=[source.url for source in sources],
                frameskip=self._frameskip,
            )
            self._analyzer.start()
            ready = self._analyzer.wait_until_ready(timeout_sec=5.0)
            BENCHMARK.record(
                "startup.full_pipeline_ready",
                (time.perf_counter_ns() - init_start) / 1_000_000.0,
                warmup=False,
            )
            if not ready:
                return None
        elif len(sources) != self._analyzer.stream_count:
            return None

        analysis_loop_start = time.perf_counter_ns()
        focus = None
        for _ in range(30):
            focus = self._analyzer.analyze_once()
            if focus is not None:
                break
        BENCHMARK.record(
            "module.analysis_loop",
            (time.perf_counter_ns() - analysis_loop_start) / 1_000_000.0,
        )

        decision_start = time.perf_counter_ns()
        scene_id = self._scene_ids_by_stream.get(focus) if focus is not None else None
        if scene_id is None or scene_id == self._last_sent_scene_id:
            BENCHMARK.record(
                "module.scene_decision",
                (time.perf_counter_ns() - decision_start) / 1_000_000.0,
            )
            BENCHMARK.record(
                "module.analyze_sources_total",
                (time.perf_counter_ns() - module_start) / 1_000_000.0,
            )
            return None

        self._last_sent_scene_id = scene_id
        decision = SceneDecision(scene_id=scene_id, confidence=self._confidence)
        BENCHMARK.record(
            "module.scene_decision",
            (time.perf_counter_ns() - decision_start) / 1_000_000.0,
        )
        BENCHMARK.record(
            "module.analyze_sources_total",
            (time.perf_counter_ns() - module_start) / 1_000_000.0,
        )
        return decision

    async def stop(self) -> None:
        if self._analyzer is not None:
            self._analyzer.stop()
            self._analyzer = None
            self._scene_ids_by_stream.clear()

        BENCHMARK.print_summary()
        output_path = BENCHMARK.save_json()
        print(f"[BENCHMARK] JSON saved to: {output_path.resolve()}")
