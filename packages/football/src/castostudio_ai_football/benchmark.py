from __future__ import annotations

import json
import math
import os
import statistics
import threading
import time
from collections import defaultdict
from pathlib import Path


class BenchmarkCollector:
    """Thread-safe latency collector for Python/C++ comparison.

    Values are stored in milliseconds. The first N samples of each metric are
    ignored to remove CUDA/model warm-up effects from the steady-state figures.
    """

    def __init__(self, warmup_samples: int = 20) -> None:
        self.warmup_samples = warmup_samples
        self._values: dict[str, list[float]] = defaultdict(list)
        self._all_counts: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.metadata: dict[str, object] = {}

    def reset(self) -> None:
        """Start a fresh benchmark session without cumulative data from a prior Castor session."""
        with self._lock:
            self._values.clear()
            self._all_counts.clear()
            self.metadata.clear()
            self.started_at = time.time()

    def set_metadata(self, **kwargs: object) -> None:
        with self._lock:
            self.metadata.update(kwargs)

    def record(self, metric: str, elapsed_ms: float, *, warmup: bool = True) -> None:
        with self._lock:
            self._all_counts[metric] += 1
            if warmup and self._all_counts[metric] <= self.warmup_samples:
                return
            self._values[metric].append(float(elapsed_ms))

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]
        rank = (len(ordered) - 1) * percentile
        low = math.floor(rank)
        high = math.ceil(rank)
        if low == high:
            return ordered[low]
        fraction = rank - low
        return ordered[low] * (1.0 - fraction) + ordered[high] * fraction

    def summary(self) -> dict[str, object]:
        with self._lock:
            snapshot = {name: list(values) for name, values in self._values.items()}
            counts = dict(self._all_counts)
            metadata = dict(self.metadata)

        metrics: dict[str, dict[str, float | int]] = {}
        for name, values in sorted(snapshot.items()):
            if not values:
                continue
            metrics[name] = {
                "samples": len(values),
                "warmup_discarded": min(counts.get(name, 0), self.warmup_samples),
                "mean_ms": statistics.fmean(values),
                "median_ms": statistics.median(values),
                "p95_ms": self._percentile(values, 0.95),
                "p99_ms": self._percentile(values, 0.99),
                "min_ms": min(values),
                "max_ms": max(values),
                "stddev_ms": statistics.pstdev(values) if len(values) > 1 else 0.0,
            }

        analyzer = metrics.get("analyzer.total")
        if analyzer and analyzer["mean_ms"]:
            metadata["steady_state_analysis_per_second"] = 1000.0 / float(analyzer["mean_ms"])

        return {
            "benchmark": "Castor Football AI Python latency benchmark",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "warmup_samples_per_metric": self.warmup_samples,
            "duration_sec": time.time() - self.started_at,
            "metadata": metadata,
            "metrics": metrics,
        }

    def print_summary(self) -> None:
        data = self.summary()
        print("\n========== FOOTBALL AI SPEED BENCHMARK ==========")
        print(f"duration           : {data['duration_sec']:.1f} s")
        print(f"warmup / metric    : {data['warmup_samples_per_metric']} samples")
        metadata = data.get("metadata", {})
        if metadata:
            print("metadata           : " + " | ".join(f"{k}={v}" for k, v in metadata.items()))
        print("--------------------------------------------------")
        print(
            f"{'metric':29s} {'n':>6s} {'mean':>9s} {'p50':>9s} "
            f"{'p95':>9s} {'p99':>9s} {'min':>9s} {'max':>9s}"
        )
        for name, stats in data["metrics"].items():
            print(
                f"{name:29s} {stats['samples']:6d} "
                f"{stats['mean_ms']:8.2f}ms {stats['median_ms']:8.2f}ms "
                f"{stats['p95_ms']:8.2f}ms {stats['p99_ms']:8.2f}ms "
                f"{stats['min_ms']:8.2f}ms {stats['max_ms']:8.2f}ms"
            )
        print("==================================================\n")

    def save_json(self, path: str | Path | None = None) -> Path:
        if path is None:
            configured = os.getenv("CASTOR_BENCHMARK_OUTPUT", "football_ai_benchmark_python.json")
            path = Path(configured)
        else:
            path = Path(path)
        data = self.summary()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path


BENCHMARK = BenchmarkCollector(
    warmup_samples=int(os.getenv("CASTOR_BENCHMARK_WARMUP", "20"))
)


class Timer:
    def __init__(self, metric: str, *, warmup: bool = True):
        self.metric = metric
        self.warmup = warmup
        self.start_ns = 0

    def __enter__(self):
        self.start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            elapsed_ms = (time.perf_counter_ns() - self.start_ns) / 1_000_000.0
            BENCHMARK.record(self.metric, elapsed_ms, warmup=self.warmup)
        return False
