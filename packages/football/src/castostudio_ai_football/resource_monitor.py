from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass

import torch

try:
    import psutil  # type: ignore
except ImportError:  # Optional dependency; Linux fallback is provided below.
    psutil = None


@dataclass
class ResourceSample:
    cpu_process_percent: float | None
    ram_process_mb: float | None
    ram_system_percent: float | None
    gpu_util_percent: float | None
    gpu_vram_used_mb: float | None
    gpu_vram_total_mb: float | None
    torch_allocated_mb: float | None
    torch_reserved_mb: float | None
    torch_peak_allocated_mb: float | None


class ResourceMonitor:
    """Periodic, low-overhead resource monitor for the football AI process."""

    def __init__(self, interval_sec: float = 2.0) -> None:
        self.interval_sec = max(0.5, float(interval_sec))
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._samples: list[ResourceSample] = []

        self._process = psutil.Process(os.getpid()) if psutil is not None else None
        if self._process is not None:
            # Prime psutil so subsequent cpu_percent() calls are meaningful.
            self._process.cpu_percent(interval=None)

        self._fallback_last_wall = time.monotonic()
        self._fallback_last_cpu = time.process_time()
        self._nvidia_smi = shutil.which("nvidia-smi")

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="FootballResourceMonitor",
            daemon=True,
        )
        self._thread.start()
        print(
            "[RESOURCE] monitor_started "
            f"interval={self.interval_sec:.1f}s | "
            f"psutil={'yes' if psutil is not None else 'no'} | "
            f"nvidia_smi={'yes' if self._nvidia_smi else 'no'}"
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_sec + 1.0)
            self._thread = None
        self._print_summary()

    def _run(self) -> None:
        # Wait one interval before the first sample so CPU percentage reflects
        # real work over a full observation window.
        while not self._stop_event.wait(self.interval_sec):
            try:
                sample = self._collect_sample()
                self._samples.append(sample)
                self._print_sample(sample)
            except Exception as exc:
                # Monitoring must never crash or alter the AI pipeline.
                print(f"[RESOURCE] monitor_error={type(exc).__name__}: {exc}")

    def _collect_sample(self) -> ResourceSample:
        cpu_process_percent, ram_process_mb, ram_system_percent = self._read_cpu_ram()
        gpu_util_percent, gpu_vram_used_mb, gpu_vram_total_mb = self._read_gpu()
        torch_allocated_mb, torch_reserved_mb, torch_peak_allocated_mb = self._read_torch_cuda()

        return ResourceSample(
            cpu_process_percent=cpu_process_percent,
            ram_process_mb=ram_process_mb,
            ram_system_percent=ram_system_percent,
            gpu_util_percent=gpu_util_percent,
            gpu_vram_used_mb=gpu_vram_used_mb,
            gpu_vram_total_mb=gpu_vram_total_mb,
            torch_allocated_mb=torch_allocated_mb,
            torch_reserved_mb=torch_reserved_mb,
            torch_peak_allocated_mb=torch_peak_allocated_mb,
        )

    def _read_cpu_ram(self) -> tuple[float | None, float | None, float | None]:
        if self._process is not None and psutil is not None:
            cpu_percent = self._process.cpu_percent(interval=None)
            ram_mb = self._process.memory_info().rss / (1024 * 1024)
            system_ram_percent = psutil.virtual_memory().percent
            return cpu_percent, ram_mb, system_ram_percent

        return self._read_cpu_ram_linux_fallback()

    def _read_cpu_ram_linux_fallback(self) -> tuple[float | None, float | None, float | None]:
        cpu_percent: float | None = None
        ram_mb: float | None = None
        system_ram_percent: float | None = None

        now_wall = time.monotonic()
        now_cpu = time.process_time()
        wall_delta = now_wall - self._fallback_last_wall
        cpu_delta = now_cpu - self._fallback_last_cpu
        self._fallback_last_wall = now_wall
        self._fallback_last_cpu = now_cpu

        if wall_delta > 0:
            # Like psutil.Process.cpu_percent(), this can exceed 100% when the
            # process uses more than one CPU core.
            cpu_percent = (cpu_delta / wall_delta) * 100.0

        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            with open("/proc/self/statm", "r", encoding="utf-8") as handle:
                resident_pages = int(handle.read().split()[1])
            ram_mb = resident_pages * page_size / (1024 * 1024)
        except (OSError, ValueError, IndexError):
            pass

        try:
            meminfo: dict[str, int] = {}
            with open("/proc/meminfo", "r", encoding="utf-8") as handle:
                for line in handle:
                    key, value = line.split(":", 1)
                    meminfo[key] = int(value.strip().split()[0])
            total_kb = meminfo.get("MemTotal")
            available_kb = meminfo.get("MemAvailable")
            if total_kb and available_kb is not None:
                system_ram_percent = ((total_kb - available_kb) / total_kb) * 100.0
        except (OSError, ValueError):
            pass

        return cpu_percent, ram_mb, system_ram_percent

    def _read_gpu(self) -> tuple[float | None, float | None, float | None]:
        if self._nvidia_smi is None:
            return None, None, None

        try:
            completed = subprocess.run(
                [
                    self._nvidia_smi,
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                    "--id=0",
                ],
                capture_output=True,
                text=True,
                timeout=1.0,
                check=False,
            )
            if completed.returncode != 0:
                return None, None, None

            first_line = completed.stdout.strip().splitlines()[0]
            values = [value.strip() for value in first_line.split(",")]
            if len(values) != 3:
                return None, None, None

            return float(values[0]), float(values[1]), float(values[2])
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return None, None, None

    @staticmethod
    def _read_torch_cuda() -> tuple[float | None, float | None, float | None]:
        if not torch.cuda.is_available():
            return None, None, None

        mb = 1024 * 1024
        return (
            torch.cuda.memory_allocated(0) / mb,
            torch.cuda.memory_reserved(0) / mb,
            torch.cuda.max_memory_allocated(0) / mb,
        )

    @staticmethod
    def _fmt(value: float | None, suffix: str, decimals: int = 1) -> str:
        if value is None:
            return "n/a"
        return f"{value:.{decimals}f}{suffix}"

    def _print_sample(self, sample: ResourceSample) -> None:
        print(
            "[RESOURCE] "
            f"CPU_process={self._fmt(sample.cpu_process_percent, '%')} | "
            f"RAM_process={self._fmt(sample.ram_process_mb, ' MB')} | "
            f"RAM_system={self._fmt(sample.ram_system_percent, '%')} | "
            f"GPU_util={self._fmt(sample.gpu_util_percent, '%')} | "
            f"GPU_VRAM={self._fmt(sample.gpu_vram_used_mb, ' MB')}/"
            f"{self._fmt(sample.gpu_vram_total_mb, ' MB')} | "
            f"Torch_alloc={self._fmt(sample.torch_allocated_mb, ' MB')} | "
            f"Torch_reserved={self._fmt(sample.torch_reserved_mb, ' MB')} | "
            f"Torch_peak={self._fmt(sample.torch_peak_allocated_mb, ' MB')}"
        )

    def _print_summary(self) -> None:
        if not self._samples:
            print("[RESOURCE][SUMMARY] no_samples_collected")
            return

        def stats(attribute: str) -> tuple[float | None, float | None]:
            values = [
                getattr(sample, attribute)
                for sample in self._samples
                if getattr(sample, attribute) is not None
            ]
            if not values:
                return None, None
            return sum(values) / len(values), max(values)

        cpu_avg, cpu_max = stats("cpu_process_percent")
        ram_avg, ram_max = stats("ram_process_mb")
        gpu_avg, gpu_max = stats("gpu_util_percent")
        vram_avg, vram_max = stats("gpu_vram_used_mb")
        torch_avg, torch_max = stats("torch_allocated_mb")

        print("========== FOOTBALL AI RESOURCE SUMMARY ==========")
        print(f"samples           : {len(self._samples)}")
        print(
            f"CPU process       : avg={self._fmt(cpu_avg, '%')} | "
            f"max={self._fmt(cpu_max, '%')}"
        )
        print(
            f"RAM process       : avg={self._fmt(ram_avg, ' MB')} | "
            f"max={self._fmt(ram_max, ' MB')}"
        )
        print(
            f"GPU utilization   : avg={self._fmt(gpu_avg, '%')} | "
            f"max={self._fmt(gpu_max, '%')}"
        )
        print(
            f"GPU VRAM total    : avg={self._fmt(vram_avg, ' MB')} | "
            f"max={self._fmt(vram_max, ' MB')}"
        )
        print(
            f"PyTorch allocated : avg={self._fmt(torch_avg, ' MB')} | "
            f"max={self._fmt(torch_max, ' MB')}"
        )
        print("==================================================")
