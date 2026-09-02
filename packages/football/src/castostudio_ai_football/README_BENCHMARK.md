# Castor Football AI — Python speed benchmark

This version is dedicated to latency/speed benchmarking for a future Python vs C++ comparison.
It intentionally disables the resource-monitor thread and avoids per-frame performance prints so logging does not distort timings.

## What is measured

All steady-state timings are in milliseconds and use `time.perf_counter_ns()`.
CUDA is explicitly synchronized around YOLO inference so `detector.inference` measures real GPU completion time.

Key comparable metrics:

- `capture.STREAM_N.read`: OpenCV/FFmpeg frame read/decode latency (background capture thread)
- `capture.STREAM_N.get_latest`: lock + frame copy latency
- `detector.resize`: resize to the model input size
- `detector.bgr_to_rgb`: color conversion
- `detector.inference`: synchronized YOLO inference latency (all streams aggregated)
- `detector.STREAM_N.inference`: synchronized YOLO inference latency for one specific camera
- `detector.postprocess`: box/class/confidence processing
- `detector.total`: full per-camera detector latency
- `analyzer.copy_all_frames`: copy all current camera frames
- `analyzer.detect_all_streams`: sequential inference of all cameras
- `analyzer.STREAM_N.detect`: complete detector latency for one specific camera
- `analyzer.detect_cycle_N_inferences`: detection latency grouped by the exact number of inferences executed in that cycle
- `analyzer.total_cycle_N_inferences`: complete cycle latency grouped by exact inference count
- `analyzer.focus_decision`: camera-selection logic
- `analyzer.total`: complete one-analysis-cycle latency
- `module.analysis_loop`: analysis loop seen by the Castor module
- `module.analyze_sources_total`: complete module-call latency
- startup metrics: model load, CUDA transfer, stream opening, ready time

## Statistics

At shutdown the benchmark prints, for each metric:

- sample count
- arithmetic mean
- median / p50
- p95
- p99
- min
- max
- standard deviation (JSON)

The first **20 samples of every steady-state metric are discarded by default** to avoid model/CUDA warm-up bias.
Change it with:

```bash
CASTOR_BENCHMARK_WARMUP=50
```

The full result is also saved to:

```text
football_ai_benchmark_python.json
```

Override the output path with:

```bash
CASTOR_BENCHMARK_OUTPUT=benchmarks/python_2cams.json
```

## Recommended protocol for Python vs C++

For each implementation, keep exactly the same:

1. machine and GPU;
2. driver/CUDA environment as far as the runtime allows;
3. source video(s);
4. number of cameras;
5. model weights;
6. 416x416 detector input;
7. FP16 inference;
8. confidence thresholds;
9. frameskip;
10. test duration and warm-up rule.

Run at least three repetitions for each camera count (1, 2, then optionally 5 cameras). Compare p50 and p95 in addition to the mean.

For the future C++ benchmark, reproduce the same metric boundaries. In particular, do not compare asynchronous CUDA launch time against Python's synchronized inference measurement.

## Session isolation

The collector is reset automatically in `FootballModule.start()`. A one-camera run followed by a two-camera run in the same Python process no longer mixes their samples. This is mandatory for a fair Python/C++ comparison.
