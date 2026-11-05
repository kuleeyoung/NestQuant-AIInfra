"""Repeat one HTTP workload across fresh W4A4KV4 service processes.

Each trial starts a new model-serving process, performs explicit warmup requests,
samples NVIDIA GPU state, runs the existing open-loop HTTP benchmark, and then
stops the service.  The resulting summary distinguishes within-process load
trends from cross-process reproducibility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import threading
import time
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_http_service import run_request


GPU_QUERY_FIELDS = (
    "timestamp,name,pstate,clocks.sm,clocks.mem,temperature.gpu,"
    "power.draw,utilization.gpu,memory.used"
)
GPU_KEYS = (
    "timestamp", "name", "pstate", "sm_clock_mhz", "memory_clock_mhz",
    "temperature_c", "power_w", "utilization_percent", "memory_used_mib",
)
NUMERIC_GPU_KEYS = GPU_KEYS[3:]


def parse_nvidia_smi_line(line: str) -> dict[str, object]:
    """Parse one no-units CSV row emitted by the fixed NVIDIA query."""
    parts = [part.strip() for part in line.strip().split(",")]
    if len(parts) != len(GPU_KEYS):
        raise ValueError(
            f"expected {len(GPU_KEYS)} NVIDIA fields, received {len(parts)}: {line!r}"
        )
    parsed: dict[str, object] = dict(zip(GPU_KEYS[:3], parts[:3]))
    for key, value in zip(NUMERIC_GPU_KEYS, parts[3:]):
        parsed[key] = float(value)
    return parsed


def coefficient_of_variation(values: list[float]) -> float:
    """Return population standard deviation divided by the mean."""
    if len(values) < 2:
        return 0.0
    mean = statistics.mean(values)
    return statistics.pstdev(values) / mean if mean else 0.0


def summarize_trials(trials: list[dict]) -> dict:
    if not trials:
        raise ValueError("at least one trial is required")
    metric_paths = {
        "request_throughput_per_second": ("request_throughput_per_second",),
        "token_throughput_per_second": ("token_throughput_per_second",),
        "ttft_p50_ms": ("ttft_ms", "p50"),
        "ttft_p95_ms": ("ttft_ms", "p95"),
        "tpot_p50_ms": ("tpot_ms", "p50"),
        "tpot_p95_ms": ("tpot_ms", "p95"),
        "e2e_p50_ms": ("e2e_ms", "p50"),
        "e2e_p95_ms": ("e2e_ms", "p95"),
    }
    metrics = {}
    for metric, path in metric_paths.items():
        values = []
        for trial in trials:
            value: object = trial["benchmark"]
            for key in path:
                value = value[key]  # type: ignore[index]
            values.append(float(value))
        metrics[metric] = {
            "values": values,
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "cv": coefficient_of_variation(values),
        }
    return {
        "schema_version": 1,
        "trial_count": len(trials),
        "all_trials_passed": all(
            trial["benchmark"]["status"] == "passed"
            and trial["benchmark"]["failed_requests"] == 0
            for trial in trials
        ),
        "metrics": metrics,
        "claim_boundary": (
            "Fresh-process, single-host reproducibility experiment. It records "
            "GPU state and client-observed HTTP latency, but is not a production SLA."
        ),
    }


def _write_markdown(summary: dict, output: Path) -> None:
    lines = [
        "# W4A4KV4 HTTP fresh-process reproducibility",
        "",
        f"- fresh service processes: **{summary['trial_count']}**",
        f"- all trials passed: **{summary['all_trials_passed']}**",
        "",
        "| metric | median | min | max | CV |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metric in summary["metrics"].items():
        lines.append(
            f"| {name} | {metric['median']:.3f} | {metric['min']:.3f} | "
            f"{metric['max']:.3f} | {metric['cv'] * 100.0:.2f}% |"
        )
    lines.extend([
        "",
        "CV is the coefficient of variation: population standard deviation divided "
        "by the mean. Raw benchmark JSON, GPU telemetry and server logs are retained "
        "beside the summary.",
        "",
        f"Claim boundary: {summary['claim_boundary']}",
        "",
    ])
    output.write_text("\n".join(lines), encoding="utf-8")


def _wait_for_health(base_url: str, process: subprocess.Popen, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"service exited during startup with code {process.returncode}")
        try:
            with urlopen(base_url.rstrip("/") + "/health", timeout=2.0) as response:
                payload = json.loads(response.read())
            if payload.get("status") == "ok":
                return payload
        except (OSError, URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
        time.sleep(0.5)
    raise TimeoutError(f"service did not become healthy in {timeout}s: {last_error}")


def _sample_gpu(stop: threading.Event, phase: dict, samples: list[dict], interval: float) -> None:
    command = [
        "nvidia-smi", f"--query-gpu={GPU_QUERY_FIELDS}",
        "--format=csv,noheader,nounits",
    ]
    started = time.monotonic()
    while not stop.is_set():
        sampled_at = time.time()
        try:
            completed = subprocess.run(
                command, check=True, capture_output=True, text=True, timeout=5.0
            )
            for gpu_index, line in enumerate(completed.stdout.splitlines()):
                sample = parse_nvidia_smi_line(line)
                sample.update({
                    "gpu_index": gpu_index,
                    "phase": phase["value"],
                    "wall_time_unix": sampled_at,
                    "elapsed_seconds": time.monotonic() - started,
                })
                samples.append(sample)
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            samples.append({
                "phase": phase["value"],
                "wall_time_unix": sampled_at,
                "elapsed_seconds": time.monotonic() - started,
                "error": f"{type(error).__name__}: {error}",
            })
        stop.wait(interval)


def _stop_service(process: subprocess.Popen, timeout: float = 30.0) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8771)
    parser.add_argument("--capacity", type=int, default=2)
    parser.add_argument("--max-inflight", type=int, default=8)
    parser.add_argument("--min-prefill-batch", type=int, default=2)
    parser.add_argument("--max-wait-steps", type=int, default=2)
    parser.add_argument(
        "--execution-policy",
        choices=("fixed", "decode", "full"),
        default="full",
    )
    splitk_group = parser.add_mutually_exclusive_group()
    splitk_group.add_argument(
        "--enable-m1-down-splitk", dest="enable_m1_down_splitk", action="store_true"
    )
    splitk_group.add_argument(
        "--disable-m1-down-splitk", dest="enable_m1_down_splitk", action="store_false"
    )
    parser.set_defaults(enable_m1_down_splitk=False)
    parser.add_argument(
        "--m1-down-splitk-fallback-name", action="append", default=[]
    )
    parser.add_argument(
        "--max-cache-len",
        type=int,
        default=1024,
        help="Must not exceed the current batched K1 limit of 64 pages per sequence.",
    )
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--arrival-interval-ms", type=float, default=500.0)
    parser.add_argument("--client-workers", type=int, default=8)
    parser.add_argument(
        "--prompt",
        default=(
            "The capital of France is Paris. The capital of Germany is Berlin. "
            "The capital of Italy is Rome. The capital of Spain is Madrid."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--telemetry-interval", type=float, default=0.25)
    parser.add_argument(
        "--nsys",
        action="store_true",
        help="Wrap each server in an Nsight Systems CUDA-profiler-API capture.",
    )
    parser.add_argument(
        "--torch-profiler",
        action="store_true",
        help="Capture the measured request window with PyTorch CPU/CUDA profiler.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if min(args.trials, args.warmup_requests, args.requests, args.client_workers) < 1:
        parser.error("trials, warmup requests, requests and workers must be positive")
    if args.arrival_interval_ms < 0 or args.telemetry_interval <= 0:
        parser.error("arrival interval must be non-negative and telemetry interval positive")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    if args.nsys and args.torch_profiler:
        parser.error("--nsys and --torch-profiler must be run separately")
    if args.m1_down_splitk_fallback_name and not args.enable_m1_down_splitk:
        parser.error("split-K fallback names require --enable-m1-down-splitk")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    base_url = f"http://{args.host}:{args.port}"
    trial_records = []
    for trial_index in range(1, args.trials + 1):
        prefix = args.output_dir / f"trial_{trial_index:02d}"
        benchmark_path = prefix.with_suffix(".benchmark.json")
        telemetry_path = prefix.with_suffix(".gpu.json")
        log_path = prefix.with_suffix(".server.log")
        control_path = prefix.with_suffix(".profile-control")
        torch_trace_path = prefix.with_suffix(".torch-trace.json")
        torch_summary_path = prefix.with_suffix(".torch-summary.json")
        nsys_prefix = prefix.with_suffix("")
        server_command = [
            sys.executable, str(ROOT / "scripts" / "serve_w4a4kv4.py"),
            str(args.checkpoint), "--host", args.host, "--port", str(args.port),
            "--capacity", str(args.capacity), "--max-inflight", str(args.max_inflight),
            "--max-cache-len", str(args.max_cache_len),
            "--min-prefill-batch", str(args.min_prefill_batch),
            "--max-wait-steps", str(args.max_wait_steps),
            "--execution-policy", args.execution_policy,
            "--offline",
        ]
        if args.nsys or args.torch_profiler:
            server_command.extend(["--profile-control-file", str(control_path)])
        server_command.append(
            "--enable-m1-down-splitk"
            if args.enable_m1_down_splitk
            else "--disable-m1-down-splitk"
        )
        for fallback_name in args.m1_down_splitk_fallback_name:
            server_command.extend([
                "--m1-down-splitk-fallback-name", fallback_name,
            ])
        if args.torch_profiler:
            server_command.extend([
                "--torch-profiler-output", str(torch_trace_path),
                "--torch-profiler-summary", str(torch_summary_path),
            ])
        if args.nsys:
            server_command = [
                "nsys", "profile",
                "--trace=cuda,nvtx,osrt",
                "--cuda-graph-trace=node",
                "--sample=none",
                "--capture-range=cudaProfilerApi",
                "--capture-range-end=stop",
                "--force-overwrite=true",
                "--output", str(nsys_prefix),
                *server_command,
            ]
        benchmark_command = [
            sys.executable, str(ROOT / "scripts" / "benchmark_http_service.py"),
            "--url", base_url, "--requests", str(args.requests),
            "--arrival-interval-ms", str(args.arrival_interval_ms),
            "--client-workers", str(args.client_workers), "--prompt", args.prompt,
            "--max-new-tokens", str(args.max_new_tokens),
            "--request-timeout", str(args.request_timeout),
            "--output", str(benchmark_path),
        ]
        samples: list[dict] = []
        phase = {"value": "startup"}
        sample_stop = threading.Event()
        sampler = threading.Thread(
            target=_sample_gpu,
            args=(sample_stop, phase, samples, args.telemetry_interval),
            daemon=True,
        )
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                server_command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True
            )
            sampler.start()
            try:
                _wait_for_health(base_url, process, args.startup_timeout)
                phase["value"] = "warmup"
                for warmup_index in range(args.warmup_requests):
                    warmup = run_request(
                        base_url,
                        f"fresh-{trial_index}-warmup-{warmup_index}",
                        args.prompt,
                        args.max_new_tokens,
                        time.perf_counter(),
                        args.request_timeout,
                    )
                    if warmup["http_status"] != 200 or warmup["terminal_status"] != "finished":
                        raise RuntimeError(f"warmup request failed: {warmup}")
                if args.nsys or args.torch_profiler:
                    control_path.write_text("start\n", encoding="utf-8")
                    time.sleep(0.25)
                phase["value"] = "benchmark"
                subprocess.run(benchmark_command, cwd=ROOT, check=True)
                benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
                with urlopen(f"{base_url}/metrics", timeout=5.0) as response:
                    runtime_metrics = json.loads(response.read().decode("utf-8"))
                if args.nsys or args.torch_profiler:
                    control_path.write_text("stop\n", encoding="utf-8")
                    time.sleep(1.0)
            finally:
                phase["value"] = "shutdown"
                if (args.nsys or args.torch_profiler) and process.poll() is None:
                    control_path.write_text("shutdown\n", encoding="utf-8")
                    try:
                        process.wait(timeout=30.0)
                    except subprocess.TimeoutExpired:
                        _stop_service(process)
                else:
                    _stop_service(process)
                sample_stop.set()
                sampler.join(timeout=10.0)
                telemetry_path.write_text(
                    json.dumps({"schema_version": 1, "samples": samples}, indent=2) + "\n",
                    encoding="utf-8",
                )
        trial_records.append({
            "trial": trial_index,
            "benchmark": benchmark,
            "runtime_metrics": runtime_metrics,
            "benchmark_path": str(benchmark_path),
            "telemetry_path": str(telemetry_path),
            "server_log_path": str(log_path),
        })
        print(
            f"trial {trial_index}/{args.trials}: TPOT P95 "
            f"{benchmark['tpot_ms']['p95']:.2f} ms, TTFT P95 "
            f"{benchmark['ttft_ms']['p95']:.2f} ms",
            flush=True,
        )

    summary = summarize_trials(trial_records)
    summary.update({
        "checkpoint": str(args.checkpoint),
        "config": {
            "url": base_url,
            "capacity": args.capacity,
            "max_inflight": args.max_inflight,
            "max_cache_len": args.max_cache_len,
            "min_prefill_batch": args.min_prefill_batch,
            "max_wait_steps": args.max_wait_steps,
            "execution_policy": args.execution_policy,
            "enable_m1_down_splitk": args.enable_m1_down_splitk,
            "m1_down_splitk_fallback_names": args.m1_down_splitk_fallback_name,
            "warmup_requests": args.warmup_requests,
            "requests": args.requests,
            "arrival_interval_ms": args.arrival_interval_ms,
            "prompt": args.prompt,
            "max_new_tokens": args.max_new_tokens,
            "telemetry_interval_seconds": args.telemetry_interval,
            "nsys": args.nsys,
            "torch_profiler": args.torch_profiler,
        },
        "trials": trial_records,
    })
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_markdown(summary, args.report)
    print(json.dumps({
        "all_trials_passed": summary["all_trials_passed"],
        "trial_count": summary["trial_count"],
        "tpot_p95_ms": summary["metrics"]["tpot_p95_ms"],
        "ttft_p95_ms": summary["metrics"]["ttft_p95_ms"],
    }, indent=2))
    if not summary["all_trials_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
