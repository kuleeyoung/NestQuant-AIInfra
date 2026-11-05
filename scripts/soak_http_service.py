"""Long-running open-loop HTTP soak with service resource snapshots."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_http_service import distribution, run_request


def get_json(url: str) -> dict:
    with urlopen(url, timeout=5) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--duration-seconds", type=float, default=600.0)
    parser.add_argument("--offered-requests-per-second", type=float, default=8.0)
    parser.add_argument("--client-workers", type=int, default=32)
    parser.add_argument("--sample-interval-seconds", type=float, default=5.0)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument(
        "--require-nqkv-pages", action="store_true",
        help="Fail unless the worker observes at least one allocated cold page.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.duration_seconds <= 0 or args.offered_requests_per_second <= 0:
        parser.error("duration and offered rate must be positive")

    base = args.url.rstrip("/")
    if get_json(base + "/health").get("status") != "ok":
        raise RuntimeError("service is not healthy")
    interval = 1.0 / args.offered_requests_per_second
    started = time.perf_counter()
    deadline = started + args.duration_seconds
    next_arrival = started
    next_sample = started
    submitted = 0
    futures = []
    samples = []
    with ThreadPoolExecutor(max_workers=args.client_workers) as executor:
        while next_arrival < deadline:
            now = time.perf_counter()
            wake_at = min(next_arrival, next_sample)
            if now < wake_at:
                time.sleep(wake_at - now)
                now = time.perf_counter()
            if now >= next_sample:
                metrics = get_json(base + "/metrics")
                samples.append({
                    "elapsed_seconds": now - started,
                    "metrics": metrics,
                })
                next_sample += args.sample_interval_seconds
            if now >= next_arrival:
                futures.append(executor.submit(
                    run_request,
                    base,
                    f"http-soak-{submitted}",
                    args.prompt,
                    args.max_new_tokens,
                    next_arrival,
                    args.request_timeout,
                ))
                submitted += 1
                next_arrival = started + submitted * interval
        runs = [future.result() for future in as_completed(futures)]
    finished = time.perf_counter()
    final_metrics = get_json(base + "/metrics")
    samples.append({
        "elapsed_seconds": finished - started,
        "metrics": final_metrics,
    })
    completed = [
        run for run in runs
        if run["http_status"] == 200 and run["terminal_status"] == "finished"
    ]
    overloaded = [run for run in runs if run["http_status"] == 429]
    failed = [run for run in runs if run not in completed and run not in overloaded]
    ttft = [float(run["ttft_ms"]) for run in completed]
    tpot = [float(run["tpot_ms"]) for run in completed]
    e2e = [float(run["e2e_ms"]) for run in completed]
    steady_samples = [
        sample for sample in samples if sample["elapsed_seconds"] >= 60.0
    ] or samples

    def resource_values(name):
        return [
            int(sample["metrics"]["runtime"][name])
            for sample in steady_samples
            if sample["metrics"]["runtime"].get(name) is not None
        ]

    resources = {}
    for name in (
        "cpu_rss_bytes", "cuda_allocated_bytes", "cuda_reserved_bytes",
        "nqkv_allocated_pages", "nqkv_free_pages", "registered_requests",
    ):
        values = resource_values(name)
        resources[name] = {
            "min": min(values, default=None),
            "max": max(values, default=None),
            "range": max(values) - min(values) if values else None,
            "final": final_metrics["runtime"].get(name),
        }
    final_runtime = final_metrics["runtime"]
    passed = (
        not failed
        and int(final_metrics["inflight"]) == 0
        and int(final_runtime["registered_requests"]) == 0
        and int(final_runtime["nqkv_allocated_pages"] or 0) == 0
        and (
            not args.require_nqkv_pages
            or int(final_runtime.get("nqkv_allocated_pages_peak") or 0) > 0
        )
    )
    report = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "config": vars(args) | {"output": str(args.output)},
        "wall_seconds": finished - started,
        "scheduled_requests": len(runs),
        "completed_requests": len(completed),
        "overloaded_requests": len(overloaded),
        "failed_requests": len(failed),
        "completion_rate": len(completed) / len(runs),
        "overload_rate": len(overloaded) / len(runs),
        "completed_request_throughput_per_second": len(completed) / (finished - started),
        "completed_token_throughput_per_second": sum(
            int(run["token_count"]) for run in completed
        ) / (finished - started),
        "ttft_ms": distribution(ttft),
        "tpot_ms": distribution(tpot),
        "e2e_ms": distribution(e2e),
        "resources_after_60_seconds": resources,
        "final_metrics": final_metrics,
        "samples": samples,
        "failed_runs": failed,
        "claim_boundary": (
            "Single-host HTTP/1.1 NDJSON soak at one offered load; includes "
            "tokenizer, queueing and socket delivery, not a production SLA."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        key: report[key] for key in (
            "status", "wall_seconds", "scheduled_requests", "completed_requests",
            "overloaded_requests", "failed_requests", "completion_rate",
            "completed_request_throughput_per_second",
            "completed_token_throughput_per_second", "ttft_ms", "tpot_ms",
            "e2e_ms", "resources_after_60_seconds", "final_metrics",
        )
    }, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
