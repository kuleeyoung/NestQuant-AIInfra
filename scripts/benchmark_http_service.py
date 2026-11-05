"""Open-loop HTTP benchmark for the NestQuant NDJSON generation endpoint."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def identity_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "count": len(values),
        "mean": statistics.mean(values) if values else 0.0,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values, default=0.0),
    }


def run_request(
    base_url: str,
    request_id: str,
    prompt: str,
    max_new_tokens: int,
    scheduled_at: float,
    timeout_seconds: float,
) -> dict:
    now = time.perf_counter()
    if scheduled_at > now:
        time.sleep(scheduled_at - now)
    submitted_at = time.perf_counter()
    body = json.dumps({
        "request_id": request_id,
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "timeout_seconds": timeout_seconds,
    }).encode("utf-8")
    request = Request(
        base_url.rstrip("/") + "/v1/generate/stream",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    token_times = []
    token_ids = []
    terminal_status = None
    server_timings_ms = None
    http_status = None
    error = None
    try:
        with urlopen(request, timeout=timeout_seconds + 5.0) as response:
            http_status = int(response.status)
            for raw_line in response:
                received_at = time.perf_counter()
                event = json.loads(raw_line)
                if "token_id" in event:
                    token_ids.append(int(event["token_id"]))
                    token_times.append(received_at)
                if event.get("done"):
                    terminal_status = event.get("status")
                    server_timings_ms = event.get("timings_ms")
    except HTTPError as exception:
        http_status = int(exception.code)
        error = exception.read().decode("utf-8", errors="replace")
    except (TimeoutError, URLError, OSError, json.JSONDecodeError) as exception:
        error = f"{type(exception).__name__}: {exception}"
    finished_at = time.perf_counter()
    gaps_ms = [
        (right - left) * 1000.0
        for left, right in zip(token_times, token_times[1:])
    ]
    return {
        "request_id": request_id,
        "http_status": http_status,
        "terminal_status": terminal_status,
        "token_ids": token_ids,
        "token_count": len(token_ids),
        "scheduled_lag_ms": (submitted_at - scheduled_at) * 1000.0,
        "ttft_ms": (
            (token_times[0] - submitted_at) * 1000.0 if token_times else None
        ),
        "tpot_ms": statistics.mean(gaps_ms) if gaps_ms else 0.0,
        "e2e_ms": (finished_at - submitted_at) * 1000.0,
        "error": error,
        "server_timings_ms": server_timings_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--arrival-interval-ms", type=float, default=100.0)
    parser.add_argument("--client-workers", type=int, default=32)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.requests < 1 or args.client_workers < 1:
        parser.error("--requests and --client-workers must be positive")
    if args.arrival_interval_ms < 0 or args.max_new_tokens < 1:
        parser.error("arrival interval must be non-negative and tokens positive")

    health_url = args.url.rstrip("/") + "/health"
    with urlopen(health_url, timeout=5) as response:
        health = json.loads(response.read())
    if health.get("status") != "ok":
        raise RuntimeError(f"service is not healthy: {health}")

    benchmark_started = time.perf_counter()
    interval = args.arrival_interval_ms / 1000.0
    futures = []
    with ThreadPoolExecutor(max_workers=args.client_workers) as executor:
        for index in range(args.requests):
            futures.append(executor.submit(
                run_request,
                args.url,
                f"http-bench-{index}",
                args.prompt,
                args.max_new_tokens,
                benchmark_started + index * interval,
                args.request_timeout,
            ))
        runs = [future.result() for future in as_completed(futures)]
    benchmark_finished = time.perf_counter()
    runs.sort(key=lambda item: int(item["request_id"].rsplit("-", 1)[1]))
    completed = [
        item for item in runs
        if item["http_status"] == 200 and item["terminal_status"] == "finished"
    ]
    overloaded = [item for item in runs if item["http_status"] == 429]
    failed = [item for item in runs if item not in completed and item not in overloaded]
    ttft = [float(item["ttft_ms"]) for item in completed if item["ttft_ms"] is not None]
    tpot = [float(item["tpot_ms"]) for item in completed]
    e2e = [float(item["e2e_ms"]) for item in completed]
    scheduled_lag = [float(item["scheduled_lag_ms"]) for item in runs]
    server_timing_keys = (
        "command_queue", "waiting_queue", "prefill_compute",
        "post_prefill_to_first_token", "admission_to_first_token",
        "server_ttft", "server_e2e",
    )
    server_timing_ms = {
        key: distribution([
            float(item["server_timings_ms"][key])
            for item in completed
            if item.get("server_timings_ms") is not None
            and item["server_timings_ms"].get(key) is not None
        ])
        for key in server_timing_keys
    }
    total_tokens = sum(int(item["token_count"]) for item in completed)
    wall_seconds = benchmark_finished - benchmark_started
    workload_identity_sha256 = identity_sha256([
        {
            "request_id": f"http-bench-{index}",
            "prompt": args.prompt,
            "max_new_tokens": args.max_new_tokens,
            "scheduled_offset_seconds": index * interval,
        }
        for index in range(args.requests)
    ])
    output_identity_sha256 = identity_sha256([
        {
            "request_id": item["request_id"],
            "http_status": item["http_status"],
            "terminal_status": item["terminal_status"],
            "token_ids": item["token_ids"],
        }
        for item in runs
    ])
    report = {
        "schema_version": 1,
        "status": "passed" if not failed else "failed",
        "url": args.url,
        "protocol": "HTTP/1.1 NDJSON client-observed timings",
        "config": {
            "requests": args.requests,
            "arrival_interval_ms": args.arrival_interval_ms,
            "client_workers": args.client_workers,
            "prompt": args.prompt,
            "max_new_tokens": args.max_new_tokens,
            "request_timeout": args.request_timeout,
        },
        "scheduled_requests": len(runs),
        "completed_requests": len(completed),
        "overloaded_requests": len(overloaded),
        "failed_requests": len(failed),
        "completion_rate": len(completed) / len(runs),
        "overload_rate": len(overloaded) / len(runs),
        "wall_seconds": wall_seconds,
        "request_throughput_per_second": len(completed) / wall_seconds,
        "token_throughput_per_second": total_tokens / wall_seconds,
        "workload_identity_sha256": workload_identity_sha256,
        "output_identity_sha256": output_identity_sha256,
        "ttft_ms": distribution(ttft),
        "tpot_ms": distribution(tpot),
        "e2e_ms": distribution(e2e),
        "scheduled_lag_ms": distribution(scheduled_lag),
        "server_timing_ms": server_timing_ms,
        "runs": runs,
        "claim_boundary": (
            "Client-observed single-host HTTP/1.1 NDJSON benchmark including "
            "tokenizer, socket and service queueing; not a production SLA."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        key: report[key] for key in (
            "status", "completed_requests", "overloaded_requests",
            "failed_requests", "request_throughput_per_second",
            "token_throughput_per_second", "ttft_ms", "tpot_ms", "e2e_ms",
        )
    }, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
