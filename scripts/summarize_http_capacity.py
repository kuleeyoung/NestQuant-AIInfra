"""Validate and summarize a fixed-workload HTTP offered-load sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(paths: list[Path]) -> dict:
    if len(paths) < 2:
        raise ValueError("capacity sweep requires at least two input points")
    points = []
    reference = None
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload["config"]
        signature = (
            int(config["requests"]), str(config["prompt"]),
            int(config["max_new_tokens"]), int(config["client_workers"]),
        )
        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError("all capacity points must share workload shape")
        if int(payload["failed_requests"]) != 0:
            raise ValueError(f"capacity point contains failures: {path}")
        interval_ms = float(config["arrival_interval_ms"])
        if interval_ms <= 0:
            raise ValueError("capacity interval must be positive")
        scheduled = int(payload["scheduled_requests"])
        completed = int(payload["completed_requests"])
        overloaded = int(payload["overloaded_requests"])
        if completed + overloaded != scheduled:
            raise ValueError("completed plus overloaded must equal scheduled")
        points.append({
            "source": str(path),
            "offered_requests_per_second": 1000.0 / interval_ms,
            "scheduled_requests": scheduled,
            "completed_requests": completed,
            "overloaded_requests": overloaded,
            "overload_rate": float(payload["overload_rate"]),
            "request_throughput_per_second": float(
                payload["request_throughput_per_second"]
            ),
            "token_throughput_per_second": float(
                payload["token_throughput_per_second"]
            ),
            "ttft_p50_ms": float(payload["ttft_ms"]["p50"]),
            "ttft_p95_ms": float(payload["ttft_ms"]["p95"]),
            "tpot_p95_ms": float(payload["tpot_ms"]["p95"]),
            "e2e_p95_ms": float(payload["e2e_ms"]["p95"]),
        })
    points.sort(key=lambda item: item["offered_requests_per_second"])
    first_overload = next(
        (point for point in points if point["overloaded_requests"] > 0), None
    )
    last_lossless = next(
        (point for point in reversed(points) if point["overloaded_requests"] == 0),
        None,
    )
    return {
        "status": "passed",
        "points": points,
        "last_lossless_offered_requests_per_second": (
            None if last_lossless is None
            else last_lossless["offered_requests_per_second"]
        ),
        "first_overloaded_offered_requests_per_second": (
            None if first_overload is None
            else first_overload["offered_requests_per_second"]
        ),
        "peak_completed_request_throughput_per_second": max(
            point["request_throughput_per_second"] for point in points
        ),
        "peak_completed_token_throughput_per_second": max(
            point["token_throughput_per_second"] for point in points
        ),
        "claim_boundary": (
            "Finite single-host HTTP/1.1 NDJSON offered-load sweep; "
            "not a steady-state production SLA."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Llama-3.2-1B packed W4A4KV4 HTTP capacity sweep", "",
        "| offered req/s | completed | HTTP 429 | overload rate | completed req/s | token/s | TTFT P50/P95 ms | TPOT P95 ms | E2E P95 ms |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for point in result["points"]:
        lines.append(
            f"| {point['offered_requests_per_second']:.0f} | "
            f"{point['completed_requests']}/{point['scheduled_requests']} | "
            f"{point['overloaded_requests']} | {point['overload_rate']:.1%} | "
            f"{point['request_throughput_per_second']:.2f} | "
            f"{point['token_throughput_per_second']:.2f} | "
            f"{point['ttft_p50_ms']:.2f}/{point['ttft_p95_ms']:.2f} | "
            f"{point['tpot_p95_ms']:.2f} | {point['e2e_p95_ms']:.2f} |"
        )
    lines.extend([
        "", "## Decision", "",
        f"The highest lossless offered rate in this finite sweep is "
        f"{result['last_lossless_offered_requests_per_second']:.0f} request/s. "
        f"The first point with explicit HTTP 429 backpressure is "
        f"{result['first_overloaded_offered_requests_per_second']:.0f} request/s. "
        f"Peak completed throughput is "
        f"{result['peak_completed_request_throughput_per_second']:.2f} request/s "
        f"or {result['peak_completed_token_throughput_per_second']:.2f} token/s.",
        "", "## Boundary", "",
        result["claim_boundary"], "Each point contains only 24 scheduled requests; "
        "the sweep locates a local knee but does not establish a long-duration SLA.",
    ])
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
