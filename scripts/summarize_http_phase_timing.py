"""Summarize server-side TTFT phases across HTTP offered-load points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PHASES = (
    "command_queue",
    "waiting_queue",
    "prefill_compute",
    "post_prefill_to_first_token",
    "server_ttft",
    "server_e2e",
)


def summarize(paths: list[Path]) -> dict:
    if len(paths) < 2:
        raise ValueError("phase comparison requires at least two inputs")
    points = []
    reference = None
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload["config"]
        signature = (
            int(config["requests"]),
            str(config["prompt"]),
            int(config["max_new_tokens"]),
            int(config["client_workers"]),
        )
        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError("all phase points must share workload shape")
        if payload["failed_requests"] or payload["overloaded_requests"]:
            raise ValueError("phase timing accepts only lossless points")
        timings = payload.get("server_timing_ms", {})
        missing = [phase for phase in PHASES if phase not in timings]
        if missing:
            raise ValueError(f"missing server timing phases: {missing}")
        server_ttft_mean = float(timings["server_ttft"]["mean"])
        queue_mean = (
            float(timings["command_queue"]["mean"])
            + float(timings["waiting_queue"]["mean"])
        )
        points.append({
            "source": str(path),
            "offered_requests_per_second": 1000.0 / float(
                config["arrival_interval_ms"]
            ),
            "client_ttft_ms": payload["ttft_ms"],
            "client_e2e_ms": payload["e2e_ms"],
            "server_timing_ms": {phase: timings[phase] for phase in PHASES},
            "queue_mean_ms": queue_mean,
            "queue_fraction_of_server_ttft_mean": (
                queue_mean / server_ttft_mean if server_ttft_mean else 0.0
            ),
            "client_minus_server_ttft_p95_ms": (
                float(payload["ttft_ms"]["p95"])
                - float(timings["server_ttft"]["p95"])
            ),
        })
    points.sort(key=lambda point: point["offered_requests_per_second"])
    return {
        "schema_version": 1,
        "status": "passed",
        "points": points,
        "interpretation": (
            "Compare mean phases for additive attribution and P95 phases for tail "
            "behavior. Percentiles from different phase distributions must not be added."
        ),
        "claim_boundary": (
            "Worker-owned monotonic timestamps for one finite single-host HTTP run "
            "per offered-load point; not a production SLA or a causal scheduler A/B."
        ),
    }


def write_markdown(summary: dict, output: Path) -> None:
    lines = [
        "# W4A4KV4 HTTP server-side TTFT phase decomposition",
        "",
        "## Additive mean decomposition",
        "",
        "| offered req/s | client TTFT mean | command queue | scheduler waiting | prefill compute | post-prefill | server TTFT mean | queue share |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for point in summary["points"]:
        timing = point["server_timing_ms"]
        lines.append(
            f"| {point['offered_requests_per_second']:.0f} | "
            f"{point['client_ttft_ms']['mean']:.2f} | "
            f"{timing['command_queue']['mean']:.2f} | "
            f"{timing['waiting_queue']['mean']:.2f} | "
            f"{timing['prefill_compute']['mean']:.2f} | "
            f"{timing['post_prefill_to_first_token']['mean']:.2f} | "
            f"{timing['server_ttft']['mean']:.2f} | "
            f"{point['queue_fraction_of_server_ttft_mean']:.1%} |"
        )
    lines.extend([
        "",
        "## Tail decomposition",
        "",
        "| offered req/s | client TTFT P95 | server TTFT P95 | command queue P95 | scheduler waiting P95 | prefill P95 | client-server P95 gap |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for point in summary["points"]:
        timing = point["server_timing_ms"]
        lines.append(
            f"| {point['offered_requests_per_second']:.0f} | "
            f"{point['client_ttft_ms']['p95']:.2f} | "
            f"{timing['server_ttft']['p95']:.2f} | "
            f"{timing['command_queue']['p95']:.2f} | "
            f"{timing['waiting_queue']['p95']:.2f} | "
            f"{timing['prefill_compute']['p95']:.2f} | "
            f"{point['client_minus_server_ttft_p95_ms']:.2f} |"
        )
    lines.extend([
        "",
        "Mean phases are additive for attribution. P95 columns describe each phase's "
        "own tail and are not added because their P95 values can come from different requests.",
        "",
        f"Claim boundary: {summary['claim_boundary']}",
        "",
    ])
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_markdown(result, args.markdown)
    print(json.dumps({
        "status": result["status"],
        "points": [{
            "offered_rps": point["offered_requests_per_second"],
            "queue_fraction": point["queue_fraction_of_server_ttft_mean"],
        } for point in result["points"]],
    }, indent=2))


if __name__ == "__main__":
    main()
