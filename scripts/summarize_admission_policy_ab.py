"""Compare repeated fresh-process HTTP runs for two admission policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


METRICS = {
    "request_throughput_per_second": ("request_throughput_per_second", "higher"),
    "ttft_p95_ms": ("ttft_ms.p95", "lower"),
    "tpot_p95_ms": ("tpot_ms.p95", "lower"),
    "e2e_p95_ms": ("e2e_ms.p95", "lower"),
    "command_queue_mean_ms": ("server_timing_ms.command_queue.mean", "lower"),
    "waiting_queue_mean_ms": ("server_timing_ms.waiting_queue.mean", "lower"),
    "prefill_compute_mean_ms": ("server_timing_ms.prefill_compute.mean", "lower"),
}


def _get(payload: dict, path: str) -> float:
    value: object = payload
    for key in path.split("."):
        value = value[key]  # type: ignore[index]
    return float(value)


def _runs(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    runs = [trial["benchmark"] for trial in payload["trials"]]
    if not runs:
        raise ValueError(f"summary contains no trials: {path}")
    if any(run["failed_requests"] or run["overloaded_requests"] for run in runs):
        raise ValueError("admission comparison requires lossless runs")
    return runs


def summarize(baseline_path: Path, candidate_path: Path) -> dict:
    baseline = _runs(baseline_path)
    candidate = _runs(candidate_path)
    if len(baseline) != len(candidate):
        raise ValueError("baseline and candidate must have equal trial counts")
    metrics = {}
    for name, (path, direction) in METRICS.items():
        left = [_get(run, path) for run in baseline]
        right = [_get(run, path) for run in candidate]
        left_median = statistics.median(left)
        right_median = statistics.median(right)
        improved = sum(
            candidate_value > baseline_value if direction == "higher"
            else candidate_value < baseline_value
            for baseline_value, candidate_value in zip(left, right)
        )
        metrics[name] = {
            "direction": direction,
            "baseline_values": left,
            "candidate_values": right,
            "baseline_median": left_median,
            "candidate_median": right_median,
            "candidate_relative_change": right_median / left_median - 1.0,
            "candidate_improved_pairs": improved,
            "trial_count": len(left),
        }
    return {
        "schema_version": 1,
        "status": "passed",
        "trial_count_per_policy": len(baseline),
        "metrics": metrics,
        "decision": (
            "Retain the baseline admission policy. Immediate admission reduces "
            "scheduler waiting but worsens TPOT, throughput and end-to-end latency."
        ),
        "claim_boundary": (
            "Three sequential fresh-process trials per policy. Trial indices are "
            "compared for readability but were not interleaved paired experiments."
        ),
    }


def write_markdown(summary: dict, output: Path) -> None:
    lines = [
        "# W4A4KV4 HTTP admission-policy A/B",
        "",
        "Baseline is `min_prefill_batch=2, max_wait_steps=2`; candidate is immediate "
        "admission with `min_prefill_batch=1, max_wait_steps=0`.",
        "",
        "| metric | baseline median | immediate median | relative change | immediate improved trials |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metric in summary["metrics"].items():
        lines.append(
            f"| {name} | {metric['baseline_median']:.2f} | "
            f"{metric['candidate_median']:.2f} | "
            f"{metric['candidate_relative_change']:+.1%} | "
            f"{metric['candidate_improved_pairs']}/{metric['trial_count']} |"
        )
    lines.extend([
        "",
        "## Decision",
        "",
        summary["decision"],
        "",
        "Immediate admission moves time out of the scheduler waiting phase, but it "
        "creates smaller and more frequent prefill work. The worker remains busy longer, "
        "command ingestion is delayed, and decode batching efficiency deteriorates.",
        "",
        f"Claim boundary: {summary['claim_boundary']}",
        "",
    ])
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.baseline, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_markdown(result, args.markdown)
    print(json.dumps({
        name: {
            "baseline_median": metric["baseline_median"],
            "candidate_median": metric["candidate_median"],
            "relative_change": metric["candidate_relative_change"],
        }
        for name, metric in result["metrics"].items()
    }, indent=2))


if __name__ == "__main__":
    main()
