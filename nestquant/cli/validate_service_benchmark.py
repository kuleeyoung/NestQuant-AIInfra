"""Validate the completeness and invariants of a service benchmark report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


LATENCY_PREFIXES = ("ttft_ms", "tpot_ms", "e2e_ms", "step_latency_ms")


def validate_report(report: dict) -> dict:
    if report.get("status") != "passed":
        raise ValueError("benchmark status is not passed")
    profiles = {profile["name"] for profile in report["profiles"]}
    policies = {policy["name"] for policy in report["policies"]}
    summaries = report["summaries"]
    repetitions = {int(summary["repetitions"]) for summary in summaries}
    if len(repetitions) != 1:
        raise ValueError("summary repetition counts disagree")
    repetition_count = repetitions.pop()
    expected_pairs = {(profile, policy) for profile in profiles for policy in policies}
    actual_pairs = {(item["profile"], item["policy"]) for item in summaries}
    if actual_pairs != expected_pairs:
        raise ValueError("profile/policy summary matrix is incomplete")
    runs = report["runs"]
    if len(runs) != len(expected_pairs) * repetition_count:
        raise ValueError("run count does not match profile/policy/repetition matrix")

    run_ids = [run["run_id"] for run in runs]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("run IDs are not unique")
    expected_repetitions = set(range(repetition_count))
    for pair in expected_pairs:
        pair_runs = [
            run for run in runs
            if (run["profile"], run["policy"]) == pair
        ]
        actual_repetitions = {int(run["repetition"]) for run in pair_runs}
        if (len(pair_runs) != repetition_count
                or actual_repetitions != expected_repetitions):
            raise ValueError(
                "profile/policy run repetitions are incomplete or duplicated"
            )

    for run in runs:
        if (run["profile"], run["policy"]) not in expected_pairs:
            raise ValueError(f"unexpected profile/policy in {run['run_id']}")
        if run["scheduled_requests"] < 1:
            raise ValueError(f"no scheduled requests in {run['run_id']}")
        if run["completed_requests"] != run["scheduled_requests"]:
            raise ValueError(f"incomplete requests in {run['run_id']}")
        if (run["failed_requests"] != 0 or run.get("rejected_requests")
                or run["final_registered_requests"] != 0):
            raise ValueError(
                f"failure, rejection, or retained state in {run['run_id']}"
            )
        if run["completion_rate"] != 1.0:
            raise ValueError(f"completion rate is not one in {run['run_id']}")
        for name in (
            "request_throughput_per_second", "token_throughput_per_second",
            "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes",
        ):
            if name not in run:
                continue
            value = run[name]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {name} in {run['run_id']}")
        for prefix in LATENCY_PREFIXES:
            values = [run[f"{prefix}_{name}"] for name in ("p50", "p95", "p99")]
            if not all(math.isfinite(value) and value >= 0 for value in values):
                raise ValueError(f"invalid {prefix} values in {run['run_id']}")
            if values != sorted(values):
                raise ValueError(f"unordered {prefix} percentiles in {run['run_id']}")

    configured_lengths = set()
    for profile in report["profiles"]:
        if profile.get("length_mode", "uniform") == "uniform":
            bounds = [profile["prompt_length"]]
        else:
            bounds = [profile["short"]["prompt_length"], profile["long"]["prompt_length"]]
        for lower, upper in bounds:
            configured_lengths.update(range(int(lower), int(upper) + 1))
    warmed = set(report.get("warmup_prompt_lengths", []))
    if not configured_lengths.issubset(warmed):
        raise ValueError("steady-state report did not warm every configured prompt length")

    return {
        "status": "passed",
        "profiles": len(profiles),
        "policies": len(policies),
        "repetitions": repetition_count,
        "runs": len(runs),
        "scheduled_requests": sum(run["scheduled_requests"] for run in runs),
        "completed_requests": sum(run["completed_requests"] for run in runs),
        "failed_requests": sum(run["failed_requests"] for run in runs),
        "rejected_requests": sum(len(run.get("rejected_requests", {})) for run in runs),
        "final_registered_requests": sum(
            run["final_registered_requests"] for run in runs
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_report(json.loads(args.report.read_text(encoding="utf-8")))
    payload = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
