"""Pair tiled and CUDA W4A4 service runs and summarize relative changes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def _key(run: dict) -> tuple[str, str, int]:
    return run["profile"], run["policy"], int(run["repetition"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    baseline_runs = {_key(run): run for run in baseline["runs"]}
    candidate_runs = {_key(run): run for run in candidate["runs"]}
    if baseline_runs.keys() != candidate_runs.keys():
        raise ValueError("service runs do not have identical pairing keys")

    pairs = []
    for key in sorted(baseline_runs):
        left, right = baseline_runs[key], candidate_runs[key]
        identity_fields = (
            "scheduled_requests", "completed_requests", "failed_requests",
            "generated_tokens",
        )
        mismatches = [
            field for field in identity_fields if left[field] != right[field]
        ]
        if mismatches:
            raise ValueError(f"unpaired workload {key}: {', '.join(mismatches)}")
        pairs.append({
            "profile": key[0],
            "policy": key[1],
            "repetition": key[2],
            "generated_tokens": left["generated_tokens"],
            "token_throughput_gain_fraction": (
                right["token_throughput_per_second"]
                / left["token_throughput_per_second"] - 1.0
            ),
            "ttft_p95_reduction_fraction": (
                1.0 - right["ttft_ms_p95"] / left["ttft_ms_p95"]
            ),
            "tpot_p95_reduction_fraction": (
                1.0 - right["tpot_ms_p95"] / left["tpot_ms_p95"]
            ),
            "e2e_p95_reduction_fraction": (
                1.0 - right["e2e_ms_p95"] / left["e2e_ms_p95"]
            ),
            "baseline_token_throughput_per_second": (
                left["token_throughput_per_second"]
            ),
            "candidate_token_throughput_per_second": (
                right["token_throughput_per_second"]
            ),
        })
    metric_names = (
        "token_throughput_gain_fraction",
        "ttft_p95_reduction_fraction",
        "tpot_p95_reduction_fraction",
        "e2e_p95_reduction_fraction",
    )
    summary = {
        name: {
            "median": statistics.median(pair[name] for pair in pairs),
            "min": min(pair[name] for pair in pairs),
            "max": max(pair[name] for pair in pairs),
        }
        for name in metric_names
    }
    profiles = sorted({pair["profile"] for pair in pairs})
    summary_by_profile = {
        profile: {
            name: {
                "median": statistics.median(
                    pair[name] for pair in pairs if pair["profile"] == profile
                ),
                "min": min(
                    pair[name] for pair in pairs if pair["profile"] == profile
                ),
                "max": max(
                    pair[name] for pair in pairs if pair["profile"] == profile
                ),
            }
            for name in metric_names
        }
        for profile in profiles
    }
    payload = {
        "schema_version": 1,
        "baseline_backend": baseline["compute_backend"],
        "candidate_backend": candidate["compute_backend"],
        "pair_count": len(pairs),
        "all_requests_completed": all(
            pair["generated_tokens"] > 0 for pair in pairs
        ),
        "all_pairs_improve_throughput": all(
            pair["token_throughput_gain_fraction"] > 0 for pair in pairs
        ),
        "summary": summary,
        "summary_by_profile": summary_by_profile,
        "pairs": pairs,
        "claim_boundary": (
            "Paired single-process synthetic token-ID service runs. No network, "
            "tokenizer, random sampling, multi-GPU execution, or production SLA."
        ),
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
