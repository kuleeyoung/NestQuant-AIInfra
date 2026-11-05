"""Summarize paired BF16-KV versus NQKV4 service-stage measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot summarize an empty measurement group")
    return statistics.median(values)


def _run_key(run: dict) -> tuple[str, int]:
    return str(run["profile"]), int(run["repetition"])


def _stage_value(run: dict, stage: str) -> float | None:
    records = run["step_records"]
    if stage == "boundary":
        values = [
            float(record["latency_ms"])
            for record in records
            if int(record["nqkv_boundary_replays"]) > 0
        ]
    else:
        values = [
            float(record["latency_ms"])
            for record in records
            if record["kind"] == stage
        ]
    return _median(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nqkv", type=Path)
    parser.add_argument("bf16", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    nqkv_payload = json.loads(args.nqkv.read_text(encoding="utf-8"))
    bf16_payload = json.loads(args.bf16.read_text(encoding="utf-8"))
    nqkv_runs = {_run_key(run): run for run in nqkv_payload["runs"]}
    bf16_runs = {_run_key(run): run for run in bf16_payload["runs"]}
    if nqkv_runs.keys() != bf16_runs.keys():
        raise RuntimeError("NQKV4 and BF16 runs do not have identical paired keys")

    stages = ("prefill_only", "decode_only", "mixed_prefill_decode", "boundary")
    profiles: dict[str, dict] = {}
    for profile in sorted({key[0] for key in nqkv_runs}):
        keys = sorted(key for key in nqkv_runs if key[0] == profile)
        profile_result = {"pairs": len(keys), "stages": {}}
        nqkv_throughput = [
            float(nqkv_runs[key]["token_throughput_per_second"]) for key in keys
        ]
        bf16_throughput = [
            float(bf16_runs[key]["token_throughput_per_second"]) for key in keys
        ]
        throughput_ratios = [
            nqkv / bf16 for nqkv, bf16 in zip(nqkv_throughput, bf16_throughput)
        ]
        profile_result["throughput"] = {
            "nqkv4_median_tokens_per_second": _median(nqkv_throughput),
            "bf16_median_tokens_per_second": _median(bf16_throughput),
            "paired_ratio_median": _median(throughput_ratios),
            "nqkv4_deficit_percent": (1.0 - _median(throughput_ratios)) * 100.0,
        }
        for stage in stages:
            pairs = []
            for key in keys:
                nqkv_value = _stage_value(nqkv_runs[key], stage)
                bf16_value = _stage_value(bf16_runs[key], stage)
                if nqkv_value is None and bf16_value is None:
                    continue
                if nqkv_value is None or bf16_value is None:
                    # BF16 has no NQKV boundary event. Pair its decode step at
                    # the same ordinal position as the NQKV boundary replay.
                    if stage != "boundary" or nqkv_value is None:
                        raise RuntimeError(f"unpaired {stage} measurement for {key}")
                    nqkv_records = nqkv_runs[key]["step_records"]
                    boundary_indices = [
                        index for index, record in enumerate(nqkv_records)
                        if int(record["nqkv_boundary_replays"]) > 0
                    ]
                    bf16_records = bf16_runs[key]["step_records"]
                    bf16_values = [
                        float(bf16_records[index]["latency_ms"])
                        for index in boundary_indices
                    ]
                    bf16_value = _median(bf16_values)
                pairs.append((nqkv_value, bf16_value))
            if not pairs:
                continue
            nqkv_values = [pair[0] for pair in pairs]
            bf16_values = [pair[1] for pair in pairs]
            ratios = [nqkv / bf16 for nqkv, bf16 in pairs]
            profile_result["stages"][stage] = {
                "pair_count": len(pairs),
                "nqkv4_median_ms": _median(nqkv_values),
                "bf16_median_ms": _median(bf16_values),
                "paired_ratio_median": _median(ratios),
                "nqkv4_overhead_percent": (_median(ratios) - 1.0) * 100.0,
            }
        profiles[profile] = profile_result

    result = {
        "status": "passed",
        "nqkv_source": str(args.nqkv),
        "bf16_source": str(args.bf16),
        "profiles": profiles,
        "claim_boundary": (
            "Same requests, seeds, shapes, GPU and W4A4 path; backends were run "
            "in separate processes rather than alternating inside one process."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# NQKV4 versus BF16 KV stage A/B", "",
        "Five fixed-seed repetitions use identical requests and the same packed W4A4 path.",
        "Only the KV backend changes. Runs are paired by profile and repetition.", "",
    ]
    for profile, data in profiles.items():
        throughput = data["throughput"]
        lines.extend([
            f"## {profile}", "",
            f"- NQKV4 throughput median: {throughput['nqkv4_median_tokens_per_second']:.2f} token/s.",
            f"- BF16 KV throughput median: {throughput['bf16_median_tokens_per_second']:.2f} token/s.",
            f"- Paired NQKV4 throughput deficit: {throughput['nqkv4_deficit_percent']:.2f}%.", "",
            "| stage | NQKV4 median | BF16 median | paired NQKV4 overhead |",
            "|---|---:|---:|---:|",
        ])
        for stage, stage_data in data["stages"].items():
            lines.append(
                f"| {stage} | {stage_data['nqkv4_median_ms']:.2f} ms | "
                f"{stage_data['bf16_median_ms']:.2f} ms | "
                f"{stage_data['nqkv4_overhead_percent']:.2f}% |"
            )
        lines.append("")
    lines.extend([
        "## Claim boundary", "",
        "The requests, random seeds, fixed shapes, GPU and W4A4 implementation are identical. "
        "The two KV backends were executed in separate processes, not alternated inside one "
        "process, so this is paired-by-seed stage evidence rather than a randomized timing trial.",
        "",
    ])
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
