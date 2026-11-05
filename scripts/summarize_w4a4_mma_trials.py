"""Validate and summarize independent full-model W4A4 MMA benchmark trials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


BASELINE_MODE = "packed_w4a4_tiled"
PROTOCOL_KEYS = (
    "model_id", "model_revision", "device_name", "environment", "input_seed",
    "sequence_lengths", "batch_sizes", "decode_tokens", "repeats", "modes",
    "activation_q", "activation_betas",
)


def _load(path: Path) -> dict[str, object]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if len(report.get("results", [])) != 1:
        raise ValueError(f"{path}: expected exactly one shape result")
    shape = report["results"][0]
    paths = {item["mode"]: item for item in shape.get("results", [])}
    modes = tuple(paths)
    if (
        len(modes) != 2
        or modes[0] != BASELINE_MODE
        or not modes[1].startswith("packed_w4a4_cuda_")
    ):
        raise ValueError(
            f"{path}: expected tiled baseline followed by one CUDA candidate"
        )
    return report


def _range(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def summarize_trials(paths: list[Path]) -> dict[str, object]:
    if len(paths) < 3:
        raise ValueError("at least three independent trials are required")
    reports = [_load(path) for path in paths]
    reference = reports[0]
    candidate_mode = reference["modes"][1]
    for path, report in zip(paths[1:], reports[1:]):
        for key in PROTOCOL_KEYS:
            if report.get(key) != reference.get(key):
                raise ValueError(f"{path}: protocol mismatch for {key}")
        if report["results"][0]["input_ids_sha256"] != reference["results"][0]["input_ids_sha256"]:
            raise ValueError(f"{path}: input token hash mismatch")

    trials = []
    tiled_medians = []
    cuda_medians = []
    tiled_ttft = []
    cuda_ttft = []
    tiled_tpot = []
    cuda_tpot = []
    ttft_speedups = []
    tpot_speedups = []
    speedups = []
    for path, report in zip(paths, reports):
        by_mode = {
            item["mode"]: item for item in report["results"][0]["results"]
        }
        tiled = by_mode[BASELINE_MODE]["steady_prefill_seconds"]["median"]
        cuda = by_mode[candidate_mode]["steady_prefill_seconds"]["median"]
        speedup = tiled / cuda
        tiled_medians.append(tiled)
        cuda_medians.append(cuda)
        tiled_ttft.append(by_mode[BASELINE_MODE]["ttft_seconds"])
        cuda_ttft.append(by_mode[candidate_mode]["ttft_seconds"])
        tiled_tpot.append(by_mode[BASELINE_MODE]["tpot_seconds"])
        cuda_tpot.append(by_mode[candidate_mode]["tpot_seconds"])
        ttft_speedups.append(
            by_mode[BASELINE_MODE]["ttft_seconds"]
            / by_mode[candidate_mode]["ttft_seconds"]
        )
        tpot_speedups.append(
            by_mode[BASELINE_MODE]["tpot_seconds"]
            / by_mode[candidate_mode]["tpot_seconds"]
        )
        speedups.append(speedup)
        trials.append({
            "path": str(path),
            "tiled_median_seconds": tiled,
            "cuda_mma_median_seconds": cuda,
            "speedup": speedup,
            "cuda_dispatch_counts": by_mode[candidate_mode]["w4a4_dispatch_counts"],
            "max_abs_logit_error": by_mode[candidate_mode]["max_abs_logit_error_vs_first_mode"],
            "relative_l2_logit_error": by_mode[candidate_mode]["relative_l2_logit_error_vs_first_mode"],
        })

    return {
        "scope": "independent-process Llama-3.2-1B W4A4 CUDA benchmark summary",
        "trial_count": len(trials),
        "candidate_mode": candidate_mode,
        "protocol": {key: reference[key] for key in PROTOCOL_KEYS},
        "input_ids_sha256": reference["results"][0]["input_ids_sha256"],
        "trials": trials,
        "tiled_median_seconds_across_trials": _range(tiled_medians),
        "cuda_mma_median_seconds_across_trials": _range(cuda_medians),
        "tiled_ttft_seconds_across_trials": _range(tiled_ttft),
        "cuda_mma_ttft_seconds_across_trials": _range(cuda_ttft),
        "tiled_tpot_seconds_across_trials": _range(tiled_tpot),
        "cuda_mma_tpot_seconds_across_trials": _range(cuda_tpot),
        "ttft_speedup_across_trials": _range(ttft_speedups),
        "tpot_speedup_across_trials": _range(tpot_speedups),
        "speedup_across_trials": _range(speedups),
        "aggregation": (
            "Each trial reports the median of seven steady repetitions; the "
            "headline value is the median of three independent trial medians."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trials", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = summarize_trials(args.trials)
    except ValueError as error:
        parser.error(str(error))
    rendered = json.dumps(output, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
