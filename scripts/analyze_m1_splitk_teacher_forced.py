"""Analyze paired teacher-forced split-K quality with context bootstrap."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def analyze(
    payload: dict[str, Any], *, seed: int, bootstrap_repetitions: int
) -> dict[str, Any]:
    samples = payload["samples"]
    if not samples:
        raise ValueError("teacher-forced payload has no samples")
    deltas = [
        statistics.mean(
            step["candidate_nll"] - step["baseline_nll"]
            for step in sample["steps"]
        )
        for sample in samples
    ]
    generator = random.Random(seed)
    bootstrap = [
        statistics.mean(generator.choice(deltas) for _ in deltas)
        for _ in range(bootstrap_repetitions)
    ]
    lower = percentile(bootstrap, 0.025)
    upper = percentile(bootstrap, 0.975)
    mean = statistics.mean(deltas)
    return {
        "schema_version": 1,
        "protocol": "paired context-bootstrap teacher-forced split-K analysis v1",
        "source_protocol": payload["protocol"],
        "sample_plan_sha256": payload["sample_plan_sha256"],
        "token_ids_sha256": payload["token_ids_sha256"],
        "sample_count": len(samples),
        "predicted_token_count": payload["summary"]["predicted_token_count"],
        "bootstrap_seed": seed,
        "bootstrap_repetitions": bootstrap_repetitions,
        "paired_context_mean_nll_delta": mean,
        "paired_context_median_nll_delta": statistics.median(deltas),
        "paired_context_min_nll_delta": min(deltas),
        "paired_context_max_nll_delta": max(deltas),
        "candidate_worse_context_count": sum(delta > 0.0 for delta in deltas),
        "candidate_better_context_count": sum(delta < 0.0 for delta in deltas),
        "mean_nll_delta_bootstrap_95ci": [lower, upper],
        "relative_ppl_delta": math.exp(mean) - 1.0,
        "relative_ppl_delta_bootstrap_95ci": [
            math.exp(lower) - 1.0,
            math.exp(upper) - 1.0,
        ],
        "argmax_agreement_rate": payload["summary"]["argmax_agreement_rate"],
        "mean_kl": payload["summary"]["mean_kl"],
        "reproducibility_max_abs_logit_error": payload["summary"][
            "reproducibility_max_abs_logit_error"
        ],
        "claim_boundary": (
            "The bootstrap resamples complete contexts, not individual correlated "
            "tokens. This is a sampled teacher-forced decode quality estimate, not "
            "full-corpus perplexity or downstream accuracy."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    args = parser.parse_args()
    if args.bootstrap_repetitions < 1000:
        parser.error("bootstrap repetitions must be at least 1000")
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = analyze(
        payload, seed=args.seed,
        bootstrap_repetitions=args.bootstrap_repetitions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
