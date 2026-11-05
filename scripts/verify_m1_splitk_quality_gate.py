"""Verify the predeclared M1 split-K teacher-forced non-inferiority gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def verify(
    analysis: dict[str, Any], *, margin: float, min_contexts: int
) -> dict[str, Any]:
    if margin <= 0.0 or min_contexts < 1:
        raise ValueError("margin and minimum contexts must be positive")
    lower, upper = analysis["relative_ppl_delta_bootstrap_95ci"]
    contexts = int(analysis["sample_count"])
    enough_contexts = contexts >= min_contexts
    passed = enough_contexts and upper <= margin
    failed = enough_contexts and lower > margin
    status = "passed" if passed else "failed" if failed else "inconclusive"
    return {
        "schema_version": 1,
        "status": status,
        "accepted": passed,
        "thresholds": {
            "relative_ppl_non_inferiority_margin": margin,
            "minimum_contexts": min_contexts,
            "confidence_level": 0.95,
        },
        "checks": {
            "minimum_contexts": enough_contexts,
            "ci_upper_within_margin": upper <= margin,
            "ci_lower_above_margin": lower > margin,
            "deterministic_replay": (
                analysis["reproducibility_max_abs_logit_error"] == 0.0
            ),
        },
        "evidence": {
            "sample_count": contexts,
            "predicted_token_count": analysis["predicted_token_count"],
            "relative_ppl_delta": analysis["relative_ppl_delta"],
            "relative_ppl_delta_bootstrap_95ci": [lower, upper],
            "argmax_agreement_rate": analysis["argmax_agreement_rate"],
            "mean_kl": analysis["mean_kl"],
        },
        "interpretation": (
            "The 95% CI upper bound is within the predeclared margin."
            if passed else
            "The 95% CI lower bound is above the predeclared margin."
            if failed else
            "The evidence is insufficient for pass or fail at the predeclared margin."
        ),
        "claim_boundary": (
            "This gate covers deterministic sampled teacher-forced M1 decode on the "
            "production NQKV Graph runner. It is not full-corpus PPL or a guarantee "
            "of identical autoregressive token sequences."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--margin", type=float, default=0.01)
    parser.add_argument("--min-contexts", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    result = verify(
        analysis, margin=args.margin, min_contexts=args.min_contexts
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] == "failed":
        raise SystemExit(2)
    if result["status"] == "inconclusive":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
