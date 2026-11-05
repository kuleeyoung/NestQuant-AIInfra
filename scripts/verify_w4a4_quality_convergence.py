"""Verify paired W4A4 WikiText quality and performance acceptance gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError(f"incomplete evaluation: {path}")
    return payload


def _pair(baseline: dict, candidate: dict) -> dict[str, float | int | bool]:
    baseline_signature = baseline["run_signature"]
    candidate_signature = candidate["run_signature"]
    compared_signature_fields = (
        "checkpoint_sha256", "model_revision", "dataset_revision",
        "context_length", "logit_chunk_size",
    )
    mismatches = [
        field for field in compared_signature_fields
        if baseline_signature[field] != candidate_signature[field]
    ]
    if baseline["data"]["token_ids_sha256"] != candidate["data"]["token_ids_sha256"]:
        mismatches.append("token_ids_sha256")
    if baseline["data"]["evaluated_chunk_count"] != candidate["data"]["evaluated_chunk_count"]:
        mismatches.append("evaluated_chunk_count")
    if baseline["data"]["predicted_token_count"] != candidate["data"]["predicted_token_count"]:
        mismatches.append("predicted_token_count")
    if mismatches:
        raise ValueError("unpaired evaluations: " + ", ".join(mismatches))

    baseline_ppl = float(baseline["result"]["perplexity"])
    candidate_ppl = float(candidate["result"]["perplexity"])
    baseline_seconds = float(baseline["result"]["elapsed_seconds"])
    candidate_seconds = float(candidate["result"]["elapsed_seconds"])
    return {
        "windows": int(candidate["data"]["evaluated_chunk_count"]),
        "predicted_tokens": int(candidate["data"]["predicted_token_count"]),
        "baseline_ppl": baseline_ppl,
        "candidate_ppl": candidate_ppl,
        "relative_ppl_delta": candidate_ppl / baseline_ppl - 1.0,
        "evaluation_speedup": baseline_seconds / candidate_seconds,
        "token_hash_equal": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--prior-baseline", type=Path)
    parser.add_argument("--prior-candidate", type=Path)
    parser.add_argument("--min-windows", type=int, default=512)
    parser.add_argument("--max-absolute-relative-ppl-delta", type=float, default=0.005)
    parser.add_argument("--require-speedup", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (args.prior_baseline is None) != (args.prior_candidate is None):
        parser.error("prior baseline and candidate must be provided together")

    current = _pair(_load(args.baseline), _load(args.candidate))
    prior = None
    if args.prior_baseline is not None:
        prior = _pair(_load(args.prior_baseline), _load(args.prior_candidate))

    checks = {
        "minimum_windows": current["windows"] >= args.min_windows,
        "ppl_delta_within_limit": (
            abs(current["relative_ppl_delta"])
            <= args.max_absolute_relative_ppl_delta
        ),
        "speedup_above_one": (
            current["evaluation_speedup"] > 1.0
            if args.require_speedup else True
        ),
        "delta_not_growing_from_prior": (
            abs(current["relative_ppl_delta"])
            <= abs(prior["relative_ppl_delta"])
            if prior is not None else True
        ),
    }
    payload = {
        "accepted": all(checks.values()),
        "thresholds": {
            "min_windows": args.min_windows,
            "max_absolute_relative_ppl_delta": (
                args.max_absolute_relative_ppl_delta
            ),
            "require_speedup": args.require_speedup,
        },
        "checks": checks,
        "current": current,
        "prior": prior,
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if not payload["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
