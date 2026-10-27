"""Promote validated hybrid-residency evidence into a compact summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .validate_hybrid_pareto import sha256_file, validate_hybrid_report


def build_summary(report: dict, validation: dict, *, hashes: dict[str, str]) -> dict:
    points = []
    for item in report["results"]:
        points.append({
            "budget_mib": item["budget_mib"],
            "resident_projection_count": item["resident_projection_count"],
            "packed_projection_count": item["packed_projection_count"],
            "cuda_allocated_bytes_after_warmup": item[
                "cuda_allocated_bytes_after_warmup"
            ],
            "actual_packed_device_bytes": item["actual_packed_device_bytes"],
            "steady_prefill_median_ms": item["steady_prefill_seconds"]["median"] * 1000,
            "ttft_median_ms": item["ttft"]["median_ms"],
            "ttft_p95_ms": item["ttft"]["p95_ms"],
            "tpot_median_ms": item["tpot"]["median_ms"],
            "tpot_p95_ms": item["tpot"]["p95_ms"],
            "decode_tokens_per_second_per_request": item[
                "decode_tokens_per_second_per_request"
            ],
            "relative_l2_logit_error_vs_bf16": item[
                "relative_l2_logit_error_vs_bf16"
            ],
        })
    return {
        "schema_version": 1,
        "status": "passed",
        "device": report["device"],
        "torch_version": report["torch_version"],
        "model_id": report["model_id"],
        "model_revision": report["model_revision"],
        "source_raw_sha256": hashes["raw"],
        "source_profile_sha256": hashes["profile"],
        "source_config_sha256": hashes["config"],
        "checkpoint_sha256": hashes["checkpoint"],
        "validation": validation,
        "pareto_budget_mib": report["pareto_budget_mib"],
        "points": points,
        "claim_boundary": report["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    hashes = {
        "raw": sha256_file(args.report),
        "profile": sha256_file(args.profile),
        "config": sha256_file(args.config),
        "checkpoint": sha256_file(args.checkpoint),
    }
    validation = validate_hybrid_report(
        report, profile, config,
        checkpoint_sha256=hashes["checkpoint"],
        profile_sha256=hashes["profile"],
        config_sha256=hashes["config"],
    )
    summary = build_summary(report, validation, hashes=hashes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "passed", "output": str(args.output),
        "source_raw_sha256": hashes["raw"],
    }, indent=2))


if __name__ == "__main__":
    main()
