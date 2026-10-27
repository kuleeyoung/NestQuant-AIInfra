"""Validate a full-model packed/BF16 hybrid-residency Pareto report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from ..runtime.hybrid_residency import (
    nondominated_indices,
    residency_groups_from_payload,
    select_residency_plan,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_hybrid_report(
    report: dict,
    profile: dict,
    config: dict,
    *,
    checkpoint_sha256: str,
    profile_sha256: str,
    config_sha256: str,
) -> dict:
    if report.get("status") != "passed" or profile.get("status") != "passed":
        raise ValueError("report and profile status must be passed")
    expected_hashes = {
        "checkpoint_sha256": checkpoint_sha256,
        "profile_sha256": profile_sha256,
        "config_sha256": config_sha256,
    }
    for name, expected in expected_hashes.items():
        if report.get(name) != expected:
            raise ValueError(f"{name} does not match the supplied evidence")
    if profile.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("profile checkpoint hash does not match")
    if report.get("config") != config:
        raise ValueError("embedded benchmark config does not match")

    groups = residency_groups_from_payload(profile["groups_ranked"])
    all_names = tuple(
        profile_item["name"] for profile_item in profile["profiles"]
    )
    if len(set(all_names)) != len(all_names):
        raise ValueError("profile projection names are not unique")
    if {name for group in groups for name in group.projection_names} != set(all_names):
        raise ValueError("profile groups do not cover every projection")

    budgets = config["budget_mib"]
    results = report.get("results")
    if not isinstance(results, list) or [item.get("budget_mib") for item in results] != budgets:
        raise ValueError("report budget matrix does not match the config")
    total = len(all_names)
    prior_resident = -1
    prior_allocated = -1
    prior_packed_bytes = math.inf
    for item in results:
        budget = int(item["budget_mib"])
        plan = select_residency_plan(
            groups,
            budget_bytes=budget * 1024 * 1024,
            all_projection_names=all_names,
        )
        if tuple(item["resident_projection_names"]) != plan.resident_projection_names:
            raise ValueError(f"resident selection does not match the policy at {budget} MiB")
        if (item["resident_projection_count"] != len(plan.resident_projection_names)
                or item["packed_projection_count"] != len(plan.packed_projection_names)
                or item["resident_projection_count"] + item["packed_projection_count"] != total):
            raise ValueError(f"projection counts are inconsistent at {budget} MiB")
        if item["converted_projection_count"] != item["packed_projection_count"]:
            raise ValueError(f"converted projection count is inconsistent at {budget} MiB")
        resident = int(item["resident_projection_count"])
        allocated = int(item["cuda_allocated_bytes_after_warmup"])
        packed_bytes = int(item["actual_packed_device_bytes"])
        if resident < prior_resident or allocated < prior_allocated or packed_bytes > prior_packed_bytes:
            raise ValueError("memory/residency curve is not monotonic")
        prior_resident, prior_allocated, prior_packed_bytes = resident, allocated, packed_bytes
        for section, fields in (("ttft", ("median_ms", "p95_ms")),
                                ("tpot", ("median_ms", "p95_ms"))):
            for field in fields:
                value = item[section][field]
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"invalid {section}.{field} at {budget} MiB")
    if results[0]["resident_projection_count"] != 0:
        raise ValueError("first point must be fully packed")
    if results[-1]["resident_projection_count"] != total:
        raise ValueError("last point must be fully BF16 resident")
    if results[0]["relative_l2_logit_error_vs_packed"] != 0.0:
        raise ValueError("packed endpoint is not self-consistent")
    if results[-1]["relative_l2_logit_error_vs_bf16"] != 0.0:
        raise ValueError("BF16 endpoint is not self-consistent")

    pareto_inputs = [{
        "memory": item["cuda_allocated_bytes_after_warmup"],
        "ttft": item["ttft"]["median_ms"],
        "tpot": item["tpot"]["median_ms"],
    } for item in results]
    indices = nondominated_indices(
        pareto_inputs, metric_names=("memory", "ttft", "tpot")
    )
    budgets_on_frontier = [results[index]["budget_mib"] for index in indices]
    if report.get("pareto_budget_mib") != budgets_on_frontier:
        raise ValueError("reported Pareto frontier is stale or incorrect")
    return {
        "status": "passed",
        "projection_count": total,
        "group_count": len(groups),
        "budget_points": len(results),
        "pareto_budget_mib": budgets_on_frontier,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = validate_hybrid_report(
        report, profile, config,
        checkpoint_sha256=sha256_file(args.checkpoint),
        profile_sha256=sha256_file(args.profile),
        config_sha256=sha256_file(args.config),
    )
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
