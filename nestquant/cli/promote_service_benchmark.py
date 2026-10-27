"""Promote a validated raw service benchmark into a compact reviewable summary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .validate_service_benchmark import validate_report
from ..runtime.service_benchmark import validate_benchmark_config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_config_report_match(config: dict, report: dict) -> None:
    validate_benchmark_config(config)
    expected = {
        "capacity": config["capacity"],
        "max_cache_len": config["max_cache_len"],
        "profiles": config["profiles"],
        "policies": config["policies"],
    }
    actual = {key: report[key] for key in expected}
    if actual != expected:
        raise ValueError("raw report does not match the supplied benchmark config")
    if any(int(item["repetitions"]) != int(config["repetitions"])
           for item in report["summaries"]):
        raise ValueError("raw report repetition count does not match the supplied config")


def build_summary(
    report: dict, config: dict, *, raw_sha256: str, config_sha256: str
) -> dict:
    validation = validate_report(report)
    validate_config_report_match(config, report)
    return {
        "schema_version": 1,
        "status": "passed",
        "device": report["device"],
        "torch_version": report["torch_version"],
        "transformers_version": report["transformers_version"],
        "capacity": report["capacity"],
        "max_cache_len": report["max_cache_len"],
        "source_raw_sha256": raw_sha256,
        "source_config_sha256": config_sha256,
        "validation": validation,
        "profiles": report["profiles"],
        "policies": report["policies"],
        "summaries": report["summaries"],
        "claim_boundary": report["claim_boundary"],
    }


def verify_summary(
    report: dict,
    config: dict,
    summary: dict,
    *,
    raw_sha256: str,
    config_sha256: str,
) -> dict:
    """Verify that an existing compact summary is exactly reproducible."""
    expected = build_summary(
        report,
        config,
        raw_sha256=raw_sha256,
        config_sha256=config_sha256,
    )
    if summary != expected:
        raise ValueError(
            "promoted summary does not match the validated raw report and config"
        )
    return {
        "status": "passed",
        "source_raw_sha256": raw_sha256,
        "source_config_sha256": config_sha256,
        "runs": expected["validation"]["runs"],
        "completed_requests": expected["validation"]["completed_requests"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    summary = build_summary(
        report, config,
        raw_sha256=sha256_file(args.report),
        config_sha256=sha256_file(args.config),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "passed",
        "output": str(args.output),
        "source_raw_sha256": summary["source_raw_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
