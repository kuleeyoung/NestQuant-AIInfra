import hashlib
from pathlib import Path

import pytest

from nestquant.cli.promote_service_benchmark import (
    build_summary,
    sha256_file,
    validate_config_report_match,
    verify_summary,
)


def valid_report() -> dict:
    run = {
        "run_id": "p-a-r0",
        "profile": "p",
        "policy": "a",
        "scheduled_requests": 2,
        "completed_requests": 2,
        "failed_requests": 0,
        "rejected_requests": {},
        "completion_rate": 1.0,
        "final_registered_requests": 0,
        "repetition": 0,
    }
    for prefix in ("ttft_ms", "tpot_ms", "e2e_ms", "step_latency_ms"):
        run.update({
            f"{prefix}_p50": 1.0,
            f"{prefix}_p95": 2.0,
            f"{prefix}_p99": 3.0,
        })
    return {
        "status": "passed",
        "profiles": [{
            "name": "p", "length_mode": "uniform", "prompt_length": [4, 5],
        }],
        "policies": [{"name": "a"}],
        "warmup_prompt_lengths": [4, 5],
        "runs": [run],
        "summaries": [{"profile": "p", "policy": "a", "repetitions": 1}],
    }


def test_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / "sample.json"
    path.write_bytes(b"evidence")
    assert sha256_file(path) == hashlib.sha256(b"evidence").hexdigest()


def test_build_summary_keeps_reviewable_evidence_only() -> None:
    report = valid_report()
    report["policies"] = [{
        "name": "a", "min_prefill_batch": 1, "max_wait_steps": 0,
    }]
    report.update({
        "device": "gpu", "torch_version": "x", "transformers_version": "y",
        "capacity": 1, "max_cache_len": 8, "claim_boundary": "test",
    })
    config = {
        "seed": 1, "capacity": 1, "max_cache_len": 8, "repetitions": 1,
        "policies": report["policies"],
        "profiles": [{
            **report["profiles"][0], "request_count": 2,
            "arrival_mode": "fixed", "arrival_interval_ms": 10,
            "output_length": [1, 2], "max_cache_len": 8,
        }],
    }
    report["profiles"] = config["profiles"]
    summary = build_summary(
        report, config, raw_sha256="a" * 64, config_sha256="b" * 64
    )
    assert summary["validation"]["completed_requests"] == 2
    assert summary["source_raw_sha256"] == "a" * 64
    assert "runs" not in summary


def test_promote_rejects_wrong_config() -> None:
    report = valid_report()
    report["policies"] = [{
        "name": "a", "min_prefill_batch": 1, "max_wait_steps": 0,
    }]
    report.update({"capacity": 1, "max_cache_len": 8})
    config = {
        "seed": 1, "capacity": 2, "max_cache_len": 8, "repetitions": 1,
        "policies": report["policies"],
        "profiles": [{
            **report["profiles"][0], "request_count": 2,
            "arrival_mode": "fixed", "arrival_interval_ms": 10,
            "output_length": [1, 2], "max_cache_len": 8,
        }],
    }
    report["profiles"] = config["profiles"]
    with pytest.raises(ValueError, match="does not match"):
        validate_config_report_match(config, report)


def test_verify_summary_rejects_tampering() -> None:
    report = valid_report()
    report["policies"] = [{
        "name": "a", "min_prefill_batch": 1, "max_wait_steps": 0,
    }]
    report.update({
        "device": "gpu", "torch_version": "x", "transformers_version": "y",
        "capacity": 1, "max_cache_len": 8, "claim_boundary": "test",
    })
    config = {
        "seed": 1, "capacity": 1, "max_cache_len": 8, "repetitions": 1,
        "policies": report["policies"],
        "profiles": [{
            **report["profiles"][0], "request_count": 2,
            "arrival_mode": "fixed", "arrival_interval_ms": 10,
            "output_length": [1, 2], "max_cache_len": 8,
        }],
    }
    report["profiles"] = config["profiles"]
    summary = build_summary(
        report, config, raw_sha256="a" * 64, config_sha256="b" * 64
    )
    assert verify_summary(
        report, config, summary,
        raw_sha256="a" * 64, config_sha256="b" * 64,
    )["completed_requests"] == 2
    summary["source_raw_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="does not match"):
        verify_summary(
            report, config, summary,
            raw_sha256="a" * 64, config_sha256="b" * 64,
        )
