from __future__ import annotations

import pytest

from scripts.benchmark_http_fresh_process import (
    coefficient_of_variation,
    parse_nvidia_smi_line,
    summarize_trials,
)


def _trial(tpot_p95: float, ttft_p95: float = 100.0) -> dict:
    return {
        "benchmark": {
            "status": "passed",
            "failed_requests": 0,
            "request_throughput_per_second": 2.0,
            "token_throughput_per_second": 6.0,
            "ttft_ms": {"p50": 80.0, "p95": ttft_p95},
            "tpot_ms": {"p50": 50.0, "p95": tpot_p95},
            "e2e_ms": {"p50": 200.0, "p95": 250.0},
        }
    }


def test_parse_nvidia_smi_line() -> None:
    parsed = parse_nvidia_smi_line(
        "2026/11/03 00:24:47.374, NVIDIA GeForce RTX 5070, P0, "
        "2677, 14001, 37.11, 39.57, 0, 1253"
    )
    assert parsed["name"] == "NVIDIA GeForce RTX 5070"
    assert parsed["pstate"] == "P0"
    assert parsed["sm_clock_mhz"] == pytest.approx(2677.0)
    assert parsed["memory_used_mib"] == pytest.approx(1253.0)


def test_parse_nvidia_smi_line_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="expected 9"):
        parse_nvidia_smi_line("P0, 2000")


def test_coefficient_of_variation() -> None:
    assert coefficient_of_variation([10.0]) == 0.0
    assert coefficient_of_variation([10.0, 20.0]) == pytest.approx(1.0 / 3.0)


def test_summarize_trials_preserves_values_and_cv() -> None:
    summary = summarize_trials([_trial(60.0), _trial(180.0)])
    assert summary["all_trials_passed"] is True
    assert summary["metrics"]["tpot_p95_ms"]["values"] == [60.0, 180.0]
    assert summary["metrics"]["tpot_p95_ms"]["median"] == 120.0
    assert summary["metrics"]["tpot_p95_ms"]["cv"] == pytest.approx(0.5)


def test_summarize_trials_marks_failure() -> None:
    failed = _trial(60.0)
    failed["benchmark"]["status"] = "failed"
    assert summarize_trials([failed])["all_trials_passed"] is False
