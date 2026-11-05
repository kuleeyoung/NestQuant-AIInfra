from __future__ import annotations

import json

import pytest

from scripts.summarize_admission_policy_ab import summarize


def _benchmark(throughput: float, tpot: float) -> dict:
    timing = {
        "command_queue": {"mean": 10.0},
        "waiting_queue": {"mean": 20.0},
        "prefill_compute": {"mean": 30.0},
    }
    return {
        "failed_requests": 0, "overloaded_requests": 0,
        "request_throughput_per_second": throughput,
        "ttft_ms": {"p95": 100.0}, "tpot_ms": {"p95": tpot},
        "e2e_ms": {"p95": 200.0}, "server_timing_ms": timing,
    }


def _write(path, values) -> None:
    path.write_text(json.dumps({
        "trials": [{"benchmark": value} for value in values]
    }), encoding="utf-8")


def test_admission_summary_respects_metric_direction(tmp_path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write(baseline, [_benchmark(4.0, 60.0), _benchmark(4.0, 62.0)])
    _write(candidate, [_benchmark(3.8, 120.0), _benchmark(3.9, 125.0)])
    result = summarize(baseline, candidate)
    assert result["metrics"]["request_throughput_per_second"]["candidate_improved_pairs"] == 0
    assert result["metrics"]["tpot_p95_ms"]["candidate_improved_pairs"] == 0
    assert result["metrics"]["tpot_p95_ms"]["candidate_relative_change"] > 0


def test_admission_summary_requires_equal_trial_counts(tmp_path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write(baseline, [_benchmark(4.0, 60.0)])
    _write(candidate, [_benchmark(4.0, 60.0), _benchmark(4.0, 60.0)])
    with pytest.raises(ValueError, match="equal trial counts"):
        summarize(baseline, candidate)
