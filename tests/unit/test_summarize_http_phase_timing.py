from __future__ import annotations

import json

import pytest

from scripts.summarize_http_phase_timing import PHASES, summarize


def _payload(interval: float, *, overloaded: int = 0) -> dict:
    timing = {
        phase: {"count": 2, "mean": 10.0, "p50": 9.0, "p95": 12.0,
                "p99": 13.0, "max": 14.0}
        for phase in PHASES
    }
    timing["command_queue"]["mean"] = 20.0
    timing["waiting_queue"]["mean"] = 30.0
    timing["server_ttft"]["mean"] = 100.0
    timing["server_ttft"]["p95"] = 110.0
    return {
        "config": {
            "requests": 2, "prompt": "p", "max_new_tokens": 3,
            "client_workers": 4, "arrival_interval_ms": interval,
        },
        "failed_requests": 0,
        "overloaded_requests": overloaded,
        "ttft_ms": {"mean": 103.0, "p95": 113.0},
        "e2e_ms": {"mean": 120.0, "p95": 130.0},
        "server_timing_ms": timing,
    }


def test_phase_summary_uses_additive_means_and_preserves_tail_gap(tmp_path) -> None:
    paths = []
    for index, interval in enumerate((500.0, 250.0)):
        path = tmp_path / f"{index}.json"
        path.write_text(json.dumps(_payload(interval)), encoding="utf-8")
        paths.append(path)
    result = summarize(paths)
    assert result["points"][0]["queue_fraction_of_server_ttft_mean"] == 0.5
    assert result["points"][0]["client_minus_server_ttft_p95_ms"] == 3.0
    assert result["points"][1]["offered_requests_per_second"] == 4.0


def test_phase_summary_rejects_overloaded_point(tmp_path) -> None:
    paths = []
    for index in range(2):
        path = tmp_path / f"{index}.json"
        path.write_text(
            json.dumps(_payload(500.0, overloaded=index)), encoding="utf-8"
        )
        paths.append(path)
    with pytest.raises(ValueError, match="lossless"):
        summarize(paths)
