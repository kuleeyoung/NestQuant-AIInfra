from __future__ import annotations

import pytest

from scripts.summarize_http_nsys_ab import summarize


def _run(rps: float, ttft: float, tpot: float, graph_calls: int) -> dict:
    return {
        "offered_rps": rps,
        "benchmark": {
            "failed_requests": 0,
            "request_throughput_per_second": rps,
            "ttft_p95_ms": ttft,
            "tpot_p95_ms": tpot,
            "e2e_p95_ms": ttft + 100,
        },
        "gpu": {"utilization_mean_percent": rps * 10},
        "cuda_apis": {
            "cudaGraphLaunch_v10000": {"calls_per_request": graph_calls / 24},
            "cudaLaunchKernel": {"calls_per_request": graph_calls * 100 / 24},
        },
    }


def test_summarize_reports_queueing_and_batching_ratios() -> None:
    result = summarize(_run(2, 180, 70, 48), _run(4, 450, 71, 40))
    ratios = result["candidate_over_baseline"]
    assert result["status"] == "passed"
    assert ratios["throughput"] == pytest.approx(2.0)
    assert ratios["tpot_p95"] == pytest.approx(71 / 70)
    assert ratios["cuda_graph_launch_calls_per_request"] == pytest.approx(40 / 48)


def test_summarize_propagates_failure() -> None:
    candidate = _run(4, 450, 71, 40)
    candidate["benchmark"]["failed_requests"] = 1
    assert summarize(_run(2, 180, 70, 48), candidate)["status"] == "failed"
