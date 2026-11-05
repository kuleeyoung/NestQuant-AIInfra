import json

import pytest

from scripts.summarize_http_capacity import summarize


def _write(path, *, interval, completed, overloaded, failed=0):
    scheduled = completed + overloaded
    path.write_text(json.dumps({
        "config": {
            "requests": scheduled, "prompt": "x", "max_new_tokens": 3,
            "client_workers": 4, "arrival_interval_ms": interval,
        },
        "scheduled_requests": scheduled,
        "completed_requests": completed,
        "overloaded_requests": overloaded,
        "failed_requests": failed,
        "overload_rate": overloaded / scheduled,
        "request_throughput_per_second": completed / 2,
        "token_throughput_per_second": completed * 3 / 2,
        "ttft_ms": {"p50": 10, "p95": 20},
        "tpot_ms": {"p95": 5},
        "e2e_ms": {"p95": 30},
    }))


def test_summarize_http_capacity_finds_lossless_and_overload_knee(tmp_path) -> None:
    low = tmp_path / "low.json"
    high = tmp_path / "high.json"
    _write(low, interval=250, completed=4, overloaded=0)
    _write(high, interval=62.5, completed=3, overloaded=1)
    result = summarize([high, low])
    assert result["last_lossless_offered_requests_per_second"] == 4
    assert result["first_overloaded_offered_requests_per_second"] == 16
    assert result["points"][0]["offered_requests_per_second"] == 4


def test_summarize_http_capacity_rejects_failures(tmp_path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write(first, interval=250, completed=4, overloaded=0)
    _write(second, interval=125, completed=4, overloaded=0, failed=1)
    with pytest.raises(ValueError, match="failures"):
        summarize([first, second])
