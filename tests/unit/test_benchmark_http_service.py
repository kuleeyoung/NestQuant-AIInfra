from scripts.benchmark_http_service import distribution, identity_sha256, percentile


def test_http_benchmark_percentile_interpolates() -> None:
    assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert percentile([1.0, 3.0], 0.5) == 2.0
    assert percentile([], 0.95) == 0.0


def test_http_benchmark_distribution_has_tail_metrics() -> None:
    result = distribution([1.0, 2.0, 10.0])
    assert result["count"] == 3
    assert result["p50"] == 2.0
    assert result["p95"] > result["p50"]
    assert result["max"] == 10.0


def test_http_identity_sha256_is_order_stable_and_token_sensitive() -> None:
    left = {"request_id": "r0", "token_ids": [1, 2], "status": "finished"}
    reordered = {"status": "finished", "token_ids": [1, 2], "request_id": "r0"}
    changed = {"request_id": "r0", "token_ids": [1, 3], "status": "finished"}
    assert identity_sha256(left) == identity_sha256(reordered)
    assert identity_sha256(left) != identity_sha256(changed)
