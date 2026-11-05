from scripts.benchmark_service_workloads import _identity_sha256


def test_identity_sha256_is_order_stable_but_value_sensitive() -> None:
    left = {"prompt": [1, 2, 3], "output": {"b": 2, "a": 1}}
    reordered = {"output": {"a": 1, "b": 2}, "prompt": [1, 2, 3]}
    changed = {"prompt": [1, 2, 4], "output": {"a": 1, "b": 2}}

    assert _identity_sha256(left) == _identity_sha256(reordered)
    assert _identity_sha256(left) != _identity_sha256(changed)
