import pytest

from nestquant.runtime.service_benchmark import (
    distribution,
    generate_workload,
    percentile,
    summarize_repetitions,
    validate_benchmark_config,
    workload_seed,
)


def test_percentile_interpolates_and_validates() -> None:
    assert percentile([0, 10], 0.5) == 5
    assert percentile([3], 0.99) == 3
    with pytest.raises(ValueError):
        percentile([], 0.5)


def test_fixed_and_burst_arrivals_are_deterministic() -> None:
    base = {
        "request_count": 5,
        "length_mode": "uniform",
        "prompt_length": [2, 2],
        "output_length": [1, 1],
        "max_cache_len": 8,
    }
    fixed = generate_workload({
        **base, "arrival_mode": "fixed", "arrival_interval_ms": 20,
    }, seed=1)
    burst = generate_workload({
        **base, "arrival_mode": "burst", "burst_size": 2,
        "burst_interval_ms": 100,
    }, seed=1)
    assert [item.arrival_seconds for item in fixed] == [0, .02, .04, .06, .08]
    assert [item.arrival_seconds for item in burst] == [0, 0, .1, .1, .2]


def test_bimodal_workload_respects_cache_limit() -> None:
    profile = {
        "request_count": 100,
        "arrival_mode": "poisson",
        "mean_arrival_interval_ms": 50,
        "length_mode": "bimodal",
        "long_fraction": 0.25,
        "short": {"prompt_length": [4, 8], "output_length": [2, 4]},
        "long": {"prompt_length": [32, 48], "output_length": [6, 12]},
        "max_cache_len": 64,
    }
    first = generate_workload(profile, seed=7)
    second = generate_workload(profile, seed=7)
    assert first == second
    assert all(item.prompt_length + item.output_length <= 64 for item in first)
    assert any(item.prompt_length >= 32 for item in first)
    assert any(item.prompt_length <= 8 for item in first)


def test_distribution_and_repetition_summary() -> None:
    assert distribution([1, 2, 3])["p50"] == 2
    with pytest.raises(ValueError, match="finite"):
        distribution([1, float("nan")])
    template = {
        "request_throughput_per_second": 10.0,
        "token_throughput_per_second": 20.0,
        "completion_rate": 1.0,
        "ttft_p50_ms": 1.0, "ttft_p95_ms": 2.0, "ttft_p99_ms": 3.0,
        "tpot_p50_ms": 4.0, "tpot_p95_ms": 5.0, "tpot_p99_ms": 6.0,
        "e2e_p50_ms": 7.0, "e2e_p95_ms": 8.0, "e2e_p99_ms": 9.0,
        "arrival_lag_p95_ms": 0.5, "queue_depth_p95": 2.0,
        "cuda_peak_allocated_bytes": 100.0,
        "cuda_peak_reserved_bytes": 120.0,
    }
    result = summarize_repetitions([template, {**template, "ttft_p95_ms": 4.0}])
    assert result["ttft_p95_ms"]["mean"] == 3.0


def test_capacity_sweep_can_pair_workloads_across_profiles() -> None:
    assert workload_seed(100, 0, 2, couple_profiles=True) == 102
    assert workload_seed(100, 5, 2, couple_profiles=True) == 102
    assert workload_seed(100, 5, 2, couple_profiles=False) == 5102
    base = {
        "request_count": 20, "arrival_mode": "fixed", "length_mode": "uniform",
        "prompt_length": [2, 6], "output_length": [1, 3], "max_cache_len": 10,
    }
    slow = generate_workload({**base, "arrival_interval_ms": 250}, seed=102)
    fast = generate_workload({**base, "arrival_interval_ms": 20}, seed=102)
    assert [
        (item.prompt_length, item.output_length) for item in slow
    ] == [
        (item.prompt_length, item.output_length) for item in fast
    ]


@pytest.mark.parametrize("patch", [
    {"arrival_interval_ms": 0},
    {"prompt_length": [8, 4]},
    {"output_length": [0, 1]},
    {"max_cache_len": 1},
    {"max_cache_len": 8.5},
    {"request_count": True},
])
def test_workload_rejects_invalid_ranges_before_gpu_execution(patch: dict) -> None:
    profile = {
        "request_count": 2,
        "arrival_mode": "fixed",
        "arrival_interval_ms": 20,
        "length_mode": "uniform",
        "prompt_length": [2, 4],
        "output_length": [1, 2],
        "max_cache_len": 8,
        **patch,
    }
    with pytest.raises(ValueError):
        generate_workload(profile, seed=1)


@pytest.mark.parametrize("long_fraction", [-0.1, 1.1, float("nan"), True])
def test_bimodal_workload_rejects_invalid_fraction(long_fraction: float) -> None:
    profile = {
        "request_count": 2,
        "arrival_mode": "fixed",
        "arrival_interval_ms": 20,
        "length_mode": "bimodal",
        "long_fraction": long_fraction,
        "short": {"prompt_length": [2, 2], "output_length": [1, 1]},
        "long": {"prompt_length": [3, 3], "output_length": [1, 1]},
        "max_cache_len": 8,
    }
    with pytest.raises(ValueError, match="long_fraction"):
        generate_workload(profile, seed=1)


def benchmark_config() -> dict:
    return {
        "seed": 1, "capacity": 2, "max_cache_len": 8, "repetitions": 1,
        "policies": [{
            "name": "p", "min_prefill_batch": 1, "max_wait_steps": 0,
        }],
        "profiles": [{
            "name": "w", "request_count": 2, "arrival_mode": "fixed",
            "arrival_interval_ms": 10, "length_mode": "uniform",
            "prompt_length": [2, 3], "output_length": [1, 2], "max_cache_len": 8,
        }],
    }


def test_benchmark_config_accepts_consistent_matrix() -> None:
    validate_benchmark_config(benchmark_config())


@pytest.mark.parametrize("mutation", ["duplicate_profile", "duplicate_policy", "cache", "capacity"])
def test_benchmark_config_rejects_matrix_errors(mutation: str) -> None:
    config = benchmark_config()
    if mutation == "duplicate_profile":
        config["profiles"].append(dict(config["profiles"][0]))
    elif mutation == "duplicate_policy":
        config["policies"].append(dict(config["policies"][0]))
    elif mutation == "cache":
        config["profiles"][0]["max_cache_len"] = 9
    else:
        config["capacity"] = 0
    with pytest.raises(ValueError):
        validate_benchmark_config(config)
