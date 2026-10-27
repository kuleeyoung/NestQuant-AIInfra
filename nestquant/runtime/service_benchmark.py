"""Deterministic workload generation and metric aggregation for engine benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import statistics
from typing import Iterable


def percentile(values: Iterable[float], fraction: float) -> float:
    """Return a linearly interpolated percentile without third-party dependencies."""
    items = sorted(float(value) for value in values)
    if not items:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    position = (len(items) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return items[lower]
    weight = position - lower
    return items[lower] * (1.0 - weight) + items[upper] * weight


def distribution(values: Iterable[float]) -> dict[str, float]:
    items = [float(value) for value in values]
    if not items:
        raise ValueError("distribution requires at least one value")
    if not all(math.isfinite(value) for value in items):
        raise ValueError("distribution requires finite values")
    return {
        "count": len(items),
        "mean": statistics.fmean(items),
        "p50": percentile(items, 0.50),
        "p95": percentile(items, 0.95),
        "p99": percentile(items, 0.99),
        "max": max(items),
    }


@dataclass(frozen=True)
class RequestSpec:
    arrival_seconds: float
    prompt_length: int
    output_length: int


def workload_seed(
    base_seed: int, profile_index: int, repetition: int, *, couple_profiles: bool
) -> int:
    """Choose paired seeds for one-factor sweeps or independent profile seeds."""
    return int(base_seed) + (0 if couple_profiles else int(profile_index) * 1000) + int(repetition)


def _lengths(profile: dict, rng: random.Random) -> tuple[int, int]:
    def sample(bounds, name: str) -> int:
        if (not isinstance(bounds, list) or len(bounds) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) for value in bounds)
                or bounds[0] < 1 or bounds[1] < bounds[0]):
            raise ValueError(f"{name} must be [positive_min, max] with min <= max")
        return rng.randint(bounds[0], bounds[1])

    length_mode = profile.get("length_mode", "uniform")
    if length_mode == "uniform":
        return (
            sample(profile["prompt_length"], "prompt_length"),
            sample(profile["output_length"], "output_length"),
        )
    if length_mode == "bimodal":
        long_fraction = profile["long_fraction"]
        if (isinstance(long_fraction, bool)
                or not isinstance(long_fraction, (int, float))
                or not math.isfinite(long_fraction)
                or not 0.0 <= long_fraction <= 1.0):
            raise ValueError("long_fraction must be finite and in [0, 1]")
        branch = "long" if rng.random() < float(long_fraction) else "short"
        selected = profile[branch]
        return (
            sample(selected["prompt_length"], f"{branch}.prompt_length"),
            sample(selected["output_length"], f"{branch}.output_length"),
        )
    raise ValueError(f"unsupported length_mode: {length_mode}")


def generate_workload(profile: dict, *, seed: int) -> list[RequestSpec]:
    """Generate scheduled arrivals and lengths from a JSON-compatible profile."""
    count = profile["request_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("request_count must be a positive integer")
    rng = random.Random(seed)
    arrival_mode = profile["arrival_mode"]
    arrivals: list[float] = []
    if arrival_mode == "fixed":
        interval = float(profile["arrival_interval_ms"]) / 1000.0
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("arrival_interval_ms must be finite and positive")
        arrivals = [index * interval for index in range(count)]
    elif arrival_mode == "poisson":
        mean = float(profile["mean_arrival_interval_ms"]) / 1000.0
        if not math.isfinite(mean) or mean <= 0:
            raise ValueError("mean_arrival_interval_ms must be finite and positive")
        elapsed = 0.0
        for index in range(count):
            if index:
                elapsed += rng.expovariate(1.0 / mean)
            arrivals.append(elapsed)
    elif arrival_mode == "burst":
        burst_size = profile["burst_size"]
        burst_interval = float(profile["burst_interval_ms"]) / 1000.0
        if (isinstance(burst_size, bool) or not isinstance(burst_size, int)
                or burst_size < 1):
            raise ValueError("burst_size must be a positive integer")
        if not math.isfinite(burst_interval) or burst_interval <= 0:
            raise ValueError("burst_interval_ms must be finite and positive")
        arrivals = [(index // burst_size) * burst_interval for index in range(count)]
    else:
        raise ValueError(f"unsupported arrival_mode: {arrival_mode}")

    specs = []
    max_cache_len = profile["max_cache_len"]
    if (isinstance(max_cache_len, bool) or not isinstance(max_cache_len, int)
            or max_cache_len < 2):
        raise ValueError("max_cache_len must be an integer of at least two")
    for arrival in arrivals:
        prompt_length, output_length = _lengths(profile, rng)
        if prompt_length + output_length > max_cache_len:
            raise ValueError("prompt_length + output_length exceeds max_cache_len")
        specs.append(RequestSpec(arrival, prompt_length, output_length))
    return specs


def validate_benchmark_config(config: dict) -> None:
    """Reject inconsistent experiment matrices before model or GPU initialization."""
    for name in ("capacity", "max_cache_len", "repetitions"):
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config["max_cache_len"] < 2:
        raise ValueError("max_cache_len must be at least two")
    policies = config.get("policies", [])
    profiles = config.get("profiles", [])
    if not policies or not profiles:
        raise ValueError("at least one policy and profile are required")
    policy_names = [item.get("name") for item in policies]
    profile_names = [item.get("name") for item in profiles]
    if any(not name for name in policy_names) or len(set(policy_names)) != len(policy_names):
        raise ValueError("policy names must be non-empty and unique")
    if any(not name for name in profile_names) or len(set(profile_names)) != len(profile_names):
        raise ValueError("profile names must be non-empty and unique")
    for policy in policies:
        for field in ("min_prefill_batch", "max_wait_steps"):
            value = policy.get(field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{field} must be an integer")
        if policy["min_prefill_batch"] < 1 or policy["max_wait_steps"] < 0:
            raise ValueError("invalid admission policy")
    for index, profile in enumerate(profiles):
        if profile.get("max_cache_len") != config["max_cache_len"]:
            raise ValueError("profile max_cache_len must match the runner max_cache_len")
        generate_workload(profile, seed=int(config.get("seed", 0)) + index)


def summarize_repetitions(items: list[dict]) -> dict:
    """Aggregate like-for-like repetition-level scalars."""
    if not items:
        raise ValueError("at least one repetition is required")
    scalar_keys = (
        "request_throughput_per_second",
        "token_throughput_per_second",
        "completion_rate",
        "ttft_p50_ms", "ttft_p95_ms", "ttft_p99_ms",
        "tpot_p50_ms", "tpot_p95_ms", "tpot_p99_ms",
        "e2e_p50_ms", "e2e_p95_ms", "e2e_p99_ms",
        "arrival_lag_p95_ms", "queue_depth_p95",
        "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes",
    )
    return {
        key: distribution(item[key] for item in items)
        for key in scalar_keys
    }
