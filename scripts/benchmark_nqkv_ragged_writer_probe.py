"""Benchmark transactional and GPU ragged sub-page NQKV writers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nestquant.runtime.nqkv_batched_writer import BatchedNQKVWriter
from nestquant.runtime.nqkv_graph_writer import (
    _launch_append_hot_ragged_subpage_graph_safe,
    append_hot_ragged_subpage,
)
from nestquant.runtime.nqkv_hot_cache import NQKVHotColdCache
from nestquant.runtime.nqkv_pool import NQKVPagePool
from nestquant.runtime.paged_nqkv_cache import PagedNQKVCache


def _cache() -> NQKVHotColdCache:
    pool = NQKVPagePool(
        capacity=4,
        page_size=16,
        num_kv_heads=8,
        head_dim=64,
        q=14,
        betas=(0.5, 0.75, 1.0, 1.25),
        device="cuda",
        output_dtype=torch.bfloat16,
    )
    cache = NQKVHotColdCache(
        PagedNQKVCache(pool, max_sequences=2, max_pages_per_sequence=2)
    )
    cache.register("left", slot=0)
    cache.register("right", slot=1)
    return cache


def _reset(cache: NQKVHotColdCache) -> None:
    cache.hot_lengths.zero_()
    cache.cold_cache.sequence_lengths.zero_()
    cache.synchronize_host_length(0, 0)
    cache.synchronize_host_length(1, 0)
    torch.cuda.synchronize()


def _sample(operation, reset, repetitions: int) -> list[float]:
    values = []
    for _ in range(repetitions):
        reset()
        started = time.perf_counter_ns()
        operation()
        torch.cuda.synchronize()
        values.append((time.perf_counter_ns() - started) / 1000.0)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    generator = torch.Generator(device="cuda").manual_seed(20261102)
    shape = (2, 15, 8, 64)
    key = torch.randn(shape, generator=generator, device="cuda").bfloat16()
    value = torch.randn(shape, generator=generator, device="cuda").bfloat16()
    slots = torch.arange(2, device="cuda", dtype=torch.int32)
    mask = torch.arange(15, device="cuda")[None, :] < torch.tensor(
        [[15], [8]], device="cuda"
    )

    transactional_cache = _cache()
    checked_cache = _cache()
    launch_cache = _cache()
    transactional = lambda: BatchedNQKVWriter(transactional_cache).append(
        key, value, slot_ids=slots, write_mask=mask
    )
    checked = lambda: append_hot_ragged_subpage(
        key, value, checked_cache, slot_ids=slots, write_mask=mask
    )
    launch = lambda: _launch_append_hot_ragged_subpage_graph_safe(
        key, value, launch_cache, slot_ids=slots, write_mask=mask
    )
    operations = {
        "transactional": (transactional, lambda: _reset(transactional_cache)),
        "checked_gpu": (checked, lambda: _reset(checked_cache)),
        "gpu_launch": (launch, lambda: _reset(launch_cache)),
    }
    for operation, reset in operations.values():
        _sample(operation, reset, args.warmup)
    samples = {
        name: _sample(operation, reset, args.repetitions)
        for name, (operation, reset) in operations.items()
    }
    medians = {name: statistics.median(values) for name, values in samples.items()}
    report = {
        "status": "passed",
        "shape_bthd": list(shape),
        "row_counts": [15, 8],
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "latency_us_median": medians,
        "speedup_vs_transactional": {
            name: medians["transactional"] / latency
            for name, latency in medians.items()
            if name != "transactional"
        },
        "latency_us_samples": samples,
        "claim_boundary": (
            "Synthetic one-layer host-wall probe. gpu_launch excludes checked-API "
            "validation and host-length mirroring."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "latency_us_samples"}, indent=2))


if __name__ == "__main__":
    main()
