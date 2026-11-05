"""Benchmark the correctness-first GPU NQKV page and attention pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nestquant.runtime.nqkv_attention import nqkv_decode_attention_reference
from nestquant.runtime.nqkv_attention_k1 import (
    NQKVAttentionGraphRunner,
    NQKVAttentionWorkspace,
    nqkv_decode_attention_k1,
)
from nestquant.runtime.nqkv_hot_cache import NQKVHotColdCache
from nestquant.runtime.nqkv_page import (
    dequantize_nqkv_page_triton,
    quantize_nqkv_page_cuda,
)
from nestquant.runtime.nqkv_pool import NQKVPagePool
from nestquant.runtime.paged_nqkv_cache import PagedNQKVCache


def _time_cuda(function, *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--context-pages", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.query_heads % args.kv_heads:
        raise SystemExit("query-heads must be divisible by kv-heads")

    key_betas = (0.25, 0.3271428571428571, 0.46214285714285713, 1.2185714285714284)
    value_betas = (0.2521428571428571, 0.3992857142857143, 0.687142857142857, 2.073571428571429)
    shape = (args.page_size, args.kv_heads, args.head_dim)
    key = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    prepared = quantize_nqkv_page_cuda(
        key, value, page_size=args.page_size, q=14,
        key_betas=key_betas, value_betas=value_betas,
    )
    dequantize_nqkv_page_triton(prepared)

    quantize_us = _time_cuda(
        lambda: quantize_nqkv_page_cuda(
            key, value, page_size=args.page_size, q=14,
            key_betas=key_betas, value_betas=value_betas,
        ),
        warmup=args.warmup, iterations=args.iterations,
    )
    decode_us = _time_cuda(
        lambda: dequantize_nqkv_page_triton(prepared),
        warmup=args.warmup, iterations=args.iterations,
    )

    pool = NQKVPagePool(
        capacity=args.context_pages, page_size=args.page_size,
        num_kv_heads=args.kv_heads, head_dim=args.head_dim, q=14,
        key_betas=key_betas, value_betas=value_betas,
    )
    cold = PagedNQKVCache(
        pool, max_sequences=1, max_pages_per_sequence=args.context_pages
    )
    cache = NQKVHotColdCache(cold)
    cache.register("benchmark")
    for _ in range(args.context_pages):
        cache.append("benchmark", key, value)
    cache.finalize("benchmark")
    query = torch.randn(
        args.query_heads, args.head_dim, device="cuda", dtype=torch.bfloat16
    )
    nqkv_decode_attention_reference(query, cache, "benchmark")
    attention_us = _time_cuda(
        lambda: nqkv_decode_attention_reference(query, cache, "benchmark"),
        warmup=args.warmup, iterations=args.iterations,
    )
    workspace = NQKVAttentionWorkspace.create(cache, query_heads=args.query_heads)
    nqkv_decode_attention_k1(query, cache, "benchmark", workspace=workspace)
    attention_k1_us = _time_cuda(
        lambda: nqkv_decode_attention_k1(
            query, cache, "benchmark", workspace=workspace
        ),
        warmup=args.warmup, iterations=args.iterations,
    )
    graph_runner = NQKVAttentionGraphRunner(
        cache, "benchmark", query_heads=args.query_heads
    )
    graph_runner.capture(query)
    graph_runner.replay(query)
    attention_graph_us = _time_cuda(
        lambda: graph_runner.replay(query),
        warmup=args.warmup, iterations=args.iterations,
    )

    table_bytes = pool.beta_table.numel() * 4 + pool.value_beta_table.numel() * 4
    packed_per_page = (pool.resident_bytes - table_bytes) / pool.capacity
    bf16_per_page = args.page_size * args.kv_heads * args.head_dim * 2 * 2
    result = {
        "status": "completed",
        "protocol": "nqkv-gpu-k1-v2",
        "device": torch.cuda.get_device_name(),
        "page_shape": list(shape),
        "context_pages": args.context_pages,
        "context_tokens": args.context_pages * args.page_size,
        "query_heads": args.query_heads,
        "key_betas": list(key_betas),
        "value_betas": list(value_betas),
        "packed_bytes_per_page": int(packed_per_page),
        "bf16_bytes_per_page": bf16_per_page,
        "resident_reduction_percent": (1.0 - packed_per_page / bf16_per_page) * 100.0,
        "gpu_quantize_pack_us_per_page": quantize_us,
        "triton_decode_us_per_page": decode_us,
        "k0_paged_attention_us": attention_us,
        "k1_fused_attention_us": attention_k1_us,
        "k1_speedup_over_k0": attention_us / attention_k1_us,
        "k1_cuda_graph_us": attention_graph_us,
        "cuda_graph_speedup_over_k1_eager": attention_k1_us / attention_graph_us,
        "notes": [
            "K0 decodes one packed page at a time and uses FP32 online softmax.",
            "Timings include temporary output allocation and Python page dispatch.",
            "This is a diagnostic baseline, not a fused-attention performance claim.",
            "K1 fuses packed page decode with page-local attention and reduces page statistics in a second kernel.",
        ],
    }
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
