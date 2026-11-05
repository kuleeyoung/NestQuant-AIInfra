"""Benchmark scalar and tiled direct packed W4xA4 kernels on a real record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nestquant.core.model_checkpoint_file import NQMCFileReader
from nestquant.core.tensor_checkpoint import PackedNestQuantTensor
from nestquant.runtime.triton_fused_linear import prepare_fused_linear
from nestquant.runtime.triton_w4a4_linear import (
    linear_packed_w4a4_tiled_triton,
    linear_packed_w4a4_triton,
    quantize_activation_cuda,
)


BETAS = tuple(value / 14 for value in (3.47, 4.74, 6.90, 18.11))


def _time_cuda(operation, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--weight", default="model.layers.0.self_attn.q_proj.weight"
    )
    parser.add_argument("--tokens", type=int, nargs="+", default=(1, 17))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--skip-scalar", action="store_true",
        help="Skip the very slow scalar correctness oracle during shape profiling.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reader = NQMCFileReader(args.checkpoint, verify_checksum=False)
    _, record = reader.read_named_record(args.weight)
    if not isinstance(record, PackedNestQuantTensor):
        raise TypeError("benchmark weight must be a packed record")
    weight = prepare_fused_linear(record)
    results = []
    for token_count in args.tokens:
        inputs = torch.randn(
            token_count, record.feature_count,
            device="cuda", dtype=torch.bfloat16,
        )
        activation = quantize_activation_cuda(inputs, q=14, betas=BETAS)
        tiled = linear_packed_w4a4_tiled_triton(activation, weight)
        if args.skip_scalar:
            scalar_us = None
            max_abs_error = None
            mean_abs_error = None
        else:
            scalar = linear_packed_w4a4_triton(activation, weight)
            error = (scalar.float() - tiled.float()).abs()
            scalar_us = _time_cuda(
                lambda: linear_packed_w4a4_triton(activation, weight),
                args.warmup, args.iterations,
            )
            max_abs_error = error.max().item()
            mean_abs_error = error.mean().item()
        tiled_us = _time_cuda(
            lambda: linear_packed_w4a4_tiled_triton(activation, weight),
            args.warmup, args.iterations,
        )
        encode_us = _time_cuda(
            lambda: quantize_activation_cuda(inputs, q=14, betas=BETAS),
            args.warmup, args.iterations,
        )
        results.append({
            "tokens": token_count,
            "shape": [token_count, record.feature_count, record.indices.shape[0]],
            "scalar_us": scalar_us,
            "tiled_us": tiled_us,
            "tiled_speedup": None if scalar_us is None else scalar_us / tiled_us,
            "activation_encode_us": encode_us,
            "max_abs_error": max_abs_error,
            "mean_abs_error": mean_abs_error,
            "packed_activation_bytes": activation.resident_bytes,
        })
    payload = {
        "protocol": "packed-w4a4-kernel-v1",
        "checkpoint": str(args.checkpoint),
        "weight": args.weight,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "results": results,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
