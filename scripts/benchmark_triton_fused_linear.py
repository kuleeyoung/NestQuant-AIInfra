"""Benchmark materialized BF16 and packed-resident fused NQRT Linear paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nestquant.core.model_checkpoint_file import NQMCFileReader
from nestquant.core.tensor_checkpoint import PackedNestQuantTensor
from nestquant.runtime.triton_fused_linear import (
    linear_fused_triton,
    prepare_fused_linear,
)
from nestquant.runtime.triton_tiled_decode import dequantize_tiled_triton


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _measure(operation, repeats: int) -> tuple[dict[str, float], int]:
    output = operation()
    torch.cuda.synchronize()
    del output
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    values = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = operation()
        torch.cuda.synchronize()
        values.append(time.perf_counter() - started)
        del output
    return _stats(values), max(0, torch.cuda.max_memory_allocated() - baseline)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("weight_name")
    parser.add_argument("--sequence-length", type=int, nargs="+", default=[1, 8, 16])
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or any(length < 1 for length in args.sequence_length):
        parser.error("repeats and sequence lengths must be positive")

    reader = NQMCFileReader(args.checkpoint, verify_checksum=False)
    _, record = reader.read_named_record(args.weight_name)
    if not isinstance(record, PackedNestQuantTensor):
        raise TypeError("benchmark weight must be a packed NQRT record")
    prepared = prepare_fused_linear(record)
    bf16_weight = dequantize_tiled_triton(
        record, device="cuda", output_dtype=torch.bfloat16
    )
    torch.cuda.synchronize()
    packed_resident_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (
            prepared.code_payload,
            prepared.beta_payload,
            prepared.row_norms,
            prepared.beta_table,
        )
    )
    results = []
    generator = torch.Generator(device="cuda").manual_seed(20261025)
    for sequence_length in args.sequence_length:
        inputs = torch.randn(
            (1, sequence_length, record.feature_count),
            device="cuda", dtype=torch.bfloat16, generator=generator,
        )
        reference = F.linear(inputs, bf16_weight)
        fused = linear_fused_triton(inputs, prepared)
        torch.cuda.synchronize()
        error = (reference.float() - fused.float()).abs()
        relative_l2 = float(
            (torch.linalg.vector_norm(error)
             / torch.linalg.vector_norm(reference.float())).item()
        )
        bf16_stats, bf16_peak = _measure(
            lambda: F.linear(inputs, bf16_weight), args.repeats
        )
        decode_stats, decode_peak = _measure(
            lambda: F.linear(
                inputs,
                dequantize_tiled_triton(
                    record, device="cuda", output_dtype=torch.bfloat16
                ),
            ),
            args.repeats,
        )
        fused_stats, fused_peak = _measure(
            lambda: linear_fused_triton(inputs, prepared), args.repeats
        )
        results.append({
            "sequence_length": sequence_length,
            "bf16_resident": {
                "seconds": bf16_stats,
                "peak_increment_bytes": bf16_peak,
            },
            "decode_then_gemm": {
                "seconds": decode_stats,
                "peak_increment_bytes": decode_peak,
            },
            "packed_resident_fused": {
                "seconds": fused_stats,
                "peak_increment_bytes": fused_peak,
            },
            "correctness_vs_bf16_gemm": {
                "max_abs_error": float(error.max().item()),
                "mean_abs_error": float(error.mean().item()),
                "relative_l2_error": relative_l2,
            },
        })
    report = {
        "scope": "single packed-resident NQRT v4 Linear; input transform excluded",
        "checkpoint": str(args.checkpoint.resolve()),
        "weight_name": args.weight_name,
        "weight_shape": [record.indices.shape[0], record.feature_count],
        "device_name": torch.cuda.get_device_name(),
        "repeats": args.repeats,
        "packed_resident_bytes": packed_resident_bytes,
        "bf16_weight_bytes": bf16_weight.numel() * bf16_weight.element_size(),
        "results": results,
        "claim_boundary": (
            "The fused prototype avoids full BF16 weight materialization but is "
            "not an optimized tensor-core GEMM or a full-model runtime."
        ),
    }
    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
