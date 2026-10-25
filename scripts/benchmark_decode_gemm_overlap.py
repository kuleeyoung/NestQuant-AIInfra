"""Measure the upper bound from overlapping one Triton decode with one BF16 GEMM."""

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
    peak = max(0, torch.cuda.max_memory_allocated() - baseline)
    return _stats(values), peak


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.sequence_length < 1 or args.repeats < 1:
        parser.error("sequence length and repeats must be positive")

    reader = NQMCFileReader(args.checkpoint, verify_checksum=False)
    pairs = (
        ("q_to_k", "model.layers.0.self_attn.q_proj.weight",
         "model.layers.0.self_attn.k_proj.weight"),
        ("gate_to_up", "model.layers.0.mlp.gate_proj.weight",
         "model.layers.0.mlp.up_proj.weight"),
        ("up_to_down", "model.layers.0.mlp.up_proj.weight",
         "model.layers.0.mlp.down_proj.weight"),
    )
    generator = torch.Generator(device="cuda").manual_seed(20261025)
    decode_stream = torch.cuda.Stream()
    results = []
    for label, current_name, next_name in pairs:
        _, current = reader.read_named_record(current_name)
        _, following = reader.read_named_record(next_name)
        if not isinstance(current, PackedNestQuantTensor) or not isinstance(
            following, PackedNestQuantTensor
        ):
            raise TypeError("overlap benchmark requires packed records")
        weight = dequantize_tiled_triton(
            current, device="cuda", output_dtype=torch.bfloat16
        )
        inputs = torch.randn(
            (1, args.sequence_length, current.feature_count),
            device="cuda", dtype=torch.bfloat16, generator=generator,
        )
        torch.cuda.synchronize()

        def gemm():
            return F.linear(inputs, weight)

        def decode():
            return dequantize_tiled_triton(
                following, device="cuda", output_dtype=torch.bfloat16
            )

        def sequential():
            gemm_output = gemm()
            decoded = decode()
            return gemm_output, decoded

        def overlapped():
            gemm_output = gemm()
            with torch.cuda.stream(decode_stream):
                decoded = decode()
            return gemm_output, decoded

        gemm_stats, gemm_peak = _measure(gemm, args.repeats)
        decode_stats, decode_peak = _measure(decode, args.repeats)
        sequential_stats, sequential_peak = _measure(sequential, args.repeats)
        overlap_stats, overlap_peak = _measure(overlapped, args.repeats)
        results.append({
            "pair": label,
            "current_weight": current_name,
            "next_weight": next_name,
            "gemm_seconds": gemm_stats,
            "decode_seconds": decode_stats,
            "sequential_seconds": sequential_stats,
            "overlap_seconds": overlap_stats,
            "overlap_speedup": sequential_stats["mean"] / overlap_stats["mean"],
            "ideal_overlap_seconds": max(gemm_stats["mean"], decode_stats["mean"]),
            "peak_increment_bytes": {
                "gemm": gemm_peak,
                "decode": decode_peak,
                "sequential": sequential_peak,
                "overlap": overlap_peak,
            },
        })
        del inputs, weight
    report = {
        "scope": "GPU-only one-layer GEMM plus next-weight decode; record I/O and parse excluded",
        "checkpoint": str(args.checkpoint.resolve()),
        "device_name": torch.cuda.get_device_name(),
        "sequence_length": args.sequence_length,
        "repeats": args.repeats,
        "results": results,
        "claim_boundary": (
            "This is an isolated overlap ceiling, not a full-model asynchronous runtime."
        ),
    }
    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
