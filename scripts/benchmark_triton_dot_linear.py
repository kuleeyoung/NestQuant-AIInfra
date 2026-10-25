"""Benchmark and tune the Tensor-Core-oriented NQRT block-dot prototype."""

from __future__ import annotations

import argparse
import itertools
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
from nestquant.runtime.triton_dot_linear import (
    linear_fused_dot_ablation_triton,
    linear_fused_dot_no_e8_ablation_triton,
    linear_fused_dot_triton,
)
from nestquant.runtime.triton_fused_linear import linear_fused_triton, prepare_fused_linear
from nestquant.runtime.triton_tiled_decode import dequantize_tiled_triton


def _measure(operation, repeats: int) -> dict[str, float]:
    operation()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = operation()
        torch.cuda.synchronize()
        values.append(time.perf_counter() - started)
        del output
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("weight_name")
    parser.add_argument("--sequence-length", type=int, nargs="+", default=[1, 8, 16, 128])
    parser.add_argument("--block-m", type=int, nargs="+", default=[16])
    parser.add_argument("--block-n", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--block-k", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--num-warps", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--maxnreg", type=int, nargs="+", default=None)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--e8-ablation", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    maxnreg_values = (
        [None]
        if args.maxnreg is None
        else [None if value == 0 else value for value in args.maxnreg]
    )
    reader = NQMCFileReader(args.checkpoint, verify_checksum=False)
    _, record = reader.read_named_record(args.weight_name)
    if not isinstance(record, PackedNestQuantTensor):
        raise TypeError("benchmark weight must be packed")
    prepared = prepare_fused_linear(record)
    bf16_weight = dequantize_tiled_triton(record, output_dtype=torch.bfloat16)
    generator = torch.Generator(device="cuda").manual_seed(20261025)
    results = []
    for sequence_length in args.sequence_length:
        inputs = torch.randn(
            (1, sequence_length, record.feature_count),
            device="cuda", dtype=torch.bfloat16, generator=generator,
        )
        reference = F.linear(inputs, bf16_weight)
        scalar = linear_fused_triton(inputs, prepared)
        scalar_error = (scalar.float() - reference.float()).abs()
        scalar_stats = _measure(lambda: linear_fused_triton(inputs, prepared), args.repeats)
        bf16_stats = _measure(lambda: F.linear(inputs, bf16_weight), args.repeats)
        candidates = []
        for block_m, block_n, block_k, num_warps, maxnreg in itertools.product(
            args.block_m,
            args.block_n,
            args.block_k,
            args.num_warps,
            maxnreg_values,
        ):
            config = {
                "block_m": block_m,
                "block_n": block_n,
                "block_k": block_k,
                "num_warps": num_warps,
                "maxnreg": maxnreg,
            }
            try:
                actual = linear_fused_dot_triton(inputs, prepared, **config)
                torch.cuda.synchronize()
                error = (actual.float() - reference.float()).abs()
                relative_l2 = float(
                    (torch.linalg.vector_norm(error)
                     / torch.linalg.vector_norm(reference.float())).item()
                )
                stats = _measure(
                    lambda c=config: linear_fused_dot_triton(inputs, prepared, **c),
                    args.repeats,
                )
                candidates.append({
                    **config,
                    "seconds": stats,
                    "max_abs_error": float(error.max().item()),
                    "relative_l2_error": relative_l2,
                    "status": "ok",
                })
            except Exception as error:
                candidates.append({
                    **config,
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                })
        valid = [item for item in candidates if item["status"] == "ok"]
        valid.sort(key=lambda item: item["seconds"]["median"])
        result = {
            "sequence_length": sequence_length,
            "bf16_resident_seconds": bf16_stats,
            "scalar_fused_seconds": scalar_stats,
            "scalar_relative_l2_error": float(
                (torch.linalg.vector_norm(scalar_error)
                 / torch.linalg.vector_norm(reference.float())).item()
            ),
            "dot_best": valid[0] if valid else None,
            "dot_candidates_by_median": valid,
            "failed_candidates": [item for item in candidates if item["status"] == "error"],
        }
        if args.e8_ablation and valid:
            best_config = {
                name: valid[0][name]
                for name in (
                    "block_m", "block_n", "block_k", "num_warps", "maxnreg"
                )
            }
            result["no_e8_performance_ablation_seconds"] = _measure(
                lambda c=best_config: linear_fused_dot_no_e8_ablation_triton(
                    inputs, prepared, **c
                ),
                args.repeats,
            )
            result["no_e8_performance_ablation_warning"] = (
                "Output is invalid; timing only removes RNG and E8 projections."
            )
            result["decode_performance_ablation_seconds"] = {
                mode: _measure(
                    lambda m=mode, c=best_config: linear_fused_dot_ablation_triton(
                        inputs, prepared, m, **c
                    ),
                    args.repeats,
                )
                for mode in (
                    "no_rng",
                    "no_dither_e8",
                    "no_coarse_e8",
                    "no_rng_or_e8",
                )
            }
        results.append(result)
    report = {
        "scope": "real packed NQRT Linear block-dot feasibility and local tuning",
        "checkpoint": str(args.checkpoint.resolve()),
        "weight_name": args.weight_name,
        "weight_shape": [record.indices.shape[0], record.feature_count],
        "device_name": torch.cuda.get_device_name(),
        "repeats": args.repeats,
        "e8_performance_ablation": args.e8_ablation,
        "search_space": {
            "block_m": args.block_m,
            "block_n": args.block_n,
            "block_k": args.block_k,
            "num_warps": args.num_warps,
            "maxnreg": maxnreg_values,
        },
        "results": results,
        "claim_boundary": (
            "tl.dot expresses a Tensor-Core-oriented dataflow, but generated "
            "MMA instructions require separate compiler-code inspection."
        ),
    }
    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
