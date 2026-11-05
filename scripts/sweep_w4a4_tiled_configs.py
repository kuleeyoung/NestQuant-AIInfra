"""Sweep numerically stable launch shapes for the packed W4A4 tiled kernel."""

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
    parser.add_argument("--weight", required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=(1, 17, 128))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--block-ms", type=int, nargs="+", default=(16, 32, 64))
    parser.add_argument("--block-ns", type=int, nargs="+", default=(16, 32, 64))
    parser.add_argument("--block-ks", type=int, nargs="+", default=(32,))
    parser.add_argument("--warps", type=int, nargs="+", default=(2, 4, 8))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reader = NQMCFileReader(args.checkpoint, verify_checksum=False)
    _, record = reader.read_named_record(args.weight)
    if not isinstance(record, PackedNestQuantTensor):
        raise TypeError("weight must be a packed record")
    weight = prepare_fused_linear(record)
    results = []
    for tokens in args.tokens:
        generator = torch.Generator(device="cuda").manual_seed(20261028 + tokens)
        inputs = torch.randn(
            tokens, record.feature_count, device="cuda", dtype=torch.bfloat16,
            generator=generator,
        )
        activation = quantize_activation_cuda(inputs, q=14, betas=BETAS)
        reference = linear_packed_w4a4_tiled_triton(activation, weight)
        candidates = []
        for block_m in args.block_ms:
            for block_n in args.block_ns:
                for block_k in args.block_ks:
                    for num_warps in args.warps:
                        config = {
                            "block_m": block_m,
                            "block_n": block_n,
                            "block_k": block_k,
                            "num_warps": num_warps,
                        }
                        try:
                            operation = lambda c=config: linear_packed_w4a4_tiled_triton(
                                activation, weight, **c
                            )
                            candidate = operation()
                            error = (candidate.float() - reference.float()).abs()
                            microseconds = _time_cuda(
                                operation, args.warmup, args.iterations
                            )
                            candidates.append({
                                **config,
                                "microseconds": microseconds,
                                "max_abs_error_vs_default": float(error.max().item()),
                                "mean_abs_error_vs_default": float(error.mean().item()),
                                "status": "ok",
                            })
                        except Exception as error:
                            candidates.append({
                                **config, "status": "failed",
                                "error": f"{type(error).__name__}: {error}",
                            })
            # Release compiler temporaries between the larger M families.
            torch.cuda.empty_cache()
        valid = [item for item in candidates if item["status"] == "ok"]
        best = min(valid, key=lambda item: item["microseconds"])
        results.append({
            "tokens": tokens,
            "shape": [tokens, record.feature_count, record.indices.shape[0]],
            "best": best,
            "candidates": candidates,
        })
        print(
            f"tokens={tokens} best={best['block_m']}x{best['block_n']} "
            f"warps={best['num_warps']} time={best['microseconds']:.2f}us "
            f"max_error={best['max_abs_error_vs_default']}", flush=True,
        )
    payload = {
        "protocol": "packed-w4a4-launch-sweep-v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "weight": args.weight,
        "block_ks": args.block_ks,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
