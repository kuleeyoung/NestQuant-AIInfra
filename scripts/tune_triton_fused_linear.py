"""Sweep fused NQRT Linear tile and warp configurations on a real weight."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import statistics
import sys
import time

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nestquant.core.model_checkpoint_file import NQMCFileReader
from nestquant.core.tensor_checkpoint import PackedNestQuantTensor
from nestquant.runtime.triton_fused_linear import (
    linear_fused_triton,
    prepare_fused_linear,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("weight_name")
    parser.add_argument("--sequence-length", type=int, nargs="+", default=[1, 8, 16])
    parser.add_argument("--block-rows", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--block-tokens", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--num-warps", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or any(length < 1 for length in args.sequence_length):
        parser.error("repeats and sequence lengths must be positive")

    reader = NQMCFileReader(args.checkpoint, verify_checksum=False)
    _, record = reader.read_named_record(args.weight_name)
    if not isinstance(record, PackedNestQuantTensor):
        raise TypeError("tuning weight must be a packed NQRT record")
    prepared = prepare_fused_linear(record)
    generator = torch.Generator(device="cuda").manual_seed(20261025)
    results = []
    for sequence_length in args.sequence_length:
        inputs = torch.randn(
            (1, sequence_length, record.feature_count),
            device="cuda", dtype=torch.bfloat16, generator=generator,
        )
        candidates = []
        for block_rows, block_tokens, num_warps in itertools.product(
            args.block_rows, args.block_tokens, args.num_warps
        ):
            config = {
                "block_rows": block_rows,
                "block_tokens": block_tokens,
                "num_warps": num_warps,
            }
            try:
                linear_fused_triton(inputs, prepared, **config)
                torch.cuda.synchronize()
                values = []
                for _ in range(args.repeats):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    output = linear_fused_triton(inputs, prepared, **config)
                    torch.cuda.synchronize()
                    values.append(time.perf_counter() - started)
                    del output
                candidates.append({
                    **config,
                    "mean_seconds": statistics.mean(values),
                    "median_seconds": statistics.median(values),
                    "min_seconds": min(values),
                    "max_seconds": max(values),
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
        valid.sort(key=lambda item: item["median_seconds"])
        results.append({
            "sequence_length": sequence_length,
            "best": valid[0] if valid else None,
            "candidates_by_median": valid,
            "failed_candidates": [
                item for item in candidates if item["status"] == "error"
            ],
        })
    report = {
        "scope": "real-weight fused NQRT Linear configuration sweep",
        "checkpoint": str(args.checkpoint.resolve()),
        "weight_name": args.weight_name,
        "device_name": torch.cuda.get_device_name(),
        "repeats": args.repeats,
        "search_space": {
            "block_rows": args.block_rows,
            "block_tokens": args.block_tokens,
            "num_warps": args.num_warps,
        },
        "results": results,
        "claim_boundary": (
            "Best configurations are local to this GPU, weight shape, token count, "
            "software stack, and search space."
        ),
    }
    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
