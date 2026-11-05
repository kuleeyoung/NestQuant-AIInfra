"""Benchmark direct-packed M1 W4A4 GEMV on real Llama projection shapes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import torch

from nestquant.core.model_checkpoint_file import NQMCFileReader
from nestquant.core.tensor_checkpoint import PackedNestQuantTensor
from nestquant.runtime.cuda_w4a4_mma import linear_packed_w4a4_gemv_m1
from nestquant.runtime.triton_fused_linear import prepare_fused_linear
from nestquant.runtime.triton_w4a4_linear import (
    linear_packed_w4a4_tiled_triton,
    quantize_activation_cuda,
    select_w4a4_tiled_config,
)


BETAS = tuple(value / 14 for value in (3.47, 4.74, 6.90, 18.11))
DEFAULT_WEIGHTS = (
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
)


def _time(operation, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iterations):
        operation()
    end.record(); end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--weights", nargs="+", default=DEFAULT_WEIGHTS)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--groups", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.groups) < 1:
        parser.error("warmup, iterations, and groups must be positive")

    reader = NQMCFileReader(args.checkpoint, verify_checksum=False)
    generator = torch.Generator(device="cuda").manual_seed(20261030)
    rows = []
    for name in args.weights:
        _, record = reader.read_named_record(name)
        if not isinstance(record, PackedNestQuantTensor):
            raise TypeError(f"{name} is not packed")
        weight = prepare_fused_linear(record)
        inputs = torch.randn(
            1, record.feature_count, device="cuda", dtype=torch.bfloat16,
            generator=generator,
        )
        activation = quantize_activation_cuda(inputs, q=14, betas=BETAS)
        config = select_w4a4_tiled_config(
            1, record.indices.shape[0], record.feature_count
        )
        custom = lambda: linear_packed_w4a4_gemv_m1(activation, weight)
        tiled = lambda: linear_packed_w4a4_tiled_triton(
            activation, weight, **config
        )
        actual, reference = custom(), tiled()
        error = (actual.float() - reference.float()).abs()
        custom_samples, tiled_samples = [], []
        for group in range(args.groups):
            operations = (custom, tiled) if group % 2 == 0 else (tiled, custom)
            first = _time(operations[0], args.warmup, args.iterations)
            second = _time(operations[1], args.warmup, args.iterations)
            if group % 2 == 0:
                custom_samples.append(first); tiled_samples.append(second)
            else:
                tiled_samples.append(first); custom_samples.append(second)
        custom_median = statistics.median(custom_samples)
        tiled_median = statistics.median(tiled_samples)
        rows.append({
            "weight": name,
            "shape": [1, record.feature_count, record.indices.shape[0]],
            "tiled_config": config,
            "custom_us": custom_median,
            "tiled_us": tiled_median,
            "speedup": tiled_median / custom_median,
            "custom_us_samples": custom_samples,
            "tiled_us_samples": tiled_samples,
            "max_abs_error": float(error.max().item()),
            "mean_abs_error": float(error.mean().item()),
        })
    report = {
        "scope": "direct-packed W4A4 M1 GEMV microbenchmark",
        "device_name": torch.cuda.get_device_name(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "groups": args.groups,
        "rows": rows,
        "claim_boundary": "Kernel-only timing; not an end-to-end TPOT claim.",
    }
    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
