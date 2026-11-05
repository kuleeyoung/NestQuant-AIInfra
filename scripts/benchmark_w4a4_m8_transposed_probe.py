"""Benchmark the fixed-shape tinygemm-style W4A4 M8 Tensor Core probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import torch

from nestquant.core.model_checkpoint_file import NQMCFileReader
from nestquant.core.tensor_checkpoint import PackedNestQuantTensor
from nestquant.runtime.cuda_w4a4_mma import (
    _load_extension,
    linear_packed_w4a4_mma_small_m,
)
from nestquant.runtime.triton_fused_linear import prepare_fused_linear
from nestquant.runtime.triton_w4a4_linear import (
    linear_packed_w4a4_tiled_triton,
    quantize_activation_cuda,
    select_w4a4_tiled_config,
)


BETAS = tuple(value / 14 for value in (3.47, 4.74, 6.90, 18.11))
DEFAULT_WEIGHT = "model.layers.0.self_attn.q_proj.weight"


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
    parser.add_argument("--weight", default=DEFAULT_WEIGHT)
    parser.add_argument("--rows", type=int, choices=(2, 4, 8), default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--groups", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.groups) < 1:
        parser.error("warmup, iterations, and groups must be positive")

    reader = NQMCFileReader(args.checkpoint, verify_checksum=False)
    _, record = reader.read_named_record(args.weight)
    if not isinstance(record, PackedNestQuantTensor):
        raise TypeError("fixed q_proj weight must be packed")
    shape = [args.rows, record.feature_count, record.indices.shape[0]]
    if shape[1] % 16 or shape[2] % 16:
        raise ValueError(f"probe requires K and N divisible by 16, got {shape}")
    weight = prepare_fused_linear(record)
    generator = torch.Generator(device="cuda").manual_seed(20261031)
    inputs = torch.randn(
        args.rows, record.feature_count, generator=generator, device="cuda",
        dtype=torch.bfloat16,
    )
    activation = quantize_activation_cuda(inputs, q=14, betas=BETAS)
    probe = lambda: linear_packed_w4a4_mma_small_m(
        activation, weight
    )
    triton_config = select_w4a4_tiled_config(
        args.rows, record.indices.shape[0], record.feature_count
    )
    triton = lambda: linear_packed_w4a4_tiled_triton(
        activation, weight, **triton_config
    )
    actual, reference = probe(), triton()
    error = (actual.float() - reference.float()).abs()
    if not torch.allclose(actual.float(), reference.float(), atol=0.04, rtol=0.01):
        raise RuntimeError(
            f"M8 probe failed correctness: max={error.max().item()} "
            f"mean={error.mean().item()}"
        )

    probe_samples, triton_samples = [], []
    for group in range(args.groups):
        operations = (probe, triton) if group % 2 == 0 else (triton, probe)
        first = _time(operations[0], args.warmup, args.iterations)
        second = _time(operations[1], args.warmup, args.iterations)
        if group % 2 == 0:
            probe_samples.append(first); triton_samples.append(second)
        else:
            triton_samples.append(first); probe_samples.append(second)
    probe_us = statistics.median(probe_samples)
    triton_us = statistics.median(triton_samples)
    extension = Path(_load_extension().__file__).resolve()
    report = {
        "scope": "fixed-KN small-M transposed-operand probe",
        "weight": args.weight,
        "shape_mkn": shape,
        "device_name": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "extension": str(extension),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "groups": args.groups,
        "triton_config": triton_config,
        "probe_us": probe_us,
        "triton_us": triton_us,
        "speedup_vs_triton": triton_us / probe_us,
        "probe_us_samples": probe_samples,
        "triton_us_samples": triton_samples,
        "max_abs_error_vs_triton": float(error.max().item()),
        "mean_abs_error_vs_triton": float(error.mean().item()),
        "finite": bool(torch.isfinite(actual).all()),
        "claim_boundary": (
            "Isolated fixed-shape kernel result; not yet runtime-dispatched, "
            "not an M2-M8 family, and not an end-to-end model result."
        ),
    }
    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
