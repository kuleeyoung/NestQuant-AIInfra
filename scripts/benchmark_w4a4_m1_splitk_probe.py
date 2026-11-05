"""Benchmark deterministic split-K candidates for the M1 down projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import torch
import torch.nn.functional as F

from nestquant.core.model_checkpoint_file import NQMCFileReader
from nestquant.core.tensor_checkpoint import PackedNestQuantTensor
from nestquant.runtime.cuda_w4a4_mma import (
    linear_packed_w4a4_gemv_m1,
    linear_packed_w4a4_gemv_m1_splitk_compensated_fp32_probe,
    linear_packed_w4a4_gemv_m1_splitk_probe,
)
from nestquant.runtime.triton_fused_linear import prepare_fused_linear
from nestquant.runtime.triton_w4a4_linear import quantize_activation_cuda


BETAS = tuple(value / 14 for value in (3.47, 4.74, 6.90, 18.11))
DEFAULT_WEIGHT = "model.layers.0.mlp.down_proj.weight"


def _time(operation, warmups: int, iterations: int) -> float:
    for _ in range(warmups):
        operation()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iterations):
        operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--weight", default=DEFAULT_WEIGHT)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--groups", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.warmups, args.iterations, args.groups) < 1:
        parser.error("warmups, iterations, and groups must be positive")

    reader = NQMCFileReader(args.checkpoint, verify_checksum=False)
    _, record = reader.read_named_record(args.weight)
    if not isinstance(record, PackedNestQuantTensor):
        raise TypeError("weight is not a packed NestQuant tensor")
    weight = prepare_fused_linear(record, device=torch.device("cuda"))
    generator = torch.Generator(device="cuda").manual_seed(20261103)
    inputs = torch.randn(
        (1, record.feature_count),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    activation = quantize_activation_cuda(inputs, q=14, betas=BETAS)
    operations = {
        "baseline": lambda: linear_packed_w4a4_gemv_m1(activation, weight),
        **{
            f"split_{split}": (
                lambda split=split: linear_packed_w4a4_gemv_m1_splitk_probe(
                    activation, weight, split_k=split
                )
            )
            for split in (2, 4, 8)
        },
        **{
            f"split_{split}_compensated": (
                lambda split=split:
                linear_packed_w4a4_gemv_m1_splitk_compensated_fp32_probe(
                    activation, weight, split_k=split
                ).to(activation.output_dtype)
            )
            for split in (2, 4, 8)
        },
    }
    reference = operations["baseline"]()
    correctness = {}
    for name, operation in operations.items():
        actual = operation()
        error = (actual.float() - reference.float()).abs()
        correctness[name] = {
            "max_abs_error": float(error.max().item()),
            "mean_abs_error": float(error.mean().item()),
            "cosine_similarity": float(F.cosine_similarity(
                actual.float().flatten(), reference.float().flatten(), dim=0
            ).item()),
            "allclose_atol_0_05_rtol_0_01": bool(torch.allclose(
                actual, reference, atol=0.05, rtol=0.01
            )),
        }

    samples = {name: [] for name in operations}
    names = tuple(operations)
    for group in range(args.groups):
        order = names[group % len(names):] + names[:group % len(names)]
        for name in order:
            samples[name].append(_time(
                operations[name], args.warmups, args.iterations
            ))
    baseline = statistics.median(samples["baseline"])
    rows = []
    for name in names:
        median = statistics.median(samples[name])
        rows.append({
            "path": name,
            "median_us": median,
            "speedup_vs_baseline": baseline / median,
            "samples_us": samples[name],
            **correctness[name],
        })
        print(name, f"{median:.2f} us", f"{baseline / median:.2f}x", flush=True)
    payload = {
        "schema_version": 1,
        "status": "passed",
        "protocol": "rotated_order_m1_down_projection_split_k_probe",
        "weight": args.weight,
        "shape": [1, record.feature_count, record.indices.shape[0]],
        "warmups": args.warmups,
        "iterations": args.iterations,
        "groups": args.groups,
        "rows": rows,
        "claim_boundary": "Isolated kernel probe; no production dispatch change.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
