"""Measure raw-FP32 and BF16 M1 split-K error over deterministic activations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

import torch

from nestquant.core.model_checkpoint_file import NQMCFileReader
from nestquant.core.tensor_checkpoint import PackedNestQuantTensor
from nestquant.runtime.cuda_w4a4_mma import (
    linear_packed_w4a4_gemv_m1_fp32_probe,
    linear_packed_w4a4_gemv_m1_splitk_fp32_probe,
    linear_packed_w4a4_gemv_m1_splitk_compensated_fp32_probe,
    linear_packed_w4a4_gemv_m1_splitk_partials_probe,
)
from nestquant.runtime.triton_fused_linear import prepare_fused_linear
from nestquant.runtime.triton_tiled_decode import dequantize_tiled_triton
from nestquant.runtime.triton_w4a4_linear import (
    decode_packed_a4_triton_diagnostic,
    quantize_activation_cuda,
)


BETAS = tuple(value / 14 for value in (3.47, 4.74, 6.90, 18.11))
DEFAULT_WEIGHT = "model.layers.0.mlp.down_proj.weight"


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def reduce_partials(partials: torch.Tensor) -> dict[str, torch.Tensor]:
    """Apply diagnostic summation orders to `[split, N]` FP32 partials."""
    sequential = torch.zeros_like(partials[0])
    compensation = torch.zeros_like(partials[0])
    kahan = torch.zeros_like(partials[0])
    for partial in partials:
        sequential = sequential + partial
        adjusted = partial - compensation
        updated = kahan + adjusted
        compensation = (updated - kahan) - adjusted
        kahan = updated
    pairwise = [partial.clone() for partial in partials]
    while len(pairwise) > 1:
        pairwise = [
            pairwise[index] + pairwise[index + 1]
            for index in range(0, len(pairwise), 2)
        ]
    return {
        "fp32_sequential": sequential,
        "fp32_pairwise": pairwise[0],
        "fp32_kahan": kahan,
        "fp64_sum": partials.double().sum(dim=0).float(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--weight", default=DEFAULT_WEIGHT)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")

    reader = NQMCFileReader(args.checkpoint, verify_checksum=False)
    _, record = reader.read_named_record(args.weight)
    if not isinstance(record, PackedNestQuantTensor):
        raise TypeError("weight is not a packed NestQuant tensor")
    weight = prepare_fused_linear(record, device=torch.device("cuda"))
    decoded_weight = dequantize_tiled_triton(
        record, device="cuda", output_dtype=torch.bfloat16
    )
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    rows = {split: [] for split in (2, 4, 8)}
    for sample in range(args.samples):
        inputs = torch.randn(
            (1, record.feature_count), device="cuda", dtype=torch.bfloat16,
            generator=generator,
        )
        activation = quantize_activation_cuda(inputs, q=14, betas=BETAS)
        decoded_activation = decode_packed_a4_triton_diagnostic(activation)
        oracle = decoded_activation.double() @ decoded_weight.double().T
        baseline = linear_packed_w4a4_gemv_m1_fp32_probe(activation, weight)
        baseline_oracle_difference = baseline.double() - oracle
        baseline_bf16 = baseline.to(torch.bfloat16)
        for split in rows:
            candidate = linear_packed_w4a4_gemv_m1_splitk_fp32_probe(
                activation, weight, split_k=split
            )
            compensated = linear_packed_w4a4_gemv_m1_splitk_compensated_fp32_probe(
                activation, weight, split_k=split
            )
            partials = linear_packed_w4a4_gemv_m1_splitk_partials_probe(
                activation, weight, split_k=split
            )
            reductions = reduce_partials(partials)
            difference = candidate - baseline
            oracle_difference = candidate.double() - oracle
            candidate_bf16 = candidate.to(torch.bfloat16)
            compensated_difference = compensated - baseline
            compensated_oracle_difference = compensated.double() - oracle
            compensated_bf16 = compensated.to(torch.bfloat16)
            rows[split].append({
                "sample": sample,
                "fp32_max_abs_error": float(difference.abs().max().item()),
                "fp32_mean_abs_error": float(difference.abs().mean().item()),
                "fp32_relative_l2_error": float(
                    (torch.linalg.vector_norm(difference)
                     / torch.linalg.vector_norm(baseline)).item()
                ),
                "bf16_differing_elements": int(
                    torch.count_nonzero(candidate_bf16 != baseline_bf16).item()
                ),
                "bf16_max_abs_error": float(
                    (candidate_bf16.float() - baseline_bf16.float()).abs().max().item()
                ),
                "baseline_vs_fp64_oracle": {
                    "max_abs_error": float(
                        baseline_oracle_difference.abs().max().item()
                    ),
                    "mean_abs_error": float(
                        baseline_oracle_difference.abs().mean().item()
                    ),
                    "relative_l2_error": float(
                        (torch.linalg.vector_norm(baseline_oracle_difference)
                         / torch.linalg.vector_norm(oracle)).item()
                    ),
                },
                "split_vs_fp64_oracle": {
                    "max_abs_error": float(oracle_difference.abs().max().item()),
                    "mean_abs_error": float(oracle_difference.abs().mean().item()),
                    "relative_l2_error": float(
                        (torch.linalg.vector_norm(oracle_difference)
                         / torch.linalg.vector_norm(oracle)).item()
                    ),
                },
                "compensated": {
                    "fp32_max_abs_error": float(
                        compensated_difference.abs().max().item()
                    ),
                    "fp32_mean_abs_error": float(
                        compensated_difference.abs().mean().item()
                    ),
                    "fp32_relative_l2_error": float(
                        (torch.linalg.vector_norm(compensated_difference)
                         / torch.linalg.vector_norm(baseline)).item()
                    ),
                    "bf16_differing_elements": int(torch.count_nonzero(
                        compensated_bf16 != baseline_bf16
                    ).item()),
                    "bf16_max_abs_error": float(
                        (compensated_bf16.float() - baseline_bf16.float())
                        .abs().max().item()
                    ),
                    "vs_fp64_oracle": {
                        "max_abs_error": float(
                            compensated_oracle_difference.abs().max().item()
                        ),
                        "mean_abs_error": float(
                            compensated_oracle_difference.abs().mean().item()
                        ),
                        "relative_l2_error": float(
                            (torch.linalg.vector_norm(compensated_oracle_difference)
                             / torch.linalg.vector_norm(oracle)).item()
                        ),
                    },
                },
                "reduction_methods": {
                    name: {
                        "max_abs_error_vs_baseline": float(
                            (value.reshape_as(baseline) - baseline).abs().max().item()
                        ),
                        "max_abs_difference_vs_cuda_reducer": float(
                            (value.reshape_as(candidate) - candidate).abs().max().item()
                        ),
                    }
                    for name, value in reductions.items()
                },
            })
    torch.cuda.synchronize()

    summaries = {}
    for split, samples in rows.items():
        summaries[str(split)] = {
            "fp32_max_abs_error": distribution([
                item["fp32_max_abs_error"] for item in samples
            ]),
            "fp32_mean_abs_error": distribution([
                item["fp32_mean_abs_error"] for item in samples
            ]),
            "fp32_relative_l2_error": distribution([
                item["fp32_relative_l2_error"] for item in samples
            ]),
            "bf16_differing_elements": distribution([
                float(item["bf16_differing_elements"]) for item in samples
            ]),
            "bf16_nonzero_sample_count": sum(
                item["bf16_differing_elements"] > 0 for item in samples
            ),
            "bf16_max_abs_error": distribution([
                item["bf16_max_abs_error"] for item in samples
            ]),
            "compensated": {
                metric: distribution([
                    float(item["compensated"][metric]) for item in samples
                ])
                for metric in (
                    "fp32_max_abs_error", "fp32_mean_abs_error",
                    "fp32_relative_l2_error", "bf16_differing_elements",
                    "bf16_max_abs_error",
                )
            },
            "fp64_oracle": {
                path: {
                    metric: distribution([
                        float(
                            item[path][metric]
                            if path != "compensated"
                            else item[path]["vs_fp64_oracle"][metric]
                        )
                        for item in samples
                    ])
                    for metric in (
                        "max_abs_error", "mean_abs_error", "relative_l2_error"
                    )
                }
                for path in (
                    "baseline_vs_fp64_oracle", "split_vs_fp64_oracle",
                    "compensated",
                )
            },
            "reduction_methods": {
                method: {
                    "max_abs_error_vs_baseline": distribution([
                        item["reduction_methods"][method]["max_abs_error_vs_baseline"]
                        for item in samples
                    ]),
                    "max_abs_difference_vs_cuda_reducer": distribution([
                        item["reduction_methods"][method]["max_abs_difference_vs_cuda_reducer"]
                        for item in samples
                    ]),
                }
                for method in (
                    "fp32_sequential", "fp32_pairwise", "fp32_kahan", "fp64_sum"
                )
            },
        }
    payload = {
        "schema_version": 2,
        "protocol": "M1 split-K raw-FP32 deterministic activation stability",
        "checkpoint": str(args.checkpoint),
        "weight": args.weight,
        "shape": [1, record.feature_count, record.indices.shape[0]],
        "seed": args.seed,
        "sample_count": args.samples,
        "summaries": summaries,
        "samples": {str(key): value for key, value in rows.items()},
        "claim_boundary": (
            "Isolated layer-0 synthetic activations. FP64 oracle is the dot product "
            "of BF16-decoded A4 and W4 values; this diagnoses accumulation error "
            "but does not predict full-model token equality by itself."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
