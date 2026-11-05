"""Validate direct-packed CUDA W4A4 across every packed model projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nestquant.runtime.cuda_w4a4_cute import (
    linear_packed_w4a4_cute_large_m_probe,
)

from nestquant.core.model_checkpoint_file import NQMCFileReader
from nestquant.core.tensor_checkpoint import PackedNestQuantTensor
from nestquant.runtime.cuda_w4a4_mma import (
    linear_packed_w4a4_gemv_m1,
    linear_packed_w4a4_gemv_m1_packed_word_probe,
    linear_packed_w4a4_mma_large_m_multiwarp_probe,
    linear_packed_w4a4_mma_large_m_staged,
    linear_packed_w4a4_mma_chunked_staged_probe,
    linear_packed_w4a4_mma_small_m,
    linear_packed_w4a4_mma_m16,
)
from nestquant.runtime.triton_fused_linear import prepare_fused_linear
from nestquant.runtime.triton_w4a4_linear import (
    linear_packed_w4a4_tiled_triton,
    quantize_activation_cuda,
)


BETAS = tuple(value / 14 for value in (3.47, 4.74, 6.90, 18.11))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--tokens", type=int, default=16,
    )
    parser.add_argument(
        "--large-m-implementation",
        choices=("unstaged", "staged", "cute", "chunked"),
        default="unstaged",
        help="Large-M implementation to validate for M32/M64/M128.",
    )
    parser.add_argument(
        "--m1-implementation",
        choices=("production", "packed-word-probe"),
        default="production",
        help="M=1 CUDA implementation to validate; ignored for --tokens 16.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.tokens < 1:
        parser.error("--tokens must be positive")
    reader = NQMCFileReader(args.checkpoint, verify_checksum=False)
    names = tuple(
        entry.name for entry in reader.entries
        if entry.name.endswith(".weight") and entry.kind == "nestquant"
    )
    rows = []
    generator = torch.Generator(device="cuda").manual_seed(20261029)
    for name in names:
        _, record = reader.read_named_record(name)
        if not isinstance(record, PackedNestQuantTensor):
            continue
        weight = prepare_fused_linear(record)
        inputs = torch.randn(
            args.tokens, record.feature_count, generator=generator,
            device="cuda", dtype=torch.bfloat16,
        )
        activation = quantize_activation_cuda(inputs, q=14, betas=BETAS)
        if args.tokens == 1:
            actual = (
                linear_packed_w4a4_gemv_m1_packed_word_probe(activation, weight)
                if args.m1_implementation == "packed-word-probe" else
                linear_packed_w4a4_gemv_m1(activation, weight)
            )
        elif args.tokens in (2, 4, 8):
            actual = linear_packed_w4a4_mma_small_m(
                activation, weight
            )
        elif args.tokens == 16:
            actual = linear_packed_w4a4_mma_m16(activation, weight)
        elif args.large_m_implementation == "chunked":
            actual = linear_packed_w4a4_mma_chunked_staged_probe(
                activation, weight
            )
        elif args.tokens in (32, 64, 128):
            if args.large_m_implementation == "cute":
                actual = linear_packed_w4a4_cute_large_m_probe(
                    activation, weight
                )
            elif args.large_m_implementation == "staged":
                actual = linear_packed_w4a4_mma_large_m_staged(
                    activation, weight
                )
            else:
                actual = linear_packed_w4a4_mma_large_m_multiwarp_probe(
                    activation, weight
                )
        else:
            parser.error(
                "unsupported token count without --large-m-implementation chunked"
            )
        reference = linear_packed_w4a4_tiled_triton(
            activation, weight, block_m=16, block_n=16, block_k=32,
            num_warps=8,
        )
        error = (actual.float() - reference.float()).abs()
        reference_float = reference.float()
        relative_l2 = torch.linalg.vector_norm(actual.float() - reference_float) / torch.clamp(
            torch.linalg.vector_norm(reference_float), min=torch.finfo(torch.float32).tiny
        )
        row = {
            "name": name,
            "shape": [args.tokens, record.feature_count, record.indices.shape[0]],
            "max_abs_error": float(error.max().item()),
            "mean_abs_error": float(error.mean().item()),
            "relative_l2_error": float(relative_l2.item()),
            "reference_max_abs": float(reference_float.abs().max().item()),
            "allclose_atol_0_125_rtol_0_01": bool(torch.allclose(
                actual.float(), reference_float, atol=0.125, rtol=0.01
            )),
        }
        if args.tokens == 1 and args.m1_implementation == "packed-word-probe":
            production = linear_packed_w4a4_gemv_m1(activation, weight)
            production_error = (actual.float() - production.float()).abs()
            row.update({
                "production_exact_equal": bool(torch.equal(actual, production)),
                "production_max_abs_error": float(production_error.max().item()),
            })
        rows.append(row)
    payload = {
        "projection_count": len(rows),
        "tokens": args.tokens,
        "m1_implementation": (
            args.m1_implementation if args.tokens == 1 else None
        ),
        "large_m_implementation": (
            args.large_m_implementation
            if args.tokens in (32, 64, 128)
            or args.large_m_implementation == "chunked" else None
        ),
        "max_abs_error": max(row["max_abs_error"] for row in rows),
        "max_mean_abs_error": max(row["mean_abs_error"] for row in rows),
        "max_relative_l2_error": max(row["relative_l2_error"] for row in rows),
        "all_layers_within_quality_gate": all(
            row["allclose_atol_0_125_rtol_0_01"] for row in rows
        ),
        "production_all_exact_equal": (
            all(row["production_exact_equal"] for row in rows)
            if args.tokens == 1 and args.m1_implementation == "packed-word-probe"
            else None
        ),
        "production_max_abs_error": (
            max(row["production_max_abs_error"] for row in rows)
            if args.tokens == 1 and args.m1_implementation == "packed-word-probe"
            else None
        ),
        "rows": rows,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
