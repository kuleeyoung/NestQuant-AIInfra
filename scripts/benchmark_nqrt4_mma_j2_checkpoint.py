"""Benchmark J2 CUDA MMA against real NQRT v4 checkpoint records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nestquant.core.codec import STATELESS_DITHER_ALGORITHM, UPSTREAM_CODEC_SEMANTICS
from nestquant.core.model_checkpoint_file import NQMCFileReader
from nestquant.core.tensor_checkpoint import PackedNestQuantTensor
from nestquant.runtime.triton_fused_linear import linear_fused_triton, prepare_fused_linear
from nestquant.runtime.triton_tiled_decode import dequantize_tiled_triton


DEFAULT_CHECKPOINT = (
    ROOT / "results/raw/llama32_1b_paper_hadamard_w4_shared128_ldlq_nqrt4.nqmc"
)
DEFAULT_WEIGHTS = (
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
)


def measure(operation, *, warmup: int, iterations: int, repeats: int):
    for _ in range(warmup):
        operation()
    values = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iterations):
            operation()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000 / iterations)
    return {
        "median_us": statistics.median(values),
        "min_us": min(values),
        "max_us": max(values),
    }


def validate_record(record: PackedNestQuantTensor) -> None:
    if record.indices.q != 14 or record.indices.code_bit_width != 4:
        raise ValueError("J2 currently requires q14 with 4-bit residue codes")
    if record.indices.beta_count != 4 or record.indices.beta_index_bit_width != 2:
        raise ValueError("J2 currently requires four beta values with 2-bit IDs")
    if record.dither_algorithm != STATELESS_DITHER_ALGORITHM or record.dither_seed is None:
        raise ValueError("J2 requires NQRT v4 stateless dither")
    if record.codec_semantics != UPSTREAM_CODEC_SEMANTICS:
        raise ValueError("J2 requires upstream E8 codec semantics")
    rows, blocks_per_row, coordinates = record.indices.shape
    if coordinates != 8 or blocks_per_row * 8 != record.feature_count:
        raise ValueError("record has an incompatible E8 block shape")
    if rows % 8 or record.feature_count % 16:
        raise ValueError("J2 requires N divisible by 8 and K divisible by 16")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--weight", action="append", dest="weights")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--tokens", type=int, choices=(1, 16), default=16)
    parser.add_argument(
        "--sum8-implementation", choices=("xor4", "broadcast8"), default="xor4",
        help="select the optimized reduction or the original diagnostic reference",
    )
    parser.add_argument(
        "--parity-implementation", choices=("packed2", "separate4"), default="packed2",
        help="select packed two-bit parity reduction or the original separate reductions",
    )
    parser.add_argument(
        "--max-index-implementation", choices=("ballot", "shuffle"), default="ballot",
        help="select ballot tie resolution or the original value-and-index shuffles",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/raw/llama32_1b_nqrt4_mma_j2_checkpoint.json",
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1 or args.repeats < 1:
        parser.error("warmup must be nonnegative; iterations and repeats must be positive")

    extension = load(
        name=(f"nqrt4_mma_j2_{args.sum8_implementation}_"
              f"{args.parity_implementation}_{args.max_index_implementation}_ext"),
        sources=[
            str(ROOT / "nestquant/runtime/nqrt4_mma_j2_binding.cpp"),
            str(ROOT / "nestquant/runtime/nqrt4_mma_j2.cu"),
        ],
        extra_cuda_cflags=[
            "-O3", "-Xptxas=-v",
            *(["-DNQRT4_SUM8_BROADCAST_REFERENCE"]
              if args.sum8_implementation == "broadcast8" else []),
            *(["-DNQRT4_PARITY_SEPARATE_REFERENCE"]
              if args.parity_implementation == "separate4" else []),
            *(["-DNQRT4_MAX_INDEX_SHUFFLE_REFERENCE"]
              if args.max_index_implementation == "shuffle" else []),
        ],
        verbose=False,
    )
    reader = NQMCFileReader(args.checkpoint, verify_checksum=False)
    generator = torch.Generator(device="cuda").manual_seed(20261026)
    records = []
    for weight_name in args.weights or DEFAULT_WEIGHTS:
        _, record = reader.read_named_record(weight_name)
        if not isinstance(record, PackedNestQuantTensor):
            raise TypeError(f"{weight_name} is not a packed NQRT record")
        validate_record(record)
        prepared = prepare_fused_linear(record)
        n_size, _, _ = record.indices.shape
        k_size = record.feature_count
        inputs = torch.randn(
            (args.tokens, k_size), device="cuda", dtype=torch.bfloat16, generator=generator
        )
        bf16_weight = dequantize_tiled_triton(
            record, device="cuda", output_dtype=torch.bfloat16
        )
        j2_operation = (
            extension.run_m1_auto if args.tokens == 1 else extension.run_m16_auto
        )
        fp32_reference = inputs.float() @ bf16_weight.float().T
        bf16_reference = F.linear(inputs, bf16_weight)
        j2_fp32 = j2_operation(
            inputs, prepared.code_payload, prepared.beta_payload,
            prepared.row_norms, prepared.beta_table, 0, record.dither_seed,
        )
        j2_bf16 = j2_fp32.to(torch.bfloat16)
        triton_bf16 = linear_fused_triton(inputs, prepared)
        torch.cuda.synchronize()
        j2_fp32_error = (j2_fp32 - fp32_reference).abs()
        j2_bf16_error = (j2_bf16.float() - bf16_reference.float()).abs()
        triton_error = (triton_bf16.float() - bf16_reference.float()).abs()
        worst_flat = int(j2_fp32_error.argmax().item())
        worst_token, worst_row = divmod(worst_flat, n_size)
        reference_norm = torch.linalg.vector_norm(fp32_reference)

        timings = {
            "j2_auto_fp32": measure(
                lambda: j2_operation(
                    inputs, prepared.code_payload, prepared.beta_payload,
                    prepared.row_norms, prepared.beta_table, 0, record.dither_seed,
                ),
                warmup=args.warmup, iterations=args.iterations, repeats=args.repeats,
            ),
            "j2_auto_bf16": measure(
                lambda: j2_operation(
                    inputs, prepared.code_payload, prepared.beta_payload,
                    prepared.row_norms, prepared.beta_table, 0, record.dither_seed,
                ).to(torch.bfloat16),
                warmup=args.warmup, iterations=args.iterations, repeats=args.repeats,
            ),
            "triton_packed_bf16": measure(
                lambda: linear_fused_triton(inputs, prepared),
                warmup=args.warmup, iterations=args.iterations, repeats=args.repeats,
            ),
            "resident_bf16": measure(
                lambda: F.linear(inputs, bf16_weight),
                warmup=args.warmup, iterations=args.iterations, repeats=args.repeats,
            ),
        }
        packed_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                prepared.code_payload, prepared.beta_payload,
                prepared.row_norms, prepared.beta_table,
            )
        )
        item = {
            "weight_name": weight_name,
            "weight_shape": [n_size, k_size],
            "dither_seed": record.dither_seed,
            "packed_bytes": packed_bytes,
            "bf16_weight_bytes": bf16_weight.numel() * bf16_weight.element_size(),
            "correctness": {
                "j2_fp32_vs_fp32_reference": {
                    "max_abs_error": float(j2_fp32_error.max().item()),
                    "mean_abs_error": float(j2_fp32_error.mean().item()),
                    "relative_l2_error": float(
                        (torch.linalg.vector_norm(j2_fp32_error) / reference_norm).item()
                    ),
                    "reference_max_abs": float(fp32_reference.abs().max().item()),
                    "worst_output_index": [worst_token, worst_row],
                    "worst_reference_value": float(fp32_reference[worst_token, worst_row].item()),
                    "worst_actual_value": float(j2_fp32[worst_token, worst_row].item()),
                    "worst_row_norm": float(prepared.row_norms[worst_row].item()),
                    "count_abs_error_gt_1e_3": int((j2_fp32_error > 1e-3).sum().item()),
                },
                "j2_bf16_vs_bf16_gemm": {
                    "max_abs_error": float(j2_bf16_error.max().item()),
                    "mean_abs_error": float(j2_bf16_error.mean().item()),
                },
                "triton_bf16_vs_bf16_gemm": {
                    "max_abs_error": float(triton_error.max().item()),
                    "mean_abs_error": float(triton_error.mean().item()),
                },
            },
            "timings": timings,
        }
        records.append(item)
        print(
            f"{weight_name} shape={n_size}x{k_size} "
            f"j2_max={item['correctness']['j2_fp32_vs_fp32_reference']['max_abs_error']:.6g} "
            f"j2_us={timings['j2_auto_bf16']['median_us']:.6g} "
            f"triton_us={timings['triton_packed_bf16']['median_us']:.6g} "
            f"bf16_us={timings['resident_bf16']['median_us']:.6g}",
            flush=True,
        )
        del prepared, inputs, bf16_weight, fp32_reference, bf16_reference
        del j2_fp32, j2_bf16, triton_bf16
        torch.cuda.empty_cache()

    report = {
        "scope": f"real Llama-3.2-1B NQRT v4 records, M={args.tokens} J2 checkpoint integration",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "device": torch.cuda.get_device_name(),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "tokens": args.tokens,
        "sum8_implementation": args.sum8_implementation,
        "parity_implementation": args.parity_implementation,
        "max_index_implementation": args.max_index_implementation,
        "records": records,
        "claim_boundary": (
            "Single real packed projection records with synthetic BF16 activations; "
            "not yet a decoder-block or end-to-end model timing."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
