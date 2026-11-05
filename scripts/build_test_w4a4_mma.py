"""Build and validate the first direct-packed W4A4 CUDA MMA prototype."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import torch

from nestquant.core.codec import decode_nested
from nestquant.core.model_checkpoint_file import NQMCFileReader
from nestquant.core.tensor_checkpoint import PackedNestQuantTensor
from nestquant.runtime.cuda_w4a4_mma import (
    decode_packed_a4_m16,
    inspect_packed_a4_m16,
    linear_packed_w4a4_mma_m16,
)
from nestquant.runtime.cuda_mma_linear import linear_fused_mma_j2
from nestquant.runtime.triton_fused_linear import prepare_fused_linear
from nestquant.runtime.triton_w4a4_linear import (
    decode_packed_a4_triton_diagnostic,
    inspect_packed_a4_triton_diagnostic,
    linear_packed_w4a4_triton,
    linear_packed_w4a4_tiled_triton,
    quantize_activation_cuda,
)


BETAS = tuple(value / 14 for value in (3.47, 4.74, 6.90, 18.11))


def _time_cuda(operation, warmup: int, iterations: int) -> float:
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
    parser.add_argument("--weight", default="model.layers.0.self_attn.q_proj.weight")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reader = NQMCFileReader(args.checkpoint, verify_checksum=False)
    _, record = reader.read_named_record(args.weight)
    if not isinstance(record, PackedNestQuantTensor):
        raise TypeError("weight must be packed")
    weight = prepare_fused_linear(record)
    generator = torch.Generator(device="cuda").manual_seed(20261028)
    inputs = torch.randn(
        16, record.feature_count, device="cuda", dtype=torch.bfloat16,
        generator=generator,
    )
    activation = quantize_activation_cuda(inputs, q=14, betas=BETAS)
    reference = linear_packed_w4a4_tiled_triton(
        activation, weight, block_m=16, block_n=16, block_k=32, num_warps=8,
    )
    actual = linear_packed_w4a4_mma_m16(activation, weight)
    scalar_oracle = linear_packed_w4a4_triton(activation, weight)
    error = (actual.float() - reference.float()).abs()
    scalar_error = (actual.float() - scalar_oracle.float()).abs()
    tiled_scalar_error = (reference.float() - scalar_oracle.float()).abs()
    decoded_activation = decode_packed_a4_m16(activation)
    triton_decoded_activation = decode_packed_a4_triton_diagnostic(activation)
    cuda_inspection = inspect_packed_a4_m16(activation)
    triton_inspection = inspect_packed_a4_triton_diagnostic(activation)
    inspection_error = (cuda_inspection - triton_inspection).abs()
    half_residuals = triton_inspection[:, 16:24]
    half_maxima = half_residuals.amax(dim=1, keepdim=True)
    half_tie_counts = (half_residuals == half_maxima).sum(dim=1)
    half_parity_odd = triton_inspection[:, 25] >= 16
    half_tied_odd_blocks = torch.nonzero(
        (half_tie_counts > 1) & half_parity_odd
    ).flatten()
    code_positions = torch.arange(
        16 * record.feature_count, device="cuda", dtype=torch.long
    )
    unpacked_codes = (
        activation.code_payload[code_positions // 2].long()
        >> ((code_positions % 2) * 4)
    ) & 15
    block_count = 16 * record.feature_count // 8
    beta_positions = torch.arange(block_count, device="cuda", dtype=torch.long)
    unpacked_betas = (
        activation.beta_payload[beta_positions // 4].long()
        >> ((beta_positions % 4) * 2)
    ) & 3
    python_decoded = decode_nested(
        unpacked_codes.reshape(16, record.feature_count // 8, 8),
        14, dtype=torch.float32,
    )
    python_decoded *= activation.beta_table[unpacked_betas].reshape(
        16, record.feature_count // 8, 1
    )
    python_decoded *= (
        activation.row_norms / (record.feature_count ** 0.5)
    ).reshape(16, 1, 1)
    python_decoded = python_decoded.reshape(16, record.feature_count).bfloat16()
    activation_decode_error = (
        decoded_activation.float() - python_decoded.float()
    ).abs()
    cuda_triton_decode_error = (
        decoded_activation.float() - triton_decoded_activation.float()
    ).abs()
    cuda_triton_block_error = cuda_triton_decode_error.reshape(-1, 8).amax(dim=1)
    differing_blocks = torch.nonzero(cuda_triton_block_error != 0).flatten()
    first_differing_block = (
        int(differing_blocks[0].item()) if differing_blocks.numel() else -1
    )
    max_differing_block = int(cuda_triton_block_error.argmax().item())
    activation_max_index = int(activation_decode_error.argmax().item())
    activation_max_row = activation_max_index // record.feature_count
    activation_max_column = activation_max_index % record.feature_count
    debug_block_index = activation_max_row * (record.feature_count // 8) + activation_max_column // 8
    debug_codes = unpacked_codes.reshape(-1, 8)[debug_block_index].float()
    debug_c7 = debug_codes[7]
    debug_coset = torch.stack((
        2 * debug_codes[0] - debug_codes[1] + 0.5 * debug_c7,
        debug_codes[1] - debug_codes[2] + 0.5 * debug_c7,
        debug_codes[2] - debug_codes[3] + 0.5 * debug_c7,
        debug_codes[3] - debug_codes[4] + 0.5 * debug_c7,
        debug_codes[4] - debug_codes[5] + 0.5 * debug_c7,
        debug_codes[5] - debug_codes[6] + 0.5 * debug_c7,
        debug_codes[6] + 0.5 * debug_c7,
        0.5 * debug_c7,
    ))
    debug_x = debug_coset / 14.0
    debug_integer = torch.round(debug_x)
    if int(debug_integer.sum().item()) & 1:
        residual = debug_x - debug_integer
        debug_integer[residual.abs().argmax()] += torch.where(
            residual[residual.abs().argmax()] >= 0, 1.0, -1.0
        )
    debug_half_base = torch.round(debug_x - 0.5)
    debug_half = debug_half_base + 0.5
    if int(debug_half_base.sum().item()) & 1:
        residual = debug_x - debug_half
        debug_half[residual.abs().argmax()] += torch.where(
            residual[residual.abs().argmax()] >= 0, 1.0, -1.0
        )
    debug_integer_error = float(((debug_x-debug_integer)**2).sum().item())
    debug_half_error = float(((debug_x-debug_half)**2).sum().item())
    decoded_reference = linear_fused_mma_j2(decoded_activation, weight).bfloat16()
    decoded_error = (decoded_reference.float() - reference.float()).abs()
    mma_mapping_error = (actual.float() - decoded_reference.float()).abs()
    if args.groups < 1:
        parser.error("--groups must be positive")
    cuda_samples = []
    triton_samples = []
    for group in range(args.groups):
        operations = (
            (
                lambda: linear_packed_w4a4_mma_m16(activation, weight),
                lambda: linear_packed_w4a4_tiled_triton(
                    activation, weight, block_m=16, block_n=16, block_k=32,
                    num_warps=8,
                ),
            )
            if group % 2 == 0 else
            (
                lambda: linear_packed_w4a4_tiled_triton(
                    activation, weight, block_m=16, block_n=16, block_k=32,
                    num_warps=8,
                ),
                lambda: linear_packed_w4a4_mma_m16(activation, weight),
            )
        )
        first = _time_cuda(operations[0], args.warmup, args.iterations)
        second = _time_cuda(operations[1], args.warmup, args.iterations)
        if group % 2 == 0:
            cuda_samples.append(first); triton_samples.append(second)
        else:
            triton_samples.append(first); cuda_samples.append(second)
    cuda_us = statistics.median(cuda_samples)
    triton_us = statistics.median(triton_samples)
    payload = {
        "weight": args.weight,
        "shape": [16, record.feature_count, record.indices.shape[0]],
        "cuda_mma_us": cuda_us,
        "triton_us": triton_us,
        "speedup": triton_us / cuda_us,
        "timing_groups": args.groups,
        "cuda_us_samples": cuda_samples,
        "triton_us_samples": triton_samples,
        "cuda_us_min": min(cuda_samples),
        "cuda_us_max": max(cuda_samples),
        "triton_us_min": min(triton_samples),
        "triton_us_max": max(triton_samples),
        "max_abs_error": float(error.max().item()),
        "mean_abs_error": float(error.mean().item()),
        "cuda_vs_scalar_max_abs_error": float(scalar_error.max().item()),
        "cuda_vs_scalar_mean_abs_error": float(scalar_error.mean().item()),
        "tiled_vs_scalar_max_abs_error": float(tiled_scalar_error.max().item()),
        "tiled_vs_scalar_mean_abs_error": float(tiled_scalar_error.mean().item()),
        "decoded_then_j2_max_abs_error": float(decoded_error.max().item()),
        "decoded_then_j2_mean_abs_error": float(decoded_error.mean().item()),
        "mma_mapping_max_abs_error": float(mma_mapping_error.max().item()),
        "mma_mapping_mean_abs_error": float(mma_mapping_error.mean().item()),
        "activation_decode_max_abs_error": float(activation_decode_error.max().item()),
        "activation_decode_mean_abs_error": float(activation_decode_error.mean().item()),
        "cuda_triton_decode_max_abs_error": float(cuda_triton_decode_error.max().item()),
        "cuda_triton_decode_mean_abs_error": float(cuda_triton_decode_error.mean().item()),
        "cuda_triton_differing_block_count": int(differing_blocks.numel()),
        "cuda_triton_first_differing_block": first_differing_block,
        "cuda_triton_max_differing_block": max_differing_block,
        "cuda_triton_max_block_codes": [
            int(value) for value in unpacked_codes.reshape(-1, 8)[max_differing_block].tolist()
        ],
        "cuda_triton_max_block_cuda": decoded_activation.reshape(-1, 8)[
            max_differing_block
        ].float().tolist(),
        "cuda_triton_max_block_triton": triton_decoded_activation.reshape(-1, 8)[
            max_differing_block
        ].float().tolist(),
        "inspection_max_abs_error": float(inspection_error.max().item()),
        "inspection_differing_value_count": int((inspection_error != 0).sum().item()),
        "half_tied_odd_block_count": int(half_tied_odd_blocks.numel()),
        "half_tied_odd_blocks": half_tied_odd_blocks.tolist(),
        "decode_differing_blocks": differing_blocks.tolist(),
        "decode_differing_block_codes": [
            [int(value) for value in unpacked_codes.reshape(-1, 8)[index].tolist()]
            for index in differing_blocks.tolist()
        ],
        "max_block_cuda_x": cuda_inspection[max_differing_block, :8].tolist(),
        "max_block_triton_x": triton_inspection[max_differing_block, :8].tolist(),
        "max_block_cuda_integer_residual": cuda_inspection[
            max_differing_block, 8:16
        ].tolist(),
        "max_block_triton_integer_residual": triton_inspection[
            max_differing_block, 8:16
        ].tolist(),
        "max_block_cuda_half_residual": cuda_inspection[
            max_differing_block, 16:24
        ].tolist(),
        "max_block_triton_half_residual": triton_inspection[
            max_differing_block, 16:24
        ].tolist(),
        "max_block_cuda_selected": cuda_inspection[
            max_differing_block, 24:26
        ].tolist(),
        "max_block_triton_selected": triton_inspection[
            max_differing_block, 24:26
        ].tolist(),
        "activation_decode_max_location": [activation_max_row, activation_max_column],
        "activation_decode_cuda_value": float(
            decoded_activation[activation_max_row, activation_max_column].item()
        ),
        "activation_decode_python_value": float(
            python_decoded[activation_max_row, activation_max_column].item()
        ),
        "activation_decode_row_norm": float(
            activation.row_norms[activation_max_row].item()
        ),
        "debug_codes": [int(value) for value in debug_codes.tolist()],
        "debug_beta_id": int(unpacked_betas[debug_block_index].item()),
        "debug_coset_over_q": debug_x.tolist(),
        "debug_integer_candidate": debug_integer.tolist(),
        "debug_half_candidate": debug_half.tolist(),
        "debug_integer_error": debug_integer_error,
        "debug_half_error": debug_half_error,
        "finite": bool(torch.isfinite(actual).all()),
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
