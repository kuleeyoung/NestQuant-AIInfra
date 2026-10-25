"""Experimental NQRT fused Linear using Triton block dot products."""

from __future__ import annotations

import math
from typing import TypedDict

import torch

from .triton_fused_linear import PreparedNQRTLinear
from .triton_unpack import _TRITON_INSTALLED


class NQRTDotConfig(TypedDict):
    """Launch parameters selected from the real Llama projection sweep."""

    block_m: int
    block_n: int
    block_k: int
    num_warps: int


def select_nqrt_dot_config(
    out_features: int, in_features: int, token_count: int
) -> NQRTDotConfig:
    """Choose a conservative launch config from the RTX 5070 shape sweep.

    The measured Llama-3.2-1B shapes receive their locally fastest config.
    Other compatible shapes use a stable fallback instead of pretending that
    the small empirical table is a general-purpose autotuner.
    """
    if min(out_features, in_features, token_count) < 1:
        raise ValueError("feature dimensions and token_count must be positive")
    if token_count > 16:
        if in_features == 2048 and out_features == 2048:
            block_m = 64 if 32 < token_count <= 64 else 128
            return {
                "block_m": block_m, "block_n": 16,
                "block_k": 32, "num_warps": 8,
            }
        if in_features == 2048 and out_features == 512:
            return {
                "block_m": 64 if token_count <= 64 else 128,
                "block_n": 16,
                "block_k": 32 if token_count <= 64 else 64,
                "num_warps": 8,
            }
        if in_features == 2048 and out_features == 8192:
            return {
                "block_m": 64 if token_count <= 64 else 128,
                "block_n": 16, "block_k": 32, "num_warps": 8,
            }
        if in_features == 8192 and out_features == 2048:
            block_m = 32 if token_count <= 32 else (64 if token_count <= 64 else 128)
            block_k = 64 if token_count <= 32 else (128 if token_count <= 64 else 32)
            return {
                "block_m": block_m, "block_n": 16,
                "block_k": block_k, "num_warps": 8,
            }
        block_m = 32 if token_count <= 32 else (64 if token_count <= 64 else 128)
        return {
            "block_m": block_m, "block_n": 16,
            "block_k": 32, "num_warps": 8,
        }
    if in_features == 8192 and out_features == 2048:
        return {"block_m": 16, "block_n": 16, "block_k": 128, "num_warps": 8}
    if in_features == 2048 and out_features == 512:
        block_k = 128 if token_count > 8 else 32
        return {"block_m": 16, "block_n": 16, "block_k": block_k, "num_warps": 8}
    if in_features == 2048 and out_features == 8192:
        if token_count == 1:
            return {"block_m": 16, "block_n": 16, "block_k": 32, "num_warps": 4}
        block_k = 64 if token_count <= 8 else 32
        return {"block_m": 16, "block_n": 16, "block_k": block_k, "num_warps": 8}
    if in_features == 2048 and out_features == 2048:
        if token_count == 1:
            return {"block_m": 16, "block_n": 16, "block_k": 32, "num_warps": 8}
        return {
            "block_m": 16,
            "block_n": 16,
            "block_k": 64,
            "num_warps": 4 if token_count <= 8 else 8,
        }
    return {"block_m": 16, "block_n": 16, "block_k": 32, "num_warps": 4}


if _TRITON_INSTALLED:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def _nearest_e8_3d(x, coordinates):
        rounded = libdevice.nearbyint(x)
        odd = (tl.sum(rounded, axis=2).to(tl.int64) & 1) != 0
        residual = x - rounded
        selected = tl.argmax(tl.abs(residual), axis=2, tie_break_left=True)
        direction = tl.where(
            tl.sum(
                tl.where(coordinates == selected[:, :, None], residual, 0.0),
                axis=2,
            ) >= 0.0,
            1.0, -1.0,
        )
        correction = tl.where(
            coordinates == selected[:, :, None], direction[:, :, None], 0.0
        )
        integer = rounded + correction * odd[:, :, None]

        shifted = x - 0.5
        half_rounded = libdevice.nearbyint(shifted)
        half_odd = (tl.sum(half_rounded, axis=2).to(tl.int64) & 1) != 0
        half_residual = shifted - half_rounded
        half_selected = tl.argmax(tl.abs(half_residual), axis=2, tie_break_left=True)
        half_direction = tl.where(
            tl.sum(
                tl.where(
                    coordinates == half_selected[:, :, None], half_residual, 0.0
                ),
                axis=2,
            ) >= 0.0,
            1.0, -1.0,
        )
        half_correction = tl.where(
            coordinates == half_selected[:, :, None],
            half_direction[:, :, None], 0.0,
        )
        half_integer = half_rounded + half_correction * half_odd[:, :, None] + 0.5
        integer_error = tl.sum((x - integer) * (x - integer), axis=2)
        half_error = tl.sum((x - half_integer) * (x - half_integer), axis=2)
        return tl.where(
            (integer_error < half_error)[:, :, None], integer, half_integer
        )

    @triton.jit
    def _nqrt_v4_dot_linear_kernel(
        inputs,
        code_payload,
        beta_payload,
        row_norms,
        beta_table,
        bias,
        output,
        token_count: tl.constexpr,
        output_rows: tl.constexpr,
        feature_count: tl.constexpr,
        blocks_per_row: tl.constexpr,
        sqrt_feature: tl.constexpr,
        q: tl.constexpr,
        seed,
        code_width: tl.constexpr,
        beta_width: tl.constexpr,
        has_bias: tl.constexpr,
        full_e8_decode: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
    ):
        tokens = tl.program_id(0) * block_m + tl.arange(0, block_m)
        rows = tl.program_id(1) * block_n + tl.arange(0, block_n)
        token_mask = tokens < token_count
        row_mask = rows < output_rows
        accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
        codes_per_byte: tl.constexpr = 8 // code_width
        betas_per_byte: tl.constexpr = 8 // beta_width

        for feature_start in tl.range(0, feature_count, block_k, num_stages=1):
            features = feature_start + tl.arange(0, block_k)
            feature_mask = features < feature_count
            local_blocks = tl.arange(0, block_k // 8)
            coordinates = tl.arange(0, 8)[None, None, :]
            feature_blocks = feature_start // 8 + local_blocks
            block_ids = rows[:, None] * blocks_per_row + feature_blocks[None, :]
            code_offsets = block_ids[:, :, None] * 8 + coordinates
            packed_codes = tl.load(
                code_payload + code_offsets // codes_per_byte,
                mask=row_mask[:, None, None], other=0,
            )
            shifts = (code_offsets % codes_per_byte) * code_width
            codes = ((packed_codes >> shifts) & ((1 << code_width) - 1)).to(tl.float32)

            next_coordinates = tl.minimum(coordinates + 1, 7)
            next_offsets = block_ids[:, :, None] * 8 + next_coordinates
            packed_next = tl.load(
                code_payload + next_offsets // codes_per_byte,
                mask=row_mask[:, None, None], other=0,
            )
            next_shifts = (next_offsets % codes_per_byte) * code_width
            next_codes = (
                (packed_next >> next_shifts) & ((1 << code_width) - 1)
            ).to(tl.float32)
            c7 = tl.sum(tl.where(coordinates == 7, codes, 0.0), axis=2)[:, :, None]
            coset = tl.where(
                coordinates == 0,
                2.0 * codes - next_codes + 0.5 * c7,
                tl.where(
                    coordinates < 6,
                    codes - next_codes + 0.5 * c7,
                    tl.where(
                        coordinates == 6,
                        codes + 0.5 * c7,
                        0.5 * c7,
                    ),
                ),
            )

            if full_e8_decode:
                counters = code_offsets.to(tl.int64)
                random = (counters + seed) & 0x7FFFFFFF
                random = random ^ (random >> 16)
                random = (random * 1_103_515_245 + 12_345) & 0x7FFFFFFF
                random = random ^ (random >> 13)
                random = (random * 134_775_813 + 1) & 0x7FFFFFFF
                random = random ^ (random >> 16)
                uniform = random.to(tl.float32) * (2.0 / 2_147_483_648.0)
                dither = uniform - _nearest_e8_3d(uniform, coordinates)
                shifted_coset = coset - dither
                coarse = _nearest_e8_3d(shifted_coset / q, coordinates)
                reconstructed = shifted_coset - q * coarse
            else:
                # Performance-only ablation: preserve packed reads, coset,
                # scaling, and tl.dot while removing RNG and E8 projections.
                reconstructed = coset

            packed_beta = tl.load(
                beta_payload + block_ids // betas_per_byte,
                mask=row_mask[:, None], other=0,
            )
            beta_shifts = (block_ids % betas_per_byte) * beta_width
            beta_ids = (packed_beta >> beta_shifts) & ((1 << beta_width) - 1)
            beta = tl.load(
                beta_table + beta_ids,
                mask=row_mask[:, None], other=0.0,
            )
            row_norm = tl.load(row_norms + rows, mask=row_mask, other=0.0)
            weights = reconstructed * beta[:, :, None]
            weights *= (row_norm / sqrt_feature)[:, None, None]
            weights = tl.reshape(weights, (block_n, block_k))

            input_offsets = tokens[:, None] * feature_count + features[None, :]
            values = tl.load(
                inputs + input_offsets,
                mask=token_mask[:, None] & feature_mask[None, :], other=0.0,
            ).to(tl.bfloat16)
            accumulator = tl.dot(
                values,
                tl.trans(weights.to(tl.bfloat16)),
                accumulator,
                input_precision="ieee",
            )

        if has_bias:
            accumulator += tl.load(bias + rows, mask=row_mask, other=0.0)[None, :]
        tl.store(
            output + tokens[:, None] * output_rows + rows[None, :],
            accumulator,
            mask=token_mask[:, None] & row_mask[None, :],
        )
else:
    triton = None
    _nqrt_v4_dot_linear_kernel = None


def linear_fused_dot_triton(
    inputs: torch.Tensor,
    prepared: PreparedNQRTLinear,
    bias: torch.Tensor | None = None,
    *,
    block_m: int = 16,
    block_n: int = 16,
    block_k: int = 32,
    num_warps: int = 4,
    _full_e8_decode: bool = True,
) -> torch.Tensor:
    """Apply one prepared packed weight through a block-dot prototype."""
    if not isinstance(inputs, torch.Tensor) or not inputs.is_floating_point():
        raise TypeError("inputs must be a floating-point tensor")
    if not isinstance(prepared, PreparedNQRTLinear):
        raise TypeError("prepared must be a PreparedNQRTLinear")
    if inputs.device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("dot fused Linear requires CUDA inputs")
    if inputs.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError("inputs must use float16, bfloat16, or float32")
    if block_m not in {16, 32, 64, 128} or block_n not in {16, 32, 64}:
        raise ValueError("block_m must be 16, 32, 64, or 128; block_n must be 16, 32, or 64")
    if block_k not in {16, 32, 64, 128}:
        raise ValueError("block_k must be 16, 32, 64, or 128")
    if num_warps not in {2, 4, 8}:
        raise ValueError("num_warps must be 2, 4, or 8")
    if not isinstance(_full_e8_decode, bool):
        raise TypeError("_full_e8_decode must be a bool")
    if not _TRITON_INSTALLED:
        raise RuntimeError("Triton is not installed")
    record = prepared.record
    if inputs.ndim < 1 or inputs.shape[-1] != record.feature_count:
        raise ValueError("input feature dimension does not match the packed weight")
    if prepared.code_payload.device != inputs.device:
        raise ValueError("prepared packed state must be on the input device")
    if not inputs.is_contiguous():
        inputs = inputs.contiguous()
    rows = record.indices.shape[0]
    if bias is not None and (bias.device != inputs.device or tuple(bias.shape) != (rows,)):
        raise ValueError("bias must be on the input device with shape [out_features]")

    token_count = math.prod(inputs.shape[:-1])
    flat_inputs = inputs.reshape(token_count, record.feature_count)
    output = torch.empty((token_count, rows), device=inputs.device, dtype=inputs.dtype)
    bias_pointer = output if bias is None else bias
    assert triton is not None and _nqrt_v4_dot_linear_kernel is not None
    _nqrt_v4_dot_linear_kernel[
        (triton.cdiv(token_count, block_m), triton.cdiv(rows, block_n))
    ](
        flat_inputs,
        prepared.code_payload,
        prepared.beta_payload,
        prepared.row_norms,
        prepared.beta_table,
        bias_pointer,
        output,
        token_count=token_count,
        output_rows=rows,
        feature_count=record.feature_count,
        blocks_per_row=record.indices.shape[1],
        sqrt_feature=record.feature_count ** 0.5,
        q=record.indices.q,
        seed=record.dither_seed,
        code_width=record.indices.code_bit_width,
        beta_width=record.indices.beta_index_bit_width,
        has_bias=bias is not None,
        full_e8_decode=_full_e8_decode,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
    )
    return output.reshape(*inputs.shape[:-1], rows)


def linear_fused_dot_no_e8_ablation_triton(
    inputs: torch.Tensor,
    prepared: PreparedNQRTLinear,
    bias: torch.Tensor | None = None,
    **config: int,
) -> torch.Tensor:
    """Performance-only path with RNG/E8 projection removed.

    The output has no model semantics and must only be used to attribute kernel
    time. Packed reads, coset construction, scaling, and ``tl.dot`` remain.
    """
    return linear_fused_dot_triton(
        inputs, prepared, bias, _full_e8_decode=False, **config
    )
