"""Grouped packed NQRT kernels for projections sharing one input tensor."""

from __future__ import annotations

import math

import torch

from .triton_fused_linear import PreparedNQRTLinear
from .triton_tiled_decode import _nearest_e8_upstream
from .triton_unpack import _TRITON_INSTALLED


if _TRITON_INSTALLED:
    import triton
    import triton.language as tl

    @triton.jit
    def _nqrt_v4_fused_qkv_kernel(
        inputs,
        code_0, beta_0, norms_0, output_0,
        code_1, beta_1, norms_1, output_1,
        code_2, beta_2, norms_2, output_2,
        beta_table_0, beta_table_1, beta_table_2,
        token_count: tl.constexpr,
        rows_0: tl.constexpr,
        rows_1: tl.constexpr,
        rows_2: tl.constexpr,
        feature_count: tl.constexpr,
        blocks_per_row: tl.constexpr,
        sqrt_feature: tl.constexpr,
        q: tl.constexpr,
        seed_0, seed_1, seed_2,
        code_width: tl.constexpr,
        beta_width: tl.constexpr,
        block_rows: tl.constexpr,
        block_tokens: tl.constexpr,
    ):
        tokens = tl.program_id(0) * block_tokens + tl.arange(0, block_tokens)
        token_mask = tokens < token_count
        rows = tl.program_id(1) * block_rows + tl.arange(0, block_rows)
        group = tl.program_id(2)
        output_rows = tl.where(group == 0, rows_0, tl.where(group == 1, rows_1, rows_2))
        row_mask = rows < output_rows
        coordinates = tl.arange(0, 8)[None, :]
        accumulator = tl.zeros((block_tokens, block_rows), dtype=tl.float32)
        codes_per_byte: tl.constexpr = 8 // code_width
        betas_per_byte: tl.constexpr = 8 // beta_width

        for feature_block in tl.range(0, blocks_per_row, num_stages=1):
            block_ids = rows * blocks_per_row + feature_block
            blocks = block_ids[:, None]
            code_offsets = blocks * 8 + coordinates
            byte_offsets = code_offsets // codes_per_byte
            packed_codes_0 = tl.load(
                code_0 + byte_offsets,
                mask=row_mask[:, None] & (group == 0), other=0,
            )
            packed_codes_1 = tl.load(
                code_1 + byte_offsets,
                mask=row_mask[:, None] & (group == 1), other=0,
            )
            packed_codes_2 = tl.load(
                code_2 + byte_offsets,
                mask=row_mask[:, None] & (group == 2), other=0,
            )
            packed_codes = tl.where(
                group == 0, packed_codes_0,
                tl.where(group == 1, packed_codes_1, packed_codes_2),
            )
            code_shifts = (code_offsets % codes_per_byte) * code_width
            codes = (
                (packed_codes >> code_shifts) & ((1 << code_width) - 1)
            ).to(tl.float32)

            beta_offset = block_ids // betas_per_byte
            packed_beta_0 = tl.load(
                beta_0 + beta_offset, mask=row_mask & (group == 0), other=0,
            )
            packed_beta_1 = tl.load(
                beta_1 + beta_offset, mask=row_mask & (group == 1), other=0,
            )
            packed_beta_2 = tl.load(
                beta_2 + beta_offset, mask=row_mask & (group == 2), other=0,
            )
            packed_beta = tl.where(
                group == 0, packed_beta_0,
                tl.where(group == 1, packed_beta_1, packed_beta_2),
            )
            beta_shifts = (block_ids % betas_per_byte) * beta_width
            beta_ids = (packed_beta >> beta_shifts) & ((1 << beta_width) - 1)
            beta_value_0 = tl.load(
                beta_table_0 + beta_ids, mask=row_mask & (group == 0), other=0.0
            )
            beta_value_1 = tl.load(
                beta_table_1 + beta_ids, mask=row_mask & (group == 1), other=0.0
            )
            beta_value_2 = tl.load(
                beta_table_2 + beta_ids, mask=row_mask & (group == 2), other=0.0
            )
            beta = tl.where(
                group == 0, beta_value_0,
                tl.where(group == 1, beta_value_1, beta_value_2),
            )

            c7 = tl.sum(tl.where(coordinates == 7, codes, 0.0), axis=1)[:, None]
            next_offsets = blocks * 8 + tl.minimum(coordinates + 1, 7)
            next_bytes = next_offsets // codes_per_byte
            packed_next_0 = tl.load(
                code_0 + next_bytes,
                mask=row_mask[:, None] & (group == 0), other=0,
            )
            packed_next_1 = tl.load(
                code_1 + next_bytes,
                mask=row_mask[:, None] & (group == 1), other=0,
            )
            packed_next_2 = tl.load(
                code_2 + next_bytes,
                mask=row_mask[:, None] & (group == 2), other=0,
            )
            packed_next = tl.where(
                group == 0, packed_next_0,
                tl.where(group == 1, packed_next_1, packed_next_2),
            )
            next_shifts = (next_offsets % codes_per_byte) * code_width
            next_codes = (
                (packed_next >> next_shifts) & ((1 << code_width) - 1)
            ).to(tl.float32)
            coset = tl.where(
                coordinates == 0,
                2.0 * codes - next_codes + 0.5 * c7,
                tl.where(
                    coordinates < 6,
                    codes - next_codes + 0.5 * c7,
                    tl.where(coordinates == 6, codes + 0.5 * c7, 0.5 * c7),
                ),
            )

            # Counters deliberately remain local to each record. Concatenating
            # Q/K/V counters would reconstruct different dither vectors.
            counters = code_offsets.to(tl.int64)
            seed = tl.where(group == 0, seed_0, tl.where(group == 1, seed_1, seed_2))
            random = (counters + seed) & 0x7FFFFFFF
            random = random ^ (random >> 16)
            random = (random * 1_103_515_245 + 12_345) & 0x7FFFFFFF
            random = random ^ (random >> 13)
            random = (random * 134_775_813 + 1) & 0x7FFFFFFF
            random = random ^ (random >> 16)
            uniform = random.to(tl.float32) * (2.0 / 2_147_483_648.0)
            dither = uniform - _nearest_e8_upstream(uniform, coordinates)
            shifted_coset = coset - dither
            coarse = _nearest_e8_upstream(shifted_coset / q, coordinates)
            reconstructed = shifted_coset - q * coarse

            norm_0 = tl.load(norms_0 + rows, mask=row_mask & (group == 0), other=0.0)
            norm_1 = tl.load(norms_1 + rows, mask=row_mask & (group == 1), other=0.0)
            norm_2 = tl.load(norms_2 + rows, mask=row_mask & (group == 2), other=0.0)
            row_norm = tl.where(group == 0, norm_0, tl.where(group == 1, norm_1, norm_2))
            weights = reconstructed * beta[:, None]
            weights *= (row_norm / sqrt_feature)[:, None]
            input_offsets = tokens[:, None] * feature_count + feature_block * 8 + coordinates
            values = tl.load(
                inputs + input_offsets, mask=token_mask[:, None], other=0.0
            ).to(tl.float32)
            accumulator += tl.sum(values[:, None, :] * weights[None, :, :], axis=2)

        mask = token_mask[:, None] & row_mask[None, :]
        tl.store(
            output_0 + tokens[:, None] * rows_0 + rows[None, :], accumulator,
            mask=mask & (group == 0),
        )
        tl.store(
            output_1 + tokens[:, None] * rows_1 + rows[None, :], accumulator,
            mask=mask & (group == 1),
        )
        tl.store(
            output_2 + tokens[:, None] * rows_2 + rows[None, :], accumulator,
            mask=mask & (group == 2),
        )
else:
    triton = None
    _nqrt_v4_fused_qkv_kernel = None


def linear_fused_qkv_triton(
    inputs: torch.Tensor,
    qkv: tuple[PreparedNQRTLinear, PreparedNQRTLinear, PreparedNQRTLinear],
    *,
    block_rows: int = 16,
    block_tokens: int = 16,
    num_warps: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply three compatible packed projections in one Triton launch."""
    if not isinstance(inputs, torch.Tensor) or not inputs.is_floating_point():
        raise TypeError("inputs must be a floating-point tensor")
    if inputs.device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("grouped fused Linear requires CUDA inputs")
    if not isinstance(qkv, tuple) or len(qkv) != 3 or not all(
        isinstance(item, PreparedNQRTLinear) for item in qkv
    ):
        raise TypeError("qkv must contain three PreparedNQRTLinear values")
    if block_rows not in {4, 8, 16, 32}:
        raise ValueError("block_rows must be one of 4, 8, 16, or 32")
    if block_tokens not in {1, 2, 4, 8, 16}:
        raise ValueError("block_tokens must be one of 1, 2, 4, 8, or 16")
    if num_warps not in {1, 2, 4, 8}:
        raise ValueError("num_warps must be one of 1, 2, 4, or 8")
    if not _TRITON_INSTALLED:
        raise RuntimeError("Triton is not installed")
    if not inputs.is_contiguous():
        inputs = inputs.contiguous()

    records = tuple(item.record for item in qkv)
    first = records[0]
    compatibility = (
        first.feature_count,
        first.indices.q,
        first.indices.code_bit_width,
        first.indices.beta_index_bit_width,
        first.dither_algorithm,
        first.codec_semantics,
    )
    for item, record in zip(qkv, records, strict=True):
        current = (
            record.feature_count,
            record.indices.q,
            record.indices.code_bit_width,
            record.indices.beta_index_bit_width,
            record.dither_algorithm,
            record.codec_semantics,
        )
        if current != compatibility:
            raise ValueError("grouped projections must use compatible NQRT metadata")
        if item.code_payload.device != inputs.device:
            raise ValueError("prepared projections must be on the input device")
    if inputs.ndim < 1 or inputs.shape[-1] != first.feature_count:
        raise ValueError("input feature dimension does not match grouped weights")
    if inputs.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError("inputs must use float16, bfloat16, or float32")

    rows = tuple(record.indices.shape[0] for record in records)
    blocks_per_row = first.indices.shape[1]
    token_count = math.prod(inputs.shape[:-1])
    flat_inputs = inputs.reshape(token_count, first.feature_count)
    outputs = tuple(
        torch.empty((token_count, count), device=inputs.device, dtype=inputs.dtype)
        for count in rows
    )
    assert triton is not None and _nqrt_v4_fused_qkv_kernel is not None
    _nqrt_v4_fused_qkv_kernel[
        (
            triton.cdiv(token_count, block_tokens),
            max(triton.cdiv(count, block_rows) for count in rows),
            3,
        )
    ](
        flat_inputs,
        qkv[0].code_payload, qkv[0].beta_payload, qkv[0].row_norms, outputs[0],
        qkv[1].code_payload, qkv[1].beta_payload, qkv[1].row_norms, outputs[1],
        qkv[2].code_payload, qkv[2].beta_payload, qkv[2].row_norms, outputs[2],
        qkv[0].beta_table, qkv[1].beta_table, qkv[2].beta_table,
        token_count=token_count,
        rows_0=rows[0], rows_1=rows[1], rows_2=rows[2],
        feature_count=first.feature_count,
        blocks_per_row=blocks_per_row,
        sqrt_feature=first.feature_count ** 0.5,
        q=first.indices.q,
        seed_0=records[0].dither_seed,
        seed_1=records[1].dither_seed,
        seed_2=records[2].dither_seed,
        code_width=first.indices.code_bit_width,
        beta_width=first.indices.beta_index_bit_width,
        block_rows=block_rows,
        block_tokens=block_tokens,
        num_warps=num_warps,
    )
    leading = inputs.shape[:-1]
    return tuple(output.reshape(*leading, count) for output, count in zip(outputs, rows, strict=True))
