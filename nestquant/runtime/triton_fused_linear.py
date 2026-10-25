"""Prototype NQRT v4 packed decode fused directly into a Linear GEMM."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ..core.codec import STATELESS_DITHER_ALGORITHM, UPSTREAM_CODEC_SEMANTICS
from ..core.tensor_checkpoint import PackedNestQuantTensor
from .triton_tiled_decode import _nearest_e8_upstream
from .triton_unpack import _TRITON_INSTALLED, _payload_tensor


if _TRITON_INSTALLED:
    import triton
    import triton.language as tl

    @triton.jit
    def _nqrt_v4_fused_linear_kernel(
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
        block_rows: tl.constexpr,
        block_tokens: tl.constexpr,
    ):
        tokens = tl.program_id(0) * block_tokens + tl.arange(0, block_tokens)
        token_mask = tokens < token_count
        rows = tl.program_id(1) * block_rows + tl.arange(0, block_rows)
        row_mask = rows < output_rows
        coordinates = tl.arange(0, 8)[None, :]
        accumulator = tl.zeros((block_tokens, block_rows), dtype=tl.float32)
        codes_per_byte: tl.constexpr = 8 // code_width
        betas_per_byte: tl.constexpr = 8 // beta_width

        for feature_block in tl.range(0, blocks_per_row, num_stages=1):
            block_ids = rows * blocks_per_row + feature_block
            blocks = block_ids[:, None]
            code_offsets = blocks * 8 + coordinates
            packed_codes = tl.load(
                code_payload + code_offsets // codes_per_byte,
                mask=row_mask[:, None], other=0,
            )
            code_shifts = (code_offsets % codes_per_byte) * code_width
            codes = (
                (packed_codes >> code_shifts) & ((1 << code_width) - 1)
            ).to(tl.float32)

            packed_betas = tl.load(
                beta_payload + block_ids // betas_per_byte,
                mask=row_mask, other=0,
            )
            beta_shifts = (block_ids % betas_per_byte) * beta_width
            beta_ids = (
                (packed_betas >> beta_shifts) & ((1 << beta_width) - 1)
            )
            beta = tl.load(beta_table + beta_ids, mask=row_mask, other=0.0)

            c7 = tl.sum(tl.where(coordinates == 7, codes, 0.0), axis=1)[:, None]
            next_offsets = blocks * 8 + tl.minimum(coordinates + 1, 7)
            packed_next = tl.load(
                code_payload + next_offsets // codes_per_byte,
                mask=row_mask[:, None], other=0,
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

            counters = code_offsets.to(tl.int64)
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

            row_norm = tl.load(row_norms + rows, mask=row_mask, other=0.0)
            weights = reconstructed * beta[:, None]
            weights *= (row_norm / sqrt_feature)[:, None]
            input_offsets = (
                tokens[:, None] * feature_count
                + feature_block * 8 + coordinates
            )
            values = tl.load(
                inputs + input_offsets, mask=token_mask[:, None], other=0.0
            ).to(tl.float32)
            accumulator += tl.sum(
                values[:, None, :] * weights[None, :, :], axis=2
            )

        if has_bias:
            accumulator += tl.load(
                bias + rows, mask=row_mask, other=0.0
            )[None, :]
        output_offsets = tokens[:, None] * output_rows + rows[None, :]
        tl.store(
            output + output_offsets, accumulator,
            mask=token_mask[:, None] & row_mask[None, :],
        )
else:
    triton = None
    _nqrt_v4_fused_linear_kernel = None


@dataclass(frozen=True)
class PreparedNQRTLinear:
    """GPU-resident packed streams and metadata for the fused Linear kernel."""

    record: PackedNestQuantTensor
    code_payload: torch.Tensor
    beta_payload: torch.Tensor
    row_norms: torch.Tensor
    beta_table: torch.Tensor


def prepare_fused_linear(
    record: PackedNestQuantTensor, *, device: torch.device | str = "cuda"
) -> PreparedNQRTLinear:
    """Copy only packed NQRT streams and compact metadata to the target GPU."""
    if not isinstance(record, PackedNestQuantTensor):
        raise TypeError("record must be a PackedNestQuantTensor")
    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("fused Triton Linear preparation requires CUDA")
    if record.dither_seed is None or record.dither_algorithm != STATELESS_DITHER_ALGORITHM:
        raise ValueError("fused Triton Linear requires NQRT v4 stateless dither")
    if record.codec_semantics != UPSTREAM_CODEC_SEMANTICS:
        raise ValueError("fused Triton Linear requires upstream E8 semantics")
    return PreparedNQRTLinear(
        record=record,
        code_payload=_payload_tensor(record.indices.code_payload, target),
        beta_payload=_payload_tensor(record.indices.beta_payload, target),
        row_norms=record.row_norms.to(device=target, dtype=torch.float32),
        beta_table=torch.tensor(record.betas, device=target, dtype=torch.float32),
    )


def linear_fused_triton(
    inputs: torch.Tensor,
    record: PackedNestQuantTensor | PreparedNQRTLinear,
    bias: torch.Tensor | None = None,
    *,
    block_rows: int = 16,
    block_tokens: int | None = None,
) -> torch.Tensor:
    """Apply a packed NQRT v4 Linear without materializing its full weight."""
    if not isinstance(inputs, torch.Tensor) or not inputs.is_floating_point():
        raise TypeError("inputs must be a floating-point tensor")
    if inputs.device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("fused Triton Linear requires CUDA inputs")
    if not inputs.is_contiguous():
        inputs = inputs.contiguous()
    prepared = (
        record if isinstance(record, PreparedNQRTLinear)
        else prepare_fused_linear(record, device=inputs.device)
    )
    record = prepared.record
    if record.dither_seed is None or record.dither_algorithm != STATELESS_DITHER_ALGORITHM:
        raise ValueError("fused Triton Linear requires NQRT v4 stateless dither")
    if record.codec_semantics != UPSTREAM_CODEC_SEMANTICS:
        raise ValueError("fused Triton Linear requires upstream E8 semantics")
    if inputs.ndim < 1 or inputs.shape[-1] != record.feature_count:
        raise ValueError("input feature dimension does not match the packed weight")
    if inputs.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError("inputs must use float16, bfloat16, or float32")
    if block_rows not in {4, 8, 16, 32}:
        raise ValueError("block_rows must be one of 4, 8, 16, or 32")
    rows, blocks_per_row, _ = record.indices.shape
    if bias is not None:
        if bias.device != inputs.device or tuple(bias.shape) != (rows,):
            raise ValueError("bias must be on the input device with shape [out_features]")
        if not bias.is_floating_point():
            raise TypeError("bias must be floating point")
    if not _TRITON_INSTALLED:
        raise RuntimeError("Triton is not installed")

    assert triton is not None and _nqrt_v4_fused_linear_kernel is not None
    token_count = math.prod(inputs.shape[:-1])
    if block_tokens is None:
        block_tokens = min(16, 1 << max(0, token_count - 1).bit_length())
    if block_tokens not in {1, 2, 4, 8, 16}:
        raise ValueError("block_tokens must be one of 1, 2, 4, 8, 16, or None")
    flat_inputs = inputs.reshape(token_count, record.feature_count)
    device = inputs.device
    if prepared.code_payload.device != device:
        raise ValueError("prepared packed state must be on the input device")
    output = torch.empty((token_count, rows), device=device, dtype=inputs.dtype)
    bias_pointer = output if bias is None else bias
    _nqrt_v4_fused_linear_kernel[
        (triton.cdiv(token_count, block_tokens), triton.cdiv(rows, block_rows))
    ](
        flat_inputs, prepared.code_payload, prepared.beta_payload,
        prepared.row_norms, prepared.beta_table,
        bias_pointer, output,
        token_count=token_count,
        output_rows=rows,
        feature_count=record.feature_count,
        blocks_per_row=blocks_per_row,
        sqrt_feature=record.feature_count ** 0.5,
        q=record.indices.q,
        seed=record.dither_seed,
        code_width=record.indices.code_bit_width,
        beta_width=record.indices.beta_index_bit_width,
        has_bias=bias is not None,
        block_rows=block_rows,
        block_tokens=block_tokens,
        num_warps=4,
    )
    return output.reshape(*inputs.shape[:-1], rows)
