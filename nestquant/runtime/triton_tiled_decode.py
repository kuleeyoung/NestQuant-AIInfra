"""Fused tiled NQRT v4 decode without expanded integer/dither tensors."""

from __future__ import annotations

import torch

from ..core.codec import STATELESS_DITHER_ALGORITHM, UPSTREAM_CODEC_SEMANTICS
from ..core.tensor_checkpoint import PackedNestQuantTensor, _DTYPE_TO_NAME
from .triton_unpack import _TRITON_INSTALLED, _payload_tensor


if _TRITON_INSTALLED:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def _nearest_even_sum_integer(x, coordinates):
        rounded = libdevice.nearbyint(x)
        odd = (tl.sum(rounded, axis=1).to(tl.int64) & 1) != 0
        residual = x - rounded
        selected = tl.argmax(tl.abs(residual), axis=1, tie_break_left=True)
        selected_residual = tl.sum(
            tl.where(coordinates == selected[:, None], residual, 0.0), axis=1
        )
        direction = tl.where(selected_residual >= 0.0, 1.0, -1.0)
        correction = tl.where(
            coordinates == selected[:, None], direction[:, None], 0.0
        )
        return rounded + correction * odd[:, None]

    @triton.jit
    def _fadd_rn(left, right):
        return tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;", "=f,f,f", [left, right],
            dtype=tl.float32, is_pure=True, pack=1,
        )

    @triton.jit
    def _sum8_in_torch_order(values, coordinates):
        # Match PyTorch CUDA's vectorized contiguous reduction for length 8:
        # four lanes accumulate values spaced by four, then reduce those four
        # accumulators. Explicit RN adds prevent compiler reassociation.
        v0 = tl.sum(tl.where(coordinates == 0, values, 0.0), axis=1)
        v1 = tl.sum(tl.where(coordinates == 1, values, 0.0), axis=1)
        v2 = tl.sum(tl.where(coordinates == 2, values, 0.0), axis=1)
        v3 = tl.sum(tl.where(coordinates == 3, values, 0.0), axis=1)
        v4 = tl.sum(tl.where(coordinates == 4, values, 0.0), axis=1)
        v5 = tl.sum(tl.where(coordinates == 5, values, 0.0), axis=1)
        v6 = tl.sum(tl.where(coordinates == 6, values, 0.0), axis=1)
        v7 = tl.sum(tl.where(coordinates == 7, values, 0.0), axis=1)
        left = _fadd_rn(_fadd_rn(v0, v4), _fadd_rn(v2, v6))
        right = _fadd_rn(_fadd_rn(v1, v5), _fadd_rn(v3, v7))
        return _fadd_rn(left, right)

    @triton.jit
    def _nearest_e8_upstream(x, coordinates):
        integer = _nearest_even_sum_integer(x, coordinates)
        half_integer = _nearest_even_sum_integer(x - 0.5, coordinates) + 0.5
        integer_delta = x - integer
        half_delta = x - half_integer
        integer_error = _sum8_in_torch_order(
            integer_delta * integer_delta, coordinates
        )
        half_error = _sum8_in_torch_order(half_delta * half_delta, coordinates)
        # Upstream policy chooses the half-integer coset on exact ties.
        return tl.where((integer_error < half_error)[:, None], integer, half_integer)

    @triton.jit
    def _nqrt_v4_tiled_decode_kernel(
        code_payload,
        beta_payload,
        row_norms,
        beta_table,
        output,
        block_count: tl.constexpr,
        blocks_per_row: tl.constexpr,
        feature_count: tl.constexpr,
        sqrt_feature: tl.constexpr,
        q: tl.constexpr,
        seed,
        code_width: tl.constexpr,
        beta_width: tl.constexpr,
        tile_blocks: tl.constexpr,
    ):
        local_blocks = tl.arange(0, tile_blocks)[:, None]
        coordinates = tl.arange(0, 8)[None, :]
        blocks = tl.program_id(0) * tile_blocks + local_blocks
        block_mask = blocks < block_count

        code_offsets = blocks * 8 + coordinates
        codes_per_byte: tl.constexpr = 8 // code_width
        packed_codes = tl.load(
            code_payload + code_offsets // codes_per_byte,
            mask=block_mask,
            other=0,
        )
        code_shifts = (code_offsets % codes_per_byte) * code_width
        codes = (packed_codes >> code_shifts) & ((1 << code_width) - 1)
        codes = codes.to(tl.float32)

        betas_per_byte: tl.constexpr = 8 // beta_width
        packed_betas = tl.load(
            beta_payload + blocks // betas_per_byte,
            mask=block_mask,
            other=0,
        )
        beta_shifts = (blocks % betas_per_byte) * beta_width
        beta_ids = (packed_betas >> beta_shifts) & ((1 << beta_width) - 1)
        beta = tl.load(beta_table + beta_ids, mask=block_mask, other=0.0)

        c7 = tl.sum(tl.where(coordinates == 7, codes, 0.0), axis=1)[:, None]
        next_offsets = blocks * 8 + tl.minimum(coordinates + 1, 7)
        packed_next = tl.load(
            code_payload + next_offsets // codes_per_byte,
            mask=block_mask,
            other=0,
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

        rows = blocks // blocks_per_row
        row_norm = tl.load(row_norms + rows, mask=block_mask, other=0.0)
        normalized = reconstructed * beta
        scale = row_norm / sqrt_feature
        output_offsets = blocks * 8 + coordinates
        tl.store(output + output_offsets, normalized * scale, mask=block_mask)

    @triton.jit
    def _stateless_dither_kernel(
        output,
        block_offset,
        block_count: tl.constexpr,
        seed,
        tile_blocks: tl.constexpr,
    ):
        local_blocks = tl.arange(0, tile_blocks)[:, None]
        coordinates = tl.arange(0, 8)[None, :]
        blocks = tl.program_id(0) * tile_blocks + local_blocks
        mask = blocks < block_count
        counters = (blocks + block_offset) * 8 + coordinates
        counters = counters.to(tl.int64)
        random = (counters + seed) & 0x7FFFFFFF
        random = random ^ (random >> 16)
        random = (random * 1_103_515_245 + 12_345) & 0x7FFFFFFF
        random = random ^ (random >> 13)
        random = (random * 134_775_813 + 1) & 0x7FFFFFFF
        random = random ^ (random >> 16)
        uniform = random.to(tl.float32) * (2.0 / 2_147_483_648.0)
        dither = uniform - _nearest_e8_upstream(uniform, coordinates)
        tl.store(output + blocks * 8 + coordinates, dither, mask=mask)
else:
    triton = None
    _nqrt_v4_tiled_decode_kernel = None
    _stateless_dither_kernel = None


def _generate_stateless_dither_triton(
    count: int, seed: int, *, block_offset: int = 0,
    device: torch.device | str = "cuda", tile_blocks: int = 64,
) -> torch.Tensor:
    """Internal cross-check path for the dither arithmetic inside the kernel."""
    if not _TRITON_INSTALLED or not torch.cuda.is_available():
        raise RuntimeError("Triton CUDA runtime is unavailable")
    target = torch.device(device)
    output = torch.empty((count, 8), device=target, dtype=torch.float32)
    assert triton is not None and _stateless_dither_kernel is not None
    _stateless_dither_kernel[(triton.cdiv(count, tile_blocks),)](
        output, block_offset=block_offset, block_count=count, seed=seed,
        tile_blocks=tile_blocks, num_warps=4,
    )
    return output


def dequantize_tiled_triton(
    record: PackedNestQuantTensor,
    *,
    device: torch.device | str = "cuda",
    output_dtype: torch.dtype | None = None,
    tile_blocks: int = 64,
) -> torch.Tensor:
    """Decode one NQRT v4 record without full expanded intermediate tensors."""
    if not isinstance(record, PackedNestQuantTensor):
        raise TypeError("record must be a PackedNestQuantTensor")
    target_device = torch.device(device)
    if target_device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Triton tiled decode requires an available CUDA device")
    if not _TRITON_INSTALLED:
        raise RuntimeError("Triton is not installed")
    if record.dither_seed is None or record.dither_algorithm != STATELESS_DITHER_ALGORITHM:
        raise ValueError("Triton tiled decode requires NQRT v4 stateless dither")
    if record.codec_semantics != UPSTREAM_CODEC_SEMANTICS:
        raise ValueError("Triton tiled decode requires upstream E8 tie semantics")
    if tile_blocks not in {16, 32, 64, 128}:
        raise ValueError("tile_blocks must be one of 16, 32, 64, or 128")
    target_dtype = record.output_dtype if output_dtype is None else output_dtype
    if target_dtype not in _DTYPE_TO_NAME:
        raise ValueError("output_dtype must be float16, bfloat16, float32, or float64")
    if target_dtype == torch.float64:
        raise ValueError("Triton tiled decode does not support float64 output")

    assert triton is not None and _nqrt_v4_tiled_decode_kernel is not None
    rows, blocks_per_row, _ = record.indices.shape
    block_count = rows * blocks_per_row
    code_payload = _payload_tensor(record.indices.code_payload, target_device)
    beta_payload = _payload_tensor(record.indices.beta_payload, target_device)
    row_norms = record.row_norms.to(device=target_device, dtype=torch.float32)
    beta_table = torch.tensor(record.betas, device=target_device, dtype=torch.float32)
    output = torch.empty(
        (rows, record.feature_count), device=target_device, dtype=target_dtype
    )
    _nqrt_v4_tiled_decode_kernel[(triton.cdiv(block_count, tile_blocks),)](
        code_payload,
        beta_payload,
        row_norms,
        beta_table,
        output,
        block_count=block_count,
        blocks_per_row=blocks_per_row,
        feature_count=record.feature_count,
        sqrt_feature=record.feature_count ** 0.5,
        q=record.indices.q,
        seed=record.dither_seed,
        code_width=record.indices.code_bit_width,
        beta_width=record.indices.beta_index_bit_width,
        tile_blocks=tile_blocks,
        num_warps=4,
    )
    return output
