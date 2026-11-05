"""Correctness-first packed NQRT4-weight by packed NQA4-activation Linear."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence

import torch

from .nqkv_page import _pack_unsigned_cuda, _quantize_rows_cuda_fused
from .triton_fused_linear import PreparedNQRTLinear
from .triton_tiled_decode import _nearest_e8_upstream
from .triton_unpack import _TRITON_INSTALLED


if _TRITON_INSTALLED:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def _decode_packed_a4_diagnostic_kernel(
        activation_codes, activation_betas, activation_norms,
        activation_beta_table, output,
        block_count: tl.constexpr, blocks_per_row: tl.constexpr,
        sqrt_feature: tl.constexpr, activation_q: tl.constexpr,
        activation_code_width: tl.constexpr,
        activation_beta_width: tl.constexpr,
    ):
        """Decode one E8 activation block per program using production math."""
        block = tl.program_id(0)
        coordinates = tl.arange(0, 8)[None, :]
        codes_per_byte: tl.constexpr = 8 // activation_code_width
        betas_per_byte: tl.constexpr = 8 // activation_beta_width
        offsets = block * 8 + coordinates
        packed = tl.load(activation_codes + offsets // codes_per_byte)
        shifts = (offsets % codes_per_byte) * activation_code_width
        code = (
            (packed >> shifts) & ((1 << activation_code_width) - 1)
        ).to(tl.float32)
        next_offsets = block * 8 + tl.minimum(coordinates + 1, 7)
        packed_next = tl.load(activation_codes + next_offsets // codes_per_byte)
        next_shifts = (next_offsets % codes_per_byte) * activation_code_width
        code_next = (
            (packed_next >> next_shifts) & ((1 << activation_code_width) - 1)
        ).to(tl.float32)
        c7 = tl.sum(tl.where(coordinates == 7, code, 0.0), axis=1)[:, None]
        coset = tl.where(
            coordinates == 0,
            2.0 * code - code_next + 0.5 * c7,
            tl.where(
                coordinates < 6,
                code - code_next + 0.5 * c7,
                tl.where(coordinates == 6, code + 0.5 * c7, 0.5 * c7),
            ),
        )
        coarse = _nearest_e8_upstream(
            coset / activation_q, tl.arange(0, 8)[None, :]
        )
        reconstructed = coset - activation_q * coarse
        packed_beta = tl.load(activation_betas + block // betas_per_byte)
        beta_shift = (block % betas_per_byte) * activation_beta_width
        beta_id = (packed_beta >> beta_shift) & ((1 << activation_beta_width) - 1)
        beta = tl.load(activation_beta_table + beta_id)
        row = block // blocks_per_row
        norm = tl.load(activation_norms + row)
        values = reconstructed * beta * (norm / sqrt_feature)
        tl.store(output + offsets, values)

    @triton.jit
    def _inspect_packed_a4_diagnostic_kernel(
        activation_codes, output,
        activation_q: tl.constexpr, activation_code_width: tl.constexpr,
    ):
        block = tl.program_id(0)
        coordinates = tl.arange(0, 8)[None, :]
        codes_per_byte: tl.constexpr = 8 // activation_code_width
        offsets = block * 8 + coordinates
        packed = tl.load(activation_codes + offsets // codes_per_byte)
        shifts = (offsets % codes_per_byte) * activation_code_width
        code = ((packed >> shifts) & ((1 << activation_code_width) - 1)).to(tl.float32)
        next_offsets = block * 8 + tl.minimum(coordinates + 1, 7)
        packed_next = tl.load(activation_codes + next_offsets // codes_per_byte)
        next_shifts = (next_offsets % codes_per_byte) * activation_code_width
        code_next = ((packed_next >> next_shifts) & ((1 << activation_code_width) - 1)).to(tl.float32)
        c7 = tl.sum(tl.where(coordinates == 7, code, 0.0), axis=1)[:, None]
        coset = tl.where(
            coordinates == 0, 2.0 * code - code_next + 0.5 * c7,
            tl.where(coordinates < 6, code - code_next + 0.5 * c7,
                     tl.where(coordinates == 6, code + 0.5 * c7, 0.5 * c7)),
        )
        x = coset / activation_q
        integer = libdevice.nearbyint(x)
        half_base = libdevice.nearbyint(x - 0.5)
        half_integer = half_base + 0.5
        integer_residual = tl.abs(x - integer)
        half_residual = tl.abs((x - 0.5) - half_base)
        integer_index = tl.argmax(integer_residual, axis=1, tie_break_left=True)
        half_index = tl.argmax(half_residual, axis=1, tie_break_left=True)
        integer_parity = (tl.sum(integer, axis=1).to(tl.int64) & 1) != 0
        half_parity = (tl.sum(half_base, axis=1).to(tl.int64) & 1) != 0
        base = block * 26
        tl.store(output + base + coordinates, x)
        tl.store(output + base + 8 + coordinates, integer_residual)
        tl.store(output + base + 16 + coordinates, half_residual)
        scalar_offset = tl.arange(0, 1)
        tl.store(
            output + base + 24 + scalar_offset,
            integer_index + integer_parity * 16,
        )
        tl.store(
            output + base + 25 + scalar_offset,
            half_index + half_parity * 16,
        )

    @triton.jit
    def _packed_w4a4_tiled_kernel(
        activation_codes, activation_betas, activation_norms, activation_beta_table,
        weight_codes, weight_betas, weight_norms, weight_beta_table,
        bias, output,
        token_count: tl.constexpr, output_rows: tl.constexpr,
        feature_count: tl.constexpr, blocks_per_row: tl.constexpr,
        sqrt_feature: tl.constexpr, activation_q: tl.constexpr,
        weight_q: tl.constexpr, weight_seed,
        activation_code_width: tl.constexpr, activation_beta_width: tl.constexpr,
        weight_code_width: tl.constexpr, weight_beta_width: tl.constexpr,
        has_bias: tl.constexpr, block_m: tl.constexpr,
        block_n: tl.constexpr, block_k: tl.constexpr,
    ):
        tokens = tl.program_id(0) * block_m + tl.arange(0, block_m)
        rows = tl.program_id(1) * block_n + tl.arange(0, block_n)
        token_mask = tokens < token_count
        row_mask = rows < output_rows
        accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
        act_codes_per_byte: tl.constexpr = 8 // activation_code_width
        act_betas_per_byte: tl.constexpr = 8 // activation_beta_width
        weight_codes_per_byte: tl.constexpr = 8 // weight_code_width
        weight_betas_per_byte: tl.constexpr = 8 // weight_beta_width

        for feature_start in tl.range(0, feature_count, block_k, num_stages=1):
            local_blocks = tl.arange(0, block_k // 8)
            coordinates = tl.arange(0, 8)[None, None, :]
            feature_blocks = feature_start // 8 + local_blocks

            act_block_ids = (
                tokens[:, None] * blocks_per_row + feature_blocks[None, :]
            )
            act_offsets = act_block_ids[:, :, None] * 8 + coordinates
            packed_act = tl.load(
                activation_codes + act_offsets // act_codes_per_byte,
                mask=token_mask[:, None, None], other=0,
            )
            act_shifts = (act_offsets % act_codes_per_byte) * activation_code_width
            act_code = (
                (packed_act >> act_shifts) & ((1 << activation_code_width) - 1)
            ).to(tl.float32)
            act_next_offsets = (
                act_block_ids[:, :, None] * 8 + tl.minimum(coordinates + 1, 7)
            )
            packed_act_next = tl.load(
                activation_codes + act_next_offsets // act_codes_per_byte,
                mask=token_mask[:, None, None], other=0,
            )
            act_next_shifts = (
                act_next_offsets % act_codes_per_byte
            ) * activation_code_width
            act_next = (
                (packed_act_next >> act_next_shifts)
                & ((1 << activation_code_width) - 1)
            ).to(tl.float32)
            act_c7 = tl.sum(
                tl.where(coordinates == 7, act_code, 0.0), axis=2
            )[:, :, None]
            act_coset = tl.where(
                coordinates == 0,
                2.0 * act_code - act_next + 0.5 * act_c7,
                tl.where(
                    coordinates < 6,
                    act_code - act_next + 0.5 * act_c7,
                    tl.where(coordinates == 6, act_code + 0.5 * act_c7, 0.5 * act_c7),
                ),
            )
            flat_coordinates = tl.arange(0, 8)[None, :]
            act_coarse = tl.reshape(
                _nearest_e8_upstream(
                    tl.reshape(
                        act_coset, (block_m * (block_k // 8), 8)
                    ) / activation_q,
                    flat_coordinates,
                ),
                (block_m, block_k // 8, 8),
            )
            act_reconstructed = act_coset - activation_q * act_coarse
            packed_act_beta = tl.load(
                activation_betas + act_block_ids // act_betas_per_byte,
                mask=token_mask[:, None], other=0,
            )
            act_beta_shift = (
                act_block_ids % act_betas_per_byte
            ) * activation_beta_width
            act_beta_ids = (
                packed_act_beta >> act_beta_shift
            ) & ((1 << activation_beta_width) - 1)
            act_beta = tl.load(
                activation_beta_table + act_beta_ids,
                mask=token_mask[:, None], other=0.0,
            )
            act_norm = tl.load(
                activation_norms + tokens, mask=token_mask, other=0.0
            )
            act_values = act_reconstructed * act_beta[:, :, None]
            act_values *= (act_norm / sqrt_feature)[:, None, None]
            act_values = tl.reshape(act_values, (block_m, block_k))

            weight_block_ids = (
                rows[:, None] * blocks_per_row + feature_blocks[None, :]
            )
            weight_offsets = weight_block_ids[:, :, None] * 8 + coordinates
            packed_weight = tl.load(
                weight_codes + weight_offsets // weight_codes_per_byte,
                mask=row_mask[:, None, None], other=0,
            )
            weight_shifts = (
                weight_offsets % weight_codes_per_byte
            ) * weight_code_width
            weight_code = (
                (packed_weight >> weight_shifts) & ((1 << weight_code_width) - 1)
            ).to(tl.float32)
            weight_next_offsets = (
                weight_block_ids[:, :, None] * 8 + tl.minimum(coordinates + 1, 7)
            )
            packed_weight_next = tl.load(
                weight_codes + weight_next_offsets // weight_codes_per_byte,
                mask=row_mask[:, None, None], other=0,
            )
            weight_next_shifts = (
                weight_next_offsets % weight_codes_per_byte
            ) * weight_code_width
            weight_next = (
                (packed_weight_next >> weight_next_shifts)
                & ((1 << weight_code_width) - 1)
            ).to(tl.float32)
            weight_c7 = tl.sum(
                tl.where(coordinates == 7, weight_code, 0.0), axis=2
            )[:, :, None]
            weight_coset = tl.where(
                coordinates == 0,
                2.0 * weight_code - weight_next + 0.5 * weight_c7,
                tl.where(
                    coordinates < 6,
                    weight_code - weight_next + 0.5 * weight_c7,
                    tl.where(
                        coordinates == 6,
                        weight_code + 0.5 * weight_c7,
                        0.5 * weight_c7,
                    ),
                ),
            )
            counters = weight_offsets.to(tl.int64)
            random = (counters + weight_seed) & 0x7FFFFFFF
            random = random ^ (random >> 16)
            random = (random * 1_103_515_245 + 12_345) & 0x7FFFFFFF
            random = random ^ (random >> 13)
            random = (random * 134_775_813 + 1) & 0x7FFFFFFF
            random = random ^ (random >> 16)
            uniform = random.to(tl.float32) * (2.0 / 2_147_483_648.0)
            flat_uniform = tl.reshape(
                uniform, (block_n * (block_k // 8), 8)
            )
            dither = uniform - tl.reshape(
                _nearest_e8_upstream(flat_uniform, flat_coordinates),
                (block_n, block_k // 8, 8),
            )
            shifted_weight = weight_coset - dither
            weight_coarse = tl.reshape(
                _nearest_e8_upstream(
                    tl.reshape(
                        shifted_weight, (block_n * (block_k // 8), 8)
                    ) / weight_q,
                    flat_coordinates,
                ),
                (block_n, block_k // 8, 8),
            )
            weight_reconstructed = shifted_weight - weight_q * weight_coarse
            packed_weight_beta = tl.load(
                weight_betas + weight_block_ids // weight_betas_per_byte,
                mask=row_mask[:, None], other=0,
            )
            weight_beta_shift = (
                weight_block_ids % weight_betas_per_byte
            ) * weight_beta_width
            weight_beta_ids = (
                packed_weight_beta >> weight_beta_shift
            ) & ((1 << weight_beta_width) - 1)
            weight_beta = tl.load(
                weight_beta_table + weight_beta_ids,
                mask=row_mask[:, None], other=0.0,
            )
            weight_norm = tl.load(weight_norms + rows, mask=row_mask, other=0.0)
            weight_values = weight_reconstructed * weight_beta[:, :, None]
            weight_values *= (weight_norm / sqrt_feature)[:, None, None]
            weight_values = tl.reshape(weight_values, (block_n, block_k))
            accumulator = tl.dot(
                act_values.to(tl.bfloat16),
                tl.trans(weight_values.to(tl.bfloat16)),
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

    @triton.jit
    def _packed_w4a4_linear_kernel(
        activation_codes, activation_betas, activation_norms, activation_beta_table,
        weight_codes, weight_betas, weight_norms, weight_beta_table,
        bias, output,
        token_count: tl.constexpr, output_rows: tl.constexpr,
        feature_count: tl.constexpr, blocks_per_row: tl.constexpr,
        sqrt_feature: tl.constexpr, activation_q: tl.constexpr,
        weight_q: tl.constexpr, weight_seed,
        activation_code_width: tl.constexpr, activation_beta_width: tl.constexpr,
        weight_code_width: tl.constexpr, weight_beta_width: tl.constexpr,
        has_bias: tl.constexpr,
    ):
        token = tl.program_id(0)
        row = tl.program_id(1)
        coordinates = tl.arange(0, 8)[None, :]
        activation_codes_per_byte: tl.constexpr = 8 // activation_code_width
        activation_betas_per_byte: tl.constexpr = 8 // activation_beta_width
        weight_codes_per_byte: tl.constexpr = 8 // weight_code_width
        weight_betas_per_byte: tl.constexpr = 8 // weight_beta_width
        accumulator = tl.zeros((1,), dtype=tl.float32)

        for feature_block in tl.range(0, blocks_per_row, num_stages=1):
            activation_block = token * blocks_per_row + feature_block
            activation_offsets = activation_block * 8 + coordinates
            packed_activation = tl.load(
                activation_codes + activation_offsets // activation_codes_per_byte
            )
            activation_shifts = (
                activation_offsets % activation_codes_per_byte
            ) * activation_code_width
            activation_code = (
                (packed_activation >> activation_shifts)
                & ((1 << activation_code_width) - 1)
            ).to(tl.float32)
            activation_next_offsets = (
                activation_block * 8 + tl.minimum(coordinates + 1, 7)
            )
            packed_activation_next = tl.load(
                activation_codes
                + activation_next_offsets // activation_codes_per_byte
            )
            activation_next_shifts = (
                activation_next_offsets % activation_codes_per_byte
            ) * activation_code_width
            activation_next = (
                (packed_activation_next >> activation_next_shifts)
                & ((1 << activation_code_width) - 1)
            ).to(tl.float32)
            activation_c7 = tl.sum(
                tl.where(coordinates == 7, activation_code, 0.0), axis=1
            )[:, None]
            activation_coset = tl.where(
                coordinates == 0,
                2.0 * activation_code - activation_next + 0.5 * activation_c7,
                tl.where(
                    coordinates < 6,
                    activation_code - activation_next + 0.5 * activation_c7,
                    tl.where(
                        coordinates == 6,
                        activation_code + 0.5 * activation_c7,
                        0.5 * activation_c7,
                    ),
                ),
            )
            activation_coarse = _nearest_e8_upstream(
                activation_coset / activation_q, coordinates
            )
            activation_reconstructed = (
                activation_coset - activation_q * activation_coarse
            )
            packed_activation_beta = tl.load(
                activation_betas
                + activation_block // activation_betas_per_byte
            )
            activation_beta_shift = (
                activation_block % activation_betas_per_byte
            ) * activation_beta_width
            activation_beta_id = (
                packed_activation_beta >> activation_beta_shift
            ) & ((1 << activation_beta_width) - 1)
            activation_beta = tl.load(
                activation_beta_table + activation_beta_id
            )
            activation_norm = tl.load(activation_norms + token)
            activation_values = activation_reconstructed * activation_beta
            activation_values *= activation_norm / sqrt_feature

            weight_block = row * blocks_per_row + feature_block
            weight_offsets = weight_block * 8 + coordinates
            packed_weight = tl.load(
                weight_codes + weight_offsets // weight_codes_per_byte
            )
            weight_shifts = (
                weight_offsets % weight_codes_per_byte
            ) * weight_code_width
            weight_code = (
                (packed_weight >> weight_shifts)
                & ((1 << weight_code_width) - 1)
            ).to(tl.float32)
            weight_next_offsets = weight_block * 8 + tl.minimum(coordinates + 1, 7)
            packed_weight_next = tl.load(
                weight_codes + weight_next_offsets // weight_codes_per_byte
            )
            weight_next_shifts = (
                weight_next_offsets % weight_codes_per_byte
            ) * weight_code_width
            weight_next = (
                (packed_weight_next >> weight_next_shifts)
                & ((1 << weight_code_width) - 1)
            ).to(tl.float32)
            weight_c7 = tl.sum(
                tl.where(coordinates == 7, weight_code, 0.0), axis=1
            )[:, None]
            weight_coset = tl.where(
                coordinates == 0,
                2.0 * weight_code - weight_next + 0.5 * weight_c7,
                tl.where(
                    coordinates < 6,
                    weight_code - weight_next + 0.5 * weight_c7,
                    tl.where(
                        coordinates == 6,
                        weight_code + 0.5 * weight_c7,
                        0.5 * weight_c7,
                    ),
                ),
            )
            counters = weight_offsets.to(tl.int64)
            random = (counters + weight_seed) & 0x7FFFFFFF
            random = random ^ (random >> 16)
            random = (random * 1_103_515_245 + 12_345) & 0x7FFFFFFF
            random = random ^ (random >> 13)
            random = (random * 134_775_813 + 1) & 0x7FFFFFFF
            random = random ^ (random >> 16)
            uniform = random.to(tl.float32) * (2.0 / 2_147_483_648.0)
            dither = uniform - _nearest_e8_upstream(uniform, coordinates)
            shifted_weight = weight_coset - dither
            weight_coarse = _nearest_e8_upstream(
                shifted_weight / weight_q, coordinates
            )
            weight_reconstructed = shifted_weight - weight_q * weight_coarse
            packed_weight_beta = tl.load(
                weight_betas + weight_block // weight_betas_per_byte
            )
            weight_beta_shift = (
                weight_block % weight_betas_per_byte
            ) * weight_beta_width
            weight_beta_id = (
                packed_weight_beta >> weight_beta_shift
            ) & ((1 << weight_beta_width) - 1)
            weight_beta = tl.load(weight_beta_table + weight_beta_id)
            weight_norm = tl.load(weight_norms + row)
            weight_values = weight_reconstructed * weight_beta
            weight_values *= weight_norm / sqrt_feature
            accumulator += tl.sum(activation_values * weight_values, axis=1)

        if has_bias:
            accumulator += tl.load(bias + row)
        tl.store(
            output + token * output_rows + row,
            tl.sum(accumulator, axis=0),
        )
else:
    triton = None
    _packed_w4a4_linear_kernel = None
    _packed_w4a4_tiled_kernel = None
    _decode_packed_a4_diagnostic_kernel = None
    _inspect_packed_a4_diagnostic_kernel = None


@dataclass(frozen=True)
class PreparedNQA4Activation:
    """GPU-resident packed activation rows consumed without BF16 expansion."""

    code_payload: torch.Tensor
    beta_payload: torch.Tensor
    row_norms: torch.Tensor
    beta_table: torch.Tensor
    original_shape: tuple[int, ...]
    q: int
    code_width: int
    beta_width: int
    output_dtype: torch.dtype

    @property
    def token_count(self) -> int:
        return math.prod(self.original_shape[:-1])

    @property
    def feature_count(self) -> int:
        return self.original_shape[-1]

    @property
    def resident_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self.code_payload, self.beta_payload,
                self.row_norms, self.beta_table,
            )
        )


def decode_packed_a4_triton_diagnostic(
    activation: PreparedNQA4Activation,
) -> torch.Tensor:
    """Decode packed A4 with the exact production Triton E8 arithmetic.

    This is intentionally a correctness diagnostic, not a runtime fallback.
    """
    if not isinstance(activation, PreparedNQA4Activation):
        raise TypeError("activation must be PreparedNQA4Activation")
    if not _TRITON_INSTALLED or _decode_packed_a4_diagnostic_kernel is None:
        raise RuntimeError("Triton CUDA runtime is unavailable")
    assert triton is not None
    blocks_per_row = activation.feature_count // 8
    block_count = activation.token_count * blocks_per_row
    output = torch.empty(
        (activation.token_count, activation.feature_count),
        device=activation.code_payload.device,
        dtype=activation.output_dtype,
    )
    _decode_packed_a4_diagnostic_kernel[(block_count,)](
        activation.code_payload,
        activation.beta_payload,
        activation.row_norms,
        activation.beta_table,
        output,
        block_count=block_count,
        blocks_per_row=blocks_per_row,
        sqrt_feature=math.sqrt(activation.feature_count),
        activation_q=activation.q,
        activation_code_width=activation.code_width,
        activation_beta_width=activation.beta_width,
        num_warps=1,
    )
    return output.reshape(*activation.original_shape)


def inspect_packed_a4_triton_diagnostic(
    activation: PreparedNQA4Activation,
) -> torch.Tensor:
    """Return per-block Triton E8 intermediate values for diagnostics."""
    if not isinstance(activation, PreparedNQA4Activation):
        raise TypeError("activation must be PreparedNQA4Activation")
    if not _TRITON_INSTALLED or _inspect_packed_a4_diagnostic_kernel is None:
        raise RuntimeError("Triton CUDA runtime is unavailable")
    assert triton is not None
    block_count = activation.token_count * (activation.feature_count // 8)
    output = torch.empty(
        (block_count, 26), device=activation.code_payload.device,
        dtype=torch.float32,
    )
    _inspect_packed_a4_diagnostic_kernel[(block_count,)](
        activation.code_payload, output,
        activation_q=activation.q,
        activation_code_width=activation.code_width,
        num_warps=1,
    )
    return output


def quantize_activation_cuda(
    inputs: torch.Tensor, *, q: int, betas: Sequence[float],
    beta_table: torch.Tensor | None = None,
) -> PreparedNQA4Activation:
    """Encode floating activations into actual packed CUDA payloads."""
    if not isinstance(inputs, torch.Tensor) or not inputs.is_floating_point():
        raise TypeError("inputs must be a floating-point tensor")
    if inputs.device.type != "cuda" or inputs.ndim < 1 or inputs.numel() == 0:
        raise ValueError("inputs must be a nonempty CUDA tensor")
    if inputs.shape[-1] % 8:
        raise ValueError("activation feature dimension must be divisible by 8")
    beta_values = tuple(float(value) for value in betas)
    if not beta_values or any(not math.isfinite(value) or value <= 0 for value in beta_values):
        raise ValueError("betas must be positive finite values")
    code_width = max(1, math.ceil(math.log2(q)))
    beta_width = max(1, math.ceil(math.log2(len(beta_values))))
    if code_width not in {1, 2, 4, 8} or beta_width not in {1, 2, 4, 8}:
        raise ValueError("packed activation fields require 1, 2, 4, or 8 bits")
    rows = inputs.reshape(-1, inputs.shape[-1]).float().contiguous()
    codes, beta_indices, row_norms = _quantize_rows_cuda_fused(
        rows, q=q, betas=beta_values, beta_table=beta_table
    )
    if beta_table is None:
        beta_table = torch.tensor(
            beta_values, device=inputs.device, dtype=torch.float32
        )
    return PreparedNQA4Activation(
        code_payload=_pack_unsigned_cuda(codes, code_width),
        beta_payload=_pack_unsigned_cuda(beta_indices, beta_width),
        row_norms=row_norms,
        beta_table=beta_table,
        original_shape=tuple(inputs.shape),
        q=q,
        code_width=code_width,
        beta_width=beta_width,
        output_dtype=inputs.dtype,
    )


def linear_packed_w4a4_triton(
    activation: PreparedNQA4Activation,
    weight: PreparedNQRTLinear,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute Linear directly from packed A4 and W4 streams."""
    if not isinstance(activation, PreparedNQA4Activation):
        raise TypeError("activation must be PreparedNQA4Activation")
    if not isinstance(weight, PreparedNQRTLinear):
        raise TypeError("weight must be PreparedNQRTLinear")
    record = weight.record
    output_rows, blocks_per_row, _ = record.indices.shape
    if activation.feature_count != record.feature_count:
        raise ValueError("packed activation and weight feature dimensions differ")
    if activation.code_payload.device != weight.code_payload.device:
        raise ValueError("packed activation and weight must share a CUDA device")
    if bias is not None and (
        bias.device != activation.code_payload.device
        or tuple(bias.shape) != (output_rows,)
    ):
        raise ValueError("bias must match the output rows and CUDA device")
    if not _TRITON_INSTALLED or _packed_w4a4_linear_kernel is None:
        raise RuntimeError("Triton CUDA runtime is unavailable")
    assert triton is not None
    output = torch.empty(
        (activation.token_count, output_rows),
        device=activation.code_payload.device,
        dtype=activation.output_dtype,
    )
    bias_pointer = output if bias is None else bias
    _packed_w4a4_linear_kernel[(activation.token_count, output_rows)](
        activation.code_payload, activation.beta_payload,
        activation.row_norms, activation.beta_table,
        weight.code_payload, weight.beta_payload, weight.row_norms, weight.beta_table,
        bias_pointer, output,
        token_count=activation.token_count,
        output_rows=output_rows,
        feature_count=record.feature_count,
        blocks_per_row=blocks_per_row,
        sqrt_feature=math.sqrt(record.feature_count),
        activation_q=activation.q,
        weight_q=record.indices.q,
        weight_seed=record.dither_seed,
        activation_code_width=activation.code_width,
        activation_beta_width=activation.beta_width,
        weight_code_width=record.indices.code_bit_width,
        weight_beta_width=record.indices.beta_index_bit_width,
        has_bias=bias is not None,
        num_warps=1,
    )
    return output.reshape(*activation.original_shape[:-1], output_rows)


def select_w4a4_tiled_config(
    token_count: int, output_rows: int, feature_count: int
) -> dict[str, int]:
    """Select a launch shape while retaining the validated K accumulation order."""
    if token_count < 1 or output_rows < 1 or feature_count < 1:
        raise ValueError("token_count, output_rows and feature_count must be positive")
    block_m = 16 if token_count <= 16 else (32 if token_count <= 32 else 64)
    if token_count >= 64 or output_rows >= 4096:
        block_n = 32
    elif feature_count >= 8192 or token_count == 1:
        block_n = 16
    else:
        block_n = 32
    return {
        "block_m": block_m,
        "block_n": block_n,
        "block_k": 32,
        "num_warps": 8,
    }


def linear_packed_w4a4_tiled_triton(
    activation: PreparedNQA4Activation,
    weight: PreparedNQRTLinear,
    bias: torch.Tensor | None = None,
    *,
    block_m: int = 16,
    block_n: int = 16,
    block_k: int = 32,
    num_warps: int = 4,
) -> torch.Tensor:
    """Tile tokens/output rows and accumulate direct packed W4xA4 with tl.dot."""
    if not isinstance(activation, PreparedNQA4Activation):
        raise TypeError("activation must be PreparedNQA4Activation")
    if not isinstance(weight, PreparedNQRTLinear):
        raise TypeError("weight must be PreparedNQRTLinear")
    record = weight.record
    output_rows, blocks_per_row, _ = record.indices.shape
    if activation.feature_count != record.feature_count:
        raise ValueError("packed activation and weight feature dimensions differ")
    if activation.code_payload.device != weight.code_payload.device:
        raise ValueError("packed activation and weight must share a CUDA device")
    if bias is not None and (
        bias.device != activation.code_payload.device
        or tuple(bias.shape) != (output_rows,)
    ):
        raise ValueError("bias must match the output rows and CUDA device")
    if block_m not in {16, 32, 64, 128} or block_n not in {16, 32, 64}:
        raise ValueError("block_m must be 16, 32, 64, or 128; block_n must be 16, 32, or 64")
    if block_k not in {16, 32, 64, 128} or record.feature_count % block_k:
        raise ValueError("block_k must divide feature_count and be 16, 32, 64, or 128")
    if num_warps not in {2, 4, 8}:
        raise ValueError("num_warps must be 2, 4, or 8")
    if not _TRITON_INSTALLED or _packed_w4a4_tiled_kernel is None:
        raise RuntimeError("Triton CUDA runtime is unavailable")
    assert triton is not None
    output = torch.empty(
        (activation.token_count, output_rows),
        device=activation.code_payload.device,
        dtype=activation.output_dtype,
    )
    bias_pointer = output if bias is None else bias
    _packed_w4a4_tiled_kernel[
        (triton.cdiv(activation.token_count, block_m), triton.cdiv(output_rows, block_n))
    ](
        activation.code_payload, activation.beta_payload,
        activation.row_norms, activation.beta_table,
        weight.code_payload, weight.beta_payload, weight.row_norms, weight.beta_table,
        bias_pointer, output,
        token_count=activation.token_count,
        output_rows=output_rows,
        feature_count=record.feature_count,
        blocks_per_row=blocks_per_row,
        sqrt_feature=math.sqrt(record.feature_count),
        activation_q=activation.q,
        weight_q=record.indices.q,
        weight_seed=record.dither_seed,
        activation_code_width=activation.code_width,
        activation_beta_width=activation.beta_width,
        weight_code_width=record.indices.code_bit_width,
        weight_beta_width=record.indices.beta_index_bit_width,
        has_bias=bias is not None,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
    )
    return output.reshape(*activation.original_shape[:-1], output_rows)
