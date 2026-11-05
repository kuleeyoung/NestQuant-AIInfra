"""Lazy CUDA Tensor-Core prototype consuming packed NQA4 and NQRT4 streams."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..core.codec import STATELESS_DITHER_ALGORITHM, UPSTREAM_CODEC_SEMANTICS
from .triton_fused_linear import PreparedNQRTLinear
from .triton_w4a4_linear import PreparedNQA4Activation


_EXTENSION: Any | None = None


def _load_extension() -> Any:
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    if not torch.cuda.is_available():
        raise RuntimeError("packed W4A4 CUDA MMA requires CUDA")
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parents[2]
    _EXTENSION = load(
        name="nqrt4_w4a4_mma_ext",
        sources=[
            str(root / "nestquant/runtime/nqrt4_w4a4_mma_binding.cpp"),
            str(root / "nestquant/runtime/nqrt4_w4a4_mma.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "-DNQRT4_PARITY_SEPARATE_REFERENCE",
            "-DNQRT4_MAX_INDEX_SHUFFLE_REFERENCE",
        ],
        verbose=False,
    )
    return _EXTENSION


def can_use_w4a4_mma_m16(
    activation: PreparedNQA4Activation, weight: PreparedNQRTLinear
) -> bool:
    record = weight.record
    tensors = (
        activation.code_payload, activation.beta_payload,
        activation.row_norms, activation.beta_table,
        weight.code_payload, weight.beta_payload,
        weight.row_norms, weight.beta_table,
    )
    return bool(
        activation.token_count == 16
        and activation.feature_count == record.feature_count
        and activation.q == 14
        and activation.code_width == 4
        and activation.beta_width == 2
        and activation.beta_table.numel() == 4
        and record.feature_count % 16 == 0
        and record.indices.shape[0] % 8 == 0
        and record.indices.q == 14
        and record.indices.code_bit_width == 4
        and record.indices.beta_count == 4
        and record.indices.beta_index_bit_width == 2
        and record.dither_seed is not None
        and record.dither_algorithm == STATELESS_DITHER_ALGORITHM
        and record.codec_semantics == UPSTREAM_CODEC_SEMANTICS
        and all(tensor.is_contiguous() for tensor in tensors)
        and all(
            tensor.device == activation.code_payload.device
            for tensor in tensors
        )
    )


def can_use_w4a4_gemv_m1(
    activation: PreparedNQA4Activation, weight: PreparedNQRTLinear
) -> bool:
    if activation.token_count != 1:
        return False
    # The packed formats and device invariants are identical to M16. Reuse the
    # mature predicate by replacing only the shape-specific row requirement.
    record = weight.record
    tensors = (
        activation.code_payload, activation.beta_payload,
        activation.row_norms, activation.beta_table,
        weight.code_payload, weight.beta_payload,
        weight.row_norms, weight.beta_table,
    )
    return bool(
        activation.feature_count == record.feature_count
        and activation.q == 14
        and activation.code_width == 4
        and activation.beta_width == 2
        and activation.beta_table.numel() == 4
        and record.feature_count % 8 == 0
        and record.indices.shape[0] % 8 == 0
        and record.indices.q == 14
        and record.indices.code_bit_width == 4
        and record.indices.beta_count == 4
        and record.indices.beta_index_bit_width == 2
        and record.dither_seed is not None
        and record.dither_algorithm == STATELESS_DITHER_ALGORITHM
        and record.codec_semantics == UPSTREAM_CODEC_SEMANTICS
        and all(tensor.is_contiguous() for tensor in tensors)
        and all(
            tensor.device == activation.code_payload.device
            for tensor in tensors
        )
    )


def can_use_w4a4_mma_small_m(
    activation: PreparedNQA4Activation, weight: PreparedNQRTLinear
) -> bool:
    if activation.token_count not in (2, 4, 8):
        return False
    record = weight.record
    tensors = (
        activation.code_payload, activation.beta_payload,
        activation.row_norms, activation.beta_table,
        weight.code_payload, weight.beta_payload,
        weight.row_norms, weight.beta_table,
    )
    return bool(
        activation.feature_count == record.feature_count
        and activation.q == 14
        and activation.code_width == 4
        and activation.beta_width == 2
        and activation.beta_table.numel() == 4
        and record.feature_count % 16 == 0
        and record.indices.shape[0] % 16 == 0
        and record.indices.q == 14
        and record.indices.code_bit_width == 4
        and record.indices.beta_count == 4
        and record.indices.beta_index_bit_width == 2
        and record.dither_seed is not None
        and record.dither_algorithm == STATELESS_DITHER_ALGORITHM
        and record.codec_semantics == UPSTREAM_CODEC_SEMANTICS
        and all(tensor.is_contiguous() for tensor in tensors)
        and all(
            tensor.device == activation.code_payload.device
            for tensor in tensors
        )
    )


def can_use_w4a4_mma_large_m_staged(
    activation: PreparedNQA4Activation, weight: PreparedNQRTLinear
) -> bool:
    if activation.token_count not in (32, 64, 128):
        return False
    record = weight.record
    tensors = (
        activation.code_payload, activation.beta_payload,
        activation.row_norms, activation.beta_table,
        weight.code_payload, weight.beta_payload,
        weight.row_norms, weight.beta_table,
    )
    return bool(
        activation.feature_count == record.feature_count
        and activation.q == 14
        and activation.code_width == 4
        and activation.beta_width == 2
        and activation.beta_table.numel() == 4
        and record.feature_count % 16 == 0
        and record.indices.shape[0] % 16 == 0
        and record.indices.q == 14
        and record.indices.code_bit_width == 4
        and record.indices.beta_count == 4
        and record.indices.beta_index_bit_width == 2
        and record.dither_seed is not None
        and record.dither_algorithm == STATELESS_DITHER_ALGORITHM
        and record.codec_semantics == UPSTREAM_CODEC_SEMANTICS
        and all(tensor.is_contiguous() for tensor in tensors)
        and all(
            tensor.device == activation.code_payload.device
            for tensor in tensors
        )
    )


def can_use_w4a4_mma_m128_staged(
    activation: PreparedNQA4Activation, weight: PreparedNQRTLinear
) -> bool:
    """Backward-compatible shape-specific predicate."""
    return bool(
        activation.token_count == 128
        and can_use_w4a4_mma_large_m_staged(activation, weight)
    )


def can_use_w4a4_mma_chunked_staged(
    activation: PreparedNQA4Activation, weight: PreparedNQRTLinear
) -> bool:
    """Return whether arbitrary M can be covered by supported staged chunks."""
    record = weight.record
    tensors = (
        activation.code_payload, activation.beta_payload,
        activation.row_norms, activation.beta_table,
        weight.code_payload, weight.beta_payload,
        weight.row_norms, weight.beta_table,
    )
    return bool(
        activation.token_count >= 1
        and activation.feature_count == record.feature_count
        and activation.q == 14
        and activation.code_width == 4
        and activation.beta_width == 2
        and activation.beta_table.numel() == 4
        and record.feature_count % 16 == 0
        and record.indices.shape[0] % 16 == 0
        and record.indices.q == 14
        and record.indices.code_bit_width == 4
        and record.indices.beta_count == 4
        and record.indices.beta_index_bit_width == 2
        and record.dither_seed is not None
        and record.dither_algorithm == STATELESS_DITHER_ALGORITHM
        and record.codec_semantics == UPSTREAM_CODEC_SEMANTICS
        and all(tensor.is_contiguous() for tensor in tensors)
        and all(
            tensor.device == activation.code_payload.device
            for tensor in tensors
        )
    )


def linear_packed_w4a4_mma_m16(
    activation: PreparedNQA4Activation,
    weight: PreparedNQRTLinear,
) -> torch.Tensor:
    """Run the first direct-packed M16 CUDA MMA prototype."""
    if not can_use_w4a4_mma_m16(activation, weight):
        raise ValueError("activation or packed weight is unsupported by W4A4 M16")
    extension = _load_extension()
    output = extension.run_m16(
        activation.code_payload, activation.beta_payload,
        activation.row_norms, activation.beta_table,
        weight.code_payload, weight.beta_payload,
        weight.row_norms, weight.beta_table,
        0, weight.record.dither_seed,
    )
    return output.to(activation.output_dtype).reshape(
        *activation.original_shape[:-1], weight.record.indices.shape[0]
    )


def linear_packed_w4a4_mma_small_m(
    activation: PreparedNQA4Activation,
    weight: PreparedNQRTLinear,
) -> torch.Tensor:
    """Run small-M MMA, using the measured M2 down-projection split-K path."""
    if not can_use_w4a4_mma_small_m(activation, weight):
        raise ValueError("activation or packed weight is unsupported by small-M probe")
    if (
        activation.token_count == 2
        and activation.feature_count == 8192
        and weight.record.indices.shape[0] == 2048
    ):
        return linear_packed_w4a4_mma_small_m_splitk_probe(
            activation, weight, split_k=8
        )
    return _linear_packed_w4a4_mma_small_m_unsplit(activation, weight)


def _linear_packed_w4a4_mma_small_m_unsplit(
    activation: PreparedNQA4Activation,
    weight: PreparedNQRTLinear,
) -> torch.Tensor:
    """Run the original one-CTA-per-output-tile small-M reference path."""
    output = _load_extension().run_small_m(
        activation.code_payload, activation.beta_payload,
        activation.row_norms, activation.beta_table,
        weight.code_payload, weight.beta_payload,
        weight.row_norms, weight.beta_table,
        0, weight.record.dither_seed,
    )
    return output.to(activation.output_dtype).reshape(
        *activation.original_shape[:-1], weight.record.indices.shape[0]
    )


def linear_packed_w4a4_mma_small_m_splitk_probe(
    activation: PreparedNQA4Activation,
    weight: PreparedNQRTLinear,
    *,
    split_k: int,
) -> torch.Tensor:
    """Run the isolated deterministic M2 split-K prototype."""
    if not can_use_w4a4_mma_small_m(activation, weight):
        raise ValueError("activation or packed weight is unsupported by small-M")
    if activation.token_count != 2:
        raise ValueError("split-K probe requires exactly two activation rows")
    if split_k not in (2, 4, 8):
        raise ValueError("split_k must be 2, 4, or 8")
    output = _load_extension().run_small_m_splitk_probe(
        activation.code_payload, activation.beta_payload,
        activation.row_norms, activation.beta_table,
        weight.code_payload, weight.beta_payload,
        weight.row_norms, weight.beta_table,
        0, weight.record.dither_seed, split_k,
    )
    return output.to(activation.output_dtype).reshape(
        *activation.original_shape[:-1], weight.record.indices.shape[0]
    )


def linear_packed_w4a4_mma_small_m_transposed_probe(
    activation: PreparedNQA4Activation,
    weight: PreparedNQRTLinear,
) -> torch.Tensor:
    """Backward-compatible alias for the promoted small-M kernel."""
    return linear_packed_w4a4_mma_small_m(activation, weight)


def linear_packed_w4a4_mma_m32_multiwarp_probe(
    activation: PreparedNQA4Activation,
    weight: PreparedNQRTLinear,
) -> torch.Tensor:
    """Run the isolated four-warp M32 structure probe."""
    if activation.token_count != 32:
        raise ValueError("M32 multi-warp probe requires exactly 32 rows")
    return linear_packed_w4a4_mma_large_m_multiwarp_probe(activation, weight)


def linear_packed_w4a4_mma_large_m_multiwarp_probe(
    activation: PreparedNQA4Activation,
    weight: PreparedNQRTLinear,
) -> torch.Tensor:
    """Run the isolated M32/M64/M128 multi-warp structure family."""
    if activation.token_count not in (32, 64, 128):
        raise ValueError("large-M multi-warp probe requires 32, 64, or 128 rows")
    record = weight.record
    if (
        activation.feature_count != record.feature_count
        or record.feature_count % 16
        or record.indices.shape[0] % 16
    ):
        raise ValueError("large-M multi-warp probe requires compatible K/N multiples of 16")
    output = _load_extension().run_large_m_multiwarp_probe(
        activation.code_payload, activation.beta_payload,
        activation.row_norms, activation.beta_table,
        weight.code_payload, weight.beta_payload,
        weight.row_norms, weight.beta_table,
        0, weight.record.dither_seed,
    )
    return output.to(activation.output_dtype).reshape(
        *activation.original_shape[:-1], weight.record.indices.shape[0]
    )


def linear_packed_w4a4_mma_m128_staged_probe(
    activation: PreparedNQA4Activation,
    weight: PreparedNQRTLinear,
) -> torch.Tensor:
    """Run the correctness-first shared decoded-weight M128 probe."""
    if activation.token_count != 128:
        raise ValueError("M128 staged probe requires exactly 128 rows")
    return linear_packed_w4a4_mma_large_m_staged(activation, weight)


def linear_packed_w4a4_mma_large_m_staged(
    activation: PreparedNQA4Activation,
    weight: PreparedNQRTLinear,
) -> torch.Tensor:
    """Run shared decoded-weight staging for M32/M64/M128."""
    if not can_use_w4a4_mma_large_m_staged(activation, weight):
        raise ValueError("activation or packed weight is unsupported by staged large-M")
    output = _load_extension().run_large_m_staged(
        activation.code_payload, activation.beta_payload,
        activation.row_norms, activation.beta_table,
        weight.code_payload, weight.beta_payload,
        weight.row_norms, weight.beta_table,
        0, weight.record.dither_seed,
    )
    return output.to(activation.output_dtype).reshape(
        *activation.original_shape[:-1], weight.record.indices.shape[0]
    )


def linear_packed_w4a4_mma_m128_staged(
    activation: PreparedNQA4Activation,
    weight: PreparedNQRTLinear,
) -> torch.Tensor:
    """Run the promoted shared decoded-weight M128 kernel."""
    if not can_use_w4a4_mma_m128_staged(activation, weight):
        raise ValueError("activation or packed weight is unsupported by staged M128")
    return linear_packed_w4a4_mma_large_m_staged(activation, weight)


def _slice_pad_packed_activation_rows(
    activation: PreparedNQA4Activation,
    start: int,
    row_count: int,
    padded_row_count: int,
) -> PreparedNQA4Activation:
    """Slice row-aligned packed A4 payloads and append zero-norm padding rows."""
    if not (0 <= start < activation.token_count):
        raise ValueError("packed activation slice start is out of range")
    if row_count < 1 or start + row_count > activation.token_count:
        raise ValueError("packed activation slice row count is out of range")
    if padded_row_count < row_count:
        raise ValueError("padded row count must cover the activation slice")
    code_bytes_per_row = (
        activation.feature_count * activation.code_width // 8
    )
    beta_bytes_per_row = (
        (activation.feature_count // 8) * activation.beta_width // 8
    )

    def slice_and_pad(payload: torch.Tensor, bytes_per_row: int) -> torch.Tensor:
        begin = start * bytes_per_row
        end = (start + row_count) * bytes_per_row
        sliced = payload[begin:end]
        padding_bytes = (padded_row_count - row_count) * bytes_per_row
        if not padding_bytes:
            return sliced.contiguous()
        return torch.cat((
            sliced,
            torch.zeros(
                padding_bytes, dtype=payload.dtype, device=payload.device
            ),
        ))

    row_norms = activation.row_norms[start:start + row_count]
    if padded_row_count != row_count:
        row_norms = torch.cat((
            row_norms,
            torch.zeros(
                padded_row_count - row_count,
                dtype=activation.row_norms.dtype,
                device=activation.row_norms.device,
            ),
        ))
    return PreparedNQA4Activation(
        code_payload=slice_and_pad(activation.code_payload, code_bytes_per_row),
        beta_payload=slice_and_pad(activation.beta_payload, beta_bytes_per_row),
        row_norms=row_norms.contiguous(),
        beta_table=activation.beta_table,
        original_shape=(padded_row_count, activation.feature_count),
        q=activation.q,
        code_width=activation.code_width,
        beta_width=activation.beta_width,
        output_dtype=activation.output_dtype,
    )


def linear_packed_w4a4_mma_chunked_staged_probe(
    activation: PreparedNQA4Activation,
    weight: PreparedNQRTLinear,
) -> torch.Tensor:
    """Cover arbitrary M by M128 chunks and one padded supported-size tail."""
    supported = (1, 2, 4, 8, 16, 32, 64, 128)
    outputs = []
    for start in range(0, activation.token_count, 128):
        row_count = min(128, activation.token_count - start)
        padded_row_count = next(size for size in supported if size >= row_count)
        chunk = _slice_pad_packed_activation_rows(
            activation, start, row_count, padded_row_count
        )
        if padded_row_count in (32, 64, 128):
            output = linear_packed_w4a4_mma_large_m_staged(chunk, weight)
        elif padded_row_count == 16:
            output = linear_packed_w4a4_mma_m16(chunk, weight)
        elif padded_row_count in (2, 4, 8):
            output = linear_packed_w4a4_mma_small_m(chunk, weight)
        else:
            output = linear_packed_w4a4_gemv_m1(chunk, weight)
        outputs.append(output.reshape(padded_row_count, -1)[:row_count])
    return torch.cat(outputs, dim=0).reshape(
        *activation.original_shape[:-1], weight.record.indices.shape[0]
    )


def linear_packed_w4a4_mma_chunked_staged(
    activation: PreparedNQA4Activation,
    weight: PreparedNQRTLinear,
) -> torch.Tensor:
    """Production arbitrary-M wrapper over exact supported CUDA kernels."""
    if not can_use_w4a4_mma_chunked_staged(activation, weight):
        raise ValueError("activation or packed weight is unsupported by chunked staged")
    return linear_packed_w4a4_mma_chunked_staged_probe(activation, weight)


def linear_packed_w4a4_mma_m128_pipeline_probe(
    activation: PreparedNQA4Activation,
    weight: PreparedNQRTLinear,
) -> torch.Tensor:
    """Run the double-buffered producer/consumer M128 pipeline probe."""
    if activation.token_count != 128:
        raise ValueError("M128 pipeline probe requires exactly 128 rows")
    record = weight.record
    if (
        activation.feature_count != record.feature_count
        or record.feature_count % 16
        or record.indices.shape[0] % 16
    ):
        raise ValueError("M128 pipeline probe requires compatible K/N multiples of 16")
    output = _load_extension().run_m128_pipeline_probe(
        activation.code_payload, activation.beta_payload,
        activation.row_norms, activation.beta_table,
        weight.code_payload, weight.beta_payload,
        weight.row_norms, weight.beta_table,
        0, weight.record.dither_seed,
    )
    return output.to(activation.output_dtype).reshape(
        *activation.original_shape[:-1], weight.record.indices.shape[0]
    )


def linear_packed_w4a4_mma_m8_transposed_probe(
    activation: PreparedNQA4Activation,
    weight: PreparedNQRTLinear,
) -> torch.Tensor:
    """Backward-compatible M8-only entry point for the small-M probe."""
    if activation.token_count != 8:
        raise ValueError("M8 transposed probe requires exactly 8 rows")
    return linear_packed_w4a4_mma_small_m(activation, weight)


def linear_packed_w4a4_gemv_m1(
    activation: PreparedNQA4Activation,
    weight: PreparedNQRTLinear,
) -> torch.Tensor:
    """Run the production packed-word direct W4A4 M1 CUDA GEMV."""
    if not can_use_w4a4_gemv_m1(activation, weight):
        raise ValueError("activation or packed weight is unsupported by W4A4 M1")
    output = _load_extension().run_m1(
        activation.code_payload, activation.beta_payload,
        activation.row_norms, activation.beta_table,
        weight.code_payload, weight.beta_payload,
        weight.row_norms, weight.beta_table,
        0, weight.record.dither_seed,
    )
    return output.to(activation.output_dtype).reshape(
        *activation.original_shape[:-1], weight.record.indices.shape[0]
    )


def linear_packed_w4a4_gemv_m1_fp32_probe(
    activation: PreparedNQA4Activation,
    weight: PreparedNQRTLinear,
) -> torch.Tensor:
    """Return the production M1 accumulator before output-dtype rounding."""
    if not can_use_w4a4_gemv_m1(activation, weight):
        raise ValueError("activation or packed weight is unsupported by W4A4 M1")
    output = _load_extension().run_m1(
        activation.code_payload, activation.beta_payload,
        activation.row_norms, activation.beta_table,
        weight.code_payload, weight.beta_payload,
        weight.row_norms, weight.beta_table,
        0, weight.record.dither_seed,
    )
    return output.reshape(
        *activation.original_shape[:-1], weight.record.indices.shape[0]
    )


def linear_packed_w4a4_gemv_m1_legacy(
    activation: PreparedNQA4Activation,
    weight: PreparedNQRTLinear,
) -> torch.Tensor:
    """Run the retained pre-packed-word M1 kernel as a regression baseline."""
    if not can_use_w4a4_gemv_m1(activation, weight):
        raise ValueError("activation or packed weight is unsupported by W4A4 M1")
    output = _load_extension().run_m1_legacy(
        activation.code_payload, activation.beta_payload,
        activation.row_norms, activation.beta_table,
        weight.code_payload, weight.beta_payload,
        weight.row_norms, weight.beta_table,
        0, weight.record.dither_seed,
    )
    return output.to(activation.output_dtype).reshape(
        *activation.original_shape[:-1], weight.record.indices.shape[0]
    )


def linear_packed_w4a4_gemv_m1_packed_word_probe(
    activation: PreparedNQA4Activation,
    weight: PreparedNQRTLinear,
) -> torch.Tensor:
    """Backward-compatible alias for the promoted packed-word M1 kernel."""
    if not can_use_w4a4_gemv_m1(activation, weight):
        raise ValueError("activation or packed weight is unsupported by W4A4 M1")
    output = _load_extension().run_m1_packed_word_probe(
        activation.code_payload, activation.beta_payload,
        activation.row_norms, activation.beta_table,
        weight.code_payload, weight.beta_payload,
        weight.row_norms, weight.beta_table,
        0, weight.record.dither_seed,
    )
    return output.to(activation.output_dtype).reshape(
        *activation.original_shape[:-1], weight.record.indices.shape[0]
    )


def linear_packed_w4a4_gemv_m1_splitk_probe(
    activation: PreparedNQA4Activation,
    weight: PreparedNQRTLinear,
    *,
    split_k: int,
) -> torch.Tensor:
    """Run the deterministic long-K M1 split-K experiment."""
    if not can_use_w4a4_gemv_m1(activation, weight):
        raise ValueError("activation or packed weight is unsupported by W4A4 M1")
    if split_k not in (2, 4, 8):
        raise ValueError("split_k must be 2, 4, or 8")
    output = _load_extension().run_m1_splitk_probe(
        activation.code_payload, activation.beta_payload,
        activation.row_norms, activation.beta_table,
        weight.code_payload, weight.beta_payload,
        weight.row_norms, weight.beta_table,
        0, weight.record.dither_seed, split_k,
    )
    return output.to(activation.output_dtype).reshape(
        *activation.original_shape[:-1], weight.record.indices.shape[0]
    )


def linear_packed_w4a4_gemv_m1_splitk_fp32_probe(
    activation: PreparedNQA4Activation,
    weight: PreparedNQRTLinear,
    *,
    split_k: int,
) -> torch.Tensor:
    """Return split-K M1 accumulators before output-dtype rounding."""
    if not can_use_w4a4_gemv_m1(activation, weight):
        raise ValueError("activation or packed weight is unsupported by W4A4 M1")
    if split_k not in (2, 4, 8):
        raise ValueError("split_k must be 2, 4, or 8")
    output = _load_extension().run_m1_splitk_probe(
        activation.code_payload, activation.beta_payload,
        activation.row_norms, activation.beta_table,
        weight.code_payload, weight.beta_payload,
        weight.row_norms, weight.beta_table,
        0, weight.record.dither_seed, split_k,
    )
    return output.reshape(
        *activation.original_shape[:-1], weight.record.indices.shape[0]
    )


def linear_packed_w4a4_gemv_m1_splitk_partials_probe(
    activation: PreparedNQA4Activation,
    weight: PreparedNQRTLinear,
    *,
    split_k: int,
) -> torch.Tensor:
    """Return `[split_k, N]` raw FP32 partial sums before reduction."""
    if not can_use_w4a4_gemv_m1(activation, weight):
        raise ValueError("activation or packed weight is unsupported by W4A4 M1")
    if split_k not in (2, 4, 8):
        raise ValueError("split_k must be 2, 4, or 8")
    return _load_extension().run_m1_splitk_partials_probe(
        activation.code_payload, activation.beta_payload,
        activation.row_norms, activation.beta_table,
        weight.code_payload, weight.beta_payload,
        weight.row_norms, weight.beta_table,
        0, weight.record.dither_seed, split_k,
    )


def linear_packed_w4a4_gemv_m1_splitk_compensated_fp32_probe(
    activation: PreparedNQA4Activation,
    weight: PreparedNQRTLinear,
    *,
    split_k: int,
) -> torch.Tensor:
    """Return diagnostic split-K output with per-partition Kahan accumulation."""
    if not can_use_w4a4_gemv_m1(activation, weight):
        raise ValueError("activation or packed weight is unsupported by W4A4 M1")
    if split_k not in (2, 4, 8):
        raise ValueError("split_k must be 2, 4, or 8")
    output = _load_extension().run_m1_splitk_compensated_probe(
        activation.code_payload, activation.beta_payload,
        activation.row_norms, activation.beta_table,
        weight.code_payload, weight.beta_payload,
        weight.row_norms, weight.beta_table,
        0, weight.record.dither_seed, split_k,
    )
    return output.reshape(
        *activation.original_shape[:-1], weight.record.indices.shape[0]
    )


def decode_packed_a4_m16(activation: PreparedNQA4Activation) -> torch.Tensor:
    """Diagnostic decoder used to isolate A-fragment semantics."""
    if activation.token_count != 16 or activation.q != 14:
        raise ValueError("diagnostic decoder requires 16 rows and q=14")
    return _load_extension().decode_a4_m16(
        activation.code_payload, activation.beta_payload,
        activation.row_norms, activation.beta_table, activation.feature_count,
    ).reshape(activation.original_shape)


def inspect_packed_a4_m16(activation: PreparedNQA4Activation) -> torch.Tensor:
    """Return per-block CUDA E8 intermediate values for diagnostics."""
    if activation.token_count != 16 or activation.q != 14:
        raise ValueError("diagnostic inspector requires 16 rows and q=14")
    return _load_extension().inspect_a4_m16(
        activation.code_payload, activation.feature_count,
    )
