"""Lazy-loaded CUDA J2 Tensor Core backend for packed NQRT v4 Linear."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..core.codec import STATELESS_DITHER_ALGORITHM, UPSTREAM_CODEC_SEMANTICS
from .triton_fused_linear import PreparedNQRTLinear


_EXTENSION: Any | None = None


def _load_extension() -> Any:
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    if not torch.cuda.is_available():
        raise RuntimeError("J2 CUDA MMA backend requires CUDA")
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parents[2]
    _EXTENSION = load(
        name="nqrt4_mma_j2_ext",
        sources=[
            str(root / "nestquant/runtime/nqrt4_mma_j2_binding.cpp"),
            str(root / "nestquant/runtime/nqrt4_mma_j2.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
    return _EXTENSION


def can_use_mma_j2(inputs: torch.Tensor, prepared: PreparedNQRTLinear) -> bool:
    """Return whether the current J2 specialization supports this call."""
    record = prepared.record
    token_count = inputs.numel() // inputs.shape[-1] if inputs.ndim else 0
    rows = record.indices.shape[0]
    return bool(
        inputs.is_cuda
        and inputs.dtype == torch.bfloat16
        and 1 <= token_count <= 16
        and record.feature_count % 16 == 0
        and rows % 8 == 0
        and record.indices.q == 14
        and record.indices.code_bit_width == 4
        and record.indices.beta_count == 4
        and record.indices.beta_index_bit_width == 2
        and record.dither_seed is not None
        and record.dither_algorithm == STATELESS_DITHER_ALGORITHM
        and record.codec_semantics == UPSTREAM_CODEC_SEMANTICS
    )


def linear_fused_mma_j2(
    inputs: torch.Tensor,
    prepared: PreparedNQRTLinear,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the M=16 cooperative J2 kernel and return an FP32 output."""
    if not isinstance(prepared, PreparedNQRTLinear):
        raise TypeError("prepared must be PreparedNQRTLinear")
    if not can_use_mma_j2(inputs, prepared):
        raise ValueError("input or packed record is not supported by J2 CUDA MMA")
    record = prepared.record
    leading_shape = inputs.shape[:-1]
    token_count = inputs.numel() // record.feature_count
    flat = inputs.reshape(token_count, record.feature_count).contiguous()
    extension = _load_extension()
    if token_count == 1:
        kernel_inputs = flat
        operation = extension.run_m1_auto
    else:
        operation = extension.run_m16_auto
        if token_count == 16:
            kernel_inputs = flat
        else:
            kernel_inputs = torch.zeros(
                (16, record.feature_count), device=flat.device, dtype=flat.dtype
            )
            kernel_inputs[:token_count].copy_(flat)
    output = operation(
        kernel_inputs,
        prepared.code_payload,
        prepared.beta_payload,
        prepared.row_norms,
        prepared.beta_table,
        0,
        record.dither_seed,
    )[:token_count].reshape(*leading_shape, record.indices.shape[0])
    if bias is not None:
        if bias.device != inputs.device or tuple(bias.shape) != (record.indices.shape[0],):
            raise ValueError("bias must be on the input device and match out_features")
        output = output + bias.float()
    return output
