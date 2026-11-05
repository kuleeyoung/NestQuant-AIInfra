"""Isolated CUTLASS CuTe layout probe for the packed W4A4 M128 kernel."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from ..core.codec import STATELESS_DITHER_ALGORITHM, UPSTREAM_CODEC_SEMANTICS
from .triton_fused_linear import PreparedNQRTLinear
from .triton_w4a4_linear import PreparedNQA4Activation


_CUTE_EXTENSION: Any | None = None


def _cutlass_root() -> Path:
    configured = os.environ.get("NESTQUANT_CUTLASS_PATH")
    if configured:
        root = Path(configured).expanduser().resolve()
    else:
        root = Path(__file__).resolve().parents[2] / "tmp/vendor/cutlass-4.8.0"
    version = root / "include/cutlass/version.h"
    cute = root / "include/cute/tensor.hpp"
    if not version.is_file() or not cute.is_file():
        raise RuntimeError(
            "CUTLASS 4.8.0 headers not found; set NESTQUANT_CUTLASS_PATH "
            "or extract them under tmp/vendor/cutlass-4.8.0"
        )
    return root


def _load_cute_extension() -> Any:
    global _CUTE_EXTENSION
    if _CUTE_EXTENSION is not None:
        return _CUTE_EXTENSION
    if not torch.cuda.is_available():
        raise RuntimeError("packed W4A4 CuTe probe requires CUDA")
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parents[2]
    cutlass = _cutlass_root()
    _CUTE_EXTENSION = load(
        name="nqrt4_w4a4_cute_probe_ext",
        sources=[
            str(root / "nestquant/runtime/nqrt4_w4a4_cute_probe_binding.cpp"),
            str(root / "nestquant/runtime/nqrt4_w4a4_cute_probe.cu"),
        ],
        extra_include_paths=[str(cutlass / "include")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "-DNQRT4_PARITY_SEPARATE_REFERENCE",
            "-DNQRT4_MAX_INDEX_SHUFFLE_REFERENCE",
        ],
        verbose=False,
    )
    return _CUTE_EXTENSION


def can_use_w4a4_cute_large_m(
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


def linear_packed_w4a4_cute_large_m_probe(
    activation: PreparedNQA4Activation,
    weight: PreparedNQRTLinear,
) -> torch.Tensor:
    """Run the isolated CuTe-layout M32/M64/M128 staged family."""
    if not can_use_w4a4_cute_large_m(activation, weight):
        raise ValueError("activation or packed weight is unsupported by CuTe large-M")
    output = _load_cute_extension().run_m128(
        activation.code_payload, activation.beta_payload,
        activation.row_norms, activation.beta_table,
        weight.code_payload, weight.beta_payload,
        weight.row_norms, weight.beta_table,
        0, weight.record.dither_seed,
    )
    return output.to(activation.output_dtype).reshape(
        *activation.original_shape[:-1], weight.record.indices.shape[0]
    )


def linear_packed_w4a4_cute_m128_probe(
    activation: PreparedNQA4Activation,
    weight: PreparedNQRTLinear,
) -> torch.Tensor:
    """Run the isolated CuTe-layout version of staged M128."""
    if activation.token_count != 128:
        raise ValueError("CuTe M128 probe requires exactly 128 rows")
    return linear_packed_w4a4_cute_large_m_probe(activation, weight)
