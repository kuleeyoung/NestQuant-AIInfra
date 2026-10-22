"""Algorithm-3 fake quantization for runtime activation-like tensors."""

from __future__ import annotations

import math
from typing import Sequence

import torch

from .quantizer import quantize_rows


def fake_quantize_nestquant_rows(
    tensor: torch.Tensor,
    *,
    q: int,
    betas: Sequence[float],
    rows_per_chunk: int = 4096,
) -> tuple[torch.Tensor, dict]:
    """Quantize/dequantize vectors along the last dimension with Algorithm 3.

    This function models quantization quality.  The returned tensor has the
    original floating dtype, so it does not establish packed runtime storage or
    low-bit execution.  No dither is used because runtime chunking must not
    change the reconstructed value and paper Algorithm 3 does not require it.
    """
    if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
        raise TypeError("tensor must be a floating-point torch.Tensor")
    if tensor.ndim < 1 or tensor.numel() == 0:
        raise ValueError("tensor must be nonempty")
    if tensor.shape[-1] % 8:
        raise ValueError("the final dimension must be divisible by 8")
    if rows_per_chunk < 1:
        raise ValueError("rows_per_chunk must be positive")
    beta_values = tuple(float(beta) for beta in betas)
    if not beta_values or any(not math.isfinite(beta) or beta <= 0 for beta in beta_values):
        raise ValueError("betas must be positive finite values")

    rows = tensor.reshape(-1, tensor.shape[-1])
    restored = torch.empty_like(rows)
    beta_histogram = torch.zeros(len(beta_values), dtype=torch.int64)
    squared_error = 0.0
    signal_energy = 0.0
    max_abs_error = 0.0
    overload_count = 0
    for start in range(0, rows.shape[0], rows_per_chunk):
        end = min(start + rows_per_chunk, rows.shape[0])
        source = rows[start:end].float()
        quantized = quantize_rows(source, q, beta_values)
        reconstructed = quantized.dequantize()
        restored[start:end].copy_(reconstructed.to(dtype=tensor.dtype))
        beta_histogram += torch.bincount(
            quantized.beta_indices.detach().cpu().reshape(-1), minlength=len(beta_values)
        )
        error = reconstructed - source
        squared_error += float(error.square().sum().item())
        signal_energy += float(source.square().sum().item())
        max_abs_error = max(max_abs_error, float(error.abs().amax().item()))
        if quantized.overloaded is not None:
            overload_count += int(quantized.overloaded.sum().item())

    entries = tensor.numel()
    vector_count = rows.shape[0]
    block_count = entries // 8
    residue_bits = math.ceil(math.log2(q))
    beta_index_bits_per_block = math.ceil(math.log2(len(beta_values))) if len(beta_values) > 1 else 0
    estimated_bits = (
        residue_bits * entries
        + beta_index_bits_per_block * block_count
        + 32 * vector_count
    )
    stats = {
        "q": q,
        "betas": list(beta_values),
        "vector_count": vector_count,
        "feature_count": tensor.shape[-1],
        "entry_count": entries,
        "e8_block_count": block_count,
        "beta_histogram": beta_histogram.tolist(),
        "squared_error": squared_error,
        "signal_energy": signal_energy,
        "relative_squared_error": squared_error / signal_energy if signal_energy else 0.0,
        "max_abs_error": max_abs_error,
        "overload_count": overload_count,
        "overload_fraction": overload_count / block_count,
        "storage_estimate": {
            "status": "estimate_only_reconstructed_tensor_is_floating_point",
            "residue_bits_per_entry": residue_bits,
            "beta_index_bits_per_block": beta_index_bits_per_block,
            "row_norm_bits_per_vector": 32,
            "estimated_total_bits": estimated_bits,
            "effective_bits_per_entry": estimated_bits / entries,
        },
    }
    return restored.reshape_as(tensor), stats
