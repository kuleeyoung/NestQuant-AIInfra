"""Row-wise, blockwise E8 quantization with a fixed beta candidate set.

This implements the core reconstruction loop of paper Algorithm 3. Choosing
the beta set by dynamic programming and LDLQ feedback are separate later steps.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from .codec import decode_nested, encode_nested
from .lattice import E8_DIMENSION
from .ldlq import block_unit_lower_ldl, ldlq_quantize_blocks


@dataclass(frozen=True)
class QuantizedRows:
    """Runtime representation of row-wise quantized matrix values.

    This is a quality-simulation object. Codes are int64 tensors in memory;
    this class does not pack codes into a low-bit file format.
    """

    codes: torch.Tensor
    beta_indices: torch.Tensor
    dither: torch.Tensor | None
    row_norms: torch.Tensor
    q: int
    betas: tuple[float, ...]
    feature_count: int
    output_dtype: torch.dtype
    overloaded: torch.Tensor | None = None

    def dequantize(self) -> torch.Tensor:
        """Reconstruct a floating-point matrix with the original shape."""
        rows, blocks, _ = self.codes.shape
        beta_table = torch.tensor(self.betas, device=self.codes.device, dtype=self.output_dtype)
        selected_betas = beta_table[self.beta_indices]
        reconstructed_blocks = decode_nested(
            self.codes.reshape(-1, E8_DIMENSION), self.q, dtype=self.output_dtype,
            dither=None if self.dither is None else self.dither.reshape(-1, E8_DIMENSION),
        ).reshape(rows, blocks, E8_DIMENSION)
        normalized = (reconstructed_blocks * selected_betas.unsqueeze(-1)).reshape(rows, self.feature_count)
        scale = self.row_norms / math.sqrt(self.feature_count)
        return normalized * scale.unsqueeze(-1)


@dataclass(frozen=True)
class QuantizedBlocks:
    """Quantized already-normalized E8 blocks, without row scaling."""

    codes: torch.Tensor
    beta_indices: torch.Tensor
    dither: torch.Tensor | None
    q: int
    betas: tuple[float, ...]
    output_dtype: torch.dtype

    def dequantize(self) -> torch.Tensor:
        beta_table = torch.tensor(self.betas, device=self.codes.device, dtype=self.output_dtype)
        selected = beta_table[self.beta_indices]
        reconstructed = decode_nested(
            self.codes.reshape(-1, E8_DIMENSION), self.q, dtype=self.output_dtype,
            dither=None if self.dither is None else self.dither.reshape(-1, E8_DIMENSION),
        ).reshape_as(self.codes)
        return reconstructed * selected.unsqueeze(-1)


def quantize_e8_blocks(
    blocks: torch.Tensor,
    q: int,
    betas: Sequence[float],
    *,
    dither: torch.Tensor | None = None,
) -> QuantizedBlocks:
    """Quantize pre-normalized tensors whose final dimension is eight."""
    if not isinstance(blocks, torch.Tensor) or not blocks.is_floating_point():
        raise TypeError("blocks must be a floating-point tensor")
    if blocks.ndim < 2 or blocks.shape[-1] != E8_DIMENSION or blocks.numel() == 0:
        raise ValueError("blocks must be nonempty with final dimension 8")
    if not torch.isfinite(blocks).all().item():
        raise ValueError("blocks must contain only finite values")
    try:
        beta_values = tuple(float(beta) for beta in betas)
    except (TypeError, ValueError) as error:
        raise TypeError("betas must be a nonempty sequence of numbers") from error
    if not beta_values or any(not math.isfinite(beta) or beta <= 0 for beta in beta_values):
        raise ValueError("every beta must be positive and finite")
    if dither is not None:
        if dither.shape != blocks.shape or dither.device != blocks.device or dither.dtype != blocks.dtype:
            raise ValueError("dither must match block shape/device/dtype")
        if not torch.isfinite(dither).all().item():
            raise ValueError("dither must contain only finite values")

    candidate_codes, candidate_errors = [], []
    for beta in beta_values:
        codes, _ = encode_nested(blocks / beta, q, dither=dither)
        reconstructed = decode_nested(codes, q, dtype=blocks.dtype, dither=dither) * beta
        candidate_codes.append(codes)
        candidate_errors.append((reconstructed - blocks).square().sum(dim=-1))
    code_stack = torch.stack(candidate_codes, dim=0)
    errors = torch.stack(candidate_errors, dim=0)
    best_beta = errors.argmin(dim=0)
    gather = best_beta.unsqueeze(0).unsqueeze(-1).expand(1, *best_beta.shape, E8_DIMENSION)
    best_codes = torch.gather(code_stack, 0, gather).squeeze(0)
    return QuantizedBlocks(
        codes=best_codes, beta_indices=best_beta, dither=dither, q=q,
        betas=beta_values, output_dtype=blocks.dtype,
    )


def quantize_rows_ldlq(
    matrix: torch.Tensor,
    hessian: torch.Tensor | None,
    q: int,
    betas: Sequence[float],
    *,
    dither: torch.Tensor | None = None,
    relative_damp: float = 0.01,
    precomputed_lower: torch.Tensor | None = None,
    absolute_damping_added: float | None = None,
) -> tuple[QuantizedRows, dict[str, float | int]]:
    """Quantize normalized weight rows with 8-column block-LDLQ feedback.

    The row norm is fixed before feedback. Each callback quantizes one
    ``[rows, 8]`` target with the ordinary E8 multi-beta nearest-neighbour
    rule, while the block LDL factor redirects errors in the calibration
    Hessian metric.
    """
    if not isinstance(matrix, torch.Tensor) or not matrix.is_floating_point():
        raise TypeError("matrix must be a floating-point tensor")
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("matrix must be nonempty with shape [rows, features]")
    if matrix.shape[1] % E8_DIMENSION:
        raise ValueError("matrix feature count must be divisible by 8")
    if not torch.isfinite(matrix).all().item():
        raise ValueError("matrix must contain only finite values")
    if precomputed_lower is None:
        if hessian is None or hessian.shape != (matrix.shape[1], matrix.shape[1]):
            raise ValueError("hessian shape must match matrix in_features")
        if hessian.device != matrix.device or hessian.dtype != matrix.dtype:
            raise ValueError("hessian must match matrix device and dtype")
    else:
        if hessian is not None:
            raise ValueError("pass either hessian or precomputed_lower, not both")
        if precomputed_lower.shape != (matrix.shape[1], matrix.shape[1]):
            raise ValueError("precomputed_lower shape must match matrix in_features")
        if precomputed_lower.device != matrix.device or precomputed_lower.dtype != matrix.dtype:
            raise ValueError("precomputed_lower must match matrix device and dtype")
        if absolute_damping_added is None:
            raise ValueError("absolute_damping_added is required with precomputed_lower")

    row_norms = torch.linalg.vector_norm(matrix, ord=2, dim=-1)
    nonzero = row_norms > 0
    safe_norms = torch.where(nonzero, row_norms, torch.ones_like(row_norms))
    normalized = matrix * (math.sqrt(matrix.shape[1]) / safe_norms).unsqueeze(-1)
    normalized = torch.where(nonzero.unsqueeze(-1), normalized, torch.zeros_like(normalized))
    block_count = matrix.shape[1] // E8_DIMENSION
    if dither is not None:
        expected = (matrix.shape[0], block_count, E8_DIMENSION)
        if dither.shape != expected or dither.device != matrix.device or dither.dtype != matrix.dtype:
            raise ValueError("dither must match [rows, feature_blocks, 8] on matrix device/dtype")

    if precomputed_lower is None:
        lower, _diagonal_blocks, damping = block_unit_lower_ldl(
            hessian, block_size=E8_DIMENSION, relative_damp=relative_damp
        )
    else:
        lower = precomputed_lower
        damping = float(absolute_damping_added)
    codes = torch.empty(
        (matrix.shape[0], block_count, E8_DIMENSION), dtype=torch.int64, device=matrix.device
    )
    beta_indices = torch.empty(
        (matrix.shape[0], block_count), dtype=torch.int64, device=matrix.device
    )

    def quantize_block(target: torch.Tensor, block_index: int) -> torch.Tensor:
        block_dither = None if dither is None else dither[:, block_index, :]
        quantized = quantize_e8_blocks(target, q, betas, dither=block_dither)
        codes[:, block_index, :] = quantized.codes
        beta_indices[:, block_index] = quantized.beta_indices
        return quantized.dequantize()

    ldlq_quantize_blocks(
        normalized, lower, quantize_block, block_size=E8_DIMENSION
    )
    result = QuantizedRows(
        codes=codes,
        beta_indices=beta_indices,
        dither=dither,
        row_norms=row_norms,
        q=q,
        betas=tuple(float(beta) for beta in betas),
        feature_count=matrix.shape[1],
        output_dtype=matrix.dtype,
    )
    return result, {
        "block_size": E8_DIMENSION,
        "block_count": block_count,
        "relative_damp": relative_damp,
        "absolute_damping_added": damping,
    }


def quantize_rows(
    matrix: torch.Tensor,
    q: int,
    betas: Sequence[float],
    *,
    dither: torch.Tensor | None = None,
) -> QuantizedRows:
    """Quantize matrix rows by normalization and E8 blocks.

    For every row ``a`` with ``n`` features, the method first forms
    ``a * sqrt(n) / ||a||_2``. Each 8-value block is encoded and decoded for
    every supplied beta, and the beta with the smallest block squared error is
    retained. A zero row uses a zero normalized row and remains zero.

    Args:
        matrix: Finite floating-point matrix shaped ``[rows, features]``.
        q: Integer E8 nesting ratio.
        betas: Positive finite scale candidates. This function consumes a
            supplied set; it does not search for an optimal set.

    Returns:
        A :class:`QuantizedRows` object with integer residues, per-block beta
        indices, original row norms, and enough metadata to reconstruct.
    """
    if not isinstance(matrix, torch.Tensor):
        raise TypeError("matrix must be a torch.Tensor")
    if not matrix.is_floating_point():
        raise TypeError("matrix must have a floating-point dtype")
    if matrix.ndim != 2:
        raise ValueError("matrix must have shape [rows, features]")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0 or matrix.shape[1] % E8_DIMENSION:
        raise ValueError("matrix must have nonempty rows and a feature count divisible by 8")
    if not torch.isfinite(matrix).all().item():
        raise ValueError("matrix must contain only finite values")
    try:
        beta_values = tuple(float(beta) for beta in betas)
    except (TypeError, ValueError) as error:
        raise TypeError("betas must be a nonempty sequence of numbers") from error
    if not beta_values:
        raise ValueError("betas must be nonempty")
    if any(not math.isfinite(beta) or beta <= 0 for beta in beta_values):
        raise ValueError("every beta must be positive and finite")

    row_norms = torch.linalg.vector_norm(matrix, ord=2, dim=-1)
    nonzero = row_norms > 0
    safe_norms = torch.where(nonzero, row_norms, torch.ones_like(row_norms))
    normalized = matrix * (math.sqrt(matrix.shape[1]) / safe_norms).unsqueeze(-1)
    normalized = torch.where(nonzero.unsqueeze(-1), normalized, torch.zeros_like(normalized))
    blocks = normalized.reshape(matrix.shape[0], -1, E8_DIMENSION)

    candidate_codes = []
    candidate_errors = []
    candidate_overloads = []
    if dither is not None:
        if dither.shape != blocks.shape or dither.device != matrix.device or dither.dtype != matrix.dtype:
            raise ValueError("dither must match normalized block shape/device/dtype")
        if not torch.isfinite(dither).all().item():
            raise ValueError("dither must contain only finite values")

    for beta in beta_values:
        codes, overloaded = encode_nested(blocks / beta, q, dither=dither)
        reconstructed = decode_nested(codes, q, dtype=matrix.dtype, dither=dither) * beta
        candidate_codes.append(codes)
        candidate_errors.append((reconstructed - blocks).square().sum(dim=-1))
        candidate_overloads.append(overloaded)

    # [num_betas, rows, blocks, 8] and [num_betas, rows, blocks]
    code_stack = torch.stack(candidate_codes, dim=0)
    error_stack = torch.stack(candidate_errors, dim=0)
    best_beta = error_stack.argmin(dim=0)
    gather_index = best_beta.unsqueeze(0).unsqueeze(-1).expand(1, *best_beta.shape, E8_DIMENSION)
    best_codes = torch.gather(code_stack, dim=0, index=gather_index).squeeze(0)
    overload_stack = torch.stack(candidate_overloads, dim=0)
    best_overloaded = torch.gather(
        overload_stack, dim=0, index=best_beta.unsqueeze(0)
    ).squeeze(0)

    return QuantizedRows(
        codes=best_codes,
        beta_indices=best_beta,
        dither=dither,
        row_norms=row_norms,
        q=q,
        betas=beta_values,
        feature_count=matrix.shape[1],
        output_dtype=matrix.dtype,
        overloaded=best_overloaded,
    )
