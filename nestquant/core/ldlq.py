"""Reference activation statistics and LDLQ error-feedback primitives.

This module intentionally stays independent from the E8 encoder.  It makes the
second-order objective and recursion testable before the full model quantizer
is wired to calibration files.  The paper defines ``H = E[x x^T]``; scaling H
by the number of samples does not change the LDLQ feedback factor.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass
class ActivationCovarianceAccumulator:
    """Accumulate ``sum(x.T @ x)`` on a chosen device without retaining inputs."""

    feature_count: int
    device: torch.device | str = "cpu"
    dtype: torch.dtype = torch.float64

    def __post_init__(self) -> None:
        if self.feature_count <= 0:
            raise ValueError("feature_count must be positive")
        if not torch.empty((), dtype=self.dtype).is_floating_point():
            raise TypeError("dtype must be floating point")
        self.device = torch.device(self.device)
        self.gram = torch.zeros(
            (self.feature_count, self.feature_count),
            dtype=self.dtype,
            device=self.device,
        )
        self.sample_count = 0

    @torch.no_grad()
    def update(self, inputs: torch.Tensor) -> None:
        """Add all leading-dimension rows from ``inputs[..., feature_count]``."""
        if not isinstance(inputs, torch.Tensor) or not inputs.is_floating_point():
            raise TypeError("inputs must be a floating-point tensor")
        if inputs.ndim < 2 or inputs.shape[-1] != self.feature_count:
            raise ValueError("inputs must end with feature_count")
        if not torch.isfinite(inputs).all().item():
            raise ValueError("inputs must contain only finite values")
        rows = inputs.detach().reshape(-1, self.feature_count).to(
            device=self.device, dtype=self.dtype
        )
        self.gram.addmm_(rows.T, rows)
        self.sample_count += rows.shape[0]

    def covariance(self) -> torch.Tensor:
        if self.sample_count == 0:
            raise RuntimeError("no activation rows have been accumulated")
        return self.gram / self.sample_count

    def merge_(self, other: "ActivationCovarianceAccumulator") -> None:
        if (
            self.feature_count != other.feature_count
            or self.device != other.device
            or self.dtype != other.dtype
        ):
            raise ValueError("accumulators must have matching shape, device and dtype")
        self.gram.add_(other.gram)
        self.sample_count += other.sample_count


def regularize_hessian(
    hessian: torch.Tensor,
    *,
    relative_damp: float = 0.01,
    absolute_damp: float = 0.0,
) -> tuple[torch.Tensor, float]:
    """Symmetrize H and add ``max(relative_damp*mean(diag(H)), absolute_damp) I``."""
    _validate_square_float(hessian, "hessian")
    if relative_damp < 0 or absolute_damp < 0:
        raise ValueError("damping values must be nonnegative")
    symmetric = (hessian + hessian.T) * 0.5
    diagonal_mean = symmetric.diagonal().mean().item()
    damping = max(relative_damp * max(diagonal_mean, 0.0), absolute_damp)
    return symmetric + torch.eye(
        symmetric.shape[0], dtype=symmetric.dtype, device=symmetric.device
    ) * damping, damping


def qa_ldlq_adjustment(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    *,
    activation_variance: float,
    epsilon_constant: float = 155.0,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Apply the public NestQuant QA-LDLQ weight/Hessian adjustment.

    With ``eps² = activation_variance / epsilon_constant``, the public code
    replaces ``W`` by ``W @ (I - eps² (H + eps² I)^-1)`` and uses
    ``H + eps² I`` for LDLQ.  ``solve`` avoids forming an explicit inverse.
    """
    if not isinstance(weight, torch.Tensor) or not weight.is_floating_point():
        raise TypeError("weight must be a floating-point tensor")
    if weight.ndim != 2:
        raise ValueError("weight must have shape [out_features, in_features]")
    _validate_square_float(hessian, "hessian")
    if hessian.shape[0] != weight.shape[1]:
        raise ValueError("hessian size must match weight in_features")
    if hessian.device != weight.device or hessian.dtype != weight.dtype:
        raise ValueError("hessian must match weight device and dtype")
    if not isinstance(activation_variance, (int, float)) or not torch.isfinite(
        torch.tensor(float(activation_variance))
    ).item() or activation_variance < 0:
        raise ValueError("activation_variance must be finite and nonnegative")
    if not isinstance(epsilon_constant, (int, float)) or not torch.isfinite(
        torch.tensor(float(epsilon_constant))
    ).item() or epsilon_constant <= 0:
        raise ValueError("epsilon_constant must be positive and finite")
    epsilon_squared = float(activation_variance) / float(epsilon_constant)
    symmetric = (hessian + hessian.T) * 0.5
    adjusted_hessian = symmetric + torch.eye(
        symmetric.shape[0], device=symmetric.device, dtype=symmetric.dtype
    ) * epsilon_squared
    if epsilon_squared == 0:
        return weight.clone(), adjusted_hessian, 0.0
    correction = torch.linalg.solve(adjusted_hessian, weight.T).T
    adjusted_weight = weight - epsilon_squared * correction
    return adjusted_weight, adjusted_hessian, epsilon_squared ** 0.5


def unit_lower_ldl(
    hessian: torch.Tensor,
    *,
    relative_damp: float = 0.01,
    absolute_damp: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return unit-lower ``L`` and diagonal ``D`` with ``H_damped = L D L.T``.

    Cholesky is used for numerical reliability.  If ``C C.T = H`` and
    ``S = diag(C)``, then ``L = C S^-1`` and ``D = S^2``.
    """
    regularized, damping = regularize_hessian(
        hessian, relative_damp=relative_damp, absolute_damp=absolute_damp
    )
    cholesky = torch.linalg.cholesky(regularized)
    scale = cholesky.diagonal()
    lower = cholesky / scale.unsqueeze(0)
    diagonal = scale.square()
    return lower, diagonal, damping


def block_unit_lower_ldl(
    hessian: torch.Tensor,
    *,
    block_size: int = 8,
    relative_damp: float = 0.01,
    absolute_damp: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return block-unit-lower ``L`` and block-diagonal ``D``.

    For a Cholesky factor ``C`` partitioned into square blocks, write each
    block column as ``C[:, i] = L[:, i] B_i`` where ``B_i = C[i, i]``.
    Then ``H = L blockdiag(B_i B_i.T) L.T`` and every diagonal block of L is
    identity.  This is the factor needed when one quantizer call jointly
    handles ``block_size`` adjacent input columns.
    """
    _validate_square_float(hessian, "hessian")
    if block_size <= 0 or hessian.shape[0] % block_size:
        raise ValueError("block_size must be positive and divide the hessian size")
    regularized, damping = regularize_hessian(
        hessian, relative_damp=relative_damp, absolute_damp=absolute_damp
    )
    lower = torch.linalg.cholesky(regularized)
    block_count = lower.shape[0] // block_size
    diagonal_blocks = torch.empty(
        (block_count, block_size, block_size), dtype=lower.dtype, device=lower.device
    )
    for block in range(block_count):
        start = block * block_size
        end = start + block_size
        root = lower[start:end, start:end].clone()
        diagonal_blocks[block] = root @ root.T
        # X = C_column @ inv(root); solve root.T @ X.T = C_column.T.
        column = lower[start:, start:end]
        normalized = torch.linalg.solve_triangular(
            root.T, column.T, upper=True
        ).T
        lower[start:, start:end] = normalized
    return lower, diagonal_blocks, damping


@torch.no_grad()
def ldlq_quantize_columns(
    weight: torch.Tensor,
    lower: torch.Tensor,
    quantize: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Apply the paper Appendix-B last-column-first LDLQ recursion.

    ``quantize`` receives one target column shaped ``[out_features]`` and must
    return a tensor of identical shape/device/dtype.  This scalar-column
    reference establishes recursion direction and proxy-loss behavior.  E8
    integration needs an 8-column block LDL factor and is deliberately a
    separate step.
    """
    if not isinstance(weight, torch.Tensor) or not weight.is_floating_point():
        raise TypeError("weight must be a floating-point tensor")
    if weight.ndim != 2:
        raise ValueError("weight must have shape [out_features, in_features]")
    _validate_square_float(lower, "lower")
    if lower.shape[0] != weight.shape[1]:
        raise ValueError("lower size must match weight in_features")
    if lower.device != weight.device or lower.dtype != weight.dtype:
        raise ValueError("lower must match weight device and dtype")
    if not torch.allclose(lower, torch.tril(lower)) or not torch.allclose(
        lower.diagonal(), torch.ones_like(lower.diagonal())
    ):
        raise ValueError("lower must be unit lower triangular")

    quantized = torch.empty_like(weight)
    error = torch.zeros_like(weight)
    for column in range(weight.shape[1] - 1, -1, -1):
        if column + 1 < weight.shape[1]:
            feedback = error[:, column + 1 :] @ lower[column + 1 :, column]
        else:
            feedback = torch.zeros(weight.shape[0], dtype=weight.dtype, device=weight.device)
        # Match the public NestQuant encoder: A + (A - decoded) @ L for the
        # already-quantized suffix. The explicit error convention avoids the
        # sign ambiguity of the compact Appendix-B notation.
        target = weight[:, column] + feedback
        result = quantize(target)
        if not isinstance(result, torch.Tensor) or result.shape != target.shape:
            raise ValueError("quantize must return a tensor with the input column shape")
        if result.device != target.device or result.dtype != target.dtype:
            raise ValueError("quantize output must preserve device and dtype")
        quantized[:, column] = result
        error[:, column] = weight[:, column] - result
    return quantized


@torch.no_grad()
def ldlq_quantize_blocks(
    weight: torch.Tensor,
    lower: torch.Tensor,
    quantize: Callable[[torch.Tensor, int], torch.Tensor],
    *,
    block_size: int = 8,
) -> torch.Tensor:
    """Block LDLQ recursion for a quantizer operating on adjacent columns.

    The callback receives ``(target_block, block_index)`` where target has
    shape ``[out_features, block_size]``.  Blocks are processed from right to
    left, matching the lower-triangular factor convention in Appendix B.
    """
    if not isinstance(weight, torch.Tensor) or not weight.is_floating_point():
        raise TypeError("weight must be a floating-point tensor")
    if weight.ndim != 2 or block_size <= 0 or weight.shape[1] % block_size:
        raise ValueError("weight must be a matrix whose in_features are divisible by block_size")
    _validate_square_float(lower, "lower")
    if lower.shape[0] != weight.shape[1]:
        raise ValueError("lower size must match weight in_features")
    if lower.device != weight.device or lower.dtype != weight.dtype:
        raise ValueError("lower must match weight device and dtype")
    block_count = weight.shape[1] // block_size
    block_view = lower.reshape(block_count, block_size, block_count, block_size)
    diagonal = block_view[torch.arange(block_count), :, torch.arange(block_count), :]
    identity = torch.eye(block_size, dtype=lower.dtype, device=lower.device).expand_as(diagonal)
    if not torch.allclose(lower, torch.tril(lower)) or not torch.allclose(diagonal, identity):
        raise ValueError("lower must be block-unit lower triangular")

    quantized = torch.empty_like(weight)
    error = torch.zeros_like(weight)
    for block in range(block_count - 1, -1, -1):
        start, end = block * block_size, (block + 1) * block_size
        if end < weight.shape[1]:
            feedback = error[:, end:] @ lower[end:, start:end]
        else:
            feedback = torch.zeros(
                (weight.shape[0], block_size), dtype=weight.dtype, device=weight.device
            )
        target = weight[:, start:end] + feedback
        result = quantize(target, block)
        if not isinstance(result, torch.Tensor) or result.shape != target.shape:
            raise ValueError("quantize must preserve the target block shape")
        if result.device != target.device or result.dtype != target.dtype:
            raise ValueError("quantize output must preserve device and dtype")
        quantized[:, start:end] = result
        error[:, start:end] = weight[:, start:end] - result
    return quantized


def hessian_proxy_loss(weight: torch.Tensor, quantized: torch.Tensor, hessian: torch.Tensor) -> torch.Tensor:
    """Return ``trace((W-Q) H (W-Q).T)`` as the LDLQ calibration objective."""
    if weight.shape != quantized.shape or weight.ndim != 2:
        raise ValueError("weight and quantized must have the same matrix shape")
    _validate_square_float(hessian, "hessian")
    if hessian.shape[0] != weight.shape[1]:
        raise ValueError("hessian size must match weight in_features")
    error = weight - quantized
    return torch.sum((error @ hessian) * error)


def _validate_square_float(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    if value.ndim != 2 or value.shape[0] != value.shape[1] or value.shape[0] == 0:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must contain only finite values")
