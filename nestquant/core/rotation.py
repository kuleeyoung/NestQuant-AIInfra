"""Orthogonal signed Hadamard transforms for weight-space ablations.

For row vectors, R = D H: forward(x) = x D H and inverse(y) = y H D.
H is normalized and symmetric. This is not whole-model rotation fusion.
"""

import math

import torch


def hadamard(x: torch.Tensor) -> torch.Tensor:
    """Normalized Sylvester transform along the final, power-of-two axis."""
    if x.ndim == 0 or not x.is_floating_point():
        raise ValueError("expected a floating-point tensor with a feature axis")
    n = x.shape[-1]
    if n < 1 or n & (n - 1):
        raise ValueError("feature dimension must be a positive power of two")
    original_shape = x.shape
    y = x.reshape(-1, n)
    width = 1
    while width < n:
        pairs = y.reshape(-1, n // (2 * width), 2, width)
        left, right = pairs[:, :, 0], pairs[:, :, 1]
        y = torch.stack((left + right, left - right), dim=2).reshape(-1, n)
        width *= 2
    return (y / math.sqrt(n)).reshape(original_shape)


def signed_hadamard(x: torch.Tensor, signs: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
    """Apply x D H, or its inverse x H D, without forming a dense matrix."""
    if x.ndim == 0 or signs.ndim != 1 or signs.shape[0] != x.shape[-1]:
        raise ValueError("signs must be a vector matching the feature dimension")
    if signs.device != x.device or not ((signs == 1) | (signs == -1)).all().item():
        raise ValueError("signs must contain +/-1 on the input device")
    return hadamard(x) * signs if inverse else hadamard(x * signs)


def rotate_covariance(hessian: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Return ``R.T @ H @ R`` for the row transform ``x -> x @ R``.

    If ``H = X.T @ X / N``, the returned matrix is exactly the covariance of
    ``signed_hadamard(X, signs)`` up to floating-point roundoff.  Applying the
    transform on both matrix axes avoids materializing the dense Hadamard
    matrix.
    """
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError("hessian must be square")
    if not hessian.is_floating_point():
        raise TypeError("hessian must be floating point")
    if signs.shape != (hessian.shape[0],) or signs.device != hessian.device:
        raise ValueError("signs must match the hessian feature axis and device")
    # First multiply H on the right by R.  Applying the same row transform to
    # its transpose then gives R.T @ H.T @ R, equal to R.T @ H @ R for a
    # covariance matrix.  Symmetrization in LDLQ also removes roundoff skew.
    right_rotated = signed_hadamard(hessian, signs)
    return signed_hadamard(right_rotated.T, signs)
