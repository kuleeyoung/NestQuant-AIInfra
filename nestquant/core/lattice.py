"""Readable nearest-point reference for the eight-dimensional E8 lattice.

E8 is represented as the union of two cosets:

    D8 = {z in Z^8 : sum(z) is even}
    D8 + (1/2, ..., 1/2)

This implementation prioritizes a clear correctness reference over speed.
"""

from __future__ import annotations

import torch

E8_DIMENSION = 8


def _nearest_even_sum_integer(points: torch.Tensor) -> torch.Tensor:
    """Round to the nearest integer vector whose coordinate sum is even."""
    rounded = torch.round(points)
    odd_sum = torch.remainder(rounded.sum(dim=-1, keepdim=True), 2) != 0

    # If the rounded sum is odd, flip the coordinate closest to a half-integer.
    # Its absolute residual is largest, so changing it costs the least squared
    # distance. The sign selects whether to round that coordinate up or down.
    residual = points - rounded
    coordinate = residual.abs().argmax(dim=-1, keepdim=True)
    direction = torch.where(
        residual.gather(-1, coordinate) >= 0,
        torch.ones_like(coordinate, dtype=points.dtype),
        -torch.ones_like(coordinate, dtype=points.dtype),
    )
    correction = torch.zeros_like(rounded).scatter(-1, coordinate, direction)
    return rounded + correction * odd_sum.to(points.dtype)


def nearest_e8(
    x: torch.Tensor, *, tie_break: str = "upstream_half"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find the nearest E8 point for every vector in ``x``.

    Args:
        x: A floating-point tensor with final dimension 8. Leading dimensions
            form a batch of independent vectors.

    Returns:
        ``(point, squared_distance)``. ``point`` has the input shape, dtype,
        and device. ``squared_distance`` has shape ``x.shape[:-1]``.

    Raises:
        TypeError: If ``x`` is not a floating-point tensor.
        ValueError: If the final dimension is not 8 or ``x`` is non-finite.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if not x.is_floating_point():
        raise TypeError("x must have a floating-point dtype")
    if x.ndim == 0 or x.shape[-1] != E8_DIMENSION:
        raise ValueError(f"the final dimension of x must be {E8_DIMENSION}")
    if not torch.isfinite(x).all().item():
        raise ValueError("x must contain only finite values")
    if tie_break not in {"upstream_half", "legacy_integer"}:
        raise ValueError("unsupported E8 coset tie break")

    half = torch.full((), 0.5, dtype=x.dtype, device=x.device)
    integer_coset = _nearest_even_sum_integer(x)
    half_integer_coset = _nearest_even_sum_integer(x - half) + half

    integer_error = (x - integer_coset).square().sum(dim=-1)
    half_error = (x - half_integer_coset).square().sum(dim=-1)
    # Match the fixed upstream ``encode_e8_fast`` tie policy: when the two E8
    # cosets have exactly equal distance, choose the half-integer coset.
    # This matters during nested decoding because residue representatives can
    # land exactly on a Voronoi boundary after division by q.
    choose_integer = (
        integer_error < half_error
        if tie_break == "upstream_half"
        else integer_error <= half_error
    )
    point = torch.where(choose_integer.unsqueeze(-1), integer_coset, half_integer_coset)
    return point, torch.minimum(integer_error, half_error)
