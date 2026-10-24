"""Metrics and validation helpers for fixed-upstream numerical alignment."""

from __future__ import annotations

import math

import torch


def tensor_alignment(reference: torch.Tensor, actual: torch.Tensor) -> dict:
    """Return auditable absolute and relative differences for two tensors."""
    if not isinstance(reference, torch.Tensor) or not isinstance(actual, torch.Tensor):
        raise TypeError("reference and actual must be tensors")
    if reference.shape != actual.shape:
        raise ValueError("reference and actual shapes must match")
    if not reference.is_floating_point() or not actual.is_floating_point():
        raise TypeError("alignment metrics require floating-point tensors")
    if not torch.isfinite(reference).all().item() or not torch.isfinite(actual).all().item():
        raise ValueError("alignment tensors must be finite")
    delta = actual.to(torch.float64) - reference.to(torch.float64)
    reference64 = reference.to(torch.float64)
    squared_error = float(delta.square().sum().item())
    reference_energy = float(reference64.square().sum().item())
    return {
        "shape": list(reference.shape),
        "max_abs_error": float(delta.abs().max().item()) if delta.numel() else 0.0,
        "rmse": math.sqrt(squared_error / delta.numel()) if delta.numel() else 0.0,
        "relative_l2_error": (
            math.sqrt(squared_error / reference_energy) if reference_energy else 0.0
        ),
        "reference_l2_norm": math.sqrt(reference_energy),
    }

