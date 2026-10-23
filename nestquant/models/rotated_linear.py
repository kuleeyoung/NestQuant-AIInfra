"""Reference Linear module for paper Section 4.3 Hadamard coordinates."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..core.rotation import hadamard


class HadamardRotatedLinear(nn.Module):
    """Store ``W H`` and apply ``X H`` online before a Linear operation.

    For the normalized symmetric Hadamard matrix ``H``, ``H H.T = I``. Thus
    ``(X H) (W H).T = X W.T`` before quantization. This module is a correctness
    reference for the local weight/activation-pair rotation in paper Section
    4.3. It is not a fused kernel and currently supports only power-of-two input
    dimensions, which cover the Linear inputs used by Llama-3.2-1B.
    """

    def __init__(
        self,
        rotated_weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(rotated_weight, torch.Tensor) or not rotated_weight.is_floating_point():
            raise TypeError("rotated_weight must be a floating-point tensor")
        if rotated_weight.ndim != 2 or not rotated_weight.numel():
            raise ValueError("rotated_weight must be nonempty with shape [out_features, in_features]")
        in_features = rotated_weight.shape[1]
        if in_features & (in_features - 1):
            raise ValueError("in_features must be a power of two")
        if bias is not None:
            if bias.shape != (rotated_weight.shape[0],):
                raise ValueError("bias must match out_features")
            if bias.device != rotated_weight.device or bias.dtype != rotated_weight.dtype:
                raise ValueError("bias must match rotated_weight device and dtype")
        self.in_features = in_features
        self.out_features = rotated_weight.shape[0]
        self.weight = nn.Parameter(rotated_weight.detach().clone())
        self.bias = None if bias is None else nn.Parameter(bias.detach().clone())

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "HadamardRotatedLinear":
        """Create the exact rotated-coordinate representation of ``linear``."""
        if not isinstance(linear, nn.Linear):
            raise TypeError("linear must be torch.nn.Linear")
        if linear.in_features & (linear.in_features - 1):
            raise ValueError("linear.in_features must be a power of two")
        rotated_weight = hadamard(linear.weight.detach())
        bias = None if linear.bias is None else linear.bias.detach()
        result = cls(rotated_weight, bias)
        result.train(linear.training)
        return result

    def reconstructed_weight(self) -> torch.Tensor:
        """Return the current rotated weight mapped back to original coordinates."""
        return hadamard(self.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"input feature dimension {inputs.shape[-1]} does not match {self.in_features}"
            )
        return F.linear(hadamard(inputs), self.weight, self.bias)
