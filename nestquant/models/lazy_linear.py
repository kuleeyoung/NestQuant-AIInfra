"""Inference-only Linear backed by one lazily decoded NQMC weight record."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from ..core.model_checkpoint_file import NQMCFileReader
from ..core.rotation import hadamard


CachePolicy = Literal["none", "persistent"]


class LazyNQMCLinear(nn.Module):
    """Decode one packed NQMC weight on demand, then run a floating-point GEMM.

    This is a bounded-scope runtime prototype, not a packed low-bit kernel.  A
    ``none`` cache decodes the weight for every call and releases the temporary
    after the call.  A ``persistent`` cache retains the decoded weight after
    the first call.  Paper-coordinate checkpoints can request the matching
    online input Hadamard transform with ``input_hadamard=True``.
    """

    def __init__(
        self,
        reader: NQMCFileReader,
        weight_name: str,
        bias: torch.Tensor | None = None,
        *,
        input_hadamard: bool = False,
        compute_dtype: torch.dtype = torch.bfloat16,
        output_dtype: torch.dtype | None = None,
        rows_per_chunk: int | None = 128,
        cache_policy: CachePolicy = "none",
    ) -> None:
        super().__init__()
        if not isinstance(reader, NQMCFileReader):
            raise TypeError("reader must be an NQMCFileReader")
        if cache_policy not in {"none", "persistent"}:
            raise ValueError("cache_policy must be 'none' or 'persistent'")
        if compute_dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
            raise ValueError("compute_dtype must be a floating-point dtype")
        if output_dtype is not None and output_dtype not in {
            torch.float16, torch.bfloat16, torch.float32, torch.float64,
        }:
            raise ValueError("output_dtype must be a floating-point dtype or None")
        if rows_per_chunk is not None and (
            isinstance(rows_per_chunk, bool)
            or not isinstance(rows_per_chunk, int)
            or rows_per_chunk < 1
        ):
            raise ValueError("rows_per_chunk must be a positive integer or None")

        entry = reader.get_entry(weight_name)
        if entry.kind != "nestquant":
            raise ValueError("LazyNQMCLinear requires a packed NestQuant weight record")
        if len(entry.shape) != 2 or not all(entry.shape):
            raise ValueError("weight record must be nonempty with shape [out_features, in_features]")
        out_features, in_features = entry.shape
        if input_hadamard and in_features & (in_features - 1):
            raise ValueError("Hadamard input transform requires power-of-two in_features")
        if bias is not None:
            if not isinstance(bias, torch.Tensor) or not bias.is_floating_point():
                raise TypeError("bias must be a floating-point tensor")
            if tuple(bias.shape) != (out_features,):
                raise ValueError("bias must match out_features")

        self.reader = reader
        self.weight_name = weight_name
        self.in_features = in_features
        self.out_features = out_features
        self.input_hadamard = input_hadamard
        self.compute_dtype = compute_dtype
        self.output_dtype = output_dtype
        self.rows_per_chunk = rows_per_chunk
        self.cache_policy: CachePolicy = cache_policy
        self.register_buffer(
            "bias", None if bias is None else bias.detach().clone().to(compute_dtype)
        )
        self.register_buffer("_cached_weight", None, persistent=False)

    @property
    def cached(self) -> bool:
        """Whether a decoded weight is currently retained."""
        return self._cached_weight is not None

    @property
    def cached_weight_bytes(self) -> int:
        """Device bytes retained by the persistent cache."""
        if self._cached_weight is None:
            return 0
        return self._cached_weight.numel() * self._cached_weight.element_size()

    def clear_cache(self) -> None:
        """Release a persistent decoded weight."""
        self._cached_weight = None

    def _weight(self, device: torch.device) -> torch.Tensor:
        cached = self._cached_weight
        if cached is not None and cached.device == device and cached.dtype == self.compute_dtype:
            return cached
        # A module moved after caching must not retain an obsolete device copy.
        if cached is not None:
            self.clear_cache()
        decoded = self.reader.decode_tensor(
            self.weight_name,
            device=device,
            output_dtype=self.compute_dtype,
            rows_per_chunk=self.rows_per_chunk,
        )
        if self.cache_policy == "persistent":
            self._cached_weight = decoded
        return decoded

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not isinstance(inputs, torch.Tensor) or not inputs.is_floating_point():
            raise TypeError("inputs must be a floating-point tensor")
        if inputs.ndim == 0 or inputs.shape[-1] != self.in_features:
            actual = None if inputs.ndim == 0 else inputs.shape[-1]
            raise ValueError(
                f"input feature dimension {actual} does not match {self.in_features}"
            )
        original_dtype = inputs.dtype
        compute_inputs = inputs.to(self.compute_dtype)
        if self.input_hadamard:
            compute_inputs = hadamard(compute_inputs)
        weight = self._weight(inputs.device)
        bias = None if self.bias is None else self.bias.to(inputs.device)
        outputs = F.linear(compute_inputs, weight, bias)
        return outputs.to(original_dtype if self.output_dtype is None else self.output_dtype)
