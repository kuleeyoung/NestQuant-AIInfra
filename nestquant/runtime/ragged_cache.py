"""Per-request position control for fixed-address Transformer KV caches."""

from __future__ import annotations

import types
from typing import Any

import torch


class RaggedCacheController:
    """Install graph-safe per-row writes on an initialized ``StaticCache``.

    Positions and write masks have shape ``[batch, query_length]``. Cache
    tensors keep their fixed addresses; ``scatter_`` updates every row while
    inactive rows scatter their previously stored values back unchanged.
    """

    def __init__(self, cache: Any, *, batch_size: int, max_cache_len: int) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(max_cache_len, bool) or not isinstance(max_cache_len, int) or max_cache_len < 1:
            raise ValueError("max_cache_len must be a positive integer")
        self.cache = cache
        self.batch_size = batch_size
        self.max_cache_len = max_cache_len
        self._positions: torch.Tensor | None = None
        self._write_mask: torch.Tensor | None = None
        self._installed = False

    @property
    def installed(self) -> bool:
        return self._installed

    def set_write_state(
        self, positions: torch.Tensor, write_mask: torch.Tensor, *, validate_bounds: bool = True
    ) -> None:
        if positions.dtype != torch.long:
            raise TypeError("positions must be a torch.long tensor")
        if write_mask.dtype != torch.bool:
            raise TypeError("write_mask must be a torch.bool tensor")
        if positions.ndim != 2 or positions.shape[0] != self.batch_size:
            raise ValueError("positions must have shape [batch_size, query_length]")
        if write_mask.shape != positions.shape:
            raise ValueError("write_mask must have the same shape as positions")
        if positions.device != write_mask.device:
            raise ValueError("positions and write_mask must share a device")
        if validate_bounds:
            if positions.numel() and (positions.min().item() < 0 or positions.max().item() >= self.max_cache_len):
                raise ValueError("positions are outside the cache")
        self._positions = positions
        self._write_mask = write_mask

    def _update(self, layer, key_states, value_states, *args, **kwargs):
        del args, kwargs
        positions = self._positions
        write_mask = self._write_mask
        if positions is None or write_mask is None:
            raise RuntimeError("write state was not set")
        if key_states.shape[0] != self.batch_size or key_states.shape[-2] != positions.shape[1]:
            raise ValueError("key/value shape does not match the configured write state")

        key_index = positions[:, None, :, None].expand(
            self.batch_size, layer.keys.shape[1], positions.shape[1], layer.keys.shape[3]
        )
        value_index = positions[:, None, :, None].expand(
            self.batch_size, layer.values.shape[1], positions.shape[1], layer.values.shape[3]
        )
        active = write_mask[:, None, :, None]
        old_keys = layer.keys.gather(2, key_index)
        old_values = layer.values.gather(2, value_index)
        layer.keys.scatter_(2, key_index, torch.where(active, key_states, old_keys))
        layer.values.scatter_(2, value_index, torch.where(active, value_states, old_values))
        return layer.keys, layer.values

    def install(self) -> None:
        if self._installed:
            return
        if not self.cache.layers or any(not getattr(layer, "is_initialized", False) for layer in self.cache.layers):
            raise RuntimeError("StaticCache layers must be initialized before installing ragged updates")
        for layer in self.cache.layers:
            if layer.keys.shape[0] != self.batch_size or layer.keys.shape[2] != self.max_cache_len:
                raise ValueError("initialized cache shape does not match the controller")
            layer.update = types.MethodType(self._update, layer)
        self._installed = True

    def attention_mask(
        self,
        query_positions: torch.Tensor,
        *,
        valid_lengths: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build additive ``[B,1,Q,K]`` masks for independent row lengths."""
        if query_positions.dtype != torch.long or query_positions.ndim != 2:
            raise TypeError("query_positions must be a rank-2 torch.long tensor")
        if query_positions.shape[0] != self.batch_size:
            raise ValueError("query_positions batch does not match the controller")
        if valid_lengths.dtype != torch.long or valid_lengths.shape != (self.batch_size,):
            raise ValueError("valid_lengths must be torch.long with shape [batch_size]")
        if query_positions.device != valid_lengths.device:
            raise ValueError("query_positions and valid_lengths must share a device")
        key_positions = torch.arange(
            self.max_cache_len, device=query_positions.device, dtype=torch.long
        ).view(1, 1, 1, -1)
        causal = key_positions <= query_positions[:, None, :, None]
        within_length = key_positions < valid_lengths[:, None, None, None]
        allowed = causal & within_length
        mask = torch.zeros(allowed.shape, dtype=dtype, device=query_positions.device)
        return mask.masked_fill(~allowed, torch.finfo(dtype).min)

    def clear_slots(self, slot_mask: torch.Tensor) -> None:
        """Zero selected batch rows without replacing fixed cache tensors."""
        if slot_mask.dtype != torch.bool or slot_mask.shape != (self.batch_size,):
            raise ValueError("slot_mask must be torch.bool with shape [batch_size]")
        for layer in self.cache.layers:
            if getattr(layer, "is_initialized", False):
                keep = (~slot_mask).view(self.batch_size, 1, 1, 1)
                layer.keys.mul_(keep)
                layer.values.mul_(keep)
