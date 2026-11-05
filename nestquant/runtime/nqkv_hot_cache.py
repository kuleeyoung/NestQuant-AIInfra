"""BF16 hot tails that flush full pages into the packed NQKV pool."""

from __future__ import annotations

from typing import Hashable

import torch

from .nqkv_page import dequantize_nqkv_page_triton, quantize_nqkv_page_cuda
from .nqkv_pool import NQKVPageHandle
from .paged_nqkv_cache import PagedNQKVCache


class NQKVHotColdCache:
    """Append-only per-sequence hot buffers over a packed cold-page cache."""

    def __init__(self, cold_cache: PagedNQKVCache) -> None:
        if not isinstance(cold_cache, PagedNQKVCache):
            raise TypeError("cold_cache must be a PagedNQKVCache")
        self.cold_cache = cold_cache
        pool = cold_cache.pool
        shape = (
            cold_cache.max_sequences, pool.page_size,
            pool.num_kv_heads, pool.head_dim,
        )
        self.hot_keys = torch.zeros(shape, device=pool.device, dtype=pool.output_dtype)
        self.hot_values = torch.zeros_like(self.hot_keys)
        # The hot/cold controller and the future batched attention kernels use
        # one canonical fixed-address metadata tensor.
        self.hot_lengths = cold_cache.hot_lengths
        self.graph_target_page_ids = torch.zeros(
            cold_cache.max_sequences,
            device=cold_cache.pool.device,
            dtype=torch.int32,
        )
        self._lengths = [0] * cold_cache.max_sequences
        self._finalized: set[Hashable] = set()

    def register(self, sequence_id: Hashable, *, slot: int | None = None) -> int:
        """Register a sequence in the same physical slot as the scheduler."""
        slot = self.cold_cache.register(sequence_id, slot=slot)
        self.hot_keys[slot].zero_()
        self.hot_values[slot].zero_()
        self.hot_lengths[slot] = 0
        self.graph_target_page_ids[slot] = 0
        self._lengths[slot] = 0
        self._finalized.discard(sequence_id)
        return slot

    def append(self, sequence_id: Hashable, key: torch.Tensor, value: torch.Tensor) -> None:
        self._append(sequence_id, key, value, reserved_handles=())

    def append_preallocated(
        self,
        sequence_id: Hashable,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        reserved_handles: tuple[NQKVPageHandle, ...],
    ) -> None:
        """Append using pages reserved before any hot-tail state is modified."""
        slot = self.cold_cache.slot(sequence_id)
        self._validate_append(key, value)
        required = (self._lengths[slot] + key.shape[0]) // self.cold_cache.pool.page_size
        if len(reserved_handles) != required:
            raise ValueError("reserved handle count does not match required flushes")
        owned_tail = (
            tuple(self.cold_cache.reserved_handles(sequence_id)[-required:])
            if required else ()
        )
        if owned_tail != reserved_handles:
            raise RuntimeError("reserved handles do not belong to this sequence")
        self._append(sequence_id, key, value, reserved_handles=reserved_handles)

    def append_dense_empty(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Bulk-fill every active empty slot with one sub-page dense prefill."""
        expected_tail = self.hot_keys.shape[2:]
        if key.ndim != 4 or key.shape != value.shape:
            raise ValueError("dense key/value must share [capacity, tokens, heads, dim]")
        if key.shape[0] != self.hot_keys.shape[0] or key.shape[2:] != expected_tail:
            raise ValueError("dense key/value shape disagrees with the hot cache")
        token_count = key.shape[1]
        if token_count < 1 or token_count >= self.cold_cache.pool.page_size:
            raise ValueError("dense empty prefill must fit strictly inside one hot page")
        for slot in range(key.shape[0]):
            sequence_id = self.cold_cache.sequence_id(slot)
            if self._lengths[slot] or self.cold_cache.total_tokens(sequence_id):
                raise RuntimeError("dense empty prefill requires empty active slots")
        self.hot_keys[:, :token_count].copy_(key)
        self.hot_values[:, :token_count].copy_(value)
        self.hot_lengths.fill_(token_count)
        self.cold_cache.sequence_lengths.fill_(token_count)
        for slot in range(key.shape[0]):
            self._lengths[slot] = token_count

    def _append(
        self,
        sequence_id: Hashable,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        reserved_handles: tuple[NQKVPageHandle, ...],
    ) -> None:
        if sequence_id in self._finalized:
            raise RuntimeError("cannot append to a finalized NQKV sequence")
        slot = self.cold_cache.slot(sequence_id)
        self._validate_append(key, value)
        offset = 0
        reservation_index = 0
        while offset < key.shape[0]:
            hot_length = self._lengths[slot]
            available = self.cold_cache.pool.page_size - hot_length
            take = min(available, key.shape[0] - offset)
            self.hot_keys[slot, hot_length:hot_length + take].copy_(key[offset:offset + take])
            self.hot_values[slot, hot_length:hot_length + take].copy_(value[offset:offset + take])
            hot_length += take
            offset += take
            self._lengths[slot] = hot_length
            self.hot_lengths[slot] = hot_length
            self.cold_cache.sequence_lengths[slot] = (
                self.cold_cache.total_tokens(sequence_id) + hot_length
            )
            if hot_length == self.cold_cache.pool.page_size:
                handle = (
                    reserved_handles[reservation_index]
                    if reservation_index < len(reserved_handles) else None
                )
                self._flush(sequence_id, slot, hot_length, handle=handle)
                reservation_index += handle is not None

    def finalize(self, sequence_id: Hashable) -> None:
        """Flush a partial tail so a completed sequence is fully packed."""
        if sequence_id in self._finalized:
            return
        slot = self.cold_cache.slot(sequence_id)
        if self._lengths[slot]:
            self._flush(sequence_id, slot, self._lengths[slot])
        self._finalized.add(sequence_id)

    def release(self, sequence_id: Hashable) -> None:
        slot = self.cold_cache.slot(sequence_id)
        self.cold_cache.release(sequence_id)
        self.hot_keys[slot].zero_()
        self.hot_values[slot].zero_()
        self.hot_lengths[slot] = 0
        self.graph_target_page_ids[slot] = 0
        self._lengths[slot] = 0
        self._finalized.discard(sequence_id)

    def hot_view(self, sequence_id: Hashable) -> tuple[torch.Tensor, torch.Tensor]:
        slot = self.cold_cache.slot(sequence_id)
        length = self._lengths[slot]
        return self.hot_keys[slot, :length], self.hot_values[slot, :length]

    def total_tokens(self, sequence_id: Hashable) -> int:
        slot = self.cold_cache.slot(sequence_id)
        return self.cold_cache.total_tokens(sequence_id) + self._lengths[slot]

    def hot_length(self, slot: int) -> int:
        """Return the host-side hot-tail length for one active slot."""
        self.cold_cache.sequence_id(slot)
        return self._lengths[slot]

    def synchronize_host_length(self, slot: int, length: int) -> None:
        """Update host control state after a CUDA Graph mutated hot metadata."""
        self.cold_cache.sequence_id(slot)
        if isinstance(length, bool) or not isinstance(length, int):
            raise TypeError("length must be an integer")
        if not 0 <= length <= self.cold_cache.pool.page_size:
            raise ValueError("length is outside the hot page")
        self._lengths[slot] = length

    def materialize(self, sequence_id: Hashable) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct packed pages and append the resident BF16 hot tail."""
        key_parts: list[torch.Tensor] = []
        value_parts: list[torch.Tensor] = []
        for handle in self.cold_cache.handles(sequence_id):
            key, value = dequantize_nqkv_page_triton(
                self.cold_cache.pool.prepared(handle)
            )
            key_parts.append(key)
            value_parts.append(value)
        hot_key, hot_value = self.hot_view(sequence_id)
        if hot_key.shape[0]:
            key_parts.append(hot_key)
            value_parts.append(hot_value)
        if not key_parts:
            pool = self.cold_cache.pool
            empty = torch.empty(
                (0, pool.num_kv_heads, pool.head_dim),
                device=pool.device, dtype=pool.output_dtype,
            )
            return empty, empty.clone()
        return torch.cat(key_parts), torch.cat(value_parts)

    def buffer_pointers(self) -> tuple[int, int, int]:
        return self.hot_keys.data_ptr(), self.hot_values.data_ptr(), self.hot_lengths.data_ptr()

    def _flush(
        self,
        sequence_id: Hashable,
        slot: int,
        length: int,
        *,
        handle: NQKVPageHandle | None = None,
    ) -> None:
        pool = self.cold_cache.pool
        kwargs = (
            {"betas": pool.betas}
            if pool.format_version == 1 else
            {"key_betas": pool.betas, "value_betas": pool.value_beta_values}
        )
        prepared = quantize_nqkv_page_cuda(
            self.hot_keys[slot, :length], self.hot_values[slot, :length],
            page_size=pool.page_size, q=pool.q, **kwargs,
        )
        if handle is None:
            self.cold_cache.append_prepared_page(sequence_id, prepared)
        else:
            self.cold_cache.commit_prepared_page(sequence_id, handle, prepared)
        self.hot_keys[slot].zero_()
        self.hot_values[slot].zero_()
        self._lengths[slot] = 0
        self.hot_lengths[slot] = 0
        self.cold_cache.sequence_lengths[slot] = self.cold_cache.total_tokens(sequence_id)

    def _validate_append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        pool = self.cold_cache.pool
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise TypeError("key and value must be torch tensors")
        expected_tail = (pool.num_kv_heads, pool.head_dim)
        if key.ndim != 3 or key.shape != value.shape or tuple(key.shape[1:]) != expected_tail:
            raise ValueError("key and value must share [tokens, kv_heads, head_dim]")
        if key.shape[0] < 1:
            raise ValueError("append must contain at least one token")
        if (
            key.device != pool.device or value.device != pool.device
            or key.dtype != pool.output_dtype or value.dtype != pool.output_dtype
        ):
            raise ValueError("key and value must match the pool device and dtype")
