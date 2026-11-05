"""Correctness-first batched append controller for paged NQKV caches."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .nqkv_hot_cache import NQKVHotColdCache


_INACTIVE_ROW = object()


@dataclass(frozen=True)
class BatchedNQKVWriteResult:
    appended_tokens: int
    flushed_pages: int
    active_slots: tuple[int, ...]


class BatchedNQKVWriter:
    """Append a ragged batch after reserving every required physical page.

    This first implementation deliberately reuses the validated one-page E8
    encoder. It establishes allocation and lifecycle semantics before a later
    profiler-guided multi-page CUDA/Triton fusion changes launch structure.
    """

    def __init__(self, cache: NQKVHotColdCache) -> None:
        if not isinstance(cache, NQKVHotColdCache):
            raise TypeError("cache must be an NQKVHotColdCache")
        self.cache = cache

    def append(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        slot_ids: torch.Tensor,
        write_mask: torch.Tensor,
    ) -> BatchedNQKVWriteResult:
        self._validate_inputs(key, value, slot_ids, write_mask)
        slots = tuple(int(slot) for slot in slot_ids.detach().cpu().tolist())
        masks = write_mask.detach().cpu()
        row_counts = tuple(int(row.sum().item()) for row in masks)
        active_slots = tuple(
            slot for slot, count in zip(slots, row_counts, strict=True) if count
        )
        if len(set(active_slots)) != len(active_slots):
            raise ValueError("active slot_ids must be unique within one batched append")
        counts: list[int] = []
        sequence_ids = []
        reservations: list[tuple] = []
        total_flushes = 0

        # Validate prefix masks and reserve all pages before mutating hot data.
        try:
            for row, slot in enumerate(slots):
                row_mask = masks[row]
                count = row_counts[row]
                if count and not bool(row_mask[:count].all().item()):
                    raise ValueError("each write_mask row must be a contiguous prefix")
                if bool(row_mask[count:].any().item()):
                    raise ValueError("each write_mask row must be a contiguous prefix")
                if not count:
                    counts.append(0)
                    sequence_ids.append(_INACTIVE_ROW)
                    reservations.append(())
                    continue
                sequence_id = self.cache.cold_cache.sequence_id(slot)
                current = self.cache.hot_length(slot)
                flush_count = (current + count) // self.cache.cold_cache.pool.page_size
                handles = tuple(
                    self.cache.cold_cache.reserve_page(sequence_id)
                    for _ in range(flush_count)
                )
                counts.append(count)
                sequence_ids.append(sequence_id)
                reservations.append(handles)
                total_flushes += flush_count
        except Exception:
            self._cancel_reservations(sequence_ids, reservations)
            raise

        try:
            for row, (sequence_id, count, handles) in enumerate(
                zip(sequence_ids, counts, reservations, strict=True)
            ):
                if count:
                    assert sequence_id is not _INACTIVE_ROW
                    self.cache.append_preallocated(
                        sequence_id,
                        key[row, :count],
                        value[row, :count],
                        reserved_handles=handles,
                    )
        except Exception:
            self._cancel_reservations(sequence_ids, reservations)
            raise

        return BatchedNQKVWriteResult(
            appended_tokens=sum(counts),
            flushed_pages=total_flushes,
            active_slots=tuple(
                slot for slot, count in zip(slots, counts, strict=True) if count
            ),
        )

    def append_dense_empty(
        self, key: torch.Tensor, value: torch.Tensor
    ) -> BatchedNQKVWriteResult:
        """Bulk-fill the fixed identity-slot batch before any page is full."""
        if key.ndim != 4 or key.shape != value.shape:
            raise ValueError("key and value must share [batch, tokens, heads, dim]")
        if key.shape[0] != self.cache.cold_cache.max_sequences:
            raise ValueError("dense empty prefill must cover the full capacity")
        self.cache.append_dense_empty(key, value)
        return BatchedNQKVWriteResult(
            appended_tokens=key.shape[0] * key.shape[1],
            flushed_pages=0,
            active_slots=tuple(range(key.shape[0])),
        )

    def _cancel_reservations(self, sequence_ids: list, reservations: list[tuple]) -> None:
        for sequence_id, handles in zip(sequence_ids, reservations):
            if sequence_id is _INACTIVE_ROW:
                continue
            outstanding = set(self.cache.cold_cache.reserved_handles(sequence_id))
            for handle in handles:
                if handle in outstanding:
                    self.cache.cold_cache.cancel_reserved_page(sequence_id, handle)

    def _validate_inputs(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        slot_ids: torch.Tensor,
        write_mask: torch.Tensor,
    ) -> None:
        pool = self.cache.cold_cache.pool
        if not all(isinstance(tensor, torch.Tensor) for tensor in (key, value, slot_ids, write_mask)):
            raise TypeError("key, value, slot_ids, and write_mask must be tensors")
        if key.ndim != 4 or key.shape != value.shape:
            raise ValueError("key and value must share [batch, tokens, kv_heads, head_dim]")
        if tuple(key.shape[2:]) != (pool.num_kv_heads, pool.head_dim):
            raise ValueError("key/value head shape does not match the NQKV pool")
        if slot_ids.dtype != torch.int32 or slot_ids.shape != (key.shape[0],):
            raise ValueError("slot_ids must be int32 with shape [batch]")
        if write_mask.dtype != torch.bool or write_mask.shape != key.shape[:2]:
            raise ValueError("write_mask must be bool with shape [batch, tokens]")
        tensors = (key, value, slot_ids, write_mask)
        if any(tensor.device != pool.device for tensor in tensors):
            raise ValueError("all batched writer inputs must be on the pool device")
        if key.dtype != pool.output_dtype or value.dtype != pool.output_dtype:
            raise ValueError("key and value must match the NQKV pool dtype")
