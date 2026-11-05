"""CUDA-Graph-safe K1 appends into fixed-address NQKV hot pages."""

from __future__ import annotations

import torch

from .nqkv_hot_cache import NQKVHotColdCache
from .triton_unpack import _TRITON_INSTALLED


if _TRITON_INSTALLED:
    import triton
    import triton.language as tl

    @triton.jit
    def _append_hot_k1_data_kernel(
        key,
        value,
        slot_ids,
        active_mask,
        cache_active,
        hot_lengths,
        hot_keys,
        hot_values,
        batch_size: tl.constexpr,
        max_sequences: tl.constexpr,
        num_kv_heads: tl.constexpr,
        page_size: tl.constexpr,
        head_dim: tl.constexpr,
        block_dim: tl.constexpr,
    ):
        combined = tl.program_id(0)
        batch_row = combined // num_kv_heads
        kv_head = combined % num_kv_heads
        requested_slot = tl.load(slot_ids + batch_row)
        slot_in_range = (requested_slot >= 0) & (requested_slot < max_sequences)
        slot = tl.maximum(0, tl.minimum(requested_slot, max_sequences - 1))
        hot_length = tl.load(hot_lengths + slot)
        active = (
            tl.load(active_mask + batch_row)
            & slot_in_range
            & tl.load(cache_active + slot)
            & (hot_length < page_size)
        )
        dimensions = tl.arange(0, block_dim)
        dimension_mask = dimensions < head_dim
        source = (batch_row * num_kv_heads + kv_head) * head_dim + dimensions
        destination = (
            ((slot * page_size + hot_length) * num_kv_heads + kv_head) * head_dim
            + dimensions
        )
        key_data = tl.load(key + source, mask=dimension_mask)
        value_data = tl.load(value + source, mask=dimension_mask)
        tl.store(hot_keys + destination, key_data, mask=active & dimension_mask)
        tl.store(hot_values + destination, value_data, mask=active & dimension_mask)

    @triton.jit
    def _append_hot_k1_metadata_kernel(
        slot_ids,
        active_mask,
        cache_active,
        hot_lengths,
        sequence_lengths,
        max_sequences: tl.constexpr,
        page_size: tl.constexpr,
    ):
        batch_row = tl.program_id(0)
        requested_slot = tl.load(slot_ids + batch_row)
        slot_in_range = (requested_slot >= 0) & (requested_slot < max_sequences)
        slot = tl.maximum(0, tl.minimum(requested_slot, max_sequences - 1))
        hot_length = tl.load(hot_lengths + slot)
        active = (
            tl.load(active_mask + batch_row)
            & slot_in_range
            & tl.load(cache_active + slot)
            & (hot_length < page_size)
        )
        tl.store(hot_lengths + slot, hot_length + 1, mask=active)
        sequence_length = tl.load(sequence_lengths + slot)
        tl.store(sequence_lengths + slot, sequence_length + 1, mask=active)

    @triton.jit
    def _append_hot_ragged_data_kernel(
        key,
        value,
        slot_ids,
        write_mask,
        cache_active,
        hot_lengths,
        hot_keys,
        hot_values,
        token_count: tl.constexpr,
        max_sequences: tl.constexpr,
        num_kv_heads: tl.constexpr,
        page_size: tl.constexpr,
        head_dim: tl.constexpr,
        block_dim: tl.constexpr,
    ):
        combined = tl.program_id(0)
        kv_head = combined % num_kv_heads
        row_token = combined // num_kv_heads
        token = row_token % token_count
        batch_row = row_token // token_count
        requested_slot = tl.load(slot_ids + batch_row)
        slot_in_range = (requested_slot >= 0) & (requested_slot < max_sequences)
        slot = tl.maximum(0, tl.minimum(requested_slot, max_sequences - 1))
        hot_length = tl.load(hot_lengths + slot)
        destination_token = hot_length + token
        active = (
            tl.load(write_mask + batch_row * token_count + token)
            & slot_in_range
            & tl.load(cache_active + slot)
            & (destination_token < page_size)
        )
        dimensions = tl.arange(0, block_dim)
        dimension_mask = dimensions < head_dim
        source = (
            ((batch_row * token_count + token) * num_kv_heads + kv_head)
            * head_dim + dimensions
        )
        destination = (
            ((slot * page_size + destination_token) * num_kv_heads + kv_head)
            * head_dim + dimensions
        )
        key_data = tl.load(key + source, mask=dimension_mask)
        value_data = tl.load(value + source, mask=dimension_mask)
        tl.store(hot_keys + destination, key_data, mask=active & dimension_mask)
        tl.store(hot_values + destination, value_data, mask=active & dimension_mask)

    @triton.jit
    def _append_hot_ragged_metadata_kernel(
        slot_ids,
        write_mask,
        cache_active,
        hot_lengths,
        sequence_lengths,
        token_count: tl.constexpr,
        count_block: tl.constexpr,
        max_sequences: tl.constexpr,
        page_size: tl.constexpr,
    ):
        batch_row = tl.program_id(0)
        requested_slot = tl.load(slot_ids + batch_row)
        slot_in_range = (requested_slot >= 0) & (requested_slot < max_sequences)
        slot = tl.maximum(0, tl.minimum(requested_slot, max_sequences - 1))
        offsets = tl.arange(0, count_block)
        valid = offsets < token_count
        row_count = tl.sum(
            tl.load(
                write_mask + batch_row * token_count + offsets,
                mask=valid,
                other=0,
            ).to(tl.int32),
            axis=0,
        )
        hot_length = tl.load(hot_lengths + slot)
        active = (
            slot_in_range
            & tl.load(cache_active + slot)
            & (row_count > 0)
            & (hot_length + row_count <= page_size)
        )
        tl.store(hot_lengths + slot, hot_length + row_count, mask=active)
        sequence_length = tl.load(sequence_lengths + slot)
        tl.store(
            sequence_lengths + slot,
            sequence_length + row_count,
            mask=active,
        )

    @triton.jit
    def _append_hot_ragged_segment_data_kernel(
        key,
        value,
        slot_ids,
        source_starts,
        segment_counts,
        cache_active,
        hot_lengths,
        hot_keys,
        hot_values,
        token_count: tl.constexpr,
        max_sequences: tl.constexpr,
        num_kv_heads: tl.constexpr,
        page_size: tl.constexpr,
        head_dim: tl.constexpr,
        block_dim: tl.constexpr,
    ):
        combined = tl.program_id(0)
        kv_head = combined % num_kv_heads
        row_token = combined // num_kv_heads
        local_token = row_token % token_count
        batch_row = row_token // token_count
        requested_slot = tl.load(slot_ids + batch_row)
        slot_in_range = (requested_slot >= 0) & (requested_slot < max_sequences)
        slot = tl.maximum(0, tl.minimum(requested_slot, max_sequences - 1))
        source_start = tl.load(source_starts + batch_row)
        segment_count = tl.load(segment_counts + batch_row)
        source_token = source_start + local_token
        hot_length = tl.load(hot_lengths + slot)
        destination_token = hot_length + local_token
        active = (
            slot_in_range
            & tl.load(cache_active + slot)
            & (local_token < segment_count)
            & (source_token < token_count)
            & (destination_token < page_size)
        )
        dimensions = tl.arange(0, block_dim)
        dimension_mask = dimensions < head_dim
        source = (
            ((batch_row * token_count + source_token) * num_kv_heads + kv_head)
            * head_dim + dimensions
        )
        destination = (
            ((slot * page_size + destination_token) * num_kv_heads + kv_head)
            * head_dim + dimensions
        )
        key_data = tl.load(key + source, mask=active & dimension_mask, other=0.0)
        value_data = tl.load(value + source, mask=active & dimension_mask, other=0.0)
        tl.store(hot_keys + destination, key_data, mask=active & dimension_mask)
        tl.store(hot_values + destination, value_data, mask=active & dimension_mask)

    @triton.jit
    def _append_hot_ragged_segment_metadata_kernel(
        slot_ids,
        segment_counts,
        cache_active,
        hot_lengths,
        sequence_lengths,
        max_sequences: tl.constexpr,
        page_size: tl.constexpr,
    ):
        batch_row = tl.program_id(0)
        requested_slot = tl.load(slot_ids + batch_row)
        slot_in_range = (requested_slot >= 0) & (requested_slot < max_sequences)
        slot = tl.maximum(0, tl.minimum(requested_slot, max_sequences - 1))
        segment_count = tl.load(segment_counts + batch_row)
        hot_length = tl.load(hot_lengths + slot)
        active = (
            slot_in_range
            & tl.load(cache_active + slot)
            & (segment_count > 0)
            & (hot_length + segment_count <= page_size)
        )
        tl.store(hot_lengths + slot, hot_length + segment_count, mask=active)
        sequence_length = tl.load(sequence_lengths + slot)
        tl.store(
            sequence_lengths + slot,
            sequence_length + segment_count,
            mask=active,
        )
else:
    triton = None
    _append_hot_k1_data_kernel = None
    _append_hot_k1_metadata_kernel = None
    _append_hot_ragged_data_kernel = None
    _append_hot_ragged_metadata_kernel = None
    _append_hot_ragged_segment_data_kernel = None
    _append_hot_ragged_segment_metadata_kernel = None


def append_hot_k1_graph_safe(
    key: torch.Tensor,
    value: torch.Tensor,
    cache: NQKVHotColdCache,
    *,
    slot_ids: torch.Tensor,
    active_mask: torch.Tensor,
) -> None:
    """Append one token per active row without tensor-to-host synchronization.

    This operation intentionally refuses a full hot page at the checked API
    boundary. The low-level launch remains capture-safe and masks such rows;
    the full-page transition will be handled by a separate packed-page graph
    stage rather than silently overwriting memory.
    """
    _validate_hot_k1_inputs(key, value, cache, slot_ids, active_mask)
    active_slots = slot_ids[active_mask].detach().cpu().tolist()
    if len(active_slots) != len(set(active_slots)):
        raise ValueError("active slot_ids must be unique")
    if active_slots:
        lengths = cache.hot_lengths[slot_ids[active_mask].long()]
        if bool((lengths >= cache.cold_cache.pool.page_size).any().item()):
            raise RuntimeError("hot page is full; packed-page transition is required")
    _launch_append_hot_k1_graph_safe(
        key, value, cache, slot_ids=slot_ids, active_mask=active_mask
    )


def _launch_append_hot_k1_graph_safe(
    key: torch.Tensor,
    value: torch.Tensor,
    cache: NQKVHotColdCache,
    *,
    slot_ids: torch.Tensor,
    active_mask: torch.Tensor,
) -> None:
    """Launch only fixed-address GPU work; caller owns control-plane checks."""
    pool = cache.cold_cache.pool
    batch_size = key.shape[0]
    assert triton is not None
    assert _append_hot_k1_data_kernel is not None
    assert _append_hot_k1_metadata_kernel is not None
    block_dim = triton.next_power_of_2(pool.head_dim)
    _append_hot_k1_data_kernel[(batch_size * pool.num_kv_heads,)](
        key,
        value,
        slot_ids,
        active_mask,
        cache.cold_cache.active,
        cache.hot_lengths,
        cache.hot_keys,
        cache.hot_values,
        batch_size=batch_size,
        max_sequences=cache.cold_cache.max_sequences,
        num_kv_heads=pool.num_kv_heads,
        page_size=pool.page_size,
        head_dim=pool.head_dim,
        block_dim=block_dim,
        num_warps=4,
    )
    _append_hot_k1_metadata_kernel[(batch_size,)](
        slot_ids,
        active_mask,
        cache.cold_cache.active,
        cache.hot_lengths,
        cache.cold_cache.sequence_lengths,
        max_sequences=cache.cold_cache.max_sequences,
        page_size=pool.page_size,
        num_warps=1,
    )


def append_hot_ragged_subpage(
    key: torch.Tensor,
    value: torch.Tensor,
    cache: NQKVHotColdCache,
    *,
    slot_ids: torch.Tensor,
    write_mask: torch.Tensor,
) -> None:
    """Append prefix-masked multi-token rows that remain below one hot page."""
    _validate_hot_ragged_inputs(key, value, cache, slot_ids, write_mask)
    slots = tuple(int(slot) for slot in slot_ids.detach().cpu().tolist())
    masks = write_mask.detach().cpu()
    counts = tuple(int(row.sum().item()) for row in masks)
    active_slots = tuple(
        slot for slot, count in zip(slots, counts, strict=True) if count
    )
    if len(active_slots) != len(set(active_slots)):
        raise ValueError("active slot_ids must be unique")
    for row, count in enumerate(counts):
        row_mask = masks[row]
        if count and not bool(row_mask[:count].all().item()):
            raise ValueError("each write_mask row must be a contiguous prefix")
        if bool(row_mask[count:].any().item()):
            raise ValueError("each write_mask row must be a contiguous prefix")
        if not count:
            continue
        slot = slots[row]
        current = cache.hot_length(slot)
        if current + count >= cache.cold_cache.pool.page_size:
            raise RuntimeError("ragged subpage append would fill or cross a page")
    _launch_append_hot_ragged_subpage_graph_safe(
        key, value, cache, slot_ids=slot_ids, write_mask=write_mask
    )
    for slot, count in zip(slots, counts, strict=True):
        if count:
            cache.synchronize_host_length(slot, cache.hot_length(slot) + count)


def _launch_append_hot_ragged_subpage_graph_safe(
    key: torch.Tensor,
    value: torch.Tensor,
    cache: NQKVHotColdCache,
    *,
    slot_ids: torch.Tensor,
    write_mask: torch.Tensor,
) -> None:
    """Launch fixed-address ragged sub-page writes without host synchronization."""
    pool = cache.cold_cache.pool
    batch_size, token_count = key.shape[:2]
    assert triton is not None
    assert _append_hot_ragged_data_kernel is not None
    assert _append_hot_ragged_metadata_kernel is not None
    block_dim = triton.next_power_of_2(pool.head_dim)
    count_block = triton.next_power_of_2(token_count)
    _append_hot_ragged_data_kernel[
        (batch_size * token_count * pool.num_kv_heads,)
    ](
        key,
        value,
        slot_ids,
        write_mask,
        cache.cold_cache.active,
        cache.hot_lengths,
        cache.hot_keys,
        cache.hot_values,
        token_count=token_count,
        max_sequences=cache.cold_cache.max_sequences,
        num_kv_heads=pool.num_kv_heads,
        page_size=pool.page_size,
        head_dim=pool.head_dim,
        block_dim=block_dim,
        num_warps=4,
    )
    _append_hot_ragged_metadata_kernel[(batch_size,)](
        slot_ids,
        write_mask,
        cache.cold_cache.active,
        cache.hot_lengths,
        cache.cold_cache.sequence_lengths,
        token_count=token_count,
        count_block=count_block,
        max_sequences=cache.cold_cache.max_sequences,
        page_size=pool.page_size,
        num_warps=1,
    )


def _launch_append_hot_ragged_segment_graph_safe(
    key: torch.Tensor,
    value: torch.Tensor,
    cache: NQKVHotColdCache,
    *,
    slot_ids: torch.Tensor,
    source_starts: torch.Tensor,
    segment_counts: torch.Tensor,
) -> None:
    """Append one compact source segment per row at its current hot tail."""
    pool = cache.cold_cache.pool
    batch_size, token_count = key.shape[:2]
    if source_starts.dtype != torch.int32 or source_starts.shape != (batch_size,):
        raise ValueError("source_starts must be int32 with shape [batch]")
    if segment_counts.dtype != torch.int32 or segment_counts.shape != (batch_size,):
        raise ValueError("segment_counts must be int32 with shape [batch]")
    if source_starts.device != pool.device or segment_counts.device != pool.device:
        raise ValueError("segment metadata must be on the cache device")
    assert triton is not None
    assert _append_hot_ragged_segment_data_kernel is not None
    assert _append_hot_ragged_segment_metadata_kernel is not None
    block_dim = triton.next_power_of_2(pool.head_dim)
    _append_hot_ragged_segment_data_kernel[
        (batch_size * token_count * pool.num_kv_heads,)
    ](
        key,
        value,
        slot_ids,
        source_starts,
        segment_counts,
        cache.cold_cache.active,
        cache.hot_lengths,
        cache.hot_keys,
        cache.hot_values,
        token_count=token_count,
        max_sequences=cache.cold_cache.max_sequences,
        num_kv_heads=pool.num_kv_heads,
        page_size=pool.page_size,
        head_dim=pool.head_dim,
        block_dim=block_dim,
        num_warps=4,
    )
    _append_hot_ragged_segment_metadata_kernel[(batch_size,)](
        slot_ids,
        segment_counts,
        cache.cold_cache.active,
        cache.hot_lengths,
        cache.cold_cache.sequence_lengths,
        max_sequences=cache.cold_cache.max_sequences,
        page_size=pool.page_size,
        num_warps=1,
    )


def _validate_hot_k1_inputs(
    key: torch.Tensor,
    value: torch.Tensor,
    cache: NQKVHotColdCache,
    slot_ids: torch.Tensor,
    active_mask: torch.Tensor,
) -> None:
    if not _TRITON_INSTALLED or not torch.cuda.is_available():
        raise RuntimeError("Triton CUDA runtime is unavailable")
    if not isinstance(cache, NQKVHotColdCache):
        raise TypeError("cache must be an NQKVHotColdCache")
    if key.ndim == 4 and key.shape[1] == 1:
        key = key[:, 0]
        value = value[:, 0]
    pool = cache.cold_cache.pool
    expected = (key.shape[0], pool.num_kv_heads, pool.head_dim)
    if key.ndim != 3 or key.shape != value.shape or tuple(key.shape) != expected:
        raise ValueError("key and value must share [batch, kv_heads, head_dim]")
    if key.device != pool.device or value.device != pool.device:
        raise ValueError("key and value must be on the cache device")
    if key.dtype != pool.output_dtype or value.dtype != pool.output_dtype:
        raise ValueError("key and value must match the cache dtype")
    if slot_ids.dtype != torch.int32 or slot_ids.shape != (key.shape[0],):
        raise ValueError("slot_ids must be int32 with shape [batch]")
    if active_mask.dtype != torch.bool or active_mask.shape != (key.shape[0],):
        raise ValueError("active_mask must be bool with shape [batch]")
    if slot_ids.device != pool.device or active_mask.device != pool.device:
        raise ValueError("slot_ids and active_mask must be on the cache device")


def _validate_hot_ragged_inputs(
    key: torch.Tensor,
    value: torch.Tensor,
    cache: NQKVHotColdCache,
    slot_ids: torch.Tensor,
    write_mask: torch.Tensor,
) -> None:
    if not _TRITON_INSTALLED or not torch.cuda.is_available():
        raise RuntimeError("Triton CUDA runtime is unavailable")
    if not isinstance(cache, NQKVHotColdCache):
        raise TypeError("cache must be an NQKVHotColdCache")
    pool = cache.cold_cache.pool
    if key.ndim != 4 or key.shape != value.shape:
        raise ValueError("key and value must share [batch, tokens, kv_heads, head_dim]")
    if tuple(key.shape[2:]) != (pool.num_kv_heads, pool.head_dim):
        raise ValueError("key/value head shape does not match the cache")
    if key.device != pool.device or value.device != pool.device:
        raise ValueError("key and value must be on the cache device")
    if key.dtype != pool.output_dtype or value.dtype != pool.output_dtype:
        raise ValueError("key and value must match the cache dtype")
    if slot_ids.dtype != torch.int32 or slot_ids.shape != (key.shape[0],):
        raise ValueError("slot_ids must be int32 with shape [batch]")
    if write_mask.dtype != torch.bool or write_mask.shape != key.shape[:2]:
        raise ValueError("write_mask must be bool with shape [batch, tokens]")
    if slot_ids.device != pool.device or write_mask.device != pool.device:
        raise ValueError("slot_ids and write_mask must be on the cache device")
