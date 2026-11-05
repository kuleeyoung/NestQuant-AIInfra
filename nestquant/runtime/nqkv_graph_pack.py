"""Boundary-only fixed-address packing of full NQKV hot pages."""

from __future__ import annotations

import math

import torch

from .nqkv_hot_cache import NQKVHotColdCache
from .nqkv_page import quantize_nqkv_page_cuda, _quantize_rows_cuda_fused
from .triton_unpack import _TRITON_INSTALLED


if _TRITON_INSTALLED:
    import triton
    import triton.language as tl

    @triton.jit
    def _copy_graph_page_payload_kernel(
        source,
        destination,
        target_page_ids,
        boundary_mask,
        hot_lengths,
        payload_size: tl.constexpr,
        page_size: tl.constexpr,
        block: tl.constexpr,
    ):
        slot = tl.program_id(0)
        offsets = tl.program_id(1) * block + tl.arange(0, block)
        valid = offsets < payload_size
        full = tl.load(boundary_mask + slot) & (
            tl.load(hot_lengths + slot) == page_size
        )
        page_id = tl.load(target_page_ids + slot)
        data = tl.load(
            source + slot * payload_size + offsets, mask=valid, other=0
        )
        tl.store(
            destination + page_id * payload_size + offsets,
            data,
            mask=full & valid,
        )

    @triton.jit
    def _publish_graph_page_kernel(
        boundary_mask,
        target_page_ids,
        page_indices,
        page_counts,
        last_page_lens,
        hot_lengths,
        max_pages: tl.constexpr,
        page_size: tl.constexpr,
    ):
        slot = tl.program_id(0)
        count = tl.load(page_counts + slot)
        full = tl.load(boundary_mask + slot) & (
            tl.load(hot_lengths + slot) == page_size
        ) & (count < max_pages)
        page_id = tl.load(target_page_ids + slot)
        tl.store(
            page_indices + slot * max_pages + count,
            page_id,
            mask=full,
        )
        tl.store(page_counts + slot, count + 1, mask=full)
        tl.store(last_page_lens + slot, page_size, mask=full)
        tl.store(hot_lengths + slot, 0, mask=full)
else:
    triton = None
    _copy_graph_page_payload_kernel = None
    _publish_graph_page_kernel = None


def launch_pack_full_hot_pages_graph_safe(
    cache: NQKVHotColdCache,
    *,
    boundary_mask: torch.Tensor,
    batched_encode: bool = False,
) -> None:
    """Encode every fixed slot, publishing only full boundary-selected rows.

    This function is intended to execute while a boundary CUDA Graph is being
    captured. Allocations made by the encoder therefore belong to the graph's
    private pool and are reused at replay rather than allocated per token.
    """
    pool = cache.cold_cache.pool
    capacity = cache.cold_cache.max_sequences
    if boundary_mask.dtype != torch.bool or boundary_mask.shape != (capacity,):
        raise ValueError("boundary_mask must be bool with cache capacity shape")
    if boundary_mask.device != pool.device:
        raise ValueError("boundary_mask must be on the cache device")
    payloads = (
        _quantize_full_hot_pages(cache)
        if batched_encode else _quantize_full_hot_pages_reference(cache)
    )
    destinations = (
        pool.key_codes,
        pool.key_betas,
        pool.key_norms,
        pool.value_codes,
        pool.value_betas,
        pool.value_norms,
    )
    assert triton is not None
    assert _copy_graph_page_payload_kernel is not None
    for source, destination in zip(payloads, destinations, strict=True):
        payload_size = source.shape[1]
        block = min(256, triton.next_power_of_2(payload_size))
        _copy_graph_page_payload_kernel[
            (capacity, triton.cdiv(payload_size, block))
        ](
            source,
            destination,
            cache.graph_target_page_ids,
            boundary_mask,
            cache.hot_lengths,
            payload_size=payload_size,
            page_size=pool.page_size,
            block=block,
            num_warps=4,
        )
    assert _publish_graph_page_kernel is not None
    _publish_graph_page_kernel[(capacity,)](
        boundary_mask,
        cache.graph_target_page_ids,
        cache.cold_cache.page_indices,
        cache.cold_cache.page_counts,
        cache.cold_cache.last_page_len,
        cache.hot_lengths,
        max_pages=cache.cold_cache.max_pages_per_sequence,
        page_size=pool.page_size,
        num_warps=1,
    )


def _quantize_full_hot_pages(
    cache: NQKVHotColdCache,
) -> tuple[torch.Tensor, ...]:
    """Encode every fixed hot page in one batched Key pass and Value pass."""
    pool = cache.cold_cache.pool
    capacity = cache.cold_cache.max_sequences
    rows_per_page = pool.page_size * pool.num_kv_heads
    key_rows = cache.hot_keys.float().reshape(
        capacity * rows_per_page, pool.head_dim
    )
    value_rows = cache.hot_values.float().reshape(
        capacity * rows_per_page, pool.head_dim
    )
    key_codes, key_betas, key_norms = _quantize_rows_cuda_fused(
        key_rows,
        q=pool.q,
        betas=pool.betas,
        beta_table=pool.beta_table,
    )
    value_codes, value_betas, value_norms = _quantize_rows_cuda_fused(
        value_rows,
        q=pool.q,
        betas=pool.value_beta_values,
        beta_table=pool.value_beta_table,
    )
    code_width = max(1, math.ceil(math.log2(pool.q)))
    beta_width = max(1, math.ceil(math.log2(len(pool.betas))))
    return (
        _pack_pages_unsigned(key_codes, code_width, capacity),
        _pack_pages_unsigned(key_betas, beta_width, capacity),
        key_norms.reshape(capacity, rows_per_page),
        _pack_pages_unsigned(value_codes, code_width, capacity),
        _pack_pages_unsigned(value_betas, beta_width, capacity),
        value_norms.reshape(capacity, rows_per_page),
    )


def _pack_pages_unsigned(
    values: torch.Tensor,
    width: int,
    capacity: int,
) -> torch.Tensor:
    """Pack fixed-width values without allowing bits to cross page boundaries."""
    per_byte = 8 // width
    pages = values.reshape(capacity, -1).to(torch.uint8)
    padded_values = math.ceil(pages.shape[1] / per_byte) * per_byte
    if padded_values != pages.shape[1]:
        padded = torch.zeros(
            (capacity, padded_values), device=pages.device, dtype=torch.uint8
        )
        padded[:, :pages.shape[1]].copy_(pages)
        pages = padded
    groups = pages.reshape(capacity, -1, per_byte).to(torch.int32)
    shifts = (
        torch.arange(per_byte, device=pages.device, dtype=torch.int32) * width
    )
    return torch.sum(groups << shifts, dim=2).to(torch.uint8)


def _quantize_full_hot_pages_reference(
    cache: NQKVHotColdCache,
) -> tuple[torch.Tensor, ...]:
    """Original per-slot encoder retained as the correctness/performance control."""
    pool = cache.cold_cache.pool
    prepared = [
        quantize_nqkv_page_cuda(
            cache.hot_keys[slot],
            cache.hot_values[slot],
            page_size=pool.page_size,
            q=pool.q,
            key_betas=pool.betas,
            value_betas=pool.value_beta_values,
            key_beta_table=pool.beta_table,
            value_beta_table=pool.value_beta_table,
        )
        for slot in range(cache.cold_cache.max_sequences)
    ]
    return tuple(
        torch.stack([getattr(page, field) for page in prepared])
        for field in (
            "key_code_payload",
            "key_beta_payload",
            "key_norms",
            "value_code_payload",
            "value_beta_payload",
            "value_norms",
        )
    )

