"""Correctness-first decode attention over packed NQKV pages."""

from __future__ import annotations

from typing import Hashable
import math

import torch

from .nqkv_hot_cache import NQKVHotColdCache
from .nqkv_page import dequantize_nqkv_page_triton


def nqkv_decode_attention_reference(
    query: torch.Tensor,
    cache: NQKVHotColdCache,
    sequence_id: Hashable,
    *,
    scale: float | None = None,
    tile_blocks: int = 64,
) -> torch.Tensor:
    """Attend one decode query to packed cold pages plus the BF16 hot tail.

    The implementation decodes at most one cold page at a time and combines
    page-local softmax statistics with the numerically stable online-softmax
    recurrence. It intentionally serves as the K0 oracle for a future fused
    paged-attention kernel, rather than as the final performance path.

    Args:
        query: CUDA tensor shaped ``[query_heads, head_dim]``.
        cache: Hot/cold NQKV cache containing the target sequence.
        sequence_id: Registered logical sequence identifier.
        scale: Optional attention scale; defaults to ``1 / sqrt(head_dim)``.
        tile_blocks: Triton NQKV decoder tile size.
    """
    if not isinstance(query, torch.Tensor):
        raise TypeError("query must be a torch tensor")
    pool = cache.cold_cache.pool
    if query.ndim != 2 or query.shape[1] != pool.head_dim:
        raise ValueError("query must have shape [query_heads, head_dim]")
    if query.device != pool.device or query.device.type != "cuda":
        raise ValueError("query must be on the cache CUDA device")
    if not query.is_floating_point():
        raise TypeError("query must be floating point")
    query_heads = query.shape[0]
    if query_heads < 1 or query_heads % pool.num_kv_heads:
        raise ValueError("query_heads must be a positive multiple of num_kv_heads")
    if cache.total_tokens(sequence_id) == 0:
        raise ValueError("cannot attend to an empty sequence")
    attention_scale = 1.0 / math.sqrt(pool.head_dim) if scale is None else float(scale)
    if not math.isfinite(attention_scale):
        raise ValueError("scale must be finite")

    query_fp32 = query.float()
    kv_head = torch.arange(query_heads, device=query.device) // (
        query_heads // pool.num_kv_heads
    )
    maximum = torch.full((query_heads,), -torch.inf, device=query.device)
    denominator = torch.zeros(query_heads, device=query.device)
    numerator = torch.zeros_like(query_fp32)

    def accumulate(key: torch.Tensor, value: torch.Tensor) -> None:
        nonlocal maximum, denominator, numerator
        if key.shape[0] == 0:
            return
        selected_key = key[:, kv_head, :].float()
        selected_value = value[:, kv_head, :].float()
        scores = torch.einsum("hd,thd->ht", query_fp32, selected_key) * attention_scale
        page_maximum = scores.max(dim=1).values
        new_maximum = torch.maximum(maximum, page_maximum)
        previous_scale = torch.exp(maximum - new_maximum)
        probabilities = torch.exp(scores - new_maximum[:, None])
        denominator = denominator * previous_scale + probabilities.sum(dim=1)
        numerator = numerator * previous_scale[:, None] + torch.einsum(
            "ht,thd->hd", probabilities, selected_value
        )
        maximum = new_maximum

    for handle in cache.cold_cache.handles(sequence_id):
        key, value = dequantize_nqkv_page_triton(
            pool.prepared(handle), output_dtype=query.dtype, tile_blocks=tile_blocks
        )
        accumulate(key, value)
    hot_key, hot_value = cache.hot_view(sequence_id)
    accumulate(hot_key, hot_value)
    return (numerator / denominator[:, None]).to(query.dtype)
