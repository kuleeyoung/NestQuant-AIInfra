"""Two-stage fused decode attention over resident packed NQKV cold pages."""

from __future__ import annotations

from typing import Hashable
from dataclasses import dataclass
import math

import torch

from .nqkv_hot_cache import NQKVHotColdCache
from .triton_unpack import _TRITON_INSTALLED


if _TRITON_INSTALLED:
    import triton
    import triton.language as tl

    from .triton_tiled_decode import _nearest_e8_upstream

    @triton.jit
    def _decode_nqkv_vector(
        code_payload,
        beta_payload,
        row_norms,
        beta_table,
        page_id,
        token,
        kv_head,
        code_bytes_per_page: tl.constexpr,
        beta_bytes_per_page: tl.constexpr,
        rows_per_page: tl.constexpr,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        blocks_per_row: tl.constexpr,
        q: tl.constexpr,
        code_width: tl.constexpr,
        beta_width: tl.constexpr,
    ):
        blocks = tl.arange(0, blocks_per_row)[:, None]
        coordinates = tl.arange(0, 8)[None, :]
        row = token * num_kv_heads + kv_head
        block = row * blocks_per_row + blocks
        element = block * 8 + coordinates
        codes_per_byte: tl.constexpr = 8 // code_width
        packed = tl.load(
            code_payload + page_id * code_bytes_per_page + element // codes_per_byte
        )
        shifts = (element % codes_per_byte) * code_width
        codes = ((packed >> shifts) & ((1 << code_width) - 1)).to(tl.float32)
        betas_per_byte: tl.constexpr = 8 // beta_width
        packed_beta = tl.load(
            beta_payload + page_id * beta_bytes_per_page + block // betas_per_byte
        )
        beta_shift = (block % betas_per_byte) * beta_width
        beta_id = (packed_beta >> beta_shift) & ((1 << beta_width) - 1)
        beta = tl.load(beta_table + beta_id)
        code7 = tl.sum(tl.where(coordinates == 7, codes, 0.0), axis=1)[:, None]
        next_element = block * 8 + tl.minimum(coordinates + 1, 7)
        packed_next = tl.load(
            code_payload + page_id * code_bytes_per_page
            + next_element // codes_per_byte
        )
        next_shift = (next_element % codes_per_byte) * code_width
        next_code = (
            (packed_next >> next_shift) & ((1 << code_width) - 1)
        ).to(tl.float32)
        coset = tl.where(
            coordinates == 0,
            2.0 * codes - next_code + 0.5 * code7,
            tl.where(
                coordinates < 6,
                codes - next_code + 0.5 * code7,
                tl.where(coordinates == 6, codes + 0.5 * code7, 0.5 * code7),
            ),
        )
        coarse = _nearest_e8_upstream(coset / q, coordinates)
        reconstructed = coset - q * coarse
        norm = tl.load(row_norms + page_id * rows_per_page + row)
        vector = reconstructed * beta * (norm / tl.sqrt(float(head_dim)))
        return tl.reshape(vector, (head_dim,))

    @triton.jit
    def _nqkv_page_attention_kernel(
        query,
        page_ids,
        key_codes,
        key_betas,
        key_norms,
        value_codes,
        value_betas,
        value_norms,
        key_beta_table,
        value_beta_table,
        page_maxima,
        page_denominators,
        page_numerators,
        page_counts,
        last_page_lens,
        sequence_slot,
        query_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        page_size: tl.constexpr,
        head_dim: tl.constexpr,
        blocks_per_row: tl.constexpr,
        code_bytes_per_page: tl.constexpr,
        beta_bytes_per_page: tl.constexpr,
        rows_per_page: tl.constexpr,
        q: tl.constexpr,
        code_width: tl.constexpr,
        beta_width: tl.constexpr,
        attention_scale: tl.constexpr,
    ):
        page_slot = tl.program_id(0)
        query_head = tl.program_id(1)
        page_count = tl.load(page_counts + sequence_slot)
        last_page_len = tl.load(last_page_lens + sequence_slot)
        dimensions = tl.arange(0, head_dim)
        valid_page = page_slot < page_count
        page_id = tl.load(page_ids + page_slot, mask=valid_page, other=0)
        group_size: tl.constexpr = query_heads // num_kv_heads
        kv_head = query_head // group_size
        query_vector = tl.load(query + query_head * head_dim + dimensions).to(tl.float32)
        maximum = -float("inf")
        denominator = 0.0
        numerator = tl.zeros((head_dim,), tl.float32)
        for token in tl.static_range(0, page_size):
            token_limit = tl.where(page_slot == page_count - 1, last_page_len, page_size)
            valid_token = valid_page & (token < token_limit)
            key = _decode_nqkv_vector(
                key_codes, key_betas, key_norms, key_beta_table,
                page_id, token, kv_head,
                code_bytes_per_page, beta_bytes_per_page, rows_per_page,
                num_kv_heads, head_dim, blocks_per_row, q, code_width, beta_width,
            )
            value = _decode_nqkv_vector(
                value_codes, value_betas, value_norms, value_beta_table,
                page_id, token, kv_head,
                code_bytes_per_page, beta_bytes_per_page, rows_per_page,
                num_kv_heads, head_dim, blocks_per_row, q, code_width, beta_width,
            )
            score = tl.sum(query_vector * key, axis=0) * attention_scale
            score = tl.where(valid_token, score, -float("inf"))
            new_maximum = tl.maximum(maximum, score)
            previous = tl.where(valid_token, tl.exp(maximum - new_maximum), 1.0)
            probability = tl.where(valid_token, tl.exp(score - new_maximum), 0.0)
            denominator = denominator * previous + probability
            numerator = numerator * previous + probability * value
            maximum = tl.where(valid_token, new_maximum, maximum)
        scalar_offset = page_slot * query_heads + query_head
        tl.store(page_maxima + scalar_offset, maximum, mask=valid_page)
        tl.store(page_denominators + scalar_offset, denominator, mask=valid_page)
        tl.store(
            page_numerators + scalar_offset * head_dim + dimensions,
            numerator,
            mask=valid_page,
        )

    @triton.jit
    def _nqkv_page_reduce_kernel(
        page_maxima,
        page_denominators,
        page_numerators,
        output,
        page_counts,
        hot_lengths,
        sequence_slot,
        query_heads: tl.constexpr,
        head_dim: tl.constexpr,
        block_pages: tl.constexpr,
    ):
        query_head = tl.program_id(0)
        page_count = tl.load(page_counts + sequence_slot)
        hot_length = tl.load(hot_lengths + sequence_slot)
        total_parts = page_count + (hot_length > 0).to(tl.int32)
        pages = tl.arange(0, block_pages)
        dimensions = tl.arange(0, head_dim)
        valid = pages < total_parts
        scalar_offsets = pages * query_heads + query_head
        maxima = tl.load(page_maxima + scalar_offsets, mask=valid, other=-float("inf"))
        maximum = tl.max(maxima, axis=0)
        rescale = tl.where(valid, tl.exp(maxima - maximum), 0.0)
        denominators = tl.load(page_denominators + scalar_offsets, mask=valid, other=0.0)
        denominator = tl.sum(denominators * rescale, axis=0)
        numerator_offsets = scalar_offsets[:, None] * head_dim + dimensions[None, :]
        numerators = tl.load(
            page_numerators + numerator_offsets, mask=valid[:, None], other=0.0
        )
        numerator = tl.sum(numerators * rescale[:, None], axis=0)
        tl.store(output + query_head * head_dim + dimensions, numerator / denominator)

    @triton.jit
    def _nqkv_hot_attention_kernel(
        query,
        hot_keys,
        hot_values,
        page_maxima,
        page_denominators,
        page_numerators,
        page_counts,
        hot_lengths,
        sequence_slot,
        query_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        page_size: tl.constexpr,
        head_dim: tl.constexpr,
        attention_scale: tl.constexpr,
    ):
        query_head = tl.program_id(0)
        hot_length = tl.load(hot_lengths + sequence_slot)
        output_slot = tl.load(page_counts + sequence_slot)
        dimensions = tl.arange(0, head_dim)
        group_size: tl.constexpr = query_heads // num_kv_heads
        kv_head = query_head // group_size
        query_vector = tl.load(query + query_head * head_dim + dimensions).to(tl.float32)
        maximum = -float("inf")
        denominator = 0.0
        numerator = tl.zeros((head_dim,), tl.float32)
        sequence_base = sequence_slot * page_size * num_kv_heads * head_dim
        for token in tl.static_range(0, page_size):
            valid = token < hot_length
            offset = (
                sequence_base + (token * num_kv_heads + kv_head) * head_dim
                + dimensions
            )
            key = tl.load(hot_keys + offset, mask=valid, other=0.0).to(tl.float32)
            value = tl.load(hot_values + offset, mask=valid, other=0.0).to(tl.float32)
            score = tl.sum(query_vector * key, axis=0) * attention_scale
            score = tl.where(valid, score, -float("inf"))
            new_maximum = tl.maximum(maximum, score)
            previous = tl.where(valid, tl.exp(maximum - new_maximum), 1.0)
            probability = tl.where(valid, tl.exp(score - new_maximum), 0.0)
            denominator = denominator * previous + probability
            numerator = numerator * previous + probability * value
            maximum = tl.where(valid, new_maximum, maximum)
        scalar_offset = output_slot * query_heads + query_head
        tl.store(page_maxima + scalar_offset, maximum)
        tl.store(page_denominators + scalar_offset, denominator)
        tl.store(
            page_numerators + scalar_offset * head_dim + dimensions, numerator
        )

    @triton.jit
    def _nqkv_batched_page_attention_kernel(
        query,
        slot_ids,
        active_mask,
        page_ids,
        key_codes,
        key_betas,
        key_norms,
        value_codes,
        value_betas,
        value_norms,
        key_beta_table,
        value_beta_table,
        page_maxima,
        page_denominators,
        page_numerators,
        page_counts,
        last_page_lens,
        cache_active,
        sequence_lengths,
        query_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        max_sequences: tl.constexpr,
        max_pages: tl.constexpr,
        part_capacity: tl.constexpr,
        page_size: tl.constexpr,
        head_dim: tl.constexpr,
        blocks_per_row: tl.constexpr,
        code_bytes_per_page: tl.constexpr,
        beta_bytes_per_page: tl.constexpr,
        rows_per_page: tl.constexpr,
        q: tl.constexpr,
        code_width: tl.constexpr,
        beta_width: tl.constexpr,
        attention_scale: tl.constexpr,
    ):
        page_slot = tl.program_id(0)
        combined_head = tl.program_id(1)
        batch_row = combined_head // query_heads
        query_head = combined_head % query_heads
        requested_slot = tl.load(slot_ids + batch_row)
        slot_in_range = (requested_slot >= 0) & (requested_slot < max_sequences)
        sequence_slot = tl.maximum(0, tl.minimum(requested_slot, max_sequences - 1))
        active = (
            tl.load(active_mask + batch_row)
            & slot_in_range
            & tl.load(cache_active + sequence_slot)
            & (tl.load(sequence_lengths + sequence_slot) > 0)
        )
        page_count = tl.load(page_counts + sequence_slot)
        last_page_len = tl.load(last_page_lens + sequence_slot)
        valid_page = active & (page_slot < page_count)
        page_id = tl.load(
            page_ids + sequence_slot * max_pages + page_slot,
            mask=valid_page,
            other=0,
        )
        dimensions = tl.arange(0, head_dim)
        group_size: tl.constexpr = query_heads // num_kv_heads
        kv_head = query_head // group_size
        query_vector = tl.load(
            query + (batch_row * query_heads + query_head) * head_dim + dimensions
        ).to(tl.float32)
        maximum = -float("inf")
        denominator = 0.0
        numerator = tl.zeros((head_dim,), tl.float32)
        for token in tl.static_range(0, page_size):
            token_limit = tl.where(page_slot == page_count - 1, last_page_len, page_size)
            valid_token = valid_page & (token < token_limit)
            key = _decode_nqkv_vector(
                key_codes, key_betas, key_norms, key_beta_table,
                page_id, token, kv_head,
                code_bytes_per_page, beta_bytes_per_page, rows_per_page,
                num_kv_heads, head_dim, blocks_per_row, q, code_width, beta_width,
            )
            value = _decode_nqkv_vector(
                value_codes, value_betas, value_norms, value_beta_table,
                page_id, token, kv_head,
                code_bytes_per_page, beta_bytes_per_page, rows_per_page,
                num_kv_heads, head_dim, blocks_per_row, q, code_width, beta_width,
            )
            score = tl.sum(query_vector * key, axis=0) * attention_scale
            score = tl.where(valid_token, score, -float("inf"))
            new_maximum = tl.maximum(maximum, score)
            previous = tl.where(valid_token, tl.exp(maximum - new_maximum), 1.0)
            probability = tl.where(valid_token, tl.exp(score - new_maximum), 0.0)
            denominator = denominator * previous + probability
            numerator = numerator * previous + probability * value
            maximum = tl.where(valid_token, new_maximum, maximum)
        scalar_offset = (
            (batch_row * part_capacity + page_slot) * query_heads + query_head
        )
        tl.store(page_maxima + scalar_offset, maximum, mask=valid_page)
        tl.store(page_denominators + scalar_offset, denominator, mask=valid_page)
        tl.store(
            page_numerators + scalar_offset * head_dim + dimensions,
            numerator,
            mask=valid_page,
        )

    @triton.jit
    def _nqkv_batched_hot_attention_kernel(
        query,
        slot_ids,
        active_mask,
        hot_keys,
        hot_values,
        page_maxima,
        page_denominators,
        page_numerators,
        page_counts,
        hot_lengths,
        cache_active,
        sequence_lengths,
        query_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        max_sequences: tl.constexpr,
        part_capacity: tl.constexpr,
        page_size: tl.constexpr,
        head_dim: tl.constexpr,
        attention_scale: tl.constexpr,
    ):
        combined_head = tl.program_id(0)
        batch_row = combined_head // query_heads
        query_head = combined_head % query_heads
        requested_slot = tl.load(slot_ids + batch_row)
        slot_in_range = (requested_slot >= 0) & (requested_slot < max_sequences)
        sequence_slot = tl.maximum(0, tl.minimum(requested_slot, max_sequences - 1))
        active = (
            tl.load(active_mask + batch_row)
            & slot_in_range
            & tl.load(cache_active + sequence_slot)
            & (tl.load(sequence_lengths + sequence_slot) > 0)
        )
        hot_length = tl.load(hot_lengths + sequence_slot)
        output_slot = tl.load(page_counts + sequence_slot)
        dimensions = tl.arange(0, head_dim)
        group_size: tl.constexpr = query_heads // num_kv_heads
        kv_head = query_head // group_size
        query_vector = tl.load(
            query + (batch_row * query_heads + query_head) * head_dim + dimensions
        ).to(tl.float32)
        maximum = -float("inf")
        denominator = 0.0
        numerator = tl.zeros((head_dim,), tl.float32)
        sequence_base = sequence_slot * page_size * num_kv_heads * head_dim
        for token in tl.static_range(0, page_size):
            valid = active & (token < hot_length)
            offset = (
                sequence_base + (token * num_kv_heads + kv_head) * head_dim
                + dimensions
            )
            key = tl.load(hot_keys + offset, mask=valid, other=0.0).to(tl.float32)
            value = tl.load(hot_values + offset, mask=valid, other=0.0).to(tl.float32)
            score = tl.sum(query_vector * key, axis=0) * attention_scale
            score = tl.where(valid, score, -float("inf"))
            new_maximum = tl.maximum(maximum, score)
            previous = tl.where(valid, tl.exp(maximum - new_maximum), 1.0)
            probability = tl.where(valid, tl.exp(score - new_maximum), 0.0)
            denominator = denominator * previous + probability
            numerator = numerator * previous + probability * value
            maximum = tl.where(valid, new_maximum, maximum)
        scalar_offset = (
            (batch_row * part_capacity + output_slot) * query_heads + query_head
        )
        has_hot = active & (hot_length > 0)
        tl.store(page_maxima + scalar_offset, maximum, mask=has_hot)
        tl.store(page_denominators + scalar_offset, denominator, mask=has_hot)
        tl.store(
            page_numerators + scalar_offset * head_dim + dimensions,
            numerator,
            mask=has_hot,
        )

    @triton.jit
    def _nqkv_batched_page_reduce_kernel(
        slot_ids,
        active_mask,
        page_maxima,
        page_denominators,
        page_numerators,
        output,
        page_counts,
        hot_lengths,
        cache_active,
        sequence_lengths,
        query_heads: tl.constexpr,
        head_dim: tl.constexpr,
        max_sequences: tl.constexpr,
        part_capacity: tl.constexpr,
        block_pages: tl.constexpr,
    ):
        combined_head = tl.program_id(0)
        batch_row = combined_head // query_heads
        query_head = combined_head % query_heads
        requested_slot = tl.load(slot_ids + batch_row)
        slot_in_range = (requested_slot >= 0) & (requested_slot < max_sequences)
        sequence_slot = tl.maximum(0, tl.minimum(requested_slot, max_sequences - 1))
        active = (
            tl.load(active_mask + batch_row)
            & slot_in_range
            & tl.load(cache_active + sequence_slot)
            & (tl.load(sequence_lengths + sequence_slot) > 0)
        )
        page_count = tl.load(page_counts + sequence_slot)
        hot_length = tl.load(hot_lengths + sequence_slot)
        total_parts = page_count + (hot_length > 0).to(tl.int32)
        pages = tl.arange(0, block_pages)
        dimensions = tl.arange(0, head_dim)
        valid = active & (pages < total_parts)
        scalar_offsets = (
            (batch_row * part_capacity + pages) * query_heads + query_head
        )
        maxima = tl.load(page_maxima + scalar_offsets, mask=valid, other=-float("inf"))
        maximum = tl.max(maxima, axis=0)
        rescale = tl.where(valid, tl.exp(maxima - maximum), 0.0)
        denominators = tl.load(page_denominators + scalar_offsets, mask=valid, other=0.0)
        denominator = tl.sum(denominators * rescale, axis=0)
        numerator_offsets = scalar_offsets[:, None] * head_dim + dimensions[None, :]
        numerators = tl.load(
            page_numerators + numerator_offsets,
            mask=valid[:, None],
            other=0.0,
        )
        numerator = tl.sum(numerators * rescale[:, None], axis=0)
        result = tl.where(active & (denominator > 0), numerator / denominator, 0.0)
        output_offset = combined_head * head_dim + dimensions
        tl.store(output + output_offset, result)
else:
    triton = None
    _nqkv_page_attention_kernel = None
    _nqkv_page_reduce_kernel = None
    _nqkv_hot_attention_kernel = None
    _nqkv_batched_page_attention_kernel = None
    _nqkv_batched_hot_attention_kernel = None
    _nqkv_batched_page_reduce_kernel = None


@dataclass(frozen=True)
class NQKVAttentionWorkspace:
    """Fixed-address intermediates for repeated K1 decode attention."""

    page_maxima: torch.Tensor
    page_denominators: torch.Tensor
    page_numerators: torch.Tensor
    output: torch.Tensor
    max_pages: int
    query_heads: int
    head_dim: int

    @classmethod
    def create(
        cls,
        cache: NQKVHotColdCache,
        *,
        query_heads: int,
        output_dtype: torch.dtype = torch.bfloat16,
    ) -> "NQKVAttentionWorkspace":
        pool = cache.cold_cache.pool
        if query_heads < 1 or query_heads % pool.num_kv_heads:
            raise ValueError("query_heads must be a positive multiple of num_kv_heads")
        if output_dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError("output_dtype must be float16, bfloat16, or float32")
        part_capacity = cache.cold_cache.max_pages_per_sequence + 1
        maxima = torch.empty(
            (part_capacity, query_heads), device=pool.device, dtype=torch.float32
        )
        denominators = torch.empty_like(maxima)
        numerators = torch.empty(
            (part_capacity, query_heads, pool.head_dim),
            device=pool.device, dtype=torch.float32,
        )
        output = torch.empty(
            (query_heads, pool.head_dim), device=pool.device, dtype=output_dtype
        )
        return cls(
            maxima, denominators, numerators, output,
            cache.cold_cache.max_pages_per_sequence, query_heads, pool.head_dim,
        )

    def pointers(self) -> tuple[int, int, int, int]:
        return (
            self.page_maxima.data_ptr(), self.page_denominators.data_ptr(),
            self.page_numerators.data_ptr(), self.output.data_ptr(),
        )


@dataclass(frozen=True)
class BatchedNQKVAttentionWorkspace:
    """Fixed-address intermediates for ragged batched K1 attention."""

    page_maxima: torch.Tensor
    page_denominators: torch.Tensor
    page_numerators: torch.Tensor
    output: torch.Tensor
    batch_size: int
    max_pages: int
    query_heads: int
    head_dim: int

    @classmethod
    def create(
        cls,
        cache: NQKVHotColdCache,
        *,
        batch_size: int,
        query_heads: int,
        output_dtype: torch.dtype = torch.bfloat16,
    ) -> "BatchedNQKVAttentionWorkspace":
        pool = cache.cold_cache.pool
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if query_heads < 1 or query_heads % pool.num_kv_heads:
            raise ValueError("query_heads must be a positive multiple of num_kv_heads")
        if output_dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError("output_dtype must be float16, bfloat16, or float32")
        part_capacity = cache.cold_cache.max_pages_per_sequence + 1
        maxima = torch.empty(
            (batch_size, part_capacity, query_heads),
            device=pool.device,
            dtype=torch.float32,
        )
        denominators = torch.empty_like(maxima)
        numerators = torch.empty(
            (batch_size, part_capacity, query_heads, pool.head_dim),
            device=pool.device,
            dtype=torch.float32,
        )
        output = torch.empty(
            (batch_size, query_heads, pool.head_dim),
            device=pool.device,
            dtype=output_dtype,
        )
        return cls(
            maxima,
            denominators,
            numerators,
            output,
            batch_size,
            cache.cold_cache.max_pages_per_sequence,
            query_heads,
            pool.head_dim,
        )

    def pointers(self) -> tuple[int, int, int, int]:
        return (
            self.page_maxima.data_ptr(),
            self.page_denominators.data_ptr(),
            self.page_numerators.data_ptr(),
            self.output.data_ptr(),
        )


class NQKVAttentionGraphRunner:
    """Capture K1 once while page counts and hot length remain GPU-dynamic."""

    def __init__(
        self,
        cache: NQKVHotColdCache,
        sequence_id: Hashable,
        *,
        query_heads: int,
        query_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        pool = cache.cold_cache.pool
        # Resolve and pin the logical slot before capture. Releasing this
        # sequence invalidates the runner even if the slot is later reused.
        self.cache = cache
        self.sequence_id = sequence_id
        self.sequence_slot = cache.cold_cache.slot(sequence_id)
        self.query = torch.empty(
            (query_heads, pool.head_dim), device=pool.device, dtype=query_dtype
        )
        self.workspace = NQKVAttentionWorkspace.create(
            cache, query_heads=query_heads, output_dtype=query_dtype
        )
        self.graph: torch.cuda.CUDAGraph | None = None

    @property
    def captured(self) -> bool:
        return self.graph is not None

    def capture(self, sample_query: torch.Tensor) -> None:
        if self.captured:
            return
        self._validate_query(sample_query)
        self.query.copy_(sample_query)
        warmup_stream = torch.cuda.Stream(device=self.query.device)
        warmup_stream.wait_stream(torch.cuda.current_stream(self.query.device))
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                nqkv_decode_attention_k1(
                    self.query, self.cache, self.sequence_id,
                    workspace=self.workspace,
                )
        torch.cuda.current_stream(self.query.device).wait_stream(warmup_stream)
        torch.cuda.synchronize(self.query.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            nqkv_decode_attention_k1(
                self.query, self.cache, self.sequence_id,
                workspace=self.workspace,
            )
        self.graph = graph

    def replay(self, query: torch.Tensor) -> torch.Tensor:
        if self.graph is None:
            raise RuntimeError("capture must run before replay")
        if self.cache.cold_cache.slot(self.sequence_id) != self.sequence_slot:
            raise RuntimeError("captured NQKV sequence slot changed")
        self._validate_query(query)
        self.query.copy_(query)
        self.graph.replay()
        return self.workspace.output

    def _validate_query(self, query: torch.Tensor) -> None:
        if not isinstance(query, torch.Tensor):
            raise TypeError("query must be a torch tensor")
        if (
            query.shape != self.query.shape or query.device != self.query.device
            or query.dtype != self.query.dtype
        ):
            raise ValueError("query must match the graph runner shape, device, and dtype")


def nqkv_decode_attention_k1(
    query: torch.Tensor,
    cache: NQKVHotColdCache,
    sequence_id: Hashable,
    *,
    scale: float | None = None,
    workspace: NQKVAttentionWorkspace | None = None,
) -> torch.Tensor:
    """Run fused packed-cold plus BF16-hot decode attention."""
    if not _TRITON_INSTALLED or not torch.cuda.is_available():
        raise RuntimeError("Triton CUDA runtime is unavailable")
    if not isinstance(query, torch.Tensor) or query.ndim != 2:
        raise ValueError("query must have shape [query_heads, head_dim]")
    pool = cache.cold_cache.pool
    if query.device != pool.device or query.shape[1] != pool.head_dim:
        raise ValueError("query must match the cache CUDA device and head_dim")
    query_heads = query.shape[0]
    if query_heads < 1 or query_heads % pool.num_kv_heads:
        raise ValueError("query_heads must be a positive multiple of num_kv_heads")
    hot_key, _ = cache.hot_view(sequence_id)
    hot_length = hot_key.shape[0]
    handles = cache.cold_cache.handles(sequence_id)
    if not handles and not hot_length:
        raise ValueError("cannot attend to an empty sequence")
    page_count = len(handles)
    max_pages = cache.cold_cache.max_pages_per_sequence
    part_capacity = max_pages + 1
    block_pages = triton.next_power_of_2(part_capacity)
    if block_pages > 128:
        raise ValueError("K1 currently supports at most 64 pages per sequence")
    slot = cache.cold_cache.slot(sequence_id)
    page_ids = cache.cold_cache.page_indices[slot]
    if workspace is None:
        workspace = NQKVAttentionWorkspace.create(
            cache, query_heads=query_heads, output_dtype=query.dtype
        )
    expected = (
        max_pages, query_heads, pool.head_dim, query.device, query.dtype,
    )
    actual = (
        workspace.max_pages, workspace.query_heads, workspace.head_dim,
        workspace.output.device, workspace.output.dtype,
    )
    if actual != expected:
        raise ValueError("workspace configuration does not match query and cache")
    page_maxima = workspace.page_maxima
    page_denominators = workspace.page_denominators
    page_numerators = workspace.page_numerators
    output = workspace.output
    attention_scale = 1.0 / math.sqrt(pool.head_dim) if scale is None else float(scale)
    assert _nqkv_page_attention_kernel is not None
    assert _nqkv_page_reduce_kernel is not None
    _nqkv_page_attention_kernel[(max_pages, query_heads)](
        query, page_ids,
        pool.key_codes, pool.key_betas, pool.key_norms,
        pool.value_codes, pool.value_betas, pool.value_norms,
        pool.beta_table, pool.value_beta_table,
        page_maxima, page_denominators, page_numerators,
        cache.cold_cache.page_counts, cache.cold_cache.last_page_len, slot,
        query_heads=query_heads,
        num_kv_heads=pool.num_kv_heads,
        page_size=pool.page_size,
        head_dim=pool.head_dim,
        blocks_per_row=pool.head_dim // 8,
        code_bytes_per_page=pool.key_codes.stride(0),
        beta_bytes_per_page=pool.key_betas.stride(0),
        rows_per_page=pool.key_norms.stride(0),
        q=pool.q,
        code_width=pool.code_width,
        beta_width=pool.beta_width,
        attention_scale=attention_scale,
        num_warps=4,
    )
    assert _nqkv_hot_attention_kernel is not None
    _nqkv_hot_attention_kernel[(query_heads,)](
        query, cache.hot_keys, cache.hot_values,
        page_maxima, page_denominators, page_numerators,
        cache.cold_cache.page_counts, cache.hot_lengths, slot,
        query_heads=query_heads,
        num_kv_heads=pool.num_kv_heads,
        page_size=pool.page_size,
        head_dim=pool.head_dim,
        attention_scale=attention_scale,
        num_warps=4,
    )
    _nqkv_page_reduce_kernel[(query_heads,)](
        page_maxima, page_denominators, page_numerators, output,
        cache.cold_cache.page_counts, cache.hot_lengths, slot,
        query_heads=query_heads, head_dim=pool.head_dim, block_pages=block_pages,
        num_warps=4,
    )
    return output


def nqkv_decode_attention_batched_k1(
    query: torch.Tensor,
    cache: NQKVHotColdCache,
    *,
    slot_ids: torch.Tensor,
    active_mask: torch.Tensor,
    scale: float | None = None,
    workspace: BatchedNQKVAttentionWorkspace | None = None,
) -> torch.Tensor:
    """Run fused packed-cold plus BF16-hot attention for a ragged batch."""
    if not _TRITON_INSTALLED or not torch.cuda.is_available():
        raise RuntimeError("Triton CUDA runtime is unavailable")
    if not isinstance(query, torch.Tensor) or query.ndim != 3:
        raise ValueError("query must have shape [batch, query_heads, head_dim]")
    if not isinstance(slot_ids, torch.Tensor) or not isinstance(active_mask, torch.Tensor):
        raise TypeError("slot_ids and active_mask must be tensors")
    pool = cache.cold_cache.pool
    batch_size, query_heads, head_dim = query.shape
    if query.device != pool.device or head_dim != pool.head_dim:
        raise ValueError("query must match the cache CUDA device and head_dim")
    if query_heads < 1 or query_heads % pool.num_kv_heads:
        raise ValueError("query_heads must be a positive multiple of num_kv_heads")
    if slot_ids.dtype != torch.int32 or slot_ids.shape != (batch_size,):
        raise ValueError("slot_ids must be int32 with shape [batch]")
    if active_mask.dtype != torch.bool or active_mask.shape != (batch_size,):
        raise ValueError("active_mask must be bool with shape [batch]")
    if slot_ids.device != pool.device or active_mask.device != pool.device:
        raise ValueError("slot_ids and active_mask must be on the cache device")
    slots = tuple(int(slot) for slot in slot_ids.detach().cpu().tolist())
    active = tuple(bool(flag) for flag in active_mask.detach().cpu().tolist())
    if any(slot < 0 or slot >= cache.cold_cache.max_sequences for slot in slots):
        raise ValueError("slot_ids contain a slot outside cache capacity")
    active_slots = [slot for slot, enabled in zip(slots, active) if enabled]
    if len(set(active_slots)) != len(active_slots):
        raise ValueError("active slot_ids must be unique")
    for slot, enabled in zip(slots, active):
        if not enabled:
            continue
        sequence_id = cache.cold_cache.sequence_id(slot)
        if cache.total_tokens(sequence_id) == 0:
            raise ValueError("cannot attend to an empty active sequence")

    max_pages = cache.cold_cache.max_pages_per_sequence
    part_capacity = max_pages + 1
    block_pages = triton.next_power_of_2(part_capacity)
    if block_pages > 128:
        raise ValueError("batched K1 currently supports at most 64 pages per sequence")
    if workspace is None:
        workspace = BatchedNQKVAttentionWorkspace.create(
            cache,
            batch_size=batch_size,
            query_heads=query_heads,
            output_dtype=query.dtype,
        )
    expected = (
        batch_size,
        max_pages,
        query_heads,
        pool.head_dim,
        query.device,
        query.dtype,
    )
    actual = (
        workspace.batch_size,
        workspace.max_pages,
        workspace.query_heads,
        workspace.head_dim,
        workspace.output.device,
        workspace.output.dtype,
    )
    if actual != expected:
        raise ValueError("batched workspace configuration does not match query and cache")

    attention_scale = 1.0 / math.sqrt(pool.head_dim) if scale is None else float(scale)
    return _launch_nqkv_decode_attention_batched_k1(
        query,
        cache,
        slot_ids=slot_ids,
        active_mask=active_mask,
        attention_scale=attention_scale,
        workspace=workspace,
    )


def _launch_nqkv_decode_attention_batched_k1(
    query: torch.Tensor,
    cache: NQKVHotColdCache,
    *,
    slot_ids: torch.Tensor,
    active_mask: torch.Tensor,
    attention_scale: float,
    workspace: BatchedNQKVAttentionWorkspace,
) -> torch.Tensor:
    """Launch graph-safe batched K1 kernels after control-plane validation."""
    pool = cache.cold_cache.pool
    batch_size, query_heads, _ = query.shape
    max_pages = cache.cold_cache.max_pages_per_sequence
    part_capacity = max_pages + 1
    block_pages = triton.next_power_of_2(part_capacity)
    assert _nqkv_batched_page_attention_kernel is not None
    assert _nqkv_batched_hot_attention_kernel is not None
    assert _nqkv_batched_page_reduce_kernel is not None
    _nqkv_batched_page_attention_kernel[(max_pages, batch_size * query_heads)](
        query,
        slot_ids,
        active_mask,
        cache.cold_cache.page_indices,
        pool.key_codes,
        pool.key_betas,
        pool.key_norms,
        pool.value_codes,
        pool.value_betas,
        pool.value_norms,
        pool.beta_table,
        pool.value_beta_table,
        workspace.page_maxima,
        workspace.page_denominators,
        workspace.page_numerators,
        cache.cold_cache.page_counts,
        cache.cold_cache.last_page_len,
        cache.cold_cache.active,
        cache.cold_cache.sequence_lengths,
        query_heads=query_heads,
        num_kv_heads=pool.num_kv_heads,
        max_sequences=cache.cold_cache.max_sequences,
        max_pages=max_pages,
        part_capacity=part_capacity,
        page_size=pool.page_size,
        head_dim=pool.head_dim,
        blocks_per_row=pool.head_dim // 8,
        code_bytes_per_page=pool.key_codes.stride(0),
        beta_bytes_per_page=pool.key_betas.stride(0),
        rows_per_page=pool.key_norms.stride(0),
        q=pool.q,
        code_width=pool.code_width,
        beta_width=pool.beta_width,
        attention_scale=attention_scale,
        num_warps=4,
    )
    _nqkv_batched_hot_attention_kernel[(batch_size * query_heads,)](
        query,
        slot_ids,
        active_mask,
        cache.hot_keys,
        cache.hot_values,
        workspace.page_maxima,
        workspace.page_denominators,
        workspace.page_numerators,
        cache.cold_cache.page_counts,
        cache.hot_lengths,
        cache.cold_cache.active,
        cache.cold_cache.sequence_lengths,
        query_heads=query_heads,
        num_kv_heads=pool.num_kv_heads,
        max_sequences=cache.cold_cache.max_sequences,
        part_capacity=part_capacity,
        page_size=pool.page_size,
        head_dim=pool.head_dim,
        attention_scale=attention_scale,
        num_warps=4,
    )
    _nqkv_batched_page_reduce_kernel[(batch_size * query_heads,)](
        slot_ids,
        active_mask,
        workspace.page_maxima,
        workspace.page_denominators,
        workspace.page_numerators,
        workspace.output,
        cache.cold_cache.page_counts,
        cache.hot_lengths,
        cache.cold_cache.active,
        cache.cold_cache.sequence_lengths,
        query_heads=query_heads,
        head_dim=pool.head_dim,
        max_sequences=cache.cold_cache.max_sequences,
        part_capacity=part_capacity,
        block_pages=block_pages,
        num_warps=4,
    )
    return workspace.output


class BatchedNQKVAttentionGraphRunner:
    """Capture batched K1 with mutable fixed-address slot metadata."""

    def __init__(
        self,
        cache: NQKVHotColdCache,
        *,
        batch_size: int,
        query_heads: int,
        query_dtype: torch.dtype = torch.bfloat16,
        scale: float | None = None,
    ) -> None:
        pool = cache.cold_cache.pool
        self.cache = cache
        self.query = torch.empty(
            (batch_size, query_heads, pool.head_dim),
            device=pool.device,
            dtype=query_dtype,
        )
        self.slot_ids = torch.zeros(
            batch_size, device=pool.device, dtype=torch.int32
        )
        self.active_mask = torch.zeros(
            batch_size, device=pool.device, dtype=torch.bool
        )
        self.workspace = BatchedNQKVAttentionWorkspace.create(
            cache,
            batch_size=batch_size,
            query_heads=query_heads,
            output_dtype=query_dtype,
        )
        self.attention_scale = (
            1.0 / math.sqrt(pool.head_dim) if scale is None else float(scale)
        )
        self.graph: torch.cuda.CUDAGraph | None = None

    @property
    def captured(self) -> bool:
        return self.graph is not None

    def capture(
        self,
        sample_query: torch.Tensor,
        *,
        slot_ids: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> None:
        if self.captured:
            return
        self._validate_inputs(sample_query, slot_ids, active_mask)
        self.query.copy_(sample_query)
        self.slot_ids.copy_(slot_ids)
        self.active_mask.copy_(active_mask)
        # Run the checked path before capture. The graph body below contains no
        # tensor-to-host conversion or request-ID lookup.
        for _ in range(3):
            nqkv_decode_attention_batched_k1(
                self.query,
                self.cache,
                slot_ids=self.slot_ids,
                active_mask=self.active_mask,
                scale=self.attention_scale,
                workspace=self.workspace,
            )
        torch.cuda.synchronize(self.query.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _launch_nqkv_decode_attention_batched_k1(
                self.query,
                self.cache,
                slot_ids=self.slot_ids,
                active_mask=self.active_mask,
                attention_scale=self.attention_scale,
                workspace=self.workspace,
            )
        self.graph = graph

    def replay(
        self,
        query: torch.Tensor,
        *,
        slot_ids: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.graph is None:
            raise RuntimeError("capture must run before replay")
        self._validate_inputs(query, slot_ids, active_mask)
        self.query.copy_(query)
        self.slot_ids.copy_(slot_ids)
        self.active_mask.copy_(active_mask)
        self.graph.replay()
        return self.workspace.output

    def input_pointers(self) -> tuple[int, int, int]:
        return (
            self.query.data_ptr(),
            self.slot_ids.data_ptr(),
            self.active_mask.data_ptr(),
        )

    def _validate_inputs(
        self,
        query: torch.Tensor,
        slot_ids: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> None:
        if not all(
            isinstance(tensor, torch.Tensor)
            for tensor in (query, slot_ids, active_mask)
        ):
            raise TypeError("query, slot_ids, and active_mask must be tensors")
        if query.shape != self.query.shape or query.device != self.query.device:
            raise ValueError("query must match graph runner shape and device")
        if query.dtype != self.query.dtype:
            raise ValueError("query must match graph runner dtype")
        if slot_ids.shape != self.slot_ids.shape or slot_ids.dtype != torch.int32:
            raise ValueError("slot_ids must be int32 with graph runner batch shape")
        if active_mask.shape != self.active_mask.shape or active_mask.dtype != torch.bool:
            raise ValueError("active_mask must be bool with graph runner batch shape")
        if slot_ids.device != self.query.device or active_mask.device != self.query.device:
            raise ValueError("slot_ids and active_mask must be on the graph runner device")
