"""Tests for preallocated ragged-batch NQKV append semantics."""

import unittest

import torch

from nestquant.runtime.nqkv_batched_writer import BatchedNQKVWriter
from nestquant.runtime.nqkv_hot_cache import NQKVHotColdCache
from nestquant.runtime.nqkv_pool import NQKVPagePool
from nestquant.runtime.paged_nqkv_cache import PagedNQKVCache
from nestquant.runtime.triton_unpack import triton_available


@unittest.skipUnless(triton_available(), "Triton CUDA runtime is unavailable")
class TestBatchedNQKVWriter(unittest.TestCase):
    def _cache(self, *, pool_capacity: int = 12) -> NQKVHotColdCache:
        pool = NQKVPagePool(
            capacity=pool_capacity,
            page_size=4,
            num_kv_heads=2,
            head_dim=64,
            q=14,
            betas=(0.5, 0.75, 1.0, 1.25),
        )
        return NQKVHotColdCache(
            PagedNQKVCache(pool, max_sequences=3, max_pages_per_sequence=4)
        )

    def _batch(self, seed: int, batch: int, tokens: int):
        generator = torch.Generator().manual_seed(seed)
        shape = (batch, tokens, 2, 64)
        key = torch.randn(shape, generator=generator).to(torch.bfloat16).cuda()
        value = torch.randn(shape, generator=generator).to(torch.bfloat16).cuda()
        return key, value

    def test_ragged_append_flushes_only_full_slots_and_preserves_addresses(self):
        cache = self._cache()
        cache.register("a", slot=0)
        cache.register("b", slot=2)
        writer = BatchedNQKVWriter(cache)
        pointers = cache.cold_cache.table_pointers() + cache.buffer_pointers()
        key, value = self._batch(1, 2, 4)
        result = writer.append(
            key,
            value,
            slot_ids=torch.tensor([0, 2], dtype=torch.int32, device="cuda"),
            write_mask=torch.tensor(
                [[True, True, True, False], [True, True, True, True]],
                dtype=torch.bool,
                device="cuda",
            ),
        )
        self.assertEqual(result.appended_tokens, 7)
        self.assertEqual(result.flushed_pages, 1)
        self.assertEqual(result.active_slots, (0, 2))
        self.assertEqual(cache.total_tokens("a"), 3)
        self.assertEqual(cache.total_tokens("b"), 4)
        self.assertEqual(len(cache.cold_cache.handles("a")), 0)
        self.assertEqual(len(cache.cold_cache.handles("b")), 1)
        self.assertEqual(cache.cold_cache.reserved_handles("b"), ())
        self.assertEqual(cache.cold_cache.table_pointers() + cache.buffer_pointers(), pointers)

    def test_matches_sequential_append_across_page_boundary(self):
        batched = self._cache()
        sequential = self._cache()
        for cache in (batched, sequential):
            cache.register("a", slot=0)
            cache.register("b", slot=1)
        first_key, first_value = self._batch(2, 2, 3)
        second_key, second_value = self._batch(3, 2, 3)
        writer = BatchedNQKVWriter(batched)
        all_active = torch.ones((2, 3), dtype=torch.bool, device="cuda")
        slots = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
        writer.append(first_key, first_value, slot_ids=slots, write_mask=all_active)
        writer.append(second_key, second_value, slot_ids=slots, write_mask=all_active)
        for row, sequence_id in enumerate(("a", "b")):
            sequential.append(sequence_id, first_key[row], first_value[row])
            sequential.append(sequence_id, second_key[row], second_value[row])
            actual_key, actual_value = batched.materialize(sequence_id)
            expected_key, expected_value = sequential.materialize(sequence_id)
            self.assertTrue(torch.equal(actual_key, expected_key))
            self.assertTrue(torch.equal(actual_value, expected_value))
            self.assertEqual(batched.total_tokens(sequence_id), 6)

    def test_dense_empty_prefill_matches_regular_append(self):
        dense = self._cache()
        regular = self._cache()
        for cache in (dense, regular):
            cache.register("a", slot=0)
            cache.register("b", slot=1)
            cache.register("c", slot=2)
        key, value = self._batch(23, 3, 3)
        dense_result = BatchedNQKVWriter(dense).append_dense_empty(key, value)
        regular_result = BatchedNQKVWriter(regular).append(
            key,
            value,
            slot_ids=torch.arange(3, dtype=torch.int32, device="cuda"),
            write_mask=torch.ones((3, 3), dtype=torch.bool, device="cuda"),
        )
        self.assertEqual(dense_result, regular_result)
        for sequence_id in ("a", "b", "c"):
            dense_key, dense_value = dense.materialize(sequence_id)
            regular_key, regular_value = regular.materialize(sequence_id)
            self.assertTrue(torch.equal(dense_key, regular_key))
            self.assertTrue(torch.equal(dense_value, regular_value))

    def test_pool_exhaustion_rolls_back_all_reservations_before_write(self):
        cache = self._cache(pool_capacity=1)
        cache.register("a", slot=0)
        cache.register("b", slot=1)
        key, value = self._batch(4, 2, 4)
        writer = BatchedNQKVWriter(cache)
        with self.assertRaisesRegex(RuntimeError, "exhausted"):
            writer.append(
                key,
                value,
                slot_ids=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
                write_mask=torch.ones((2, 4), dtype=torch.bool, device="cuda"),
            )
        self.assertEqual(cache.cold_cache.pool.free_pages, 1)
        self.assertEqual(cache.total_tokens("a"), 0)
        self.assertEqual(cache.total_tokens("b"), 0)
        self.assertEqual(cache.cold_cache.reserved_handles("a"), ())
        self.assertEqual(cache.cold_cache.reserved_handles("b"), ())

    def test_rejects_nonprefix_mask_and_duplicate_slots_without_mutation(self):
        cache = self._cache()
        cache.register("a", slot=0)
        key, value = self._batch(5, 1, 3)
        writer = BatchedNQKVWriter(cache)
        with self.assertRaisesRegex(ValueError, "contiguous prefix"):
            writer.append(
                key,
                value,
                slot_ids=torch.tensor([0], dtype=torch.int32, device="cuda"),
                write_mask=torch.tensor([[True, False, True]], device="cuda"),
            )
        duplicate_key = key.expand(2, -1, -1, -1).contiguous()
        duplicate_value = value.expand(2, -1, -1, -1).contiguous()
        with self.assertRaisesRegex(ValueError, "unique"):
            writer.append(
                duplicate_key,
                duplicate_value,
                slot_ids=torch.tensor([0, 0], dtype=torch.int32, device="cuda"),
                write_mask=torch.ones((2, 3), dtype=torch.bool, device="cuda"),
            )
        self.assertEqual(cache.total_tokens("a"), 0)


if __name__ == "__main__":
    unittest.main()
