"""Tests for CUDA-Graph-safe one-token NQKV hot-page writes."""

import unittest

import torch

from nestquant.runtime.nqkv_graph_writer import (
    _launch_append_hot_k1_graph_safe,
    _launch_append_hot_ragged_segment_graph_safe,
    append_hot_k1_graph_safe,
    append_hot_ragged_subpage,
)
from nestquant.runtime.nqkv_hot_cache import NQKVHotColdCache
from nestquant.runtime.nqkv_pool import NQKVPagePool
from nestquant.runtime.paged_nqkv_cache import PagedNQKVCache
from nestquant.runtime.triton_unpack import triton_available


@unittest.skipUnless(triton_available(), "Triton CUDA runtime is unavailable")
class TestNQKVGraphWriter(unittest.TestCase):
    def _cache(self):
        pool = NQKVPagePool(
            capacity=4,
            page_size=4,
            num_kv_heads=2,
            head_dim=64,
            q=14,
            betas=(0.5, 0.75, 1.0, 1.25),
        )
        cache = NQKVHotColdCache(
            PagedNQKVCache(pool, max_sequences=2, max_pages_per_sequence=2)
        )
        cache.register("left", slot=0)
        cache.register("right", slot=1)
        return cache

    def _values(self, seed):
        generator = torch.Generator().manual_seed(seed)
        shape = (2, 2, 64)
        key = torch.randn(shape, generator=generator).bfloat16().cuda()
        value = torch.randn(shape, generator=generator).bfloat16().cuda()
        return key, value

    def test_checked_append_writes_only_active_rows(self):
        cache = self._cache()
        key, value = self._values(1)
        slots = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
        active = torch.tensor([True, False], device="cuda")
        append_hot_k1_graph_safe(
            key, value, cache, slot_ids=slots, active_mask=active
        )
        self.assertEqual(cache.hot_lengths.tolist(), [1, 0])
        self.assertEqual(cache.cold_cache.sequence_lengths.tolist(), [1, 0])
        self.assertTrue(torch.equal(cache.hot_keys[0, 0], key[0]))
        self.assertEqual(torch.count_nonzero(cache.hot_keys[1]).item(), 0)

    def test_graph_replay_observes_new_inputs_and_increments_gpu_lengths(self):
        cache = self._cache()
        static_key, static_value = self._values(2)
        slots = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
        active = torch.tensor([True, True], device="cuda")

        # Compile before capture, then erase the warmup mutation.
        append_hot_k1_graph_safe(
            static_key, static_value, cache, slot_ids=slots, active_mask=active
        )
        cache.hot_keys.zero_()
        cache.hot_values.zero_()
        cache.hot_lengths.zero_()
        cache.cold_cache.sequence_lengths.zero_()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _launch_append_hot_k1_graph_safe(
                static_key,
                static_value,
                cache,
                slot_ids=slots,
                active_mask=active,
            )
        # Capture executes the work once; reset graph-visible state before the
        # first measured replay so token positions remain deterministic.
        cache.hot_keys.zero_()
        cache.hot_values.zero_()
        cache.hot_lengths.zero_()
        cache.cold_cache.sequence_lengths.zero_()

        first_key, first_value = self._values(3)
        static_key.copy_(first_key)
        static_value.copy_(first_value)
        graph.replay()
        second_key, second_value = self._values(4)
        static_key.copy_(second_key)
        static_value.copy_(second_value)
        active.copy_(torch.tensor([False, True], device="cuda"))
        graph.replay()
        torch.cuda.synchronize()

        self.assertEqual(cache.hot_lengths.tolist(), [1, 2])
        self.assertEqual(cache.cold_cache.sequence_lengths.tolist(), [1, 2])
        self.assertTrue(torch.equal(cache.hot_keys[0, 0], first_key[0]))
        self.assertTrue(torch.equal(cache.hot_values[0, 0], first_value[0]))
        self.assertTrue(torch.equal(cache.hot_keys[1, 0], first_key[1]))
        self.assertTrue(torch.equal(cache.hot_keys[1, 1], second_key[1]))
        self.assertTrue(torch.equal(cache.hot_values[1, 1], second_value[1]))

    def test_checked_append_rejects_full_hot_page(self):
        cache = self._cache()
        key, value = self._values(5)
        slots = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
        active = torch.tensor([True, False], device="cuda")
        cache.hot_lengths[0] = cache.cold_cache.pool.page_size
        with self.assertRaisesRegex(RuntimeError, "full"):
            append_hot_k1_graph_safe(
                key, value, cache, slot_ids=slots, active_mask=active
            )

    def test_ragged_subpage_matches_regular_writer(self):
        actual = self._cache()
        expected = self._cache()
        generator = torch.Generator().manual_seed(17)
        shape = (2, 3, 2, 64)
        key = torch.randn(shape, generator=generator).bfloat16().cuda()
        value = torch.randn(shape, generator=generator).bfloat16().cuda()
        slots = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
        mask = torch.tensor(
            [[True, True, False], [True, False, False]], device="cuda"
        )
        append_hot_ragged_subpage(
            key, value, actual, slot_ids=slots, write_mask=mask
        )
        from nestquant.runtime.nqkv_batched_writer import BatchedNQKVWriter
        BatchedNQKVWriter(expected).append(
            key, value, slot_ids=slots, write_mask=mask
        )
        torch.cuda.synchronize()
        self.assertEqual(actual.hot_lengths.tolist(), [2, 1])
        self.assertEqual(actual.cold_cache.sequence_lengths.tolist(), [2, 1])
        for sequence_id in ("left", "right"):
            actual_key, actual_value = actual.materialize(sequence_id)
            expected_key, expected_value = expected.materialize(sequence_id)
            self.assertTrue(torch.equal(actual_key, expected_key))
            self.assertTrue(torch.equal(actual_value, expected_value))

    def test_ragged_subpage_rejects_nonprefix_and_page_fill(self):
        cache = self._cache()
        key = torch.zeros((2, 3, 2, 64), device="cuda", dtype=torch.bfloat16)
        slots = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "contiguous prefix"):
            append_hot_ragged_subpage(
                key, key, cache, slot_ids=slots,
                write_mask=torch.tensor(
                    [[True, False, True], [False, False, False]], device="cuda"
                ),
            )
        first_key, first_value = self._values(18)
        append_hot_k1_graph_safe(
            first_key,
            first_value,
            cache,
            slot_ids=slots,
            active_mask=torch.tensor([True, False], device="cuda"),
        )
        cache.synchronize_host_length(0, 1)
        with self.assertRaisesRegex(RuntimeError, "fill or cross"):
            append_hot_ragged_subpage(
                key, key, cache, slot_ids=slots,
                write_mask=torch.tensor(
                    [[True, True, True], [False, False, False]], device="cuda"
                ),
            )

    def test_ragged_segment_uses_independent_source_offsets(self):
        cache = self._cache()
        generator = torch.Generator().manual_seed(23)
        key = torch.randn((2, 5, 2, 64), generator=generator).bfloat16().cuda()
        value = torch.randn(
            (2, 5, 2, 64), generator=generator
        ).bfloat16().cuda()
        slots = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
        starts = torch.tensor([1, 3], device="cuda", dtype=torch.int32)
        counts = torch.tensor([3, 2], device="cuda", dtype=torch.int32)

        _launch_append_hot_ragged_segment_graph_safe(
            key,
            value,
            cache,
            slot_ids=slots,
            source_starts=starts,
            segment_counts=counts,
        )
        torch.cuda.synchronize()

        self.assertEqual(cache.hot_lengths.tolist(), [3, 2])
        self.assertEqual(cache.cold_cache.sequence_lengths.tolist(), [3, 2])
        self.assertTrue(torch.equal(cache.hot_keys[0, :3], key[0, 1:4]))
        self.assertTrue(torch.equal(cache.hot_values[0, :3], value[0, 1:4]))
        self.assertTrue(torch.equal(cache.hot_keys[1, :2], key[1, 3:5]))
        self.assertTrue(torch.equal(cache.hot_values[1, :2], value[1, 3:5]))


if __name__ == "__main__":
    unittest.main()
