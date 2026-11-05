"""Tests for boundary-only graph-safe NQKV page packing."""

import unittest

import torch

from nestquant.runtime.nqkv_batch_manager import BatchedNQKVCacheManager
from nestquant.runtime.nqkv_graph_pack import launch_pack_full_hot_pages_graph_safe
from nestquant.runtime.nqkv_graph_writer import (
    _launch_append_hot_k1_graph_safe,
    _launch_append_hot_ragged_subpage_graph_safe,
)
from nestquant.runtime.nqkv_hot_cache import NQKVHotColdCache
from nestquant.runtime.nqkv_pool import NQKVPagePool
from nestquant.runtime.paged_nqkv_cache import PagedNQKVCache
from nestquant.runtime.triton_unpack import triton_available


@unittest.skipUnless(triton_available(), "Triton CUDA runtime is unavailable")
class TestNQKVGraphPack(unittest.TestCase):
    def _cache(self, *, pool_pages=4, max_pages=2):
        pool = NQKVPagePool(
            capacity=pool_pages,
            page_size=4,
            num_kv_heads=2,
            head_dim=64,
            q=14,
            key_betas=(0.25, 0.5, 0.75, 1.0),
            value_betas=(0.3, 0.6, 0.9, 1.2),
        )
        return NQKVHotColdCache(
            PagedNQKVCache(
                pool, max_sequences=2, max_pages_per_sequence=max_pages
            )
        )

    def _tokens(self, seed, count):
        generator = torch.Generator().manual_seed(seed)
        shape = (count, 2, 64)
        key = torch.randn(shape, generator=generator).bfloat16().cuda()
        value = torch.randn(shape, generator=generator).bfloat16().cuda()
        return key, value

    def _assert_page_equal(self, actual, actual_id, expected, expected_id):
        for name in (
            "key_codes", "key_betas", "key_norms",
            "value_codes", "value_betas", "value_norms",
        ):
            self.assertTrue(torch.equal(
                getattr(actual.cold_cache.pool, name)[actual_id],
                getattr(expected.cold_cache.pool, name)[expected_id],
            ), name)

    def test_boundary_pack_matches_eager_payload_byte_for_byte(self):
        actual = self._cache()
        expected = self._cache()
        manager = BatchedNQKVCacheManager((actual,), query_heads=4)
        manager.register("left", slot=0)
        expected.register("left", slot=0)
        first_key, first_value = self._tokens(1, 3)
        last_key, last_value = self._tokens(2, 1)
        actual.append("left", first_key, first_value)
        expected.append("left", first_key, first_value)
        expected.append("left", last_key, last_value)

        manager.reserve_graph_transition_pages((0,))
        slot_ids = torch.tensor([0], device="cuda", dtype=torch.int32)
        active = torch.tensor([True], device="cuda")
        _launch_append_hot_k1_graph_safe(
            last_key,
            last_value,
            actual,
            slot_ids=slot_ids,
            active_mask=active,
        )
        boundary = torch.tensor([True, False], device="cuda")
        launch_pack_full_hot_pages_graph_safe(actual, boundary_mask=boundary)
        torch.cuda.synchronize()
        manager.commit_graph_transition_pages((0,))

        actual_id = int(actual.graph_target_page_ids[0].item())
        expected_id = expected.cold_cache.handles("left")[0].page_id
        self._assert_page_equal(actual, actual_id, expected, expected_id)
        self.assertEqual(actual.hot_lengths.tolist(), [0, 0])
        self.assertEqual(actual.cold_cache.page_counts.tolist(), [1, 0])
        self.assertEqual(actual.cold_cache.last_page_len.tolist(), [4, 0])
        actual_key, actual_value = actual.materialize("left")
        expected_key, expected_value = expected.materialize("left")
        self.assertTrue(torch.equal(actual_key, expected_key))
        self.assertTrue(torch.equal(actual_value, expected_value))

    def test_pack_operations_can_be_captured_and_replayed(self):
        cache = self._cache()
        manager = BatchedNQKVCacheManager((cache,), query_heads=4)
        manager.register("left", slot=0)
        first_key, first_value = self._tokens(3, 3)
        last_key, last_value = self._tokens(4, 1)
        cache.append("left", first_key, first_value)
        manager.reserve_graph_transition_pages((0,))
        slots = torch.tensor([0], device="cuda", dtype=torch.int32)
        active = torch.tensor([True], device="cuda")
        boundary = torch.tensor([True, False], device="cuda")

        # Compile all kernels before entering capture.
        _launch_append_hot_k1_graph_safe(
            last_key, last_value, cache, slot_ids=slots, active_mask=active
        )
        launch_pack_full_hot_pages_graph_safe(cache, boundary_mask=boundary)
        cache.hot_keys[0, :3].copy_(first_key)
        cache.hot_values[0, :3].copy_(first_value)
        cache.hot_lengths[0] = 3
        cache.cold_cache.sequence_lengths[0] = 3
        cache.cold_cache.page_counts[0] = 0
        cache.cold_cache.last_page_len[0] = 0

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _launch_append_hot_k1_graph_safe(
                last_key, last_value, cache, slot_ids=slots, active_mask=active
            )
            launch_pack_full_hot_pages_graph_safe(cache, boundary_mask=boundary)
        # Restore the exact pre-boundary state after capture execution.
        cache.hot_keys[0, :3].copy_(first_key)
        cache.hot_values[0, :3].copy_(first_value)
        cache.hot_lengths[0] = 3
        cache.cold_cache.sequence_lengths[0] = 3
        cache.cold_cache.page_counts[0] = 0
        cache.cold_cache.last_page_len[0] = 0
        graph.replay()
        torch.cuda.synchronize()
        manager.commit_graph_transition_pages((0,))
        self.assertEqual(cache.hot_lengths[0].item(), 0)
        self.assertEqual(cache.cold_cache.page_counts[0].item(), 1)
        self.assertEqual(len(cache.cold_cache.handles("left")), 1)

    def test_ragged_multi_token_fill_and_pack_matches_eager(self):
        actual = self._cache()
        expected = self._cache()
        manager = BatchedNQKVCacheManager((actual,), query_heads=4)
        manager.register("left", slot=0)
        expected.register("left", slot=0)
        first_key, first_value = self._tokens(21, 1)
        tail_key, tail_value = self._tokens(22, 3)
        actual.append("left", first_key, first_value)
        expected.append("left", first_key, first_value)
        expected.append("left", tail_key, tail_value)

        manager.reserve_graph_transition_pages((0,))
        key = torch.zeros((2, 3, 2, 64), device="cuda", dtype=torch.bfloat16)
        value = torch.zeros_like(key)
        key[0].copy_(tail_key)
        value[0].copy_(tail_value)
        slots = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
        mask = torch.tensor(
            [[True, True, True], [False, False, False]], device="cuda"
        )
        _launch_append_hot_ragged_subpage_graph_safe(
            key, value, actual, slot_ids=slots, write_mask=mask
        )
        boundary = torch.tensor([True, False], device="cuda")
        launch_pack_full_hot_pages_graph_safe(actual, boundary_mask=boundary)
        torch.cuda.synchronize()
        manager.commit_graph_transition_pages((0,))

        actual_id = actual.cold_cache.handles("left")[0].page_id
        expected_id = expected.cold_cache.handles("left")[0].page_id
        self._assert_page_equal(actual, actual_id, expected, expected_id)
        actual_key, actual_value = actual.materialize("left")
        expected_key, expected_value = expected.materialize("left")
        self.assertTrue(torch.equal(actual_key, expected_key))
        self.assertTrue(torch.equal(actual_value, expected_value))
        self.assertEqual(actual.hot_lengths.tolist(), [0, 0])
        self.assertEqual(actual.cold_cache.page_counts.tolist(), [1, 0])

    def test_graph_transition_reservation_can_roll_back(self):
        cache = self._cache()
        manager = BatchedNQKVCacheManager((cache,), query_heads=4)
        manager.register("left", slot=0)
        free_before = cache.cold_cache.pool.free_pages
        manager.reserve_graph_transition_pages((0,))
        self.assertEqual(cache.cold_cache.pool.free_pages, free_before - 1)
        self.assertEqual(len(cache.cold_cache.reserved_handles("left")), 1)
        manager.cancel_graph_transition_pages((0,))
        self.assertEqual(cache.cold_cache.pool.free_pages, free_before)
        self.assertEqual(cache.cold_cache.reserved_handles("left"), ())
        self.assertEqual(cache.graph_target_page_ids[0].item(), 0)
        self.assertEqual(cache.cold_cache.page_indices[0, 0].item(), -1)

    def test_published_graph_transition_can_roll_back_to_hot_page(self):
        cache = self._cache()
        manager = BatchedNQKVCacheManager((cache,), query_heads=4)
        manager.register("left", slot=0)
        first_key, first_value = self._tokens(31, 1)
        tail_key, tail_value = self._tokens(32, 3)
        cache.append("left", first_key, first_value)
        expected_key = torch.cat((first_key, tail_key), dim=0)
        expected_value = torch.cat((first_value, tail_value), dim=0)
        free_before = cache.cold_cache.pool.free_pages

        manager.reserve_graph_transition_pages((0,))
        key = torch.zeros((2, 3, 2, 64), device="cuda", dtype=torch.bfloat16)
        value = torch.zeros_like(key)
        key[0].copy_(tail_key)
        value[0].copy_(tail_value)
        slots = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
        mask = torch.tensor(
            [[True, True, True], [False, False, False]], device="cuda"
        )
        _launch_append_hot_ragged_subpage_graph_safe(
            key, value, cache, slot_ids=slots, write_mask=mask
        )
        boundary = torch.tensor([True, False], device="cuda")
        launch_pack_full_hot_pages_graph_safe(cache, boundary_mask=boundary)
        torch.cuda.synchronize()
        self.assertEqual(cache.cold_cache.page_counts.tolist(), [1, 0])
        self.assertEqual(cache.hot_lengths.tolist(), [0, 0])

        manager.cancel_graph_transition_pages((0,))

        self.assertEqual(cache.cold_cache.pool.free_pages, free_before)
        self.assertEqual(cache.cold_cache.reserved_handles("left"), ())
        self.assertEqual(cache.cold_cache.page_counts.tolist(), [0, 0])
        self.assertEqual(cache.cold_cache.page_indices[0, 0].item(), -1)
        self.assertEqual(cache.graph_target_page_ids[0].item(), 0)
        self.assertEqual(cache.hot_lengths.tolist(), [4, 0])
        actual_key, actual_value = cache.materialize("left")
        self.assertTrue(torch.equal(actual_key, expected_key))
        self.assertTrue(torch.equal(actual_value, expected_value))

    def test_multi_page_rounds_match_transactional_writer(self):
        actual = self._cache(pool_pages=16, max_pages=8)
        expected = self._cache(pool_pages=16, max_pages=8)
        manager = BatchedNQKVCacheManager((actual,), query_heads=4)
        manager.register_batch((("left", 0), ("right", 1)))
        expected.register("left", slot=0)
        expected.register("right", slot=1)
        left_first_key, left_first_value = self._tokens(41, 2)
        right_first_key, right_first_value = self._tokens(42, 1)
        actual.append("left", left_first_key, left_first_value)
        actual.append("right", right_first_key, right_first_value)
        expected.append("left", left_first_key, left_first_value)
        expected.append("right", right_first_key, right_first_value)
        generator = torch.Generator().manual_seed(43)
        key = torch.randn((2, 11, 2, 64), generator=generator).bfloat16().cuda()
        value = torch.randn((2, 11, 2, 64), generator=generator).bfloat16().cuda()
        row_counts = (11, 5)
        expected.append("left", key[0, :11], value[0, :11])
        expected.append("right", key[1, :5], value[1, :5])
        plan = manager.plan_ragged_multi_page(row_counts)
        manager.reserve_multi_page_append(plan)
        slots = torch.tensor([0, 1], device="cuda", dtype=torch.int32)

        for round_index in range(plan.max_rounds):
            segments = plan.round(round_index)
            starts = torch.tensor(
                [segment.source_start if segment is not None else 0 for segment in segments],
                device="cuda", dtype=torch.int32,
            )
            counts = torch.tensor(
                [segment.token_count if segment is not None else 0 for segment in segments],
                device="cuda", dtype=torch.int32,
            )
            boundary = torch.tensor(
                [bool(segment is not None and segment.flush_after) for segment in segments],
                device="cuda",
            )
            manager.prepare_multi_page_round(0, segments)
            manager.append_layer_ragged_segment(
                0, key, value, slot_ids=slots,
                source_starts=starts, segment_counts=counts,
            )
            if bool(boundary.any().item()):
                launch_pack_full_hot_pages_graph_safe(
                    actual, boundary_mask=boundary, batched_encode=True
                )
        torch.cuda.synchronize()
        manager.commit_multi_page_append(plan)

        self.assertEqual(
            actual.cold_cache.page_counts.tolist(),
            expected.cold_cache.page_counts.tolist(),
        )
        self.assertEqual(actual.hot_lengths.tolist(), expected.hot_lengths.tolist())
        for sequence_id in ("left", "right"):
            self.assertEqual(
                len(actual.cold_cache.handles(sequence_id)),
                len(expected.cold_cache.handles(sequence_id)),
            )
            for actual_handle, expected_handle in zip(
                actual.cold_cache.handles(sequence_id),
                expected.cold_cache.handles(sequence_id),
                strict=True,
            ):
                self._assert_page_equal(
                    actual, actual_handle.page_id,
                    expected, expected_handle.page_id,
                )
            actual_key, actual_value = actual.materialize(sequence_id)
            expected_key, expected_value = expected.materialize(sequence_id)
            self.assertTrue(torch.equal(actual_key, expected_key))
            self.assertTrue(torch.equal(actual_value, expected_value))


if __name__ == "__main__":
    unittest.main()
