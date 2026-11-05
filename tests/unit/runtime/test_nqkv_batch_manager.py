"""Tests for transactional multi-layer batched NQKV cache ownership."""

import unittest

import torch

from nestquant.runtime.nqkv_attention_k1 import nqkv_decode_attention_k1
from nestquant.runtime.nqkv_batch_manager import BatchedNQKVCacheManager
from nestquant.runtime.nqkv_hot_cache import NQKVHotColdCache
from nestquant.runtime.nqkv_pool import NQKVPagePool
from nestquant.runtime.paged_nqkv_cache import PagedNQKVCache
from nestquant.runtime.triton_unpack import triton_available


@unittest.skipUnless(triton_available(), "Triton CUDA runtime is unavailable")
class TestBatchedNQKVCacheManager(unittest.TestCase):
    def _layer(self, *, capacity: int = 4, pool_pages: int = 16):
        pool = NQKVPagePool(
            capacity=pool_pages,
            page_size=4,
            num_kv_heads=2,
            head_dim=64,
            q=14,
            betas=(0.5, 0.75, 1.0, 1.25),
        )
        return NQKVHotColdCache(
            PagedNQKVCache(
                pool, max_sequences=capacity, max_pages_per_sequence=4
            )
        )

    def _manager(self, layers: int = 3):
        return BatchedNQKVCacheManager(
            [self._layer() for _ in range(layers)], query_heads=4
        )

    def _batch(self, seed: int, tokens: int = 3):
        generator = torch.Generator().manual_seed(seed)
        shape = (4, tokens, 2, 64)
        key = torch.randn(shape, generator=generator).to(torch.bfloat16).cuda()
        value = torch.randn(shape, generator=generator).to(torch.bfloat16).cuda()
        return key, value

    def test_register_append_attend_and_release_stay_aligned(self):
        manager = self._manager()
        manager.register_batch((("left", 0), ("right", 2)))
        manager.assert_aligned()
        key, value = self._batch(1)
        slots = torch.arange(4, device="cuda", dtype=torch.int32)
        write_mask = torch.tensor(
            [[True, True, True], [False, False, False],
             [True, True, False], [False, False, False]],
            device="cuda",
        )
        result = manager.append_layer(
            1, key, value, slot_ids=slots, write_mask=write_mask
        )
        self.assertEqual(result.appended_tokens, 5)
        query = torch.randn(4, 4, 64, device="cuda", dtype=torch.bfloat16)
        active = torch.tensor([True, False, True, False], device="cuda")
        actual = manager.attend_layer(
            1, query, slot_ids=slots, active_mask=active
        )
        expected_left = nqkv_decode_attention_k1(
            query[0], manager.layer_caches[1], "left"
        )
        expected_right = nqkv_decode_attention_k1(
            query[2], manager.layer_caches[1], "right"
        )
        self.assertTrue(torch.equal(actual[0], expected_left))
        self.assertTrue(torch.equal(actual[2], expected_right))
        self.assertEqual(torch.count_nonzero(actual[1]).item(), 0)
        self.assertEqual(torch.count_nonzero(actual[3]).item(), 0)
        released = manager.release_slots((0, 2))
        self.assertEqual(released, ("left", "right"))
        self.assertEqual(manager.active_requests, 0)
        manager.assert_aligned()
        for cache in manager.layer_caches:
            self.assertEqual(cache.cold_cache.pool.allocated_pages, 0)

    def test_layer_registration_failure_rolls_back_prior_layers(self):
        layers = [self._layer() for _ in range(3)]
        layers[1].register("foreign", slot=2)
        manager = BatchedNQKVCacheManager(layers, query_heads=4)
        with self.assertRaisesRegex(RuntimeError, "already active"):
            manager.register("request", slot=2)
        self.assertEqual(manager.active_requests, 0)
        self.assertFalse(layers[0].cold_cache.active[2].item())
        self.assertTrue(layers[1].cold_cache.active[2].item())
        self.assertFalse(layers[2].cold_cache.active[2].item())

    def test_admission_microbatch_failure_rolls_back_earlier_requests(self):
        manager = self._manager()
        with self.assertRaisesRegex(RuntimeError, "already active"):
            manager.register_batch((("first", 0), ("second", 0)))
        self.assertEqual(manager.active_requests, 0)
        for cache in manager.layer_caches:
            self.assertEqual(cache.cold_cache.active_sequences, 0)

    def test_sixteen_layers_capacity_eight_release_and_reuse(self):
        manager = BatchedNQKVCacheManager(
            [self._layer(capacity=8, pool_pages=8) for _ in range(16)],
            query_heads=4,
        )
        manager.register_batch(tuple((f"request-{slot}", slot) for slot in range(8)))
        self.assertEqual(manager.active_requests, 8)
        manager.assert_aligned()
        released = manager.release_slots((1, 3, 5, 7))
        self.assertEqual(
            released, ("request-1", "request-3", "request-5", "request-7")
        )
        manager.register_batch(
            (("replacement-1", 1), ("replacement-3", 3),
             ("replacement-5", 5), ("replacement-7", 7))
        )
        manager.assert_aligned()
        self.assertEqual(manager.active_requests, 8)
        for slot in (1, 3, 5, 7):
            self.assertEqual(manager.sequence_id(slot), f"replacement-{slot}")
        manager.release_slots(tuple(range(8)))
        manager.assert_aligned()
        self.assertEqual(manager.active_requests, 0)
        for cache in manager.layer_caches:
            self.assertEqual(cache.cold_cache.active_sequences, 0)
            self.assertEqual(cache.cold_cache.pool.allocated_pages, 0)

    def test_multi_page_plan_reserves_and_cancels_every_layer_atomically(self):
        manager = self._manager(layers=3)
        manager.register_batch((("left", 0), ("right", 2)))
        plan = manager.plan_ragged_multi_page((11, 0, 5, 0))
        self.assertEqual(plan.required_pages_per_row, (2, 0, 1, 0))
        before = tuple(
            cache.cold_cache.pool.free_pages for cache in manager.layer_caches
        )

        manager.reserve_multi_page_append(plan)

        for layer_idx, cache in enumerate(manager.layer_caches):
            self.assertEqual(cache.cold_cache.pool.free_pages, before[layer_idx] - 3)
            self.assertEqual(len(manager.pending_multi_page_ids(layer_idx, 0)), 2)
            self.assertEqual(len(manager.pending_multi_page_ids(layer_idx, 2)), 1)
            self.assertEqual(len(cache.cold_cache.reserved_handles("left")), 2)
            self.assertEqual(len(cache.cold_cache.reserved_handles("right")), 1)

        manager.cancel_multi_page_append()

        for layer_idx, cache in enumerate(manager.layer_caches):
            self.assertEqual(cache.cold_cache.pool.free_pages, before[layer_idx])
            self.assertEqual(cache.cold_cache.reserved_handles("left"), ())
            self.assertEqual(cache.cold_cache.reserved_handles("right"), ())
            self.assertTrue(bool((cache.cold_cache.page_indices == -1).all().item()))

    def test_multi_page_reservation_rejects_capacity_before_mutation(self):
        manager = BatchedNQKVCacheManager(
            [self._layer(pool_pages=2) for _ in range(3)], query_heads=4
        )
        manager.register("left", slot=0)
        plan = manager.plan_ragged_multi_page((12, 0, 0, 0))
        before = tuple(
            cache.cold_cache.pool.free_pages for cache in manager.layer_caches
        )
        with self.assertRaisesRegex(RuntimeError, "insufficient"):
            manager.reserve_multi_page_append(plan)
        self.assertEqual(
            tuple(cache.cold_cache.pool.free_pages for cache in manager.layer_caches),
            before,
        )
        self.assertTrue(all(
            cache.cold_cache.reserved_handles("left") == ()
            for cache in manager.layer_caches
        ))


if __name__ == "__main__":
    unittest.main()
