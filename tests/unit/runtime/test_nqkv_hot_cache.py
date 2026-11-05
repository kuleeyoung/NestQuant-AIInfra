"""Tests for BF16 hot tails and GPU-only flushes into packed NQKV pages."""

import unittest

import torch

from nestquant.runtime.nqkv_hot_cache import NQKVHotColdCache
from nestquant.runtime.nqkv_page import dequantize_nqkv_page_triton
from nestquant.runtime.nqkv_pool import NQKVPagePool
from nestquant.runtime.paged_nqkv_cache import PagedNQKVCache
from nestquant.runtime.triton_unpack import triton_available


@unittest.skipUnless(triton_available(), "Triton CUDA runtime is unavailable")
class TestNQKVHotColdCache(unittest.TestCase):
    def setUp(self):
        pool = NQKVPagePool(
            capacity=8, page_size=4, num_kv_heads=2, head_dim=64,
            q=14, betas=(0.5, 0.75, 1.0, 1.25),
        )
        cold = PagedNQKVCache(pool, max_sequences=2, max_pages_per_sequence=4)
        self.cache = NQKVHotColdCache(cold)

    def _tokens(self, seed: int, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(seed)
        key = torch.randn(count, 2, 64, generator=generator).to(torch.bfloat16).cuda()
        value = torch.randn(count, 2, 64, generator=generator).to(torch.bfloat16).cuda()
        return key, value

    def test_full_page_flush_and_hot_tail_preserve_token_order(self):
        slot = self.cache.register("request")
        pointers = self.cache.buffer_pointers()
        first_key, first_value = self._tokens(1, 3)
        second_key, second_value = self._tokens(2, 3)
        all_key = torch.cat((first_key, second_key))
        all_value = torch.cat((first_value, second_value))
        self.cache.append("request", first_key, first_value)
        self.assertEqual(self.cache.hot_lengths[slot].item(), 3)
        self.assertEqual(self.cache.cold_cache.sequence_lengths[slot].item(), 3)
        self.assertEqual(len(self.cache.cold_cache.handles("request")), 0)
        self.cache.append("request", second_key, second_value)
        self.assertEqual(self.cache.total_tokens("request"), 6)
        self.assertEqual(self.cache.cold_cache.sequence_lengths[slot].item(), 6)
        self.assertEqual(len(self.cache.cold_cache.handles("request")), 1)
        handle = self.cache.cold_cache.handles("request")[0]
        cold_key, cold_value = dequantize_nqkv_page_triton(
            self.cache.cold_cache.pool.prepared(handle)
        )
        # Packed cold rows equal quantized reconstruction; the two-token tail
        # remains original BF16 until it is finalized.
        self.assertEqual(cold_key.shape[0], 4)
        hot_key, hot_value = self.cache.hot_view("request")
        self.assertTrue(torch.equal(hot_key, all_key[4:]))
        self.assertTrue(torch.equal(hot_value, all_value[4:]))
        self.assertEqual(self.cache.buffer_pointers(), pointers)

    def test_finalize_flushes_partial_tail_and_prevents_append(self):
        slot = self.cache.register("request")
        key, value = self._tokens(3, 2)
        self.cache.append("request", key, value)
        self.cache.finalize("request")
        self.assertEqual(len(self.cache.cold_cache.handles("request")), 1)
        self.assertEqual(self.cache.cold_cache.total_tokens("request"), 2)
        self.assertEqual(self.cache.cold_cache.sequence_lengths[slot].item(), 2)
        self.assertEqual(self.cache.hot_view("request")[0].shape[0], 0)
        with self.assertRaisesRegex(RuntimeError, "finalized"):
            self.cache.append("request", key[:1], value[:1])

    def test_release_clears_hot_state_and_reuses_slot(self):
        slot = self.cache.register("old")
        key, value = self._tokens(4, 1)
        self.cache.append("old", key, value)
        self.cache.release("old")
        self.assertEqual(self.cache.hot_lengths[slot].item(), 0)
        self.assertEqual(self.cache.cold_cache.sequence_lengths[slot].item(), 0)
        new_slot = self.cache.register("new")
        self.assertEqual(new_slot, slot)
        self.assertEqual(self.cache.total_tokens("new"), 0)
        self.assertEqual(torch.count_nonzero(self.cache.hot_keys[new_slot]).item(), 0)


if __name__ == "__main__":
    unittest.main()
