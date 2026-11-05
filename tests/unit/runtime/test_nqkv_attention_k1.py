"""Numerical tests for two-stage fused packed cold-page attention."""

import unittest

import torch

from nestquant.runtime.nqkv_attention import nqkv_decode_attention_reference
from nestquant.runtime.nqkv_attention_k1 import (
    NQKVAttentionGraphRunner,
    NQKVAttentionWorkspace,
    nqkv_decode_attention_k1,
)
from nestquant.runtime.nqkv_hot_cache import NQKVHotColdCache
from nestquant.runtime.nqkv_pool import NQKVPagePool
from nestquant.runtime.paged_nqkv_cache import PagedNQKVCache
from nestquant.runtime.triton_unpack import triton_available


@unittest.skipUnless(triton_available(), "Triton CUDA runtime is unavailable")
class TestNQKVAttentionK1(unittest.TestCase):
    def _cache(self, *, max_pages=8):
        pool = NQKVPagePool(
            capacity=max_pages, page_size=4, num_kv_heads=2, head_dim=64,
            q=14, betas=(0.5, 0.75, 1.0, 1.25),
        )
        cold = PagedNQKVCache(pool, max_sequences=1, max_pages_per_sequence=max_pages)
        return NQKVHotColdCache(cold)

    def _v2_cache(self, *, max_pages=8):
        pool = NQKVPagePool(
            capacity=max_pages, page_size=4, num_kv_heads=2, head_dim=64, q=14,
            key_betas=(0.25, 0.3271428571, 0.4621428571, 1.2185714286),
            value_betas=(0.2521428571, 0.3992857143, 0.6871428571, 2.0735714286),
        )
        cold = PagedNQKVCache(pool, max_sequences=1, max_pages_per_sequence=max_pages)
        return NQKVHotColdCache(cold)

    def _tokens(self, seed, count):
        generator = torch.Generator().manual_seed(seed)
        key = torch.randn(count, 2, 64, generator=generator).bfloat16().cuda()
        value = torch.randn(count, 2, 64, generator=generator).bfloat16().cuda()
        return key, value

    def test_full_and_partial_cold_pages_match_k0(self):
        for token_count in (3, 4, 9, 16):
            with self.subTest(token_count=token_count):
                cache = self._cache()
                cache.register("request")
                key, value = self._tokens(10 + token_count, token_count)
                cache.append("request", key, value)
                cache.finalize("request")
                query = self._tokens(100 + token_count, 4)[0].reshape(8, 64)
                expected = nqkv_decode_attention_reference(query, cache, "request")
                actual = nqkv_decode_attention_k1(query, cache, "request")
                torch.testing.assert_close(actual, expected, rtol=0, atol=2e-2)

    def test_cold_pages_plus_hot_tail_and_hot_only_match_k0(self):
        for token_count in (2, 6, 10):
            with self.subTest(token_count=token_count):
                cache = self._cache()
                cache.register("request")
                key, value = self._tokens(30 + token_count, token_count)
                cache.append("request", key, value)
                query = self._tokens(40 + token_count, 2)[0].reshape(4, 64)
                expected = nqkv_decode_attention_reference(query, cache, "request")
                actual = nqkv_decode_attention_k1(query, cache, "request")
                torch.testing.assert_close(actual, expected, rtol=0, atol=2e-2)

    def test_rejects_empty_sequence(self):
        cache = self._cache()
        cache.register("request")
        query = self._tokens(4, 2)[0].reshape(4, 64)
        with self.assertRaisesRegex(ValueError, "empty"):
            nqkv_decode_attention_k1(query, cache, "request")

    def test_workspace_addresses_and_output_are_reused(self):
        cache = self._cache()
        cache.register("request")
        key, value = self._tokens(50, 6)
        cache.append("request", key, value)
        query = self._tokens(51, 2)[0].reshape(4, 64)
        workspace = NQKVAttentionWorkspace.create(cache, query_heads=4)
        pointers = workspace.pointers()
        first = nqkv_decode_attention_k1(
            query, cache, "request", workspace=workspace
        ).clone()
        second = nqkv_decode_attention_k1(
            query, cache, "request", workspace=workspace
        )
        self.assertEqual(workspace.pointers(), pointers)
        self.assertEqual(second.data_ptr(), workspace.output.data_ptr())
        torch.testing.assert_close(second, first, rtol=0, atol=0)

    def test_cuda_graph_replay_observes_dynamic_hot_and_cold_lengths(self):
        cache = self._cache()
        cache.register("request")
        key, value = self._tokens(60, 2)
        cache.append("request", key, value)
        query = self._tokens(61, 2)[0].reshape(4, 64)
        runner = NQKVAttentionGraphRunner(
            cache, "request", query_heads=4
        )
        runner.capture(query)
        initial = runner.replay(query).clone()
        initial_expected = nqkv_decode_attention_reference(
            query, cache, "request"
        )
        torch.testing.assert_close(initial, initial_expected, rtol=0, atol=2e-2)

        # Cross a page boundary after capture: two old hot tokens become one
        # packed cold page and one new token remains hot. Kernel scalar state
        # must be read from the GPU tables during replay, not frozen in Graph.
        extra_key, extra_value = self._tokens(62, 3)
        cache.append("request", extra_key, extra_value)
        new_query = self._tokens(63, 2)[0].reshape(4, 64)
        actual = runner.replay(new_query)
        expected = nqkv_decode_attention_reference(new_query, cache, "request")
        torch.testing.assert_close(actual, expected, rtol=0, atol=2e-2)

    def test_v2_distinct_tables_match_k0_across_cold_and_hot(self):
        cache = self._v2_cache()
        cache.register("request")
        key, value = self._tokens(70, 10)
        cache.append("request", key, value)
        query = self._tokens(71, 2)[0].reshape(4, 64)
        expected = nqkv_decode_attention_reference(query, cache, "request")
        actual = nqkv_decode_attention_k1(query, cache, "request")
        torch.testing.assert_close(actual, expected, rtol=0, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
