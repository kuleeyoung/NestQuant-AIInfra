"""Tests for the Llama BHTD-to-NQKV decode adapter."""

import unittest

import torch

from nestquant.core.rotation import hadamard
from nestquant.runtime.nqkv_attention import nqkv_decode_attention_reference
from nestquant.runtime.nqkv_hot_cache import NQKVHotColdCache
from nestquant.runtime.nqkv_llama_adapter import NQKVLlamaDecodeAdapter
from nestquant.runtime.nqkv_pool import NQKVPagePool
from nestquant.runtime.paged_nqkv_cache import PagedNQKVCache
from nestquant.runtime.triton_unpack import triton_available


@unittest.skipUnless(triton_available(), "Triton CUDA runtime is unavailable")
class TestNQKVLlamaDecodeAdapter(unittest.TestCase):
    def _cache(self):
        pool = NQKVPagePool(
            capacity=8, page_size=4, num_kv_heads=2, head_dim=64, q=14,
            key_betas=(0.25, 0.3271428571, 0.4621428571, 1.2185714286),
            value_betas=(0.2521428571, 0.3992857143, 0.6871428571, 2.0735714286),
        )
        return NQKVHotColdCache(PagedNQKVCache(
            pool, max_sequences=1, max_pages_per_sequence=8
        ))

    def _states(self, seed, tokens):
        generator = torch.Generator().manual_seed(seed)
        query = torch.randn(1, 4, 1, 64, generator=generator).bfloat16().cuda()
        key = torch.randn(1, 2, tokens, 64, generator=generator).bfloat16().cuda()
        value = torch.randn(1, 2, tokens, 64, generator=generator).bfloat16().cuda()
        return query, key, value

    def test_rotated_bhtd_adapter_matches_manual_k0_oracle(self):
        actual_cache = self._cache()
        expected_cache = self._cache()
        actual_cache.register("request")
        expected_cache.register("request")
        query, key, value = self._states(80, 6)
        adapter = NQKVLlamaDecodeAdapter(actual_cache, query_heads=4)
        actual = adapter.append_and_attend(query, key, value, "request")

        rotated_query = hadamard(query.float()).bfloat16()
        rotated_key = hadamard(key.float()).bfloat16()
        expected_cache.append(
            "request", rotated_key[0].permute(1, 0, 2).contiguous(),
            value[0].permute(1, 0, 2).contiguous(),
        )
        expected = nqkv_decode_attention_reference(
            rotated_query[0, :, 0].contiguous(), expected_cache, "request"
        ).view_as(actual)
        torch.testing.assert_close(actual, expected, rtol=0, atol=2e-2)
        self.assertEqual(actual.shape, (1, 4, 1, 64))

    def test_repeated_decode_appends_one_token(self):
        cache = self._cache()
        cache.register("request")
        adapter = NQKVLlamaDecodeAdapter(cache, query_heads=4)
        first = self._states(81, 1)
        second = self._states(82, 1)
        adapter.append_and_attend(*first, "request")
        output = adapter.append_and_attend(*second, "request")
        self.assertEqual(cache.total_tokens("request"), 2)
        self.assertEqual(output.shape, (1, 4, 1, 64))

    def test_rejects_batch_or_query_length_outside_decode_scope(self):
        cache = self._cache()
        cache.register("request")
        adapter = NQKVLlamaDecodeAdapter(cache, query_heads=4)
        query, key, value = self._states(83, 1)
        with self.assertRaisesRegex(ValueError, "batch size one"):
            adapter.append_and_attend(
                query.expand(2, -1, -1, -1),
                key.expand(2, -1, -1, -1),
                value.expand(2, -1, -1, -1), "request",
            )
        with self.assertRaisesRegex(ValueError, "query must have shape"):
            adapter.append_and_attend(
                query.expand(-1, -1, 2, -1), key, value, "request"
            )


if __name__ == "__main__":
    unittest.main()
