"""Tests for page-streamed NQKV decode attention."""

import math
import unittest

import torch

from nestquant.runtime.nqkv_attention import nqkv_decode_attention_reference
from nestquant.runtime.nqkv_hot_cache import NQKVHotColdCache
from nestquant.runtime.nqkv_page import dequantize_nqkv_page_triton
from nestquant.runtime.nqkv_pool import NQKVPagePool
from nestquant.runtime.paged_nqkv_cache import PagedNQKVCache
from nestquant.runtime.triton_unpack import triton_available


@unittest.skipUnless(triton_available(), "Triton CUDA runtime is unavailable")
class TestNQKVDecodeAttention(unittest.TestCase):
    def setUp(self):
        pool = NQKVPagePool(
            capacity=8, page_size=4, num_kv_heads=2, head_dim=64,
            q=14, betas=(0.5, 0.75, 1.0, 1.25),
        )
        cold = PagedNQKVCache(pool, max_sequences=2, max_pages_per_sequence=4)
        self.cache = NQKVHotColdCache(cold)

    def _tokens(self, seed: int, count: int):
        generator = torch.Generator().manual_seed(seed)
        key = torch.randn(count, 2, 64, generator=generator).bfloat16().cuda()
        value = torch.randn(count, 2, 64, generator=generator).bfloat16().cuda()
        return key, value

    def _materialized_oracle(self, sequence_id, query):
        keys, values = [], []
        for handle in self.cache.cold_cache.handles(sequence_id):
            key, value = dequantize_nqkv_page_triton(
                self.cache.cold_cache.pool.prepared(handle), output_dtype=query.dtype
            )
            keys.append(key)
            values.append(value)
        hot_key, hot_value = self.cache.hot_view(sequence_id)
        if hot_key.shape[0]:
            keys.append(hot_key)
            values.append(hot_value)
        key = torch.cat(keys).float()
        value = torch.cat(values).float()
        kv_head = torch.arange(query.shape[0], device=query.device) // (
            query.shape[0] // key.shape[1]
        )
        scores = torch.einsum("hd,thd->ht", query.float(), key[:, kv_head])
        weights = torch.softmax(scores / math.sqrt(query.shape[1]), dim=1)
        return torch.einsum("ht,thd->hd", weights, value[:, kv_head]).to(query.dtype)

    def test_two_cold_pages_and_hot_tail_match_materialized_oracle(self):
        self.cache.register("request")
        key, value = self._tokens(1, 10)
        self.cache.append("request", key, value)
        query = self._tokens(2, 2)[0].reshape(4, 64)
        actual = nqkv_decode_attention_reference(query, self.cache, "request")
        expected = self._materialized_oracle("request", query)
        torch.testing.assert_close(actual, expected, rtol=0, atol=2e-2)

    def test_hot_only_supports_grouped_query_attention(self):
        self.cache.register("request")
        key, value = self._tokens(3, 3)
        self.cache.append("request", key, value)
        query = self._tokens(4, 4)[0].reshape(8, 64)
        actual = nqkv_decode_attention_reference(query, self.cache, "request")
        expected = self._materialized_oracle("request", query)
        torch.testing.assert_close(actual, expected, rtol=0, atol=2e-2)

    def test_finalized_partial_cold_page_matches_oracle(self):
        self.cache.register("request")
        key, value = self._tokens(5, 3)
        self.cache.append("request", key, value)
        self.cache.finalize("request")
        query = self._tokens(6, 2)[0].reshape(4, 64)
        actual = nqkv_decode_attention_reference(query, self.cache, "request")
        expected = self._materialized_oracle("request", query)
        torch.testing.assert_close(actual, expected, rtol=0, atol=2e-2)

    def test_rejects_empty_sequence_and_invalid_gqa_ratio(self):
        self.cache.register("empty")
        with self.assertRaisesRegex(ValueError, "empty"):
            nqkv_decode_attention_reference(
                torch.zeros(4, 64, device="cuda", dtype=torch.bfloat16),
                self.cache, "empty",
            )
        key, value = self._tokens(7, 1)
        self.cache.append("empty", key, value)
        with self.assertRaisesRegex(ValueError, "multiple"):
            nqkv_decode_attention_reference(
                torch.zeros(3, 64, device="cuda", dtype=torch.bfloat16),
                self.cache, "empty",
            )


if __name__ == "__main__":
    unittest.main()
