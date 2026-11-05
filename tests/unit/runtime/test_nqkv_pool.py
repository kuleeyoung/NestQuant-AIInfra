"""Allocation, reuse, and decoding tests for the CUDA NQKV page pool."""

import unittest

import torch

from nestquant.core.kv_format import PackedNQKVPage
from nestquant.runtime.nqkv_page import dequantize_nqkv_page_triton, quantize_nqkv_page_cuda
from nestquant.runtime.nqkv_pool import NQKVPagePool
from nestquant.runtime.triton_unpack import triton_available


@unittest.skipUnless(triton_available(), "Triton CUDA runtime is unavailable")
class TestNQKVPagePool(unittest.TestCase):
    def _pool(self, capacity: int = 2) -> NQKVPagePool:
        return NQKVPagePool(
            capacity=capacity, page_size=16, num_kv_heads=2, head_dim=64,
            q=14, betas=(0.5, 0.75, 1.0, 1.25),
        )

    def _page(self, seed: int, tokens: int = 16) -> PackedNQKVPage:
        generator = torch.Generator().manual_seed(seed)
        key = torch.randn(tokens, 2, 64, generator=generator).to(torch.bfloat16)
        value = torch.randn(tokens, 2, 64, generator=generator).to(torch.bfloat16)
        return PackedNQKVPage.from_tensors(
            key, value, page_size=16, q=14, betas=(0.5, 0.75, 1.0, 1.25)
        )

    def test_allocate_store_decode_release_and_reuse(self):
        pool = self._pool()
        first = pool.allocate()
        second = pool.allocate()
        self.assertEqual(pool.free_pages, 0)
        with self.assertRaisesRegex(RuntimeError, "exhausted"):
            pool.allocate()

        source = self._page(101, tokens=15)
        pool.store(first, source)
        actual_key, actual_value = dequantize_nqkv_page_triton(pool.prepared(first))
        expected_key, expected_value = source.dequantize(device="cuda")
        self.assertTrue(torch.equal(actual_key, expected_key))
        self.assertTrue(torch.equal(actual_value, expected_value))

        pointer = pool.prepared(first).key_code_payload.data_ptr()
        pool.release(first)
        with self.assertRaisesRegex(RuntimeError, "stale"):
            pool.prepared(first)
        reused = pool.allocate()
        self.assertEqual(reused.page_id, first.page_id)
        self.assertNotEqual(reused.generation, first.generation)
        replacement = self._page(202)
        pool.store(reused, replacement)
        self.assertEqual(pool.prepared(reused).key_code_payload.data_ptr(), pointer)
        pool.release(reused)
        pool.release(second)
        self.assertEqual(pool.free_pages, pool.capacity)

    def test_pages_are_isolated_and_have_fixed_addresses(self):
        pool = self._pool()
        left = pool.allocate()
        right = pool.allocate()
        left_source = self._page(303)
        right_source = self._page(404)
        pool.store(left, left_source)
        pool.store(right, right_source)
        left_ptr = pool.prepared(left).key_code_payload.data_ptr()
        right_ptr = pool.prepared(right).key_code_payload.data_ptr()
        self.assertNotEqual(left_ptr, right_ptr)
        left_key, _ = dequantize_nqkv_page_triton(pool.prepared(left))
        right_key, _ = dequantize_nqkv_page_triton(pool.prepared(right))
        self.assertTrue(torch.equal(left_key, left_source.dequantize(device="cuda")[0]))
        self.assertTrue(torch.equal(right_key, right_source.dequantize(device="cuda")[0]))
        self.assertFalse(torch.equal(left_key, right_key))

    def test_rejects_mismatched_page_configuration(self):
        pool = self._pool(capacity=1)
        handle = pool.allocate()
        generator = torch.Generator().manual_seed(505)
        key = torch.randn(8, 2, 32, generator=generator)
        wrong = PackedNQKVPage.from_tensors(
            key, key.clone(), page_size=16, q=14,
            betas=(0.5, 0.75, 1.0, 1.25),
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            pool.store(handle, wrong)

    def test_store_gpu_quantized_page_without_cpu_payload(self):
        pool = self._pool(capacity=1)
        handle = pool.allocate()
        generator = torch.Generator().manual_seed(606)
        key = torch.randn(15, 2, 64, generator=generator).to(torch.bfloat16).cuda()
        value = torch.randn(15, 2, 64, generator=generator).to(torch.bfloat16).cuda()
        prepared = quantize_nqkv_page_cuda(
            key, value, page_size=16, q=14,
            betas=(0.5, 0.75, 1.0, 1.25),
        )
        pool.store_prepared(handle, prepared)
        actual_key, actual_value = dequantize_nqkv_page_triton(pool.prepared(handle))
        expected_key, expected_value = dequantize_nqkv_page_triton(prepared)
        self.assertTrue(torch.equal(actual_key, expected_key))
        self.assertTrue(torch.equal(actual_value, expected_value))


if __name__ == "__main__":
    unittest.main()
