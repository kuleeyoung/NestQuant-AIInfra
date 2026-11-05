"""CUDA tests for GPU-resident NQKV pages and the Triton decoder."""

import unittest

import torch

from nestquant.core.kv_format import PackedNQKVPage
from nestquant.core.packing import pack_nestquant_indices
from nestquant.core.quantizer import quantize_rows
from nestquant.runtime.nqkv_page import (
    dequantize_nqkv_page_triton,
    prepare_nqkv_page,
    quantize_nqkv_page_cuda,
)
from nestquant.runtime.triton_unpack import triton_available


@unittest.skipUnless(triton_available(), "Triton CUDA runtime is unavailable")
class TestPreparedNQKVPage(unittest.TestCase):
    def _page(self, tokens: int = 15, page_size: int = 16) -> PackedNQKVPage:
        generator = torch.Generator().manual_seed(20261028 + tokens)
        key = torch.randn(tokens, 8, 64, generator=generator).to(torch.bfloat16)
        value = torch.randn(tokens, 8, 64, generator=generator).to(torch.bfloat16)
        return PackedNQKVPage.from_tensors(
            key, value, page_size=page_size, q=14,
            betas=(0.5, 0.75, 1.0, 1.25),
        )

    def test_prepared_page_contains_only_compact_resident_tensors(self):
        source = self._page()
        prepared = prepare_nqkv_page(source)
        tensors = (
            prepared.key_code_payload, prepared.key_beta_payload, prepared.key_norms,
            prepared.value_code_payload, prepared.value_beta_payload, prepared.value_norms,
            prepared.beta_table,
        )
        self.assertTrue(all(tensor.device.type == "cuda" for tensor in tensors))
        self.assertEqual(prepared.resident_bytes, source.payload_bytes + 4 * len(source.betas))
        bf16_cache_bytes = 2 * source.page_size * source.num_kv_heads * source.head_dim * 2
        self.assertLess(prepared.resident_bytes, bf16_cache_bytes)

    def test_triton_decode_matches_cuda_reference_at_fill_boundaries(self):
        for tokens, page_size in ((1, 16), (15, 16), (16, 16), (17, 32), (32, 32)):
            with self.subTest(tokens=tokens):
                source = self._page(tokens, page_size)
                prepared = prepare_nqkv_page(source)
                expected_key, expected_value = source.dequantize(device="cuda")
                for tile_blocks in (16, 64, 128):
                    actual_key, actual_value = dequantize_nqkv_page_triton(
                        prepared, tile_blocks=tile_blocks
                    )
                    self.assertTrue(torch.equal(actual_key, expected_key))
                    self.assertTrue(torch.equal(actual_value, expected_value))

    def test_repeated_decode_reuses_resident_payload_addresses(self):
        source = self._page()
        prepared = prepare_nqkv_page(source)
        pointers = (
            prepared.key_code_payload.data_ptr(), prepared.key_beta_payload.data_ptr(),
            prepared.key_norms.data_ptr(), prepared.value_code_payload.data_ptr(),
            prepared.value_beta_payload.data_ptr(), prepared.value_norms.data_ptr(),
            prepared.beta_table.data_ptr(),
        )
        dequantize_nqkv_page_triton(prepared)
        dequantize_nqkv_page_triton(prepared)
        self.assertEqual(pointers, (
            prepared.key_code_payload.data_ptr(), prepared.key_beta_payload.data_ptr(),
            prepared.key_norms.data_ptr(), prepared.value_code_payload.data_ptr(),
            prepared.value_beta_payload.data_ptr(), prepared.value_norms.data_ptr(),
            prepared.beta_table.data_ptr(),
        ))

    def test_cuda_quantize_and_pack_matches_cpu_format(self):
        generator = torch.Generator().manual_seed(20261029)
        key = torch.randn(15, 8, 64, generator=generator).to(torch.bfloat16)
        value = torch.randn(15, 8, 64, generator=generator).to(torch.bfloat16)
        cpu_page = PackedNQKVPage.from_tensors(
            key, value, page_size=16, q=14, betas=(0.5, 0.75, 1.0, 1.25)
        )
        cuda_page = quantize_nqkv_page_cuda(
            key.cuda(), value.cuda(), page_size=16, q=14,
            betas=(0.5, 0.75, 1.0, 1.25),
        )
        self.assertEqual(
            bytes(cuda_page.key_code_payload.cpu().tolist()),
            cpu_page.key_indices.code_payload,
        )
        self.assertEqual(
            bytes(cuda_page.key_beta_payload.cpu().tolist()),
            cpu_page.key_indices.beta_payload,
        )
        torch.testing.assert_close(
            cuda_page.key_norms.cpu(), cpu_page.key_norms, rtol=1e-6, atol=1e-6
        )
        expected_key, expected_value = cpu_page.dequantize(device="cuda")
        actual_key, actual_value = dequantize_nqkv_page_triton(cuda_page)
        self.assertTrue(torch.equal(actual_key, expected_key))
        self.assertTrue(torch.equal(actual_value, expected_value))

    def test_v2_cuda_writer_and_decoder_use_distinct_beta_tables(self):
        generator = torch.Generator().manual_seed(20261030)
        key = torch.randn(16, 8, 64, generator=generator).bfloat16()
        value = torch.randn(16, 8, 64, generator=generator).bfloat16()
        key_betas = (0.25, 0.3271428571, 0.4621428571, 1.2185714286)
        value_betas = (0.2521428571, 0.3992857143, 0.6871428571, 2.0735714286)
        cpu_page = PackedNQKVPage.from_tensors(
            key, value, page_size=16, q=14,
            key_betas=key_betas, value_betas=value_betas,
        )
        cuda_page = quantize_nqkv_page_cuda(
            key.cuda(), value.cuda(), page_size=16, q=14,
            key_betas=key_betas, value_betas=value_betas,
        )
        rows = 16 * 8
        gpu_key = quantize_rows(key.cuda().reshape(rows, 64).float(), 14, key_betas)
        gpu_value = quantize_rows(value.cuda().reshape(rows, 64).float(), 14, value_betas)
        expected_key_indices = pack_nestquant_indices(
            gpu_key.codes, gpu_key.beta_indices, q=14, beta_count=4
        )
        expected_value_indices = pack_nestquant_indices(
            gpu_value.codes, gpu_value.beta_indices, q=14, beta_count=4
        )
        self.assertEqual(
            bytes(cuda_page.key_code_payload.cpu().tolist()),
            expected_key_indices.code_payload,
        )
        self.assertEqual(
            bytes(cuda_page.value_code_payload.cpu().tolist()),
            expected_value_indices.code_payload,
        )
        expected_key = gpu_key.dequantize().reshape(16, 8, 64).bfloat16()
        expected_value = gpu_value.dequantize().reshape(16, 8, 64).bfloat16()
        actual_key, actual_value = dequantize_nqkv_page_triton(cuda_page)
        self.assertTrue(torch.equal(actual_key, expected_key))
        self.assertTrue(torch.equal(actual_value, expected_value))
        self.assertNotEqual(
            cuda_page.key_beta_table.data_ptr(),
            cuda_page.value_beta_table.data_ptr(),
        )


if __name__ == "__main__":
    unittest.main()
