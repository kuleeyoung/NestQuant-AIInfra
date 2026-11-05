"""Tests for the fixed-capacity NQKV v1 packed page contract."""

import unittest

import torch

from nestquant.core.kv_format import PackedNQKVPage
from nestquant.core.runtime_quantization import fake_quantize_nestquant_rows


class TestPackedNQKVPage(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(20261027)
        self.key = torch.randn(17, 8, 64, generator=generator, dtype=torch.float32).to(torch.bfloat16)
        self.value = torch.randn(17, 8, 64, generator=generator, dtype=torch.float32).to(torch.bfloat16)
        self.betas = (0.5, 0.75, 1.0, 1.25)

    def test_round_trip_matches_existing_fake_quant_oracle(self):
        page = PackedNQKVPage.from_tensors(
            self.key, self.value, page_size=32, q=14, betas=self.betas
        )
        restored = PackedNQKVPage.from_bytes(page.to_bytes())
        actual_key, actual_value = restored.dequantize()
        expected_key, _ = fake_quantize_nestquant_rows(
            self.key, q=14, betas=self.betas, rows_per_chunk=5
        )
        expected_value, _ = fake_quantize_nestquant_rows(
            self.value, q=14, betas=self.betas, rows_per_chunk=7
        )
        self.assertTrue(torch.equal(actual_key, expected_key))
        self.assertTrue(torch.equal(actual_value, expected_value))
        self.assertEqual(actual_key.shape, self.key.shape)
        self.assertEqual(actual_key.dtype, torch.bfloat16)

    def test_page_has_fixed_payload_size_and_measured_475_bits_per_entry(self):
        one = PackedNQKVPage.from_tensors(
            self.key[:1], self.value[:1], page_size=32, q=14, betas=self.betas
        )
        full = PackedNQKVPage.from_tensors(
            self.key, self.value, page_size=32, q=14, betas=self.betas
        )
        self.assertEqual(one.payload_bytes, full.payload_bytes)
        report = full.storage_report()
        self.assertEqual(report["format"], "NQKV-v1")
        self.assertEqual(report["valid_tokens"], 17)
        self.assertEqual(report["payload_bits_per_entry"], 4.75)
        self.assertGreater(report["container_bytes"], report["payload_bytes"])

    def test_zero_padded_rows_do_not_leak_into_valid_prefix(self):
        page = PackedNQKVPage.from_tensors(
            self.key[:15], self.value[:15], page_size=16, q=14, betas=self.betas
        )
        key, value = page.dequantize(output_dtype=torch.float32)
        self.assertEqual(key.shape, (15, 8, 64))
        self.assertEqual(value.shape, (15, 8, 64))
        self.assertTrue(torch.isfinite(key).all())
        self.assertTrue(torch.isfinite(value).all())

    def test_boundary_fill_levels_round_trip(self):
        for valid_tokens in (1, 15, 16, 17, 32):
            generator = torch.Generator().manual_seed(1000 + valid_tokens)
            key = torch.randn(valid_tokens, 2, 64, generator=generator)
            value = torch.randn(valid_tokens, 2, 64, generator=generator)
            page_size = 16 if valid_tokens <= 16 else 32
            page = PackedNQKVPage.from_tensors(
                key, value, page_size=page_size, q=14, betas=self.betas
            )
            restored_key, restored_value = PackedNQKVPage.from_bytes(
                page.to_bytes()
            ).dequantize()
            expected_key, _ = fake_quantize_nestquant_rows(
                key, q=14, betas=self.betas
            )
            expected_value, _ = fake_quantize_nestquant_rows(
                value, q=14, betas=self.betas
            )
            self.assertTrue(torch.equal(restored_key, expected_key))
            self.assertTrue(torch.equal(restored_value, expected_value))

    def test_checksum_rejects_corruption(self):
        page = PackedNQKVPage.from_tensors(
            self.key[:1], self.value[:1], page_size=16, q=14, betas=self.betas
        )
        blob = bytearray(page.to_bytes())
        blob[-33] ^= 1
        with self.assertRaisesRegex(ValueError, "checksum"):
            PackedNQKVPage.from_bytes(blob)

    def test_v2_round_trip_uses_distinct_key_and_value_beta_tables(self):
        key_betas = (0.25, 0.3271428571, 0.4621428571, 1.2185714286)
        value_betas = (0.2521428571, 0.3992857143, 0.6871428571, 2.0735714286)
        page = PackedNQKVPage.from_tensors(
            self.key[:16], self.value[:16], page_size=16, q=14,
            key_betas=key_betas, value_betas=value_betas,
        )
        restored = PackedNQKVPage.from_bytes(page.to_bytes())
        actual_key, actual_value = restored.dequantize()
        expected_key, _ = fake_quantize_nestquant_rows(
            self.key[:16], q=14, betas=key_betas
        )
        expected_value, _ = fake_quantize_nestquant_rows(
            self.value[:16], q=14, betas=value_betas
        )
        self.assertEqual(restored.format_version, 2)
        self.assertEqual(restored.storage_report()["format"], "NQKV-v2")
        self.assertTrue(torch.equal(actual_key, expected_key))
        self.assertTrue(torch.equal(actual_value, expected_value))
        self.assertEqual(restored.payload_bytes, 9728)

    def test_v2_adds_only_one_float32_beta_table_to_container(self):
        v1 = PackedNQKVPage.from_tensors(
            self.key[:16], self.value[:16], page_size=16, q=14, betas=self.betas
        )
        v2 = PackedNQKVPage.from_tensors(
            self.key[:16], self.value[:16], page_size=16, q=14,
            key_betas=self.betas, value_betas=(0.6, 0.8, 1.1, 1.4),
        )
        self.assertEqual(v2.payload_bytes, v1.payload_bytes)
        # One extra uint16 count in the v2 header and four extra float32 betas.
        self.assertEqual(v2.serialized_bytes - v1.serialized_bytes, 18)

    def test_rejects_invalid_shape_and_capacity(self):
        with self.assertRaisesRegex(ValueError, "NHD"):
            PackedNQKVPage.from_tensors(
                self.key[0], self.value[0], page_size=16, q=14, betas=self.betas
            )
        with self.assertRaisesRegex(ValueError, "token count"):
            PackedNQKVPage.from_tensors(
                self.key, self.value, page_size=16, q=14, betas=self.betas
            )
        with self.assertRaisesRegex(ValueError, "divisible by 8"):
            bad = torch.randn(2, 3, 10)
            PackedNQKVPage.from_tensors(bad, bad.clone(), page_size=16, q=14, betas=self.betas)


if __name__ == "__main__":
    unittest.main()
