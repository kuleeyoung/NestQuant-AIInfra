"""Tests for runtime Algorithm-3 quality simulation."""

import unittest

import torch

from nestquant.core.runtime_quantization import fake_quantize_nestquant_rows


class TestRuntimeQuantization(unittest.TestCase):
    def test_preserves_shape_dtype_and_is_chunk_invariant(self):
        generator = torch.Generator().manual_seed(23)
        tensor = torch.randn(2, 3, 4, 16, generator=generator, dtype=torch.float32)
        betas = (0.25, 0.5, 1.0, 2.0)
        one_chunk, one_stats = fake_quantize_nestquant_rows(
            tensor, q=14, betas=betas, rows_per_chunk=1000
        )
        many_chunks, many_stats = fake_quantize_nestquant_rows(
            tensor, q=14, betas=betas, rows_per_chunk=5
        )
        self.assertEqual(one_chunk.shape, tensor.shape)
        self.assertEqual(one_chunk.dtype, tensor.dtype)
        self.assertTrue(torch.equal(one_chunk, many_chunks))
        self.assertEqual(one_stats["beta_histogram"], many_stats["beta_histogram"])
        self.assertEqual(sum(one_stats["beta_histogram"]), tensor.numel() // 8)
        self.assertEqual(one_stats["overload_count"], many_stats["overload_count"])
        self.assertEqual(one_stats["vector_count"], 24)
        # 4 residue bits + 2/8 beta-index bits + 32/16 row-norm bits.
        self.assertEqual(one_stats["storage_estimate"]["effective_bits_per_entry"], 6.25)

    def test_rejects_non_e8_feature_dimension(self):
        with self.assertRaisesRegex(ValueError, "divisible by 8"):
            fake_quantize_nestquant_rows(
                torch.ones(2, 7), q=14, betas=(0.5, 1.0)
            )


if __name__ == "__main__":
    unittest.main()
