"""Tests for row normalization and fixed-candidate beta quantization."""

import unittest

import torch

from nestquant.core.quantizer import quantize_e8_blocks, quantize_rows, quantize_rows_ldlq


class TestRowQuantizer(unittest.TestCase):
    def test_pre_normalized_block_quantizer_selects_best_beta(self):
        blocks = torch.tensor(
            [[0.2, -0.4, 0.7, 1.1, -0.8, 0.3, 0.5, -1.2],
             [1.4, 0.1, -0.6, 0.9, -1.1, 0.2, 0.8, -0.3]], dtype=torch.float64
        )
        quantized = quantize_e8_blocks(blocks, 14, (0.25, 0.5, 1.0))
        restored = quantized.dequantize()
        self.assertEqual(restored.shape, blocks.shape)
        self.assertTrue(torch.isfinite(restored).all())
        candidate_errors = []
        for beta in (0.25, 0.5, 1.0):
            single = quantize_e8_blocks(blocks, 14, (beta,)).dequantize()
            candidate_errors.append((single - blocks).square().sum(-1))
        expected = torch.stack(candidate_errors).amin(0)
        torch.testing.assert_close((restored - blocks).square().sum(-1), expected)

    def test_ldlq_row_quantizer_identity_hessian_matches_independent_blocks(self):
        generator = torch.Generator().manual_seed(17)
        matrix = torch.randn(5, 16, generator=generator, dtype=torch.float64)
        hessian = torch.eye(16, dtype=torch.float64)
        betas = (0.25, 0.5, 1.0)
        independent = quantize_rows(matrix, 14, betas).dequantize()
        ldlq, metadata = quantize_rows_ldlq(
            matrix, hessian, 14, betas, relative_damp=0.0
        )
        torch.testing.assert_close(ldlq.dequantize(), independent)
        self.assertEqual(metadata["block_count"], 2)
        self.assertEqual(metadata["absolute_damping_added"], 0.0)
    def test_shape_metadata_and_code_ranges(self):
        matrix = torch.randn(3, 24, dtype=torch.float64)
        result = quantize_rows(matrix, q=14, betas=(0.5, 1.0, 2.0))
        self.assertEqual(result.codes.shape, (3, 3, 8))
        self.assertEqual(result.beta_indices.shape, (3, 3))
        self.assertEqual(result.codes.dtype, torch.int64)
        self.assertGreaterEqual(result.codes.min().item(), 0)
        self.assertLess(result.codes.max().item(), 14)
        self.assertTrue(torch.all(result.beta_indices < 3).item())
        self.assertEqual(result.dequantize().shape, matrix.shape)
        self.assertEqual(result.dequantize().dtype, matrix.dtype)

    def test_zero_rows_stay_zero_and_outputs_are_finite(self):
        matrix = torch.tensor([[0.0] * 16, [1.0] * 16], dtype=torch.float32)
        result = quantize_rows(matrix, q=14, betas=(0.5, 1.0, 2.0))
        reconstruction = result.dequantize()
        self.assertTrue(torch.equal(reconstruction[0], torch.zeros(16)))
        self.assertTrue(torch.isfinite(reconstruction).all().item())

    def test_more_beta_candidates_cannot_increase_selected_error(self):
        generator = torch.Generator().manual_seed(58)
        matrix = torch.randn(5, 32, generator=generator, dtype=torch.float64)
        one = quantize_rows(matrix, q=14, betas=(1.0,)).dequantize()
        several = quantize_rows(matrix, q=14, betas=(0.5, 1.0, 2.0)).dequantize()
        error_one = (matrix - one).square().sum()
        error_several = (matrix - several).square().sum()
        self.assertLessEqual(float(error_several), float(error_one) + 1e-12)

    def test_rejects_invalid_matrix_and_beta_inputs(self):
        with self.assertRaises(ValueError):
            quantize_rows(torch.randn(2, 10), q=14, betas=(1.0,))
        with self.assertRaises(ValueError):
            quantize_rows(torch.randn(2, 16), q=14, betas=())
        with self.assertRaises(ValueError):
            quantize_rows(torch.randn(2, 16), q=14, betas=(0.0,))


if __name__ == "__main__":
    unittest.main()
