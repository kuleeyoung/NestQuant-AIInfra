import unittest

import torch

from nestquant.core.ldlq import (
    ActivationCovarianceAccumulator,
    block_unit_lower_ldl,
    hessian_proxy_loss,
    ldlq_quantize_blocks,
    ldlq_quantize_columns,
    qa_ldlq_adjustment,
    regularize_hessian,
    unit_lower_ldl,
)


class TestLDLQ(unittest.TestCase):
    def test_qa_ldlq_adjustment_matches_public_inverse_formula(self):
        weight = torch.tensor([[1.0, -0.5], [0.2, 0.8]], dtype=torch.float64)
        hessian = torch.tensor([[2.0, 0.3], [0.3, 1.0]], dtype=torch.float64)
        adjusted, effective_hessian, epsilon = qa_ldlq_adjustment(
            weight, hessian, activation_variance=3.1, epsilon_constant=155.0
        )
        eps2 = 3.1 / 155.0
        expected_hessian = hessian + torch.eye(2, dtype=torch.float64) * eps2
        expected = weight @ (
            torch.eye(2, dtype=torch.float64)
            - eps2 * torch.linalg.inv(expected_hessian)
        )
        torch.testing.assert_close(adjusted, expected, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(effective_hessian, expected_hessian)
        self.assertAlmostEqual(epsilon, eps2 ** 0.5)

    def test_zero_variance_keeps_weight_and_hessian(self):
        weight = torch.randn(2, 3, dtype=torch.float64)
        hessian = torch.eye(3, dtype=torch.float64)
        adjusted, effective_hessian, epsilon = qa_ldlq_adjustment(
            weight, hessian, activation_variance=0.0
        )
        self.assertTrue(torch.equal(adjusted, weight))
        self.assertTrue(torch.equal(effective_hessian, hessian))
        self.assertEqual(epsilon, 0.0)

    def test_covariance_accumulator_matches_direct_batched_result(self):
        values = torch.tensor(
            [[[1.0, 2.0, -1.0], [0.0, 3.0, 2.0]], [[2.0, -1.0, 1.0], [4.0, 0.0, 2.0]]]
        )
        accumulator = ActivationCovarianceAccumulator(3)
        accumulator.update(values[:1])
        accumulator.update(values[1:])
        rows = values.reshape(-1, 3).double()
        self.assertEqual(accumulator.sample_count, 4)
        torch.testing.assert_close(accumulator.covariance(), rows.T @ rows / 4)

    def test_accumulator_merge_and_empty_error(self):
        left = ActivationCovarianceAccumulator(2)
        right = ActivationCovarianceAccumulator(2)
        with self.assertRaisesRegex(RuntimeError, "no activation"):
            left.covariance()
        left.update(torch.tensor([[1.0, 2.0]]))
        right.update(torch.tensor([[3.0, 4.0], [5.0, 6.0]]))
        left.merge_(right)
        direct = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float64)
        self.assertEqual(left.sample_count, 3)
        torch.testing.assert_close(left.covariance(), direct.T @ direct / 3)

    def test_regularization_and_ldl_reconstruct_damped_hessian(self):
        hessian = torch.tensor(
            [[4.0, 1.5, 0.5], [1.5, 3.0, 0.25], [0.5, 0.25, 2.0]], dtype=torch.float64
        )
        regularized, expected_damping = regularize_hessian(hessian, relative_damp=0.02)
        lower, diagonal, damping = unit_lower_ldl(hessian, relative_damp=0.02)
        reconstructed = lower @ torch.diag(diagonal) @ lower.T
        self.assertAlmostEqual(damping, expected_damping)
        torch.testing.assert_close(lower, torch.tril(lower))
        torch.testing.assert_close(lower.diagonal(), torch.ones(3, dtype=torch.float64))
        torch.testing.assert_close(reconstructed, regularized, atol=1e-12, rtol=1e-12)

    def test_ldlq_identity_quantizer_preserves_weight_exactly(self):
        weight = torch.tensor([[0.2, -1.1, 0.7], [1.3, 0.4, -0.8]], dtype=torch.float64)
        hessian = torch.tensor(
            [[2.0, 0.8, 0.2], [0.8, 1.5, 0.5], [0.2, 0.5, 1.0]], dtype=torch.float64
        )
        lower, _, _ = unit_lower_ldl(hessian, relative_damp=0.0)
        quantized = ldlq_quantize_columns(weight, lower, lambda column: column.clone())
        self.assertTrue(torch.equal(quantized, weight))

    def test_block_ldl_reconstructs_and_has_identity_diagonal_blocks(self):
        generator = torch.Generator().manual_seed(31)
        source = torch.randn(16, 16, generator=generator, dtype=torch.float64)
        hessian = source @ source.T + 0.5 * torch.eye(16, dtype=torch.float64)
        regularized, _ = regularize_hessian(hessian, relative_damp=0.01)
        lower, diagonal_blocks, _ = block_unit_lower_ldl(
            hessian, block_size=4, relative_damp=0.01
        )
        diagonal = torch.block_diag(*diagonal_blocks.unbind())
        torch.testing.assert_close(lower @ diagonal @ lower.T, regularized, atol=1e-10, rtol=1e-10)
        blocks = lower.reshape(4, 4, 4, 4)
        for index in range(4):
            torch.testing.assert_close(blocks[index, :, index, :], torch.eye(4, dtype=torch.float64))

    def test_block_ldlq_matches_manual_two_block_feedback(self):
        weight = torch.tensor([[0.49, 0.51, 1.49, -0.51]], dtype=torch.float64)
        lower = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
             [0.6, -0.2, 1.0, 0.0], [0.3, 0.7, 0.0, 1.0]], dtype=torch.float64
        )
        actual = ldlq_quantize_blocks(weight, lower, lambda block, _index: torch.round(block), block_size=2)
        q1 = torch.round(weight[:, 2:])
        e1 = weight[:, 2:] - q1
        q0 = torch.round(weight[:, :2] + e1 @ lower[2:, :2])
        torch.testing.assert_close(actual, torch.cat((q0, q1), dim=1))

    def test_ldlq_recursion_matches_manual_last_column_first_rounding(self):
        weight = torch.tensor([[0.49, 0.51, 1.49]], dtype=torch.float64)
        lower = torch.tensor(
            [[1.0, 0.0, 0.0], [0.6, 1.0, 0.0], [-0.2, 0.7, 1.0]], dtype=torch.float64
        )
        actual = ldlq_quantize_columns(weight, lower, torch.round)
        q2 = torch.round(weight[:, 2])
        e2 = weight[:, 2] - q2
        q1 = torch.round(weight[:, 1] + e2 * lower[2, 1])
        e1 = weight[:, 1] - q1
        q0 = torch.round(weight[:, 0] + e1 * lower[1, 0] + e2 * lower[2, 0])
        expected = torch.stack((q0, q1, q2), dim=1)
        self.assertTrue(torch.equal(actual, expected))

    def test_hessian_proxy_loss_matches_explicit_output_error(self):
        weight = torch.tensor([[1.0, -0.5], [0.2, 0.8]], dtype=torch.float64)
        quantized = torch.round(weight)
        activations = torch.tensor([[1.0, 2.0, -1.0], [0.5, -1.0, 3.0]], dtype=torch.float64)
        hessian = activations @ activations.T
        explicit = torch.linalg.matrix_norm((weight - quantized) @ activations).square()
        torch.testing.assert_close(hessian_proxy_loss(weight, quantized, hessian), explicit)


if __name__ == "__main__":
    unittest.main()
