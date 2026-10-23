"""Correctness tests for the paper-style local Hadamard Linear coordinates."""

import unittest

import torch
from torch import nn

from nestquant.core.rotation import hadamard
from nestquant.models.rotated_linear import HadamardRotatedLinear


class TestHadamardRotatedLinear(unittest.TestCase):
    def test_matches_linear_for_batched_inputs_and_bias(self):
        generator = torch.Generator().manual_seed(58)
        linear = nn.Linear(16, 7, bias=True, dtype=torch.float64)
        with torch.no_grad():
            linear.weight.copy_(torch.randn(linear.weight.shape, generator=generator, dtype=torch.float64))
            linear.bias.copy_(torch.randn(linear.bias.shape, generator=generator, dtype=torch.float64))
        inputs = torch.randn(2, 3, 16, generator=generator, dtype=torch.float64)

        rotated = HadamardRotatedLinear.from_linear(linear)

        torch.testing.assert_close(rotated(inputs), linear(inputs), atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(
            rotated.reconstructed_weight(), linear.weight, atol=1e-12, rtol=1e-12
        )
        self.assertEqual(rotated.training, linear.training)

    def test_preserves_input_and_weight_gradients(self):
        generator = torch.Generator().manual_seed(61)
        linear = nn.Linear(8, 4, bias=False, dtype=torch.float64)
        inputs = torch.randn(5, 8, generator=generator, dtype=torch.float64, requires_grad=True)
        reference_inputs = inputs.detach().clone().requires_grad_(True)
        rotated = HadamardRotatedLinear.from_linear(linear)

        linear(reference_inputs).square().sum().backward()
        rotated(inputs).square().sum().backward()

        torch.testing.assert_close(inputs.grad, reference_inputs.grad, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(
            rotated.weight.grad,
            hadamard(linear.weight.grad),
            atol=1e-12,
            rtol=1e-12,
        )

    def test_rejects_non_power_of_two_and_wrong_input_shape(self):
        with self.assertRaisesRegex(ValueError, "power of two"):
            HadamardRotatedLinear.from_linear(nn.Linear(12, 3))
        rotated = HadamardRotatedLinear.from_linear(nn.Linear(8, 3))
        with self.assertRaisesRegex(ValueError, "does not match"):
            rotated(torch.ones(2, 7))


if __name__ == "__main__":
    unittest.main()
