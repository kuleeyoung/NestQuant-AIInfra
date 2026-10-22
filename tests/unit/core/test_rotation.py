import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from nestquant.core.ldlq import hessian_proxy_loss
from nestquant.core.rotation import hadamard, rotate_covariance, signed_hadamard
from nestquant.models.quantization import quantize_nestquant_w4_


class TestRotation(unittest.TestCase):
    def test_identity_quantizer_leaves_bf16_weights_exactly_unchanged(self):
        model = nn.Sequential(nn.Linear(16, 4, bias=False)).to(torch.bfloat16)
        before = model[0].weight.detach().clone()
        def identity(matrix, *args, **kwargs):
            return SimpleNamespace(dequantize=lambda: matrix.clone())
        with patch("nestquant.models.quantization.quantize_rows", side_effect=identity):
            quantize_nestquant_w4_(
                model, q=14, beta_coefficients=(3.5, 6., 14.5, 50.),
                selected_beta_count=2, rotation_seed=123, rows_per_chunk=2)
        self.assertTrue(torch.equal(before, model[0].weight))

    def test_dense_reference_inverse_and_norm(self):
        h = torch.ones(1, 1, dtype=torch.float64)
        for _ in range(4):
            h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
        h /= 4
        x = torch.randn(3, 16, dtype=torch.float64)
        signs = torch.tensor([1, -1] * 8, dtype=torch.float64)
        actual = signed_hadamard(x, signs)
        torch.testing.assert_close(actual, (x * signs) @ h)
        torch.testing.assert_close(signed_hadamard(actual, signs, inverse=True), x)
        torch.testing.assert_close(actual.square().sum(-1), x.square().sum(-1))

    def test_linear_output_preservation_without_quantization(self):
        x = torch.randn(7, 16, dtype=torch.float64)
        w = torch.randn(5, 16, dtype=torch.float64)
        signs = torch.tensor([-1, 1] * 8, dtype=torch.float64)
        torch.testing.assert_close(
            signed_hadamard(x, signs) @ signed_hadamard(w, signs).T, x @ w.T)
        restored = signed_hadamard(signed_hadamard(w, signs), signs, inverse=True)
        torch.testing.assert_close(x @ restored.T, x @ w.T)

    def test_rotated_covariance_matches_explicit_rotated_activations(self):
        generator = torch.Generator().manual_seed(19)
        activations = torch.randn(23, 16, generator=generator, dtype=torch.float64)
        signs = torch.tensor([1, -1, -1, 1] * 4, dtype=torch.float64)
        hessian = activations.T @ activations / activations.shape[0]

        expected = signed_hadamard(activations, signs)
        expected = expected.T @ expected / activations.shape[0]
        actual = rotate_covariance(hessian, signs)

        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

    def test_rotated_hessian_preserves_output_error_objective(self):
        generator = torch.Generator().manual_seed(23)
        weight = torch.randn(4, 16, generator=generator, dtype=torch.float64)
        signs = torch.tensor([-1, 1] * 8, dtype=torch.float64)
        rotated_weight = signed_hadamard(weight, signs)
        quantized_rotated = torch.round(rotated_weight * 4) / 4
        activations = torch.randn(31, 16, generator=generator, dtype=torch.float64)
        hessian = activations.T @ activations
        rotated_hessian = rotate_covariance(hessian, signs)

        original_basis_quantized = signed_hadamard(
            quantized_rotated, signs, inverse=True
        )
        original_loss = hessian_proxy_loss(weight, original_basis_quantized, hessian)
        rotated_loss = hessian_proxy_loss(
            rotated_weight, quantized_rotated, rotated_hessian
        )

        torch.testing.assert_close(original_loss, rotated_loss, atol=1e-10, rtol=1e-10)

    def test_rejects_bad_shape_and_signs(self):
        with self.assertRaises(ValueError):
            hadamard(torch.ones(2, 12))
        with self.assertRaises(ValueError):
            signed_hadamard(torch.ones(2, 8), torch.zeros(8))

    def test_seeded_adapter_is_reproducible_and_preserves_tied_storage(self):
        class Tied(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(8, 16)
                self.head = nn.Linear(16, 8, bias=False)
                self.head.weight = self.embed.weight
            def get_input_embeddings(self):
                return self.embed
        first, second = Tied(), Tied()
        second.load_state_dict(first.state_dict())
        kwargs = dict(q=14, beta_coefficients=(3.5, 4.5, 6., 14.5, 25., 40., 50.),
                      selected_beta_count=4, seed=58, rotation_seed=123, rows_per_chunk=2)
        stats = quantize_nestquant_w4_(first, **kwargs)
        quantize_nestquant_w4_(second, **kwargs)
        self.assertTrue(torch.equal(first.head.weight, second.head.weight))
        self.assertEqual(first.head.weight.data_ptr(), first.embed.weight.data_ptr())
        self.assertEqual(stats["weight_rotation"], "signed_hadamard_inverse_weight")
        with self.assertRaisesRegex(ValueError, "metadata"):
            quantize_nestquant_w4_(first, output_artifacts={}, **kwargs)

    def test_rotation_and_ldlq_can_be_combined_reproducibly(self):
        first = nn.Sequential(nn.Linear(16, 4, bias=False))
        second = nn.Sequential(nn.Linear(16, 4, bias=False))
        second.load_state_dict(first.state_dict())

        def hessian_provider(_name):
            hessian = torch.eye(16)
            hessian[:8, 8:] = 0.2 * torch.eye(8)
            hessian[8:, :8] = 0.2 * torch.eye(8)
            return "shared-input", hessian

        kwargs = dict(
            q=14, beta_coefficients=(3.5, 4.5, 6., 14.5, 25., 40., 50.),
            selected_beta_count=4, seed=58, rotation_seed=123,
            rows_per_chunk=2, hessian_provider=hessian_provider,
        )
        stats = quantize_nestquant_w4_(first, **kwargs)
        quantize_nestquant_w4_(second, **kwargs)

        self.assertTrue(torch.equal(first[0].weight, second[0].weight))
        self.assertTrue(stats["ldlq"])
        self.assertEqual(
            stats["ldlq_hessian_coordinates"],
            "per_linear_signed_hadamard_input",
        )
        self.assertEqual(
            stats["layer_hessian_coordinates"]["0"],
            "signed_hadamard_input",
        )
