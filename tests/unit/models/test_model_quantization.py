"""Tests for fake-quant adapters used by causal-LM quality evaluation."""

import unittest

import torch
from torch import nn

from nestquant.models.quantization import quantize_nestquant_w4_, quantize_rtn_w4_
from nestquant.core.tensor_checkpoint import PackedNestQuantTensor


class TestModelQuantization(unittest.TestCase):
    def test_rtn_w4_updates_linear_weights_in_place(self):
        model = nn.Sequential(nn.Linear(16, 4, bias=False))
        before = model[0].weight.detach().clone()
        stats = quantize_rtn_w4_(model, group_size=8)
        self.assertEqual(stats["layers"], 1)
        self.assertEqual(stats["parameters"], 64)
        self.assertFalse(torch.equal(before, model[0].weight))
        self.assertTrue(torch.isfinite(model[0].weight).all().item())
        self.assertEqual(stats["storage_estimate"]["total_bits"], 512)
        self.assertEqual(stats["storage_estimate"]["effective_bits_per_quantized_parameter"], 8.0)
        self.assertEqual(stats["storage_estimate"]["status"], "estimate_only_no_packed_tensor_written")

    def test_nestquant_quality_simulation_updates_and_reports_beta_subset(self):
        model = nn.Sequential(nn.Linear(16, 4, bias=False))
        before = model[0].weight.detach().clone()
        artifacts = {}
        stats = quantize_nestquant_w4_(model, q=14,
            beta_coefficients=(3.0, 4.5, 6.0, 10.0, 15.0), selected_beta_count=4,
            seed=58, rows_per_chunk=2, output_artifacts=artifacts)
        self.assertEqual(stats["layers"], 1)
        self.assertEqual(stats["parameters"], 64)
        self.assertEqual(len(stats["layer_beta_subsets"]["0"]), 4)
        self.assertFalse(torch.equal(before, model[0].weight))
        self.assertTrue(torch.isfinite(model[0].weight).all().item())
        estimate = stats["storage_estimate"]
        self.assertEqual(estimate["residue_code_bits"], 256)
        self.assertEqual(estimate["beta_index_bits"], 16)
        self.assertEqual(estimate["row_norm_metadata_bits"], 128)
        self.assertEqual(estimate["total_bits"], 624)
        self.assertAlmostEqual(estimate["effective_bits_per_quantized_parameter"], 9.75)
        self.assertTrue(stats["dither_state_captured"])
        restored = artifacts["0.weight"].dequantize()
        self.assertTrue(torch.equal(restored, model[0].weight.detach()))
        self.assertIsNone(artifacts["0.weight"].dither)
        self.assertIsNotNone(artifacts["0.weight"].dither_generator_state)
        restored_artifact = PackedNestQuantTensor.from_bytes(artifacts["0.weight"].to_bytes())
        self.assertTrue(torch.equal(restored_artifact.dequantize(), model[0].weight.detach()))
        self.assertLess(len(restored_artifact.dither_generator_state), 8192)

    def test_nestquant_can_preserve_an_exact_linear_name(self):
        model = nn.Sequential(nn.Linear(16, 4, bias=False), nn.Linear(4, 8, bias=False))
        before_first = model[0].weight.detach().clone()
        before_second = model[1].weight.detach().clone()
        stats = quantize_nestquant_w4_(
            model, q=14,
            beta_coefficients=(3.0, 4.5, 6.0, 10.0, 15.0),
            selected_beta_count=4, seed=58, rows_per_chunk=2,
            excluded_linear_names=("1",),
        )
        self.assertFalse(torch.equal(before_first, model[0].weight))
        self.assertTrue(torch.equal(before_second, model[1].weight))
        self.assertEqual(stats["layers"], 1)
        self.assertEqual(stats["excluded_linear_names"], ["1"])

    def test_nestquant_rejects_unknown_linear_exclusion(self):
        model = nn.Sequential(nn.Linear(16, 4, bias=False))
        with self.assertRaisesRegex(ValueError, "unknown excluded Linear"):
            quantize_nestquant_w4_(
                model, q=14,
                beta_coefficients=(3.0, 4.5, 6.0, 10.0, 15.0),
                selected_beta_count=4, excluded_linear_names=("lm_head",),
            )

    def test_ldlq_codes_export_and_reconstruct_without_hessian(self):
        model = nn.Sequential(nn.Linear(16, 4, bias=False))
        artifacts = {}

        def hessian_provider(_name):
            hessian = torch.eye(16)
            hessian[8:, :8] = 0.15 * torch.eye(8)
            hessian[:8, 8:] = 0.15 * torch.eye(8)
            return "shared-input", hessian

        stats = quantize_nestquant_w4_(
            model, q=14,
            beta_coefficients=(3.0, 4.5, 6.0, 10.0, 15.0),
            selected_beta_count=4, seed=58, rows_per_chunk=2,
            output_artifacts=artifacts, hessian_provider=hessian_provider,
            ldlq_relative_damp=0.01,
        )

        self.assertTrue(stats["ldlq"])
        self.assertTrue(stats["dither_state_captured"])
        self.assertEqual(set(artifacts), {"0.weight"})
        # The Hessian is an encoder-only input. Codes, beta IDs, row norms and
        # dither state are sufficient to reconstruct the committed weight.
        packed = PackedNestQuantTensor.from_bytes(artifacts["0.weight"].to_bytes())
        self.assertTrue(torch.equal(packed.dequantize(), model[0].weight.detach()))

    def test_qa_ldlq_uses_activation_variance_and_exports_adjusted_codes(self):
        model = nn.Sequential(nn.Linear(16, 4, bias=False))
        before = model[0].weight.detach().clone()
        artifacts = {}

        def hessian_provider(_name):
            return "input", torch.eye(16)

        def variance_provider(name):
            self.assertEqual(name, "0")
            return "input", 0.5

        stats = quantize_nestquant_w4_(
            model, q=14,
            beta_coefficients=(3.0, 4.5, 6.0, 10.0, 15.0),
            selected_beta_count=4, seed=58, rows_per_chunk=2,
            output_artifacts=artifacts, hessian_provider=hessian_provider,
            activation_variance_provider=variance_provider,
            qa_epsilon_constant=155.0,
        )
        self.assertTrue(stats["qa_ldlq"])
        self.assertAlmostEqual(stats["layer_activation_variances"]["0"], 0.5)
        self.assertAlmostEqual(stats["layer_qa_epsilons"]["0"], (0.5 / 155.0) ** 0.5)
        self.assertFalse(torch.equal(before, model[0].weight))
        self.assertTrue(torch.equal(artifacts["0.weight"].dequantize(), model[0].weight))

    def test_tied_input_embedding_is_disclosed_and_quantized_with_lm_head(self):
        class TiedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(16, 8)
                self.lm_head = nn.Linear(8, 16, bias=False)
                self.lm_head.weight = self.embedding.weight

            def get_input_embeddings(self):
                return self.embedding

        for quantize in (
            lambda model: quantize_rtn_w4_(model, group_size=8),
            lambda model: quantize_nestquant_w4_(
                model, q=14, beta_coefficients=(3.0, 4.5, 6.0, 10.0, 15.0),
                selected_beta_count=4, seed=58, rows_per_chunk=4),
        ):
            model = TiedModel()
            before = model.embedding.weight.detach().clone()
            stats = quantize(model)
            self.assertTrue(stats["input_embedding_weight_shared_with_linear"])
            self.assertTrue(stats["storage_estimate"]["input_embedding_weight_is_included"])
            self.assertFalse(torch.equal(before, model.embedding.weight))
            self.assertTrue(torch.equal(model.embedding.weight, model.lm_head.weight))


if __name__ == "__main__":
    unittest.main()
