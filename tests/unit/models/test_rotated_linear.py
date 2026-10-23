"""Correctness tests for the paper-style local Hadamard Linear coordinates."""

import unittest

import torch
from torch import nn

from nestquant.core.rotation import hadamard
from nestquant.models.rotated_linear import (
    HadamardRotatedLinear,
    convert_linears_to_hadamard_coordinates_,
    restore_hadamard_structure_from_metadata_,
)


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

    def test_low_precision_source_uses_float32_reference_compute(self):
        linear = nn.Linear(8, 3, bias=False, dtype=torch.bfloat16)
        rotated = HadamardRotatedLinear.from_linear(linear)
        self.assertEqual(rotated.weight.dtype, torch.float32)
        self.assertEqual(rotated.output_dtype, torch.bfloat16)
        output = rotated(torch.randn(2, 8, dtype=torch.bfloat16))
        self.assertEqual(output.dtype, torch.bfloat16)

    def test_source_dtype_storage_mode_keeps_bf16_weights(self):
        linear = nn.Linear(8, 3, bias=True, dtype=torch.bfloat16)
        rotated = HadamardRotatedLinear.from_linear(
            linear, preserve_source_dtype=True
        )
        self.assertEqual(rotated.weight.dtype, torch.bfloat16)
        self.assertEqual(rotated.bias.dtype, torch.bfloat16)
        output = rotated(torch.randn(2, 8, dtype=torch.bfloat16))
        self.assertEqual(output.dtype, torch.bfloat16)

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

    def test_model_converter_preserves_outputs_and_reports_selection(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 4))
                self.other = nn.Linear(8, 3)

        model = Model().to(torch.float64)
        inputs = torch.randn(3, 8, dtype=torch.float64)
        expected = model.block(inputs)
        report = convert_linears_to_hadamard_coordinates_(
            model, include_names={"block.0", "block.2"}
        )
        self.assertEqual(report.converted, ("block.0", "block.2"))
        self.assertIsInstance(model.block[0], HadamardRotatedLinear)
        self.assertIsInstance(model.other, nn.Linear)
        torch.testing.assert_close(model.block(inputs), expected, atol=1e-12, rtol=1e-12)

    def test_model_converter_skips_tied_lm_head(self):
        class TiedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(16, 8)
                self.proj = nn.Linear(8, 8, bias=False)
                self.lm_head = nn.Linear(8, 16, bias=False)
                self.lm_head.weight = self.embed.weight

            def get_input_embeddings(self):
                return self.embed

        model = TiedModel()
        report = convert_linears_to_hadamard_coordinates_(model)
        self.assertEqual(report.converted, ("proj",))
        self.assertEqual(report.skipped_tied, ("lm_head",))
        self.assertIs(model.lm_head.weight, model.embed.weight)

    def test_model_converter_validates_before_mutation(self):
        model = nn.Sequential(nn.Linear(8, 4), nn.Linear(12, 3))
        with self.assertRaisesRegex(ValueError, "non-power-of-two"):
            convert_linears_to_hadamard_coordinates_(model)
        self.assertTrue(all(isinstance(module, nn.Linear) for module in model))
        with self.assertRaisesRegex(ValueError, "unknown Linear"):
            convert_linears_to_hadamard_coordinates_(model, include_names={"missing"})

    def test_checkpoint_metadata_rebuilds_exact_module_structure(self):
        class TiedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(16, 8)
                self.proj = nn.Linear(8, 8, bias=False)
                self.non_power = nn.Linear(12, 3, bias=False)
                self.lm_head = nn.Linear(8, 16, bias=False)
                self.lm_head.weight = self.embed.weight

            def get_input_embeddings(self):
                return self.embed

        source = TiedModel()
        report = convert_linears_to_hadamard_coordinates_(source, strict_power_of_two=False)
        restored = TiedModel()
        restored_report = restore_hadamard_structure_from_metadata_(
            restored, report.checkpoint_metadata()
        )
        self.assertEqual(restored_report, report)
        self.assertIsInstance(restored.proj, HadamardRotatedLinear)
        self.assertIsInstance(restored.non_power, nn.Linear)
        self.assertIs(restored.lm_head.weight, restored.embed.weight)

    def test_checkpoint_metadata_rejects_category_drift_before_conversion(self):
        model = nn.Sequential(nn.Linear(8, 4))
        metadata = {
            "format": "local_hadamard_linear_v1",
            "converted_linear_names": ["0"],
            "skipped_tied_linear_names": ["0"],
            "skipped_non_power_of_two_linear_names": [],
            "weight_storage": "float32_reference",
        }
        with self.assertRaisesRegex(ValueError, "disjoint"):
            restore_hadamard_structure_from_metadata_(model, metadata)
        self.assertIsInstance(model[0], nn.Linear)


if __name__ == "__main__":
    unittest.main()
