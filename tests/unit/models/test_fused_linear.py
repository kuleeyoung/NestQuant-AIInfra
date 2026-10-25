"""Tests for selective model integration of packed fused NQRT Linears."""

from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from nestquant.core.codec import generate_e8_dither_stateless
from nestquant.core.model_checkpoint import PackedNestQuantModel
from nestquant.core.model_checkpoint_file import NQMCFileReader
from nestquant.core.quantizer import quantize_rows
from nestquant.core.rotation import hadamard
from nestquant.core.tensor_checkpoint import PackedNestQuantTensor
from nestquant.models.fused_linear import (
    FusedNQRTLinear,
    convert_linears_to_fused_nqrt_,
)
from nestquant.models.rotated_linear import PAPER_HADAMARD_FORMAT
from nestquant.runtime import triton_available


class TinyProjectionPair(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(32, 16, bias=True)
        self.second = nn.Linear(16, 8, bias=False)


class TestFusedNQRTLinear(unittest.TestCase):
    def _checkpoint(self, model: TinyProjectionPair, directory: str):
        state = model.state_dict()
        seed = 20261025
        stored = hadamard(state["first.weight"])
        dither = generate_e8_dither_stateless(
            stored.numel() // 8, seed
        ).reshape(stored.shape[0], stored.shape[1] // 8, 8)
        rows = quantize_rows(
            stored, q=14, betas=(0.5, 0.75, 1.0, 1.25), dither=dither
        )
        record = PackedNestQuantTensor.from_quantized_rows(
            "first.weight", rows, dither_seed=seed
        )
        metadata = {
            "hadamard_rotation": {
                "format": PAPER_HADAMARD_FORMAT,
                "converted_linear_names": ["first"],
                "skipped_tied_linear_names": [],
                "skipped_non_power_of_two_linear_names": [],
                "weight_storage": "source_model_dtype",
            }
        }
        path = Path(directory) / "tiny.nqmc"
        PackedNestQuantModel.from_state_dict(
            {**state, "first.weight": stored},
            {"first.weight": record},
            model_metadata=metadata,
        ).write_to_file(path)
        return NQMCFileReader(path), record

    def test_selective_conversion_preserves_unselected_modules(self):
        model = TinyProjectionPair().eval()
        with tempfile.TemporaryDirectory() as directory:
            reader, _ = self._checkpoint(model, directory)
            original_second = model.second
            report = convert_linears_to_fused_nqrt_(model, reader, ["first"])

            self.assertIsInstance(model.first, FusedNQRTLinear)
            self.assertIs(model.second, original_second)
            self.assertEqual(report.replaced_linear_names, ("first",))
            self.assertEqual(report.input_hadamard_names, ("first",))
            self.assertFalse(model.first.prepared)

    def test_validation_is_transactional(self):
        model = TinyProjectionPair().eval()
        with tempfile.TemporaryDirectory() as directory:
            reader, _ = self._checkpoint(model, directory)
            original_first = model.first
            with self.assertRaisesRegex(ValueError, "not packed"):
                convert_linears_to_fused_nqrt_(
                    model, reader, ["first", "second"]
                )
            self.assertIs(model.first, original_first)

    @unittest.skipUnless(
        torch.cuda.is_available() and triton_available(),
        "CUDA and Triton are required",
    )
    def test_cuda_forward_matches_dequantized_reference(self):
        torch.manual_seed(71)
        model = TinyProjectionPair().eval()
        with tempfile.TemporaryDirectory() as directory:
            reader, record = self._checkpoint(model, directory)
            bias = model.first.bias.detach().clone().to("cuda", torch.bfloat16)
            layer = FusedNQRTLinear(
                reader,
                "first.weight",
                bias,
                input_hadamard=True,
                output_dtype=torch.bfloat16,
            ).cuda()
            inputs = torch.randn(2, 3, 32, device="cuda", dtype=torch.bfloat16)
            expected = torch.nn.functional.linear(
                hadamard(inputs),
                record.dequantize_chunked(
                    device="cuda", output_dtype=torch.bfloat16, rows_per_chunk=4
                ),
                bias,
            )
            actual = layer(inputs)
            error = (actual.float() - expected.float()).abs()

            self.assertEqual(actual.shape, (2, 3, 16))
            self.assertLessEqual(error.max().item(), 0.0625)
            self.assertTrue(layer.prepared)
            self.assertGreater(layer.packed_device_bytes, 0)
            layer.to(dtype=torch.float32)
            self.assertFalse(layer.prepared)


if __name__ == "__main__":
    unittest.main()
