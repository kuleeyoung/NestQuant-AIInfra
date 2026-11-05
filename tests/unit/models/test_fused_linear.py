"""Tests for selective model integration of packed fused NQRT Linears."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

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
    GroupedQKVCoordinator,
    SharedHadamardCache,
    SharedPackedActivationCache,
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
            self.assertEqual(report.shared_hadamard_groups, ())
            self.assertEqual(report.grouped_qkv_groups, ())
            self.assertFalse(model.first.prepared)
            self.assertEqual(model.first.compute_backend, "tensor_core_dot")

    def test_rejects_unknown_compute_backend(self):
        model = TinyProjectionPair().eval()
        with tempfile.TemporaryDirectory() as directory:
            reader, _ = self._checkpoint(model, directory)
            with self.assertRaisesRegex(ValueError, "compute_backend"):
                FusedNQRTLinear(reader, "first.weight", compute_backend="unknown")

    def test_splitk_fallback_is_explicit_and_reported(self):
        model = TinyProjectionPair().eval()
        with tempfile.TemporaryDirectory() as directory:
            reader, _ = self._checkpoint(model, directory)
            report = convert_linears_to_fused_nqrt_(
                model,
                reader,
                ["first"],
                enable_m1_down_splitk=True,
                m1_down_splitk_fallback_names=("first",),
            )

            self.assertFalse(model.first.enable_m1_down_splitk)
            self.assertEqual(report.m1_down_splitk_fallback_names, ("first",))

    def test_splitk_fallback_requires_enabled_candidate(self):
        model = TinyProjectionPair().eval()
        with tempfile.TemporaryDirectory() as directory:
            reader, _ = self._checkpoint(model, directory)
            with self.assertRaisesRegex(ValueError, "requires enable"):
                convert_linears_to_fused_nqrt_(
                    model,
                    reader,
                    ["first"],
                    m1_down_splitk_fallback_names=("first",),
                )

    def test_shared_hadamard_cache_reuses_only_the_same_input_object(self):
        cache = SharedHadamardCache(2)
        inputs = torch.randn(2, 8)
        with torch.inference_mode():
            first = cache.transform(inputs)
            second = cache.transform(inputs)
            third = cache.transform(inputs)
        self.assertIs(first, second)
        self.assertIsNot(second, third)
        self.assertEqual(cache.hits, 1)
        self.assertEqual(cache.misses, 2)
        cache.clear()
        with torch.inference_mode():
            cache.transform(inputs)
        self.assertEqual(cache.misses, 3)

    @unittest.skipUnless(
        torch.cuda.is_available() and triton_available(),
        "CUDA and Triton are required",
    )
    def test_shared_packed_activation_cache_encodes_once(self):
        cache = SharedPackedActivationCache(2)
        inputs = torch.randn(2, 32, device="cuda", dtype=torch.bfloat16)
        betas = (0.25, 0.5, 0.75, 1.0)
        with torch.inference_mode():
            first, first_encoded = cache.encode(inputs, q=14, betas=betas)
            second, second_encoded = cache.encode(inputs, q=14, betas=betas)
        self.assertIs(first, second)
        self.assertTrue(first_encoded)
        self.assertFalse(second_encoded)
        self.assertEqual((cache.misses, cache.hits), (1, 1))

    def test_grouped_qkv_coordinator_launches_once_and_clears_after_consumption(self):
        coordinator = GroupedQKVCoordinator(
            Mock(spec=NQMCFileReader), ("q", "k", "v")
        )
        inputs = torch.randn(2, 8)
        outputs = tuple(torch.randn(2, size) for size in (6, 4, 4))
        coordinator._prepared_for = lambda device: ("q", "k", "v")
        with (
            patch("nestquant.models.fused_linear.hadamard", side_effect=lambda value: value),
            patch(
                "nestquant.models.fused_linear.linear_fused_qkv_triton",
                return_value=outputs,
            ) as grouped,
            torch.inference_mode(),
        ):
            self.assertIs(coordinator.project(0, inputs), outputs[0])
            with self.assertRaisesRegex(RuntimeError, "more than once"):
                coordinator.project(0, inputs)
            self.assertIs(coordinator.project(1, inputs), outputs[1])
            self.assertIs(coordinator.project(2, inputs), outputs[2])
            self.assertIs(coordinator.project(0, inputs), outputs[0])
        self.assertEqual(grouped.call_count, 2)
        self.assertEqual(coordinator.launches, 2)

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

    @unittest.skipUnless(
        torch.cuda.is_available() and triton_available(),
        "CUDA and Triton are required",
    )
    def test_packed_w4a4_backend_quantizes_activation_inside_linear(self):
        model = TinyProjectionPair().eval()
        with tempfile.TemporaryDirectory() as directory:
            reader, _ = self._checkpoint(model, directory)
            layer = FusedNQRTLinear(
                reader,
                "first.weight",
                model.first.bias.detach().to("cuda", torch.bfloat16),
                input_hadamard=True,
                output_dtype=torch.bfloat16,
                compute_backend="packed_w4a4",
                activation_q=14,
                activation_betas=(0.25, 0.5, 0.75, 1.0),
            ).cuda()
            inputs = torch.randn(2, 32, device="cuda", dtype=torch.bfloat16)
            output = layer(inputs)
            self.assertEqual(output.shape, (2, 16))
            self.assertTrue(torch.isfinite(output).all())
            self.assertEqual(layer.compute_backend, "packed_w4a4")

    @unittest.skipUnless(
        torch.cuda.is_available() and triton_available(),
        "CUDA and Triton are required",
    )
    def test_packed_w4a4_cuda_mma_dispatch_and_shape_fallback(self):
        model = TinyProjectionPair().eval()
        betas = (0.25, 0.5, 0.75, 1.0)
        with tempfile.TemporaryDirectory() as directory:
            reader, _ = self._checkpoint(model, directory)
            cuda_layer = FusedNQRTLinear(
                reader, "first.weight", None, output_dtype=torch.bfloat16,
                compute_backend="packed_w4a4_cuda_mma", activation_q=14,
                activation_betas=betas,
            ).cuda()
            tiled_layer = FusedNQRTLinear(
                reader, "first.weight", None, output_dtype=torch.bfloat16,
                compute_backend="packed_w4a4_tiled", activation_q=14,
                activation_betas=betas,
            ).cuda()
            m1_only_layer = FusedNQRTLinear(
                reader, "first.weight", None, output_dtype=torch.bfloat16,
                compute_backend="packed_w4a4_cuda_gemv_m1", activation_q=14,
                activation_betas=betas,
            ).cuda()
            sensitive_layer = FusedNQRTLinear(
                reader, "first.weight", None, output_dtype=torch.bfloat16,
                compute_backend="packed_w4a4_cuda_mma", activation_q=14,
                activation_betas=betas, force_w4a4_tiled_fallback=True,
            ).cuda()
            inputs = torch.randn(16, 32, device="cuda", dtype=torch.bfloat16)
            actual = cuda_layer(inputs)
            expected = tiled_layer(inputs)
            self.assertEqual(cuda_layer.w4a4_cuda_mma_dispatch_count, 1)
            self.assertEqual(cuda_layer.w4a4_cuda_mma_m16_dispatch_count, 1)
            self.assertEqual(cuda_layer.w4a4_cuda_mma_m1_dispatch_count, 0)
            self.assertEqual(cuda_layer.w4a4_tiled_fallback_count, 0)
            self.assertTrue(
                torch.allclose(actual.float(), expected.float(), atol=0.02, rtol=0.01)
            )
            m1 = cuda_layer(inputs[:1])
            self.assertEqual(cuda_layer.w4a4_cuda_mma_dispatch_count, 2)
            self.assertEqual(cuda_layer.w4a4_cuda_mma_m1_dispatch_count, 1)
            self.assertEqual(m1.shape, (1, 16))
            self.assertTrue(torch.isfinite(m1).all())
            fallback = cuda_layer(inputs[:2])
            self.assertEqual(cuda_layer.w4a4_cuda_mma_dispatch_count, 3)
            self.assertEqual(cuda_layer.w4a4_cuda_mma_small_m_dispatch_count, 1)
            self.assertEqual(cuda_layer.w4a4_tiled_fallback_count, 0)
            self.assertEqual(fallback.shape, (2, 16))
            self.assertTrue(torch.isfinite(fallback).all())
            sensitive = sensitive_layer(inputs[:2])
            self.assertEqual(
                sensitive_layer.w4a4_cuda_mma_small_m_dispatch_count, 0
            )
            self.assertEqual(sensitive_layer.w4a4_tiled_fallback_count, 1)
            self.assertTrue(torch.allclose(
                sensitive.float(), fallback.float(), atol=0.02, rtol=0.01
            ))
            m16_fallback = m1_only_layer(inputs)
            self.assertEqual(m1_only_layer.w4a4_cuda_mma_m16_dispatch_count, 0)
            self.assertEqual(m1_only_layer.w4a4_tiled_fallback_count, 1)
            self.assertTrue(torch.allclose(
                m16_fallback.float(), expected.float(), atol=0.02, rtol=0.01
            ))
            m128_inputs = torch.randn(
                128, 32, device="cuda", dtype=torch.bfloat16
            )
            m128 = cuda_layer(m128_inputs)
            m128_expected = tiled_layer(m128_inputs)
            self.assertEqual(
                cuda_layer.w4a4_cuda_mma_large_m_staged_dispatch_count, 1
            )
            self.assertEqual(
                cuda_layer.w4a4_cuda_mma_m128_staged_dispatch_count, 1
            )
            self.assertEqual(cuda_layer.w4a4_cuda_mma_dispatch_count, 4)
            self.assertTrue(torch.allclose(
                m128.float(), m128_expected.float(), atol=0.02, rtol=0.01
            ))
            for rows, expected_count in ((32, 2), (64, 3)):
                with self.subTest(rows=rows):
                    large_m_inputs = m128_inputs[:rows]
                    large_m = cuda_layer(large_m_inputs)
                    large_m_expected = tiled_layer(large_m_inputs)
                    self.assertEqual(
                        cuda_layer.w4a4_cuda_mma_large_m_staged_dispatch_count,
                        expected_count,
                    )
                    self.assertTrue(torch.allclose(
                        large_m.float(), large_m_expected.float(),
                        atol=0.02, rtol=0.01,
                    ))
            chunked_inputs = m128_inputs[:40]
            chunked = cuda_layer(chunked_inputs)
            chunked_expected = tiled_layer(chunked_inputs)
            self.assertEqual(
                cuda_layer.w4a4_cuda_mma_chunked_staged_dispatch_count, 1
            )
            self.assertTrue(torch.allclose(
                chunked.float(), chunked_expected.float(), atol=0.04, rtol=0.01
            ))
            sensitive_chunked = sensitive_layer(chunked_inputs)
            self.assertEqual(
                sensitive_layer.w4a4_cuda_mma_chunked_staged_dispatch_count, 0
            )
            self.assertTrue(torch.allclose(
                sensitive_chunked.float(), chunked_expected.float(),
                atol=0.02, rtol=0.01,
            ))
            sensitive_m128 = sensitive_layer(m128_inputs)
            self.assertEqual(
                sensitive_layer.w4a4_cuda_mma_m128_staged_dispatch_count, 0
            )
            self.assertEqual(sensitive_layer.w4a4_tiled_fallback_count, 3)
            self.assertTrue(torch.allclose(
                sensitive_m128.float(), m128_expected.float(),
                atol=0.02, rtol=0.01,
            ))
            m1_only = m1_only_layer(inputs[:1])
            self.assertEqual(m1_only_layer.w4a4_cuda_mma_m1_dispatch_count, 1)
            self.assertEqual(m1_only_layer.w4a4_tiled_fallback_count, 1)
            self.assertTrue(torch.isfinite(m1_only).all())


if __name__ == "__main__":
    unittest.main()
