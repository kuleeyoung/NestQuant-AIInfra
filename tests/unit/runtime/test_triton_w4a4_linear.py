"""Correctness tests for direct packed-W4 by packed-A4 execution."""

import unittest

import torch

from nestquant.core.codec import generate_e8_dither_stateless
from nestquant.core.quantizer import quantize_rows
from nestquant.core.tensor_checkpoint import PackedNestQuantTensor
from nestquant.runtime.triton_fused_linear import linear_fused_triton, prepare_fused_linear
from nestquant.runtime.triton_unpack import triton_available
from nestquant.runtime.triton_w4a4_linear import (
    linear_packed_w4a4_triton,
    linear_packed_w4a4_tiled_triton,
    quantize_activation_cuda,
    select_w4a4_tiled_config,
)


@unittest.skipUnless(triton_available(), "Triton CUDA runtime is unavailable")
class TestPackedW4A4Linear(unittest.TestCase):
    def test_launch_config_scales_tiles_without_changing_block_k(self):
        self.assertEqual(
            select_w4a4_tiled_config(1, 2048, 2048),
            {"block_m": 16, "block_n": 16, "block_k": 32, "num_warps": 8},
        )
        self.assertEqual(
            select_w4a4_tiled_config(17, 8192, 2048),
            {"block_m": 32, "block_n": 32, "block_k": 32, "num_warps": 8},
        )
        self.assertEqual(
            select_w4a4_tiled_config(17, 2048, 8192),
            {"block_m": 32, "block_n": 16, "block_k": 32, "num_warps": 8},
        )
        self.assertEqual(
            select_w4a4_tiled_config(128, 2048, 8192),
            {"block_m": 64, "block_n": 32, "block_k": 32, "num_warps": 8},
        )

    def test_launch_config_rejects_nonpositive_shapes(self):
        with self.assertRaises(ValueError):
            select_w4a4_tiled_config(0, 2048, 2048)

    def test_weight_stationary_m128_launch_is_accepted(self):
        inputs = torch.randn(128, 32, device="cuda", dtype=torch.bfloat16)
        packed = quantize_activation_cuda(
            inputs, q=14, betas=(0.25, 0.5, 0.75, 1.0)
        )
        output = linear_packed_w4a4_tiled_triton(
            packed, self._weight(rows=32), block_m=128, block_n=16,
            block_k=32, num_warps=8,
        )
        self.assertEqual(output.shape, (128, 32))

    def _weight(self, rows=7, features=32):
        matrix = torch.randn(
            rows, features, generator=torch.Generator().manual_seed(110)
        ).cuda()
        seed = 20261028
        dither = generate_e8_dither_stateless(
            rows * features // 8, seed, device="cuda"
        ).reshape(rows, features // 8, 8)
        quantized = quantize_rows(
            matrix, q=14, betas=(0.25, 0.5, 0.75, 1.0), dither=dither
        )
        record = PackedNestQuantTensor.from_quantized_rows(
            "weight", quantized, dither_seed=seed
        )
        return prepare_fused_linear(record)

    def test_packed_activation_is_smaller_than_bf16_and_stays_uint8(self):
        inputs = torch.randn(3, 32, device="cuda", dtype=torch.bfloat16)
        packed = quantize_activation_cuda(
            inputs, q=14, betas=(0.25, 0.5, 0.75, 1.0)
        )
        self.assertEqual(packed.code_payload.dtype, torch.uint8)
        self.assertEqual(packed.beta_payload.dtype, torch.uint8)
        self.assertLess(packed.resident_bytes, inputs.numel() * inputs.element_size())
        self.assertEqual(packed.original_shape, (3, 32))

    def test_direct_packed_dot_matches_dequantize_then_w4_reference(self):
        inputs = torch.randn(
            3, 32, generator=torch.Generator().manual_seed(111)
        ).bfloat16().cuda()
        betas = (0.25, 0.5, 0.75, 1.0)
        packed = quantize_activation_cuda(inputs, q=14, betas=betas)
        weight = self._weight()
        actual = linear_packed_w4a4_triton(packed, weight)
        restored = quantize_rows(inputs.float(), q=14, betas=betas).dequantize()
        expected = linear_fused_triton(restored.bfloat16(), weight)
        self.assertEqual(actual.shape, (3, 7))
        self.assertTrue(torch.allclose(actual.float(), expected.float(), atol=0.08, rtol=0.02))

    def test_tiled_kernel_matches_scalar_packed_oracle(self):
        inputs = torch.randn(
            19, 32, generator=torch.Generator().manual_seed(112)
        ).bfloat16().cuda()
        packed = quantize_activation_cuda(
            inputs, q=14, betas=(0.25, 0.5, 0.75, 1.0)
        )
        weight = self._weight(rows=23)
        scalar = linear_packed_w4a4_triton(packed, weight)
        tiled = linear_packed_w4a4_tiled_triton(
            packed, weight, block_m=16, block_n=16, block_k=32,
        )
        self.assertEqual(tiled.shape, (19, 23))
        difference = (tiled.float() - scalar.float()).abs()
        self.assertTrue(
            torch.allclose(tiled.float(), scalar.float(), atol=0.08, rtol=0.02),
            f"max={difference.max().item()} mean={difference.mean().item()} "
            f"scalar={scalar[:2, :4]} tiled={tiled[:2, :4]}",
        )


if __name__ == "__main__":
    unittest.main()
