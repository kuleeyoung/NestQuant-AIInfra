"""Correctness gates for the experimental direct-packed CUDA W4A4 MMA path."""

import math
import unittest
from dataclasses import replace

import torch

from nestquant.core.codec import decode_nested, generate_e8_dither_stateless
from nestquant.core.quantizer import quantize_rows
from nestquant.core.tensor_checkpoint import PackedNestQuantTensor
from nestquant.runtime.cuda_w4a4_mma import (
    decode_packed_a4_m16,
    linear_packed_w4a4_gemv_m1,
    linear_packed_w4a4_gemv_m1_packed_word_probe,
    linear_packed_w4a4_gemv_m1_splitk_probe,
    linear_packed_w4a4_gemv_m1_splitk_compensated_fp32_probe,
    linear_packed_w4a4_mma_m8_transposed_probe,
    linear_packed_w4a4_mma_m32_multiwarp_probe,
    linear_packed_w4a4_mma_large_m_multiwarp_probe,
    linear_packed_w4a4_mma_large_m_staged,
    linear_packed_w4a4_mma_chunked_staged_probe,
    linear_packed_w4a4_mma_m128_staged_probe,
    linear_packed_w4a4_mma_m128_pipeline_probe,
    linear_packed_w4a4_mma_small_m_transposed_probe,
    linear_packed_w4a4_mma_small_m_splitk_probe,
    linear_packed_w4a4_mma_m16,
)
from nestquant.runtime.cuda_w4a4_cute import (
    _cutlass_root,
    linear_packed_w4a4_cute_large_m_probe,
    linear_packed_w4a4_cute_m128_probe,
)
from nestquant.runtime.triton_fused_linear import prepare_fused_linear
from nestquant.runtime.triton_unpack import triton_available
from nestquant.runtime.triton_w4a4_linear import (
    PreparedNQA4Activation,
    linear_packed_w4a4_tiled_triton,
    quantize_activation_cuda,
)


BETAS = (0.25, 0.5, 0.75, 1.0)


def _cute_headers_available() -> bool:
    try:
        _cutlass_root()
    except RuntimeError:
        return False
    return True


@unittest.skipUnless(triton_available(), "Triton CUDA runtime is unavailable")
class TestCudaW4A4MMA(unittest.TestCase):
    @staticmethod
    def _pack_nibbles(values: torch.Tensor) -> torch.Tensor:
        flat = values.reshape(-1).to(torch.uint8)
        return (flat[0::2] | (flat[1::2] << 4)).contiguous()

    @staticmethod
    def _weight(rows: int = 16, features: int = 32):
        generator = torch.Generator().manual_seed(20261029)
        matrix = torch.randn(rows, features, generator=generator).cuda()
        seed = 20261029
        dither = generate_e8_dither_stateless(
            rows * features // 8, seed, device="cuda"
        ).reshape(rows, features // 8, 8)
        quantized = quantize_rows(
            matrix, q=14, betas=BETAS, dither=dither
        )
        return prepare_fused_linear(
            PackedNestQuantTensor.from_quantized_rows(
                "weight", quantized, dither_seed=seed
            )
        )

    def test_activation_decoder_matches_upstream_boundary_order_exactly(self):
        feature_count = 32
        codes = torch.zeros((16, feature_count // 8, 8), dtype=torch.uint8)
        # These blocks previously exposed (x - .5) - round(x - .5) versus
        # x - (round(x - .5) + .5) non-associativity in FP32.
        codes[0, 0] = torch.tensor([5, 12, 3, 0, 7, 3, 9, 2])
        codes[5, 1] = torch.tensor([5, 5, 2, 7, 2, 4, 10, 2])
        codes[10, 2] = torch.tensor([3, 0, 2, 12, 4, 9, 7, 2])
        activation = PreparedNQA4Activation(
            code_payload=self._pack_nibbles(codes).cuda(),
            beta_payload=torch.zeros(16 * feature_count // 32, dtype=torch.uint8, device="cuda"),
            row_norms=torch.ones(16, dtype=torch.float32, device="cuda"),
            beta_table=torch.tensor(BETAS, dtype=torch.float32, device="cuda"),
            original_shape=(16, feature_count), q=14, code_width=4,
            beta_width=2, output_dtype=torch.bfloat16,
        )
        actual = decode_packed_a4_m16(activation)
        expected = decode_nested(codes.cuda().long(), 14)
        expected *= BETAS[0] / math.sqrt(feature_count)
        self.assertTrue(torch.equal(actual, expected.reshape(16, feature_count).bfloat16()))

    def test_direct_mma_matches_tiled_reference(self):
        generator = torch.Generator(device="cuda").manual_seed(20261029)
        inputs = torch.randn(
            16, 32, generator=generator, device="cuda", dtype=torch.bfloat16
        )
        activation = quantize_activation_cuda(inputs, q=14, betas=BETAS)
        weight = self._weight()
        actual = linear_packed_w4a4_mma_m16(activation, weight)
        expected = linear_packed_w4a4_tiled_triton(
            activation, weight, block_m=16, block_n=16, block_k=32,
            num_warps=8,
        )
        difference = (actual.float() - expected.float()).abs()
        self.assertTrue(
            torch.allclose(actual.float(), expected.float(), atol=0.02, rtol=0.01),
            f"max={difference.max().item()} mean={difference.mean().item()}",
        )

    def test_direct_m1_gemv_matches_tiled_reference(self):
        generator = torch.Generator(device="cuda").manual_seed(20261030)
        inputs = torch.randn(
            1, 32, generator=generator, device="cuda", dtype=torch.bfloat16
        )
        activation = quantize_activation_cuda(inputs, q=14, betas=BETAS)
        weight = self._weight()
        actual = linear_packed_w4a4_gemv_m1(activation, weight)
        expected = linear_packed_w4a4_tiled_triton(
            activation, weight, block_m=16, block_n=16, block_k=32,
            num_warps=8,
        )
        difference = (actual.float() - expected.float()).abs()
        self.assertTrue(
            torch.allclose(actual.float(), expected.float(), atol=0.02, rtol=0.01),
            f"max={difference.max().item()} mean={difference.mean().item()}",
        )

    def test_m8_transposed_probe_matches_tiled_reference(self):
        generator = torch.Generator(device="cuda").manual_seed(20261031)
        inputs = torch.randn(
            8, 32, generator=generator, device="cuda", dtype=torch.bfloat16
        )
        activation = quantize_activation_cuda(inputs, q=14, betas=BETAS)
        weight = self._weight()
        actual = linear_packed_w4a4_mma_m8_transposed_probe(activation, weight)
        expected = linear_packed_w4a4_tiled_triton(
            activation, weight, block_m=16, block_n=16, block_k=32,
            num_warps=8,
        )
        difference = (actual.float() - expected.float()).abs()
        self.assertTrue(
            torch.allclose(actual.float(), expected.float(), atol=0.02, rtol=0.01),
            f"max={difference.max().item()} mean={difference.mean().item()}",
        )

    def test_m2_m4_transposed_probes_match_tiled_reference(self):
        generator = torch.Generator(device="cuda").manual_seed(20261031)
        weight = self._weight()
        for rows in (2, 4):
            with self.subTest(rows=rows):
                inputs = torch.randn(
                    rows, 32, generator=generator, device="cuda",
                    dtype=torch.bfloat16,
                )
                activation = quantize_activation_cuda(inputs, q=14, betas=BETAS)
                actual = linear_packed_w4a4_mma_small_m_transposed_probe(
                    activation, weight
                )
                expected = linear_packed_w4a4_tiled_triton(
                    activation, weight, block_m=16, block_n=16, block_k=32,
                    num_warps=8,
                )
                difference = (actual.float() - expected.float()).abs()
                self.assertTrue(
                    torch.allclose(
                        actual.float(), expected.float(), atol=0.02, rtol=0.01
                    ),
                    f"rows={rows} max={difference.max().item()} "
                    f"mean={difference.mean().item()}",
                )

    def test_m2_splitk_probe_matches_small_m_reference(self):
        generator = torch.Generator(device="cuda").manual_seed(20261102)
        inputs = torch.randn(
            2, 128, generator=generator, device="cuda",
            dtype=torch.bfloat16,
        )
        activation = quantize_activation_cuda(inputs, q=14, betas=BETAS)
        weight = self._weight(rows=32, features=128)
        expected = linear_packed_w4a4_mma_small_m_transposed_probe(
            activation, weight
        )
        for split_k in (2, 4, 8):
            with self.subTest(split_k=split_k):
                actual = linear_packed_w4a4_mma_small_m_splitk_probe(
                    activation, weight, split_k=split_k
                )
                difference = (actual.float() - expected.float()).abs()
                self.assertTrue(
                    torch.allclose(
                        actual.float(), expected.float(), atol=0.03, rtol=0.01
                    ),
                    f"split_k={split_k} max={difference.max().item()} "
                    f"mean={difference.mean().item()}",
                )

    def test_m32_multiwarp_probe_matches_tiled_reference(self):
        generator = torch.Generator(device="cuda").manual_seed(20261101)
        inputs = torch.randn(
            32, 32, generator=generator, device="cuda", dtype=torch.bfloat16
        )
        activation = quantize_activation_cuda(inputs, q=14, betas=BETAS)
        weight = self._weight()
        actual = linear_packed_w4a4_mma_m32_multiwarp_probe(activation, weight)
        expected = linear_packed_w4a4_tiled_triton(
            activation, weight, block_m=32, block_n=16, block_k=32,
            num_warps=8,
        )
        difference = (actual.float() - expected.float()).abs()
        self.assertTrue(
            torch.allclose(actual.float(), expected.float(), atol=0.02, rtol=0.01),
            f"max={difference.max().item()} mean={difference.mean().item()}",
        )

    def test_m32_multiwarp_matches_four_m8_slices_exactly(self):
        generator = torch.Generator(device="cuda").manual_seed(20261101)
        inputs = torch.randn(
            32, 32, generator=generator, device="cuda", dtype=torch.bfloat16
        )
        weight = self._weight()
        activation = quantize_activation_cuda(inputs, q=14, betas=BETAS)
        actual = linear_packed_w4a4_mma_m32_multiwarp_probe(activation, weight)
        slices = []
        for start in range(0, 32, 8):
            packed_slice = quantize_activation_cuda(
                inputs[start:start + 8], q=14, betas=BETAS
            )
            slices.append(
                linear_packed_w4a4_mma_m8_transposed_probe(packed_slice, weight)
            )
        self.assertTrue(torch.equal(actual, torch.cat(slices, dim=0)))

    def test_m64_m128_multiwarp_family_matches_m8_slices_exactly(self):
        generator = torch.Generator(device="cuda").manual_seed(20261101)
        weight = self._weight()
        for rows in (64, 128):
            with self.subTest(rows=rows):
                inputs = torch.randn(
                    rows, 32, generator=generator, device="cuda",
                    dtype=torch.bfloat16,
                )
                activation = quantize_activation_cuda(inputs, q=14, betas=BETAS)
                actual = linear_packed_w4a4_mma_large_m_multiwarp_probe(
                    activation, weight
                )
                slices = [
                    linear_packed_w4a4_mma_m8_transposed_probe(
                        quantize_activation_cuda(
                            inputs[start:start + 8], q=14, betas=BETAS
                        ),
                        weight,
                    )
                    for start in range(0, rows, 8)
                ]
                self.assertTrue(torch.equal(actual, torch.cat(slices, dim=0)))

    def test_large_m_staged_family_matches_unstaged_exactly(self):
        generator = torch.Generator(device="cuda").manual_seed(20261101)
        weight = self._weight()
        for rows in (32, 64, 128):
            with self.subTest(rows=rows):
                inputs = torch.randn(
                    rows, 32, generator=generator, device="cuda",
                    dtype=torch.bfloat16,
                )
                activation = quantize_activation_cuda(inputs, q=14, betas=BETAS)
                staged = linear_packed_w4a4_mma_large_m_staged(
                    activation, weight
                )
                unstaged = linear_packed_w4a4_mma_large_m_multiwarp_probe(
                    activation, weight
                )
                self.assertTrue(torch.equal(staged, unstaged))

        activation = quantize_activation_cuda(inputs, q=14, betas=BETAS)
        compatibility = linear_packed_w4a4_mma_m128_staged_probe(
            activation, weight
        )
        self.assertTrue(torch.equal(compatibility, staged))

    def test_m128_pipeline_probe_matches_staged_exactly(self):
        generator = torch.Generator(device="cuda").manual_seed(20261101)
        inputs = torch.randn(
            128, 32, generator=generator, device="cuda", dtype=torch.bfloat16
        )
        weight = self._weight()
        activation = quantize_activation_cuda(inputs, q=14, betas=BETAS)
        pipeline = linear_packed_w4a4_mma_m128_pipeline_probe(
            activation, weight
        )
        staged = linear_packed_w4a4_mma_m128_staged_probe(activation, weight)
        self.assertTrue(torch.equal(pipeline, staged))

    @unittest.skipUnless(_cute_headers_available(), "CUTLASS CuTe headers unavailable")
    def test_m128_cute_probe_matches_staged_exactly(self):
        generator = torch.Generator(device="cuda").manual_seed(20261101)
        inputs = torch.randn(
            128, 32, generator=generator, device="cuda", dtype=torch.bfloat16
        )
        weight = self._weight()
        activation = quantize_activation_cuda(inputs, q=14, betas=BETAS)
        cute = linear_packed_w4a4_cute_m128_probe(activation, weight)
        staged = linear_packed_w4a4_mma_m128_staged_probe(activation, weight)
        self.assertTrue(torch.equal(cute, staged))

    @unittest.skipUnless(_cute_headers_available(), "CUTLASS CuTe headers unavailable")
    def test_m32_m64_cute_family_matches_existing_kernels_exactly(self):
        generator = torch.Generator(device="cuda").manual_seed(20261101)
        weight = self._weight()
        for rows in (32, 64):
            with self.subTest(rows=rows):
                inputs = torch.randn(
                    rows, 32, generator=generator, device="cuda",
                    dtype=torch.bfloat16,
                )
                activation = quantize_activation_cuda(inputs, q=14, betas=BETAS)
                cute = linear_packed_w4a4_cute_large_m_probe(activation, weight)
                existing = linear_packed_w4a4_mma_large_m_multiwarp_probe(
                    activation, weight
                )
                self.assertTrue(torch.equal(cute, existing))

    def test_m1_packed_word_probe_matches_production_exactly(self):
        generator = torch.Generator(device="cuda").manual_seed(20261030)
        inputs = torch.randn(
            1, 32, generator=generator, device="cuda", dtype=torch.bfloat16
        )
        activation = quantize_activation_cuda(inputs, q=14, betas=BETAS)
        weight = self._weight()
        production = linear_packed_w4a4_gemv_m1(activation, weight)
        probe = linear_packed_w4a4_gemv_m1_packed_word_probe(activation, weight)
        self.assertTrue(torch.equal(probe, production))

    def test_m1_splitk_probe_matches_production_reference(self):
        generator = torch.Generator(device="cuda").manual_seed(20261103)
        inputs = torch.randn(
            1, 128, generator=generator, device="cuda", dtype=torch.bfloat16
        )
        activation = quantize_activation_cuda(inputs, q=14, betas=BETAS)
        weight = self._weight(rows=32, features=128)
        expected = linear_packed_w4a4_gemv_m1(activation, weight)
        for split_k in (2, 4, 8):
            with self.subTest(split_k=split_k):
                actual = linear_packed_w4a4_gemv_m1_splitk_probe(
                    activation, weight, split_k=split_k
                )
                difference = (actual.float() - expected.float()).abs()
                self.assertTrue(torch.allclose(
                    actual.float(), expected.float(), atol=0.03, rtol=0.01
                ), f"split_k={split_k} max={difference.max().item()}")
                compensated = (
                    linear_packed_w4a4_gemv_m1_splitk_compensated_fp32_probe(
                        activation, weight, split_k=split_k
                    )
                )
                self.assertEqual(compensated.dtype, torch.float32)
                self.assertTrue(torch.isfinite(compensated).all())
                self.assertTrue(torch.allclose(
                    compensated, expected.float(), atol=0.03, rtol=0.01
                ))

        with self.assertRaisesRegex(ValueError, "split_k"):
            linear_packed_w4a4_gemv_m1_splitk_probe(
                activation, weight, split_k=3
            )

    def test_chunked_staged_covers_dynamic_service_m(self):
        generator = torch.Generator(device="cuda").manual_seed(20261101)
        weight = self._weight()
        for rows in (24, 40, 120, 152, 240, 384):
            with self.subTest(rows=rows):
                inputs = torch.randn(
                    rows, 32, generator=generator, device="cuda",
                    dtype=torch.bfloat16,
                )
                activation = quantize_activation_cuda(inputs, q=14, betas=BETAS)
                actual = linear_packed_w4a4_mma_chunked_staged_probe(
                    activation, weight
                )
                expected = linear_packed_w4a4_tiled_triton(
                    activation, weight, block_m=16, block_n=16,
                    block_k=32, num_warps=8,
                )
                self.assertEqual(actual.shape, expected.shape)
                self.assertTrue(torch.isfinite(actual).all())
                self.assertTrue(torch.allclose(
                    actual.float(), expected.float(), atol=0.04, rtol=0.01
                ))

    def test_diagnostic_decoder_rejects_noncontiguous_payload(self):
        inputs = torch.zeros(16, 32, device="cuda", dtype=torch.bfloat16)
        activation = quantize_activation_cuda(inputs, q=14, betas=BETAS)
        noncontiguous = activation.code_payload.reshape(16, -1).transpose(0, 1)
        self.assertFalse(noncontiguous.is_contiguous())
        malformed = replace(activation, code_payload=noncontiguous)
        with self.assertRaisesRegex(RuntimeError, "contiguous"):
            decode_packed_a4_m16(malformed)


if __name__ == "__main__":
    unittest.main()
