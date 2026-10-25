"""CUDA correctness tests for the experimental Triton NQPK unpacker."""

import unittest

import torch

from nestquant.core.codec import generate_e8_dither, generate_e8_dither_stateless
from nestquant.core.packing import pack_nestquant_indices
from nestquant.core.quantizer import quantize_rows
from nestquant.core.tensor_checkpoint import PackedNestQuantTensor
from nestquant.runtime.triton_decode import dequantize_with_triton_unpack
from nestquant.runtime.triton_unpack import triton_available, unpack_indices_triton
from nestquant.runtime.triton_tiled_decode import (
    _generate_stateless_dither_triton,
    dequantize_tiled_triton,
)
from nestquant.runtime.triton_fused_linear import (
    linear_fused_triton,
    prepare_fused_linear,
)
from nestquant.runtime.triton_grouped_linear import (
    linear_fused_pair_triton,
    linear_fused_qkv_triton,
)
from nestquant.runtime.triton_dot_linear import (
    linear_fused_dot_triton,
    select_nqrt_dot_config,
)


class TestNQRTDotDispatch(unittest.TestCase):
    def test_selects_measured_llama_projection_configs(self):
        cases = (
            ((2048, 2048, 1), (32, 8)),
            ((2048, 2048, 8), (64, 4)),
            ((2048, 2048, 16), (64, 8)),
            ((512, 2048, 16), (128, 8)),
            ((8192, 2048, 8), (64, 8)),
            ((2048, 8192, 1), (128, 8)),
        )
        for dimensions, expected in cases:
            with self.subTest(dimensions=dimensions):
                config = select_nqrt_dot_config(*dimensions)
                self.assertEqual((config["block_k"], config["num_warps"]), expected)
                self.assertEqual(config["block_m"], 16)
                self.assertEqual(config["block_n"], 16)

    def test_selects_large_m_tiles_for_long_prefill(self):
        cases = (
            ((2048, 2048, 32), (128, 32)),
            ((2048, 2048, 64), (64, 32)),
            ((512, 2048, 128), (128, 64)),
            ((8192, 2048, 64), (64, 32)),
            ((2048, 8192, 32), (32, 64)),
            ((2048, 8192, 64), (64, 128)),
            ((2048, 8192, 128), (128, 32)),
        )
        for dimensions, expected in cases:
            with self.subTest(dimensions=dimensions):
                config = select_nqrt_dot_config(*dimensions)
                self.assertEqual((config["block_m"], config["block_k"]), expected)
                self.assertEqual(config["num_warps"], 8)

    def test_rejects_nonpositive_dimensions(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            select_nqrt_dot_config(16, 0, 1)


@unittest.skipUnless(triton_available(), "Triton CUDA runtime is unavailable")
class TestTritonUnpack(unittest.TestCase):
    def test_matches_reference_for_supported_fixed_widths_and_padding(self):
        generator = torch.Generator().manual_seed(19)
        cases = ((14, 4), (4, 3), (2, 1), (256, 16))
        for q, beta_count in cases:
            with self.subTest(q=q, beta_count=beta_count):
                codes = torch.randint(q, (3, 5, 8), generator=generator)
                betas = torch.randint(beta_count, (3, 5), generator=generator)
                packed = pack_nestquant_indices(
                    codes, betas, q=q, beta_count=beta_count
                )
                expected_codes, expected_betas = packed.unpack(device="cuda")
                actual_codes, actual_betas = unpack_indices_triton(
                    packed, output_dtype=torch.int64
                )
                self.assertTrue(torch.equal(actual_codes, expected_codes))
                self.assertTrue(torch.equal(actual_betas, expected_betas))

    def test_compact_output_dtypes_preserve_values(self):
        codes = torch.arange(80).reshape(2, 5, 8) % 14
        betas = torch.arange(10).reshape(2, 5) % 4
        packed = pack_nestquant_indices(codes, betas, q=14, beta_count=4)
        for dtype in (torch.int16, torch.int32):
            actual_codes, actual_betas = unpack_indices_triton(
                packed, output_dtype=dtype
            )
            self.assertEqual(actual_codes.dtype, dtype)
            self.assertTrue(torch.equal(actual_codes.cpu().long(), codes))
            self.assertTrue(torch.equal(actual_betas.cpu().long(), betas))

    def test_hybrid_decoder_matches_chunked_reference(self):
        generator = torch.Generator().manual_seed(23)
        matrix = torch.randn(7, 16, generator=generator)
        dither_generator = torch.Generator().manual_seed(29)
        dither_state = bytes(dither_generator.get_state().untyped_storage())
        dither = generate_e8_dither(
            14, generator=dither_generator, device="cuda"
        ).reshape(7, 2, 8)
        rows = quantize_rows(
            matrix.cuda(), q=14, betas=(0.25, 0.5, 0.75, 1.0), dither=dither
        )
        record = PackedNestQuantTensor.from_quantized_rows(
            "weight", rows, dither_generator_state=dither_state
        )
        expected = record.dequantize_chunked(
            device="cuda", output_dtype=torch.bfloat16, rows_per_chunk=3
        )
        actual = dequantize_with_triton_unpack(
            record, device="cuda", output_dtype=torch.bfloat16, rows_per_chunk=3
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_hybrid_decoder_supports_v4_random_access_dither(self):
        matrix = torch.randn(
            9, 24, generator=torch.Generator().manual_seed(31)
        ).cuda()
        seed = 20261025
        dither = generate_e8_dither_stateless(
            27, seed, device="cuda"
        ).reshape(9, 3, 8)
        rows = quantize_rows(
            matrix, q=14, betas=(0.25, 0.5, 0.75, 1.0), dither=dither
        )
        record = PackedNestQuantTensor.from_quantized_rows(
            "weight", rows, dither_seed=seed
        )
        expected = record.dequantize_chunked(
            device="cuda", output_dtype=torch.bfloat16, rows_per_chunk=4
        )
        actual = dequantize_with_triton_unpack(
            record, device="cuda", output_dtype=torch.bfloat16, rows_per_chunk=2
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_tiled_v4_decoder_matches_reference_without_expanded_streams(self):
        matrix = torch.randn(
            17, 40, generator=torch.Generator().manual_seed(41)
        ).cuda()
        seed = 20261025
        dither = generate_e8_dither_stateless(
            85, seed, device="cuda"
        ).reshape(17, 5, 8)
        rows = quantize_rows(
            matrix, q=14, betas=(0.25, 0.5, 0.75, 1.0), dither=dither
        )
        record = PackedNestQuantTensor.from_quantized_rows(
            "weight", rows, dither_seed=seed
        )
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                expected = record.dequantize_chunked(
                    device="cuda", output_dtype=dtype, rows_per_chunk=3
                )
                for tile_blocks in (16, 64, 128):
                    actual = dequantize_tiled_triton(
                        record, output_dtype=dtype, tile_blocks=tile_blocks
                    )
                    self.assertTrue(torch.equal(actual, expected))

    def test_tiled_dither_matches_reference_on_voronoi_boundaries(self):
        cases = (
            (1_283_774_271, 267_894),
            (852_565_332, 475_261),
            (730_069_157, 34_102),
        )
        for seed, block_offset in cases:
            with self.subTest(seed=seed, block_offset=block_offset):
                expected = generate_e8_dither_stateless(
                    1, seed, block_offset=block_offset, device="cuda"
                )
                actual = _generate_stateless_dither_triton(
                    1, seed, block_offset=block_offset
                )
                self.assertTrue(torch.equal(actual, expected))

    def test_fused_linear_uses_packed_resident_state_without_full_weight(self):
        matrix = torch.randn(
            12, 32, generator=torch.Generator().manual_seed(53)
        ).cuda()
        seed = 20261025
        dither = generate_e8_dither_stateless(
            48, seed, device="cuda"
        ).reshape(12, 4, 8)
        rows = quantize_rows(
            matrix, q=14, betas=(0.25, 0.5, 0.75, 1.0), dither=dither
        )
        record = PackedNestQuantTensor.from_quantized_rows(
            "weight", rows, dither_seed=seed
        )
        prepared = prepare_fused_linear(record)
        inputs = torch.randn(
            2, 3, 32, device="cuda", dtype=torch.bfloat16
        )
        reference_weight = record.dequantize_chunked(
            device="cuda", output_dtype=torch.bfloat16, rows_per_chunk=4
        )
        expected = torch.nn.functional.linear(inputs, reference_weight)
        actual = linear_fused_triton(inputs, prepared)
        error = (actual.float() - expected.float()).abs()
        relative_l2 = torch.linalg.vector_norm(error) / torch.linalg.vector_norm(
            expected.float()
        )
        self.assertEqual(actual.shape, (2, 3, 12))
        self.assertLessEqual(error.max().item(), 0.0625)
        self.assertLess(relative_l2.item(), 0.01)
        packed_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                prepared.code_payload,
                prepared.beta_payload,
                prepared.row_norms,
                prepared.beta_table,
            )
        )
        self.assertLess(packed_bytes, reference_weight.numel() * reference_weight.element_size())

    def test_grouped_qkv_matches_three_independent_fused_linears(self):
        prepared = []
        generator = torch.Generator().manual_seed(79)
        for index, (rows_count, seed) in enumerate(zip(
            (12, 8, 4), (20261025, 1703560109, 71600901), strict=True
        )):
            matrix = torch.randn(rows_count, 32, generator=generator).cuda()
            dither = generate_e8_dither_stateless(
                matrix.numel() // 8, seed, device="cuda"
            ).reshape(rows_count, 4, 8)
            rows = quantize_rows(
                matrix, q=14, betas=(0.25, 0.5, 0.75, 1.0), dither=dither
            )
            record = PackedNestQuantTensor.from_quantized_rows(
                f"projection_{index}.weight", rows, dither_seed=seed
            )
            prepared.append(prepare_fused_linear(record))

        inputs = torch.randn(2, 3, 32, device="cuda", dtype=torch.bfloat16)
        expected = tuple(linear_fused_triton(inputs, item) for item in prepared)
        actual = linear_fused_qkv_triton(inputs, tuple(prepared))
        torch.cuda.synchronize()

        self.assertEqual(tuple(output.shape for output in actual), (
            (2, 3, 12), (2, 3, 8), (2, 3, 4),
        ))
        for grouped, independent in zip(actual, expected, strict=True):
            self.assertTrue(torch.equal(grouped, independent))

    def test_grouped_pair_matches_two_independent_fused_linears(self):
        prepared = []
        generator = torch.Generator().manual_seed(83)
        for index, seed in enumerate((1889849671, 1703560109)):
            matrix = torch.randn(12, 32, generator=generator).cuda()
            dither = generate_e8_dither_stateless(
                matrix.numel() // 8, seed, device="cuda"
            ).reshape(12, 4, 8)
            rows = quantize_rows(
                matrix, q=14, betas=(0.25, 0.5, 0.75, 1.0), dither=dither
            )
            record = PackedNestQuantTensor.from_quantized_rows(
                f"pair_{index}.weight", rows, dither_seed=seed
            )
            prepared.append(prepare_fused_linear(record))
        inputs = torch.randn(2, 3, 32, device="cuda", dtype=torch.bfloat16)
        expected = tuple(linear_fused_triton(inputs, item) for item in prepared)
        actual = linear_fused_pair_triton(inputs, tuple(prepared))
        torch.cuda.synchronize()
        for grouped, independent in zip(actual, expected, strict=True):
            self.assertTrue(torch.equal(grouped, independent))

    def test_dot_fused_linear_matches_dequantized_reference(self):
        seed = 1889849671
        matrix = torch.randn(
            24, 64, generator=torch.Generator().manual_seed(89)
        ).cuda()
        dither = generate_e8_dither_stateless(
            matrix.numel() // 8, seed, device="cuda"
        ).reshape(24, 8, 8)
        rows = quantize_rows(
            matrix, q=14, betas=(0.25, 0.5, 0.75, 1.0), dither=dither
        )
        record = PackedNestQuantTensor.from_quantized_rows(
            "dot.weight", rows, dither_seed=seed
        )
        prepared = prepare_fused_linear(record)
        inputs = torch.randn(2, 3, 64, device="cuda", dtype=torch.bfloat16)
        reference_weight = record.dequantize_chunked(
            device="cuda", output_dtype=torch.bfloat16, rows_per_chunk=8
        )
        expected = torch.nn.functional.linear(inputs, reference_weight)
        actual = linear_fused_dot_triton(inputs, prepared)
        torch.cuda.synchronize()
        error = (actual.float() - expected.float()).abs()
        relative_l2 = torch.linalg.vector_norm(error) / torch.linalg.vector_norm(
            expected.float()
        )
        self.assertEqual(actual.shape, (2, 3, 24))
        self.assertLessEqual(error.max().item(), 0.0625)
        self.assertLess(relative_l2.item(), 0.01)


if __name__ == "__main__":
    unittest.main()
