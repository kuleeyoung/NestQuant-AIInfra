"""Tests for post-RoPE KV-cache fake quantization."""

import unittest

import torch
from transformers import LlamaConfig
from transformers import LlamaForCausalLM
from transformers.cache_utils import DynamicCache

from nestquant.core.runtime_quantization import fake_quantize_nestquant_rows
from nestquant.evaluation.kv_cache import (
    KVBlockSampler,
    KVCalibrationCache,
    KVQuantizationStats,
    NestQuantDynamicCache,
    RotatedKeyDynamicCache,
    enable_kv_attention_rotation,
    hadamard_attention_heads,
)


class TestNestQuantDynamicCache(unittest.TestCase):
    def test_quantizes_before_cache_storage_and_accumulates_stats(self):
        config = LlamaConfig(
            hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2, head_dim=8,
        )
        collector = KVQuantizationStats()
        cache = NestQuantDynamicCache(
            config=config, q=14, default_betas=(0.25, 0.5, 1.0, 2.0),
            rows_per_chunk=2, stats=collector,
        )
        generator = torch.Generator().manual_seed(31)
        key = torch.randn(1, 2, 3, 8, generator=generator)
        value = torch.randn(1, 2, 3, 8, generator=generator)
        expected_key, _ = fake_quantize_nestquant_rows(
            hadamard_attention_heads(key), q=14,
            betas=(0.25, 0.5, 1.0, 2.0), rows_per_chunk=2
        )
        cached_key, cached_value = cache.update(key, value, 0)
        self.assertTrue(torch.equal(cached_key, expected_key))
        self.assertEqual(cached_value.shape, value.shape)

        second_key = key[:, :, :1]
        second_value = value[:, :, :1]
        concatenated_key, _ = cache.update(second_key, second_value, 0)
        self.assertEqual(concatenated_key.shape[-2], 4)
        summary = collector.summary()
        self.assertEqual(summary["layer_count"], 1)
        self.assertEqual(summary["layers"]["0"]["key"]["updates"], 2)
        self.assertEqual(summary["entry_count"], 2 * (key.numel() + second_key.numel()))
        self.assertEqual(
            summary["storage_estimate"]["status"],
            "estimate_only_no_packed_kv_cache_allocated",
        )

    def test_calibration_cache_collects_bounded_post_rope_samples(self):
        config = LlamaConfig(
            hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2, head_dim=8,
        )
        sampler = KVBlockSampler(calibration_chunks=2, sample_limit_per_tensor=6)
        cache = KVCalibrationCache(config=config, sampler=sampler)
        generator = torch.Generator().manual_seed(37)
        for _ in range(2):
            key = torch.randn(1, 2, 4, 8, generator=generator)
            value = torch.randn(1, 2, 4, 8, generator=generator)
            cache.update(key, value, 0)
        result = sampler.search(
            q=14, candidate_betas=(0.25, 0.5, 1.0, 2.0),
            selected_beta_count=2, device="cpu", rows_per_chunk=3,
            overload_margin=0.25,
        )
        self.assertEqual(result["0"]["key"]["sample_count"], 6)
        self.assertEqual(result["0"]["key"]["seen_block_count"], 16)
        self.assertEqual(len(result["0"]["key"]["beta_values"]), 2)
        self.assertEqual(len(result["0"]["key"]["sample_sha256"]), 64)
        self.assertGreaterEqual(
            result["0"]["key"]["beta_values"][-1],
            result["0"]["key"]["beta_values_before_overload_margin"][-1] + 0.25,
        )

    def test_matched_query_key_rotation_preserves_tiny_llama_logits(self):
        torch.manual_seed(41)
        config = LlamaConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=2, head_dim=8,
        )
        model = LlamaForCausalLM(config).eval()
        input_ids = torch.tensor([[1, 2, 3, 4]])
        with torch.inference_mode():
            baseline = model(
                input_ids=input_ids, use_cache=True,
                past_key_values=DynamicCache(config=config),
            ).logits
            enable_kv_attention_rotation(model)
            rotated = model(
                input_ids=input_ids, use_cache=True,
                past_key_values=RotatedKeyDynamicCache(config=config),
            ).logits
        self.assertTrue(torch.allclose(baseline, rotated, atol=2e-6, rtol=2e-5))


if __name__ == "__main__":
    unittest.main()
