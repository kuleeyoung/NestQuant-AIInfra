"""Integration tests for the registered Transformers Llama NQKV backend."""

import unittest

import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaForCausalLM,
    LlamaModel,
)

from nestquant.runtime.nqkv_hf_backend import (
    NQKVLlamaCache,
    install_nqkv_llama_backend,
)
from nestquant.runtime.triton_unpack import triton_available


class _OneAttentionModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention = LlamaAttention(config, layer_idx=0)


@unittest.skipUnless(triton_available(), "Triton CUDA runtime is unavailable")
class TestNQKVHFBackend(unittest.TestCase):
    def setUp(self):
        self.config = LlamaConfig(
            hidden_size=256, intermediate_size=512, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, head_dim=64,
            attention_dropout=0.0,
        )
        self.model = _OneAttentionModel(self.config).eval().bfloat16().cuda()
        self.cache = NQKVLlamaCache(
            num_layers=1, num_query_heads=4, num_kv_heads=2, head_dim=64,
            max_cache_len=32, page_size=4, q=14,
            key_betas=(0.25, 0.3271428571, 0.4621428571, 1.2185714286),
            value_betas=(0.2521428571, 0.3992857143, 0.6871428571, 2.0735714286),
            dtype=torch.bfloat16, device="cuda",
        )
        install_nqkv_llama_backend(self.model, self.cache)

    def _forward_token(self, seed):
        hidden = torch.randn(
            1, 1, 256, generator=torch.Generator().manual_seed(seed)
        ).bfloat16().cuda()
        cos = torch.ones(1, 1, 64, device="cuda", dtype=torch.bfloat16)
        sin = torch.zeros_like(cos)
        return self.model.attention(
            hidden, position_embeddings=(cos, sin), past_key_values=self.cache
        )[0]

    def test_real_llama_attention_forward_uses_registered_backend(self):
        first = self._forward_token(90)
        second = self._forward_token(91)
        self.assertEqual(first.shape, (1, 1, 256))
        self.assertEqual(second.shape, (1, 1, 256))
        self.assertTrue(torch.isfinite(first).all())
        self.assertEqual(self.cache.get_seq_length(0), 2)

    def test_reset_releases_all_layer_pages_and_hot_tokens(self):
        self._forward_token(92)
        self.cache.reset()
        self.assertEqual(self.cache.get_seq_length(0), 0)
        self._forward_token(93)
        self.assertEqual(self.cache.get_seq_length(0), 1)

    def test_beam_reorder_is_explicitly_unsupported(self):
        with self.assertRaises(NotImplementedError):
            self.cache.reorder_cache(torch.tensor([0], device="cuda"))

    def test_complete_llama_model_runs_two_decode_steps(self):
        model = LlamaModel(self.config).eval().bfloat16().cuda()
        cache = NQKVLlamaCache(
            num_layers=1, num_query_heads=4, num_kv_heads=2, head_dim=64,
            max_cache_len=32, page_size=4, q=14,
            key_betas=(0.25, 0.3271428571, 0.4621428571, 1.2185714286),
            value_betas=(0.2521428571, 0.3992857143, 0.6871428571, 2.0735714286),
            dtype=torch.bfloat16, device="cuda",
        )
        install_nqkv_llama_backend(model, cache)
        with torch.no_grad():
            first = model(
                input_ids=torch.tensor([[1]], device="cuda"),
                past_key_values=cache, use_cache=True,
            ).last_hidden_state
            second = model(
                input_ids=torch.tensor([[2]], device="cuda"),
                past_key_values=cache, use_cache=True,
            ).last_hidden_state
        self.assertEqual(first.shape, (1, 1, 256))
        self.assertEqual(second.shape, (1, 1, 256))
        self.assertEqual(cache.get_seq_length(), 2)

    def test_complete_llama_model_prefill_then_decode(self):
        model = LlamaModel(self.config).eval().bfloat16().cuda()
        cache = NQKVLlamaCache(
            num_layers=1, num_query_heads=4, num_kv_heads=2, head_dim=64,
            max_cache_len=32, page_size=4, q=14,
            key_betas=(0.25, 0.3271428571, 0.4621428571, 1.2185714286),
            value_betas=(0.2521428571, 0.3992857143, 0.6871428571, 2.0735714286),
            dtype=torch.bfloat16, device="cuda",
        )
        install_nqkv_llama_backend(model, cache)
        with torch.no_grad():
            prompt = model(
                input_ids=torch.tensor([[1, 2, 3, 4, 5]], device="cuda"),
                past_key_values=cache, use_cache=True,
            ).last_hidden_state
            decoded = model(
                input_ids=torch.tensor([[6]], device="cuda"),
                past_key_values=cache, use_cache=True,
            ).last_hidden_state
        self.assertEqual(prompt.shape, (1, 5, 256))
        self.assertEqual(decoded.shape, (1, 1, 256))
        self.assertTrue(torch.isfinite(prompt).all())
        self.assertEqual(cache.get_seq_length(), 6)

    def test_generate_uses_nqkv_prefill_and_decode(self):
        model = LlamaForCausalLM(self.config).eval().bfloat16().cuda()
        cache = NQKVLlamaCache(
            num_layers=1, num_query_heads=4, num_kv_heads=2, head_dim=64,
            max_cache_len=32, page_size=4, q=14,
            key_betas=(0.25, 0.3271428571, 0.4621428571, 1.2185714286),
            value_betas=(0.2521428571, 0.3992857143, 0.6871428571, 2.0735714286),
            dtype=torch.bfloat16, device="cuda",
        )
        install_nqkv_llama_backend(model, cache)
        with torch.no_grad():
            generated = model.generate(
                input_ids=torch.tensor([[1, 2, 3, 4, 5]], device="cuda"),
                past_key_values=cache,
                max_new_tokens=3,
                do_sample=False,
            )
        self.assertEqual(generated.shape, (1, 8))
        # Generation does not run the final sampled token back through the model.
        self.assertEqual(cache.get_seq_length(), 7)


if __name__ == "__main__":
    unittest.main()
