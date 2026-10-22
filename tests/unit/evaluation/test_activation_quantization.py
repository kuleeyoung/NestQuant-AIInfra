import unittest
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from nestquant.evaluation.activation_quantization import (
    ActivationQuantizationStats,
    enable_activation_quantization,
)


class TestActivationQuantization(unittest.TestCase):
    def test_attaches_only_to_seven_layer_projection_types(self):
        config = LlamaConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=1, max_position_embeddings=32,
        )
        model = LlamaForCausalLM(config).eval()
        stats = ActivationQuantizationStats()
        _, handles = enable_activation_quantization(
            model, q=14, betas=(0.25, 0.5, 0.75, 1.5),
            rows_per_chunk=8, stats=stats,
        )
        with patch(
            "nestquant.evaluation.activation_quantization.fake_quantize_nestquant_rows",
            wraps=__import__(
                "nestquant.evaluation.activation_quantization",
                fromlist=["fake_quantize_nestquant_rows"],
            ).fake_quantize_nestquant_rows,
        ) as quantize_mock, torch.inference_mode():
            output = model(torch.tensor([[1, 2, 3, 4]]))
        summary = stats.summary()
        self.assertEqual(output.logits.shape, (1, 4, 32))
        self.assertEqual(len(handles), 7)
        self.assertEqual(summary["module_count"], 7)
        self.assertNotIn("lm_head", summary["modules"])
        self.assertGreater(summary["entry_count"], 0)
        self.assertGreater(summary["squared_error"], 0)
        self.assertEqual(quantize_mock.call_count, 4)
        for handle in handles:
            handle.remove()

    def test_rejects_model_without_llama_projection_names(self):
        with self.assertRaisesRegex(ValueError, "no supported Llama"):
            enable_activation_quantization(
                torch.nn.Sequential(torch.nn.Linear(8, 8)),
                q=14, betas=(0.25, 0.5),
            )


if __name__ == "__main__":
    unittest.main()
