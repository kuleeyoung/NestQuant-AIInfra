"""Tests for explicit corpus boundaries and aggregated token-level PPL."""

import math
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from nestquant.evaluation.perplexity import evaluate_perplexity, tokenize_corpus


class TinyTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    model_max_length = 3

    def __call__(self, text, *, add_special_tokens, return_attention_mask):
        return {"input_ids": [3, 4, 5, 6, 7]}


class UniformLogitModel(nn.Module):
    def forward(self, *, input_ids, use_cache, past_key_values=None):
        if past_key_values is not None:
            past_key_values.mark_used()
        return SimpleNamespace(logits=torch.zeros((*input_ids.shape, 10), device=input_ids.device))


class RecordingCache:
    def __init__(self):
        self.used = False

    def mark_used(self):
        self.used = True


class TestPerplexity(unittest.TestCase):
    def test_corpus_hashes_and_boundary_tokens_are_stable(self):
        tokenizer = TinyTokenizer()
        result = tokenize_corpus(["one", "two"], tokenizer, context_length=4)
        self.assertEqual(result.token_ids, (1, 3, 4, 5, 6, 7, 2))
        self.assertEqual(result.chunk_count, 1)
        self.assertEqual(result.predicted_token_count, 3)
        self.assertEqual(result.dropped_tail_tokens, 3)
        self.assertEqual(tokenizer.model_max_length, len("one\n\ntwo"))
        self.assertEqual(len(result.text_sha256), 64)
        self.assertEqual(len(result.token_ids_sha256), 64)

    def test_aggregates_nll_and_ppl_by_predicted_tokens(self):
        result = evaluate_perplexity(UniformLogitModel(), [1, 2, 3, 4, 5, 6],
                                     context_length=3, device="cpu", logit_chunk_size=1)
        self.assertEqual(result["predicted_token_count"], 4)
        self.assertAlmostEqual(result["mean_nll_per_predicted_token"], math.log(10), places=6)
        self.assertAlmostEqual(result["perplexity"], 10.0, places=5)

    def test_constructs_a_fresh_cache_for_each_context_window(self):
        caches = []

        def cache_factory():
            cache = RecordingCache()
            caches.append(cache)
            return cache

        result = evaluate_perplexity(
            UniformLogitModel(), [1, 2, 3, 4, 5, 6], context_length=3,
            device="cpu", cache_factory=cache_factory,
        )
        self.assertEqual(result["chunk_count"], 2)
        self.assertEqual(len(caches), 2)
        self.assertTrue(all(cache.used for cache in caches))

    def test_batches_independent_context_windows(self):
        caches = []

        def cache_factory():
            cache = RecordingCache()
            caches.append(cache)
            return cache

        result = evaluate_perplexity(
            UniformLogitModel(), [value % 10 for value in range(12)], context_length=3,
            device="cpu", cache_factory=cache_factory, batch_size=3,
        )
        self.assertEqual(result["chunk_count"], 4)
        self.assertEqual(result["predicted_token_count"], 8)
        self.assertEqual(len(caches), 2)
        self.assertTrue(all(cache.used for cache in caches))

    def test_reports_completed_chunk_progress_after_each_batch(self):
        progress = []
        evaluate_perplexity(
            UniformLogitModel(), [value % 10 for value in range(15)],
            context_length=3, device="cpu", batch_size=2,
            progress_callback=lambda completed, total: progress.append((completed, total)),
        )
        self.assertEqual(progress, [(2, 5), (4, 5), (5, 5)])


if __name__ == "__main__":
    unittest.main()
