"""Tests for downstream-evaluation runtime integration."""

import unittest

import torch
from torch import nn

from nestquant.experiments.downstream_eval import _enable_fresh_quantized_cache


class RecordingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, inputs, *, past_key_values=None, use_cache=False):
        self.calls.append((past_key_values, use_cache))
        return inputs


class TestDownstreamEval(unittest.TestCase):
    def test_injects_a_fresh_cache_for_each_harness_forward(self):
        model = RecordingModel()
        caches = []

        def factory():
            cache = object()
            caches.append(cache)
            return cache

        _enable_fresh_quantized_cache(model, factory)
        model(torch.ones(1))
        model(torch.ones(1))
        self.assertEqual(len(caches), 2)
        self.assertIs(model.calls[0][0], caches[0])
        self.assertIs(model.calls[1][0], caches[1])
        self.assertTrue(all(use_cache for _, use_cache in model.calls))

    def test_preserves_an_explicit_cache(self):
        model = RecordingModel()
        explicit = object()
        _enable_fresh_quantized_cache(model, lambda: self.fail("factory should not run"))
        model(torch.ones(1), past_key_values=explicit)
        self.assertIs(model.calls[0][0], explicit)


if __name__ == "__main__":
    unittest.main()
