"""Compatibility import; use :mod:`nestquant.evaluation.perplexity` in new code."""

from .evaluation.perplexity import TokenizedCorpus, evaluate_perplexity, tokenize_corpus

__all__ = ["TokenizedCorpus", "evaluate_perplexity", "tokenize_corpus"]
