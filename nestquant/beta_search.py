"""Compatibility import; use :mod:`nestquant.core.beta_search` in new code."""

from .core.beta_search import BetaSearchResult, build_candidate_tables, select_beta_subset

__all__ = ["BetaSearchResult", "build_candidate_tables", "select_beta_subset"]
