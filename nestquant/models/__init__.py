"""Adapters applying algorithm primitives to neural-network modules."""

from .quantization import quantize_nestquant_w4_, quantize_rtn_w4_

__all__ = ["quantize_nestquant_w4_", "quantize_rtn_w4_"]
