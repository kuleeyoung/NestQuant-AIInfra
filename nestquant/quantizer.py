"""Compatibility import; use :mod:`nestquant.core.quantizer` in new code."""

from .core.quantizer import QuantizedRows, quantize_rows

__all__ = ["QuantizedRows", "quantize_rows"]
