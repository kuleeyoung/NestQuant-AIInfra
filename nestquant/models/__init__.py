"""Adapters applying algorithm primitives to neural-network modules."""

from .lazy_linear import CachePolicy, LazyNQMCLinear, LazyNQMCWeightCache
from .quantization import quantize_nestquant_w4_, quantize_rtn_w4_
from .rotated_linear import (
    HadamardConversionReport,
    HadamardRotatedLinear,
    PAPER_HADAMARD_FORMAT,
    convert_linears_to_hadamard_coordinates_,
    restore_hadamard_structure_from_metadata_,
)

__all__ = [
    "CachePolicy",
    "HadamardConversionReport",
    "HadamardRotatedLinear",
    "LazyNQMCLinear",
    "LazyNQMCWeightCache",
    "PAPER_HADAMARD_FORMAT",
    "convert_linears_to_hadamard_coordinates_",
    "restore_hadamard_structure_from_metadata_",
    "quantize_nestquant_w4_",
    "quantize_rtn_w4_",
]
