"""Adapters applying algorithm primitives to neural-network modules."""

from .lazy_linear import (
    CachePolicy,
    LazyNQMCLinear,
    LazyNQMCWeightCache,
    NQMCRecordPrefetcher,
)
from .lazy_model import LazyNQMCConversionReport, convert_model_to_lazy_nqmc_
from .fused_linear import (
    FusedNQRTConversionReport,
    FusedNQRTLinear,
    SharedHadamardCache,
    convert_linears_to_fused_nqrt_,
)
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
    "FusedNQRTConversionReport",
    "FusedNQRTLinear",
    "SharedHadamardCache",
    "LazyNQMCLinear",
    "LazyNQMCConversionReport",
    "LazyNQMCWeightCache",
    "NQMCRecordPrefetcher",
    "PAPER_HADAMARD_FORMAT",
    "convert_linears_to_hadamard_coordinates_",
    "convert_linears_to_fused_nqrt_",
    "convert_model_to_lazy_nqmc_",
    "restore_hadamard_structure_from_metadata_",
    "quantize_nestquant_w4_",
    "quantize_rtn_w4_",
]
