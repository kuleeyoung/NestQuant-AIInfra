"""Experimental compiled runtime primitives for packed NestQuant artifacts."""

from .triton_unpack import triton_available, unpack_indices_triton
from .triton_decode import dequantize_with_triton_unpack
from .triton_tiled_decode import dequantize_tiled_triton
from .triton_fused_linear import (
    PreparedNQRTLinear,
    linear_fused_triton,
    prepare_fused_linear,
)
from .triton_grouped_linear import linear_fused_pair_triton, linear_fused_qkv_triton
from .triton_dot_linear import linear_fused_dot_triton, select_nqrt_dot_config
from .cuda_e8 import cuda_e8_available, e8_nearest_cuda
from .cuda_mma_linear import can_use_mma_j2, linear_fused_mma_j2
from .cuda_graph_decode import NQRTGraphDecodePool, NQRTGraphDecodeRunner
from .ragged_cache import RaggedCacheController
from .ragged_graph_decode import NQRTRaggedGraphDecodeRunner
from .continuous_batching import ContinuousBatchScheduler, DecodeRequest, RequestStatus
from .inference_engine import AdmissionPolicy, ContinuousBatchInferenceEngine, EngineStep, GreedyHostSampler
from .hybrid_residency import (
    ResidencyGroup,
    ResidencyPlan,
    atomic_projection_groups,
    make_residency_groups,
    nondominated_indices,
    residency_groups_from_payload,
    select_residency_plan,
)

__all__ = [
    "dequantize_with_triton_unpack",
    "dequantize_tiled_triton",
    "linear_fused_triton",
    "linear_fused_qkv_triton",
    "linear_fused_pair_triton",
    "linear_fused_dot_triton",
    "select_nqrt_dot_config",
    "prepare_fused_linear",
    "PreparedNQRTLinear",
    "triton_available",
    "unpack_indices_triton",
    "cuda_e8_available",
    "e8_nearest_cuda",
    "can_use_mma_j2",
    "linear_fused_mma_j2",
    "NQRTGraphDecodeRunner",
    "NQRTGraphDecodePool",
    "RaggedCacheController",
    "NQRTRaggedGraphDecodeRunner",
    "ContinuousBatchScheduler",
    "DecodeRequest",
    "RequestStatus",
    "ContinuousBatchInferenceEngine",
    "EngineStep",
    "GreedyHostSampler",
    "AdmissionPolicy",
    "ResidencyGroup",
    "ResidencyPlan",
    "atomic_projection_groups",
    "make_residency_groups",
    "residency_groups_from_payload",
    "select_residency_plan",
    "nondominated_indices",
]
