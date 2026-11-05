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
from .cuda_w4a4_mma import (
    can_use_w4a4_gemv_m1,
    can_use_w4a4_mma_small_m,
    can_use_w4a4_mma_m128_staged,
    can_use_w4a4_mma_large_m_staged,
    can_use_w4a4_mma_chunked_staged,
    can_use_w4a4_mma_m16,
    linear_packed_w4a4_gemv_m1,
    linear_packed_w4a4_mma_small_m,
    linear_packed_w4a4_mma_m128_staged,
    linear_packed_w4a4_mma_large_m_staged,
    linear_packed_w4a4_mma_chunked_staged,
    linear_packed_w4a4_mma_m16,
)
from .cuda_graph_decode import NQRTGraphDecodePool, NQRTGraphDecodeRunner
from .ragged_cache import RaggedCacheController
from .ragged_graph_decode import NQRTRaggedGraphDecodeRunner
from .continuous_batching import ContinuousBatchScheduler, DecodeRequest, RequestStatus
from .inference_engine import AdmissionPolicy, ContinuousBatchInferenceEngine, EngineStep, GreedyHostSampler
from .online_service import (
    OnlineInferenceService,
    OnlineRequestHandle,
    OnlineServiceResult,
    OnlineServiceStatus,
    ServiceOverloadedError,
)
from .http_service import NestQuantHTTPServer, TokenizerProtocol
from .hybrid_residency import (
    ResidencyGroup,
    ResidencyPlan,
    atomic_projection_groups,
    make_residency_groups,
    nondominated_indices,
    residency_groups_from_payload,
    select_residency_plan,
)
from .nqkv_page import (
    PreparedNQKVPage,
    dequantize_nqkv_page_triton,
    prepare_nqkv_page,
    quantize_nqkv_page_cuda,
)
from .nqkv_pool import NQKVPageHandle, NQKVPagePool
from .paged_nqkv_cache import BatchedNQKVMetadata, PagedNQKVCache
from .nqkv_hot_cache import NQKVHotColdCache
from .nqkv_batched_writer import BatchedNQKVWriteResult, BatchedNQKVWriter
from .nqkv_batch_manager import BatchedNQKVCacheManager
from .nqkv_batched_hf_backend import (
    BATCHED_NQKV_ATTENTION_NAME,
    BatchedNQKVLlamaCache,
    install_batched_nqkv_llama_backend,
)
from .nqkv_batched_runner import BatchedNQKVLlamaRunner
from .nqkv_graph_writer import append_hot_k1_graph_safe
from .nqkv_graph_pack import launch_pack_full_hot_pages_graph_safe
from .nqkv_attention import nqkv_decode_attention_reference
from .nqkv_attention_k1 import (
    BatchedNQKVAttentionGraphRunner,
    BatchedNQKVAttentionWorkspace,
    NQKVAttentionGraphRunner,
    NQKVAttentionWorkspace,
    nqkv_decode_attention_batched_k1,
    nqkv_decode_attention_k1,
)
from .nqkv_llama_adapter import NQKVLlamaDecodeAdapter
from .nqkv_hf_backend import (
    NQKV_ATTENTION_NAME,
    NQKVLlamaCache,
    install_nqkv_llama_backend,
)
from .triton_w4a4_linear import (
    PreparedNQA4Activation,
    linear_packed_w4a4_triton,
    linear_packed_w4a4_tiled_triton,
    quantize_activation_cuda,
    select_w4a4_tiled_config,
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
    "can_use_w4a4_gemv_m1",
    "can_use_w4a4_mma_small_m",
    "can_use_w4a4_mma_m128_staged",
    "can_use_w4a4_mma_large_m_staged",
    "can_use_w4a4_mma_chunked_staged",
    "can_use_w4a4_mma_m16",
    "linear_packed_w4a4_gemv_m1",
    "linear_packed_w4a4_mma_small_m",
    "linear_packed_w4a4_mma_m128_staged",
    "linear_packed_w4a4_mma_large_m_staged",
    "linear_packed_w4a4_mma_chunked_staged",
    "linear_packed_w4a4_mma_m16",
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
    "OnlineInferenceService",
    "OnlineRequestHandle",
    "OnlineServiceResult",
    "OnlineServiceStatus",
    "ServiceOverloadedError",
    "NestQuantHTTPServer",
    "TokenizerProtocol",
    "ResidencyGroup",
    "ResidencyPlan",
    "atomic_projection_groups",
    "make_residency_groups",
    "residency_groups_from_payload",
    "select_residency_plan",
    "nondominated_indices",
    "PreparedNQKVPage",
    "prepare_nqkv_page",
    "quantize_nqkv_page_cuda",
    "dequantize_nqkv_page_triton",
    "NQKVPageHandle",
    "NQKVPagePool",
    "BatchedNQKVMetadata",
    "PagedNQKVCache",
    "NQKVHotColdCache",
    "BatchedNQKVWriteResult",
    "BatchedNQKVWriter",
    "BatchedNQKVCacheManager",
    "BATCHED_NQKV_ATTENTION_NAME",
    "BatchedNQKVLlamaCache",
    "install_batched_nqkv_llama_backend",
    "BatchedNQKVLlamaRunner",
    "append_hot_k1_graph_safe",
    "launch_pack_full_hot_pages_graph_safe",
    "nqkv_decode_attention_reference",
    "nqkv_decode_attention_k1",
    "nqkv_decode_attention_batched_k1",
    "NQKVAttentionWorkspace",
    "BatchedNQKVAttentionWorkspace",
    "BatchedNQKVAttentionGraphRunner",
    "NQKVAttentionGraphRunner",
    "NQKVLlamaDecodeAdapter",
    "NQKV_ATTENTION_NAME",
    "NQKVLlamaCache",
    "install_nqkv_llama_backend",
    "PreparedNQA4Activation",
    "quantize_activation_cuda",
    "linear_packed_w4a4_triton",
    "linear_packed_w4a4_tiled_triton",
    "select_w4a4_tiled_config",
]
