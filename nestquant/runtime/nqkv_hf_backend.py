"""Transformers Llama Cache and AttentionInterface integration for NQKV K1."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ..core.rotation import hadamard
from .nqkv_attention_k1 import NQKVAttentionWorkspace, nqkv_decode_attention_k1
from .nqkv_hot_cache import NQKVHotColdCache
from .nqkv_pool import NQKVPagePool
from .paged_nqkv_cache import PagedNQKVCache


NQKV_ATTENTION_NAME = "nestquant_nqkv_k1"
_SEQUENCE_ID = "batch0"


class NQKVLlamaCache(Cache):
    """One-sequence, multi-layer NQKV cache for Transformers Llama decode."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_cache_len: int,
        page_size: int,
        q: int,
        key_betas: Sequence[float],
        value_betas: Sequence[float],
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        super().__init__(layers=[])
        for name, value in (
            ("num_layers", num_layers), ("num_query_heads", num_query_heads),
            ("num_kv_heads", num_kv_heads), ("head_dim", head_dim),
            ("max_cache_len", max_cache_len), ("page_size", page_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if num_query_heads % num_kv_heads:
            raise ValueError("num_query_heads must be divisible by num_kv_heads")
        max_pages = math.ceil(max_cache_len / page_size)
        self.num_layers = num_layers
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_cache_len = max_cache_len
        self.page_size = page_size
        self.dtype = dtype
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.layer_caches: list[NQKVHotColdCache] = []
        self.workspaces: list[NQKVAttentionWorkspace] = []
        for _ in range(num_layers):
            pool = NQKVPagePool(
                capacity=max_pages, page_size=page_size,
                num_kv_heads=num_kv_heads, head_dim=head_dim, q=q,
                key_betas=key_betas, value_betas=value_betas,
                output_dtype=dtype, device=self.device,
            )
            logical = PagedNQKVCache(
                pool, max_sequences=1, max_pages_per_sequence=max_pages
            )
            cache = NQKVHotColdCache(logical)
            cache.register(_SEQUENCE_ID)
            self.layer_caches.append(cache)
            self.workspaces.append(NQKVAttentionWorkspace.create(
                cache, query_heads=num_query_heads, output_dtype=dtype
            ))

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del args, kwargs
        self._validate_layer(layer_idx)
        if key_states.ndim != 4 or key_states.shape != value_states.shape:
            raise ValueError("NQKV cache update requires matching BHTD Key/Value")
        if key_states.shape[0] != 1:
            raise ValueError("NQKV Llama backend currently supports batch size one")
        if key_states.shape[1] != self.num_kv_heads or key_states.shape[3] != self.head_dim:
            raise ValueError("Key/Value heads or head_dim disagree with NQKV cache")
        if key_states.device != self.device or key_states.dtype != self.dtype:
            raise ValueError("Key/Value must match NQKV cache device and dtype")
        rotated_key = hadamard(key_states.float()).to(self.dtype)
        self.layer_caches[layer_idx].append(
            _SEQUENCE_ID,
            rotated_key[0].permute(1, 0, 2).contiguous(),
            value_states[0].permute(1, 0, 2).contiguous(),
        )
        # The registered AttentionInterface consumes the resident cache rather
        # than these current-token tensors. Returning tensors preserves the
        # Transformers Cache.update contract without materializing history.
        return rotated_key, value_states

    def attend(
        self, layer_idx: int, query_states: torch.Tensor, *, scale: float
    ) -> torch.Tensor:
        self._validate_layer(layer_idx)
        if query_states.shape != (1, self.num_query_heads, 1, self.head_dim):
            raise ValueError(
                "NQKV K1 currently requires query shape [1, query_heads, 1, head_dim]"
            )
        rotated_query = hadamard(query_states.float()).to(self.dtype)
        output = nqkv_decode_attention_k1(
            rotated_query[0, :, 0].contiguous(),
            self.layer_caches[layer_idx], _SEQUENCE_ID,
            scale=scale, workspace=self.workspaces[layer_idx],
        )
        return output.view(1, self.num_query_heads, 1, self.head_dim)

    def prefill(
        self, layer_idx: int, query_states: torch.Tensor, *, scale: float
    ) -> torch.Tensor:
        """Run an initial causal prompt over the packed-cold/BF16-hot cache."""
        self._validate_layer(layer_idx)
        if query_states.ndim != 4 or query_states.shape[:2] != (
            1, self.num_query_heads
        ) or query_states.shape[3] != self.head_dim:
            raise ValueError("NQKV prefill query has an incompatible BHTD shape")
        query_length = query_states.shape[2]
        cache = self.layer_caches[layer_idx]
        if cache.total_tokens(_SEQUENCE_ID) != query_length:
            raise ValueError("NQKV prefill currently requires an empty initial cache")
        key, value = cache.materialize(_SEQUENCE_ID)
        rotated_query = hadamard(query_states.float()).to(self.dtype)
        return F.scaled_dot_product_attention(
            rotated_query,
            key.permute(1, 0, 2).unsqueeze(0),
            value.permute(1, 0, 2).unsqueeze(0),
            dropout_p=0.0,
            is_causal=True,
            scale=scale,
            enable_gqa=self.num_query_heads != self.num_kv_heads,
        )

    def get_seq_length(self, layer_idx: int = 0) -> int:
        self._validate_layer(layer_idx)
        return self.layer_caches[layer_idx].total_tokens(_SEQUENCE_ID)

    def get_max_length(self, layer_idx: int | None = None) -> int:
        if layer_idx is not None:
            self._validate_layer(layer_idx)
        return self.max_cache_len

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:
        self._validate_layer(layer_idx)
        return self.get_seq_length(layer_idx) + query_length, 0

    @property
    def batch_size(self) -> int:
        return 1

    @property
    def is_compileable(self) -> bool:
        return False

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        del beam_idx
        raise NotImplementedError("NQKV backend does not support beam search")

    def reset(self) -> None:
        for cache in self.layer_caches:
            cache.release(_SEQUENCE_ID)
            cache.register(_SEQUENCE_ID)

    def _validate_layer(self, layer_idx: int) -> None:
        if isinstance(layer_idx, bool) or not isinstance(layer_idx, int):
            raise TypeError("layer_idx must be an integer")
        if not 0 <= layer_idx < self.num_layers:
            raise ValueError("layer_idx is outside the NQKV layer range")


def _nqkv_attention_interface(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    del key, value
    cache = getattr(module, "_nestquant_nqkv_cache", None)
    if not isinstance(cache, NQKVLlamaCache):
        raise RuntimeError("Llama attention module has no installed NQKV cache")
    if kwargs.get("dropout", 0.0) != 0.0:
        raise ValueError("NQKV backend supports inference dropout=0 only")
    scale = float(kwargs.get("scaling", module.scaling))
    if query.shape[2] == 1:
        return cache.attend(module.layer_idx, query, scale=scale), None
    if attention_mask is not None:
        raise ValueError("NQKV prefill currently supports unpadded prompts only")
    return cache.prefill(module.layer_idx, query, scale=scale), None


def install_nqkv_llama_backend(model: torch.nn.Module, cache: NQKVLlamaCache) -> str:
    """Register K1 and attach one cache controller to every Llama layer."""
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch module")
    if not isinstance(cache, NQKVLlamaCache):
        raise TypeError("cache must be an NQKVLlamaCache")
    ALL_ATTENTION_FUNCTIONS.register(NQKV_ATTENTION_NAME, _nqkv_attention_interface)
    attached = 0
    for module in model.modules():
        if module.__class__.__name__ == "LlamaAttention":
            if not 0 <= module.layer_idx < cache.num_layers:
                raise ValueError("Llama layer index disagrees with NQKV cache")
            module._nestquant_nqkv_cache = cache
            module.config._attn_implementation = NQKV_ATTENTION_NAME
            attached += 1
    if attached != cache.num_layers:
        raise ValueError(
            f"expected {cache.num_layers} LlamaAttention modules, found {attached}"
        )
    if hasattr(model, "config"):
        model.config._attn_implementation = NQKV_ATTENTION_NAME
    return NQKV_ATTENTION_NAME
