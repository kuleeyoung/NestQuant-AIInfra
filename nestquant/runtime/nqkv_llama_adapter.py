"""Shape and rotation adapter from Llama post-RoPE states to NQKV K1."""

from __future__ import annotations

from typing import Hashable

import torch

from ..core.rotation import hadamard
from .nqkv_attention_k1 import NQKVAttentionWorkspace, nqkv_decode_attention_k1
from .nqkv_hot_cache import NQKVHotColdCache


class NQKVLlamaDecodeAdapter:
    """Consume one Llama decode query and append its post-RoPE K/V states.

    Inputs follow the Transformers attention layout ``[batch, heads, tokens,
    head_dim]``. Version one intentionally supports a single logical sequence
    and one query token while allowing multiple K/V tokens for prompt-to-cache
    bootstrap tests.
    """

    def __init__(
        self,
        cache: NQKVHotColdCache,
        *,
        query_heads: int,
        rotate_query_key: bool = True,
        output_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        pool = cache.cold_cache.pool
        if query_heads < 1 or query_heads % pool.num_kv_heads:
            raise ValueError("query_heads must be a positive multiple of num_kv_heads")
        self.cache = cache
        self.query_heads = query_heads
        self.rotate_query_key = bool(rotate_query_key)
        self.workspace = NQKVAttentionWorkspace.create(
            cache, query_heads=query_heads, output_dtype=output_dtype
        )

    def append_and_attend(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        sequence_id: Hashable,
        *,
        scale: float | None = None,
    ) -> torch.Tensor:
        """Append K/V and return one-token attention in BHTD layout."""
        self._validate_states(query_states, key_states, value_states)
        query = query_states
        key = key_states
        if self.rotate_query_key:
            query = hadamard(query.float()).to(query_states.dtype)
            key = hadamard(key.float()).to(key_states.dtype)
        # BHTD -> THD for the page writer. Batch is deliberately fixed to one.
        key_nhd = key[0].permute(1, 0, 2).contiguous()
        value_nhd = value_states[0].permute(1, 0, 2).contiguous()
        self.cache.append(sequence_id, key_nhd, value_nhd)
        query_hd = query[0, :, 0, :].contiguous()
        output = nqkv_decode_attention_k1(
            query_hd, self.cache, sequence_id, scale=scale,
            workspace=self.workspace,
        )
        return output.view(1, self.query_heads, 1, output.shape[-1])

    def _validate_states(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        pool = self.cache.cold_cache.pool
        if not all(isinstance(tensor, torch.Tensor) for tensor in (query, key, value)):
            raise TypeError("query, key, and value must be torch tensors")
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError("query, key, and value must use BHTD layout")
        if query.shape[0] != 1 or key.shape[0] != 1 or value.shape[0] != 1:
            raise ValueError("the first Llama adapter supports batch size one")
        if query.shape[1:] != (self.query_heads, 1, pool.head_dim):
            raise ValueError("query must have shape [1, query_heads, 1, head_dim]")
        if key.shape != value.shape or key.shape[1] != pool.num_kv_heads:
            raise ValueError("key/value must share [1, kv_heads, tokens, head_dim]")
        if key.shape[2] < 1 or key.shape[3] != pool.head_dim:
            raise ValueError("key/value token count and head_dim are invalid")
        for tensor in (query, key, value):
            if tensor.device != pool.device or tensor.dtype != pool.output_dtype:
                raise ValueError("query, key, and value must match cache device and dtype")
