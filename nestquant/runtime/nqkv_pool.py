"""Fixed-address GPU page pool for NQKV v1 payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from ..core.kv_format import PackedNQKVPage
from .nqkv_page import PreparedNQKVPage


@dataclass(frozen=True)
class NQKVPageHandle:
    """Generation-checked ownership token for one physical page."""

    page_id: int
    generation: int


class NQKVPagePool:
    """Preallocated SoA page storage with host-managed allocation metadata.

    The underlying CUDA tensors never move. Allocation and release modify only
    host metadata, so prepared page views retain graph-compatible addresses for
    the lifetime of the pool.
    """

    def __init__(
        self,
        *,
        capacity: int,
        page_size: int,
        num_kv_heads: int,
        head_dim: int,
        q: int,
        betas: Sequence[float] | None = None,
        key_betas: Sequence[float] | None = None,
        value_betas: Sequence[float] | None = None,
        output_dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cuda",
    ) -> None:
        for name, value in (
            ("capacity", capacity), ("page_size", page_size),
            ("num_kv_heads", num_kv_heads), ("head_dim", head_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if head_dim % 8:
            raise ValueError("head_dim must be divisible by 8")
        if betas is not None:
            if key_betas is not None or value_betas is not None:
                raise ValueError("pass legacy betas or key_betas/value_betas, not both")
            key_beta_values = tuple(float(beta) for beta in betas)
            value_beta_values = key_beta_values
            legacy_shared = True
        else:
            if key_betas is None or value_betas is None:
                raise ValueError("v2 requires both key_betas and value_betas")
            key_beta_values = tuple(float(beta) for beta in key_betas)
            value_beta_values = tuple(float(beta) for beta in value_betas)
            legacy_shared = False
        if not key_beta_values or not value_beta_values:
            raise ValueError("beta tables must be nonempty")
        target = torch.device(device)
        if target.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("NQKVPagePool requires an available CUDA device")
        if target.index is None:
            target = torch.device("cuda", torch.cuda.current_device())

        # Derive exact stream sizes from the canonical format rather than
        # duplicating packing-width arithmetic in the allocator.
        zero = torch.zeros((1, num_kv_heads, head_dim), dtype=output_dtype)
        template = (
            PackedNQKVPage.from_tensors(
                zero, zero, page_size=page_size, q=q, betas=key_beta_values
            )
            if legacy_shared else PackedNQKVPage.from_tensors(
                zero, zero, page_size=page_size, q=q,
                key_betas=key_beta_values, value_betas=value_beta_values,
            )
        )
        self.capacity = capacity
        self.page_size = page_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q = q
        self.betas = key_beta_values
        self.value_beta_values = value_beta_values
        self.format_version = 1 if legacy_shared else 2
        self.output_dtype = output_dtype
        self.device = target
        self.code_width = template.key_indices.code_bit_width
        self.beta_width = template.key_indices.beta_index_bit_width
        self._code_bytes = len(template.key_indices.code_payload)
        self._beta_bytes = len(template.key_indices.beta_payload)
        self._row_count = page_size * num_kv_heads

        self.key_codes = torch.zeros((capacity, self._code_bytes), device=target, dtype=torch.uint8)
        self.key_betas = torch.zeros((capacity, self._beta_bytes), device=target, dtype=torch.uint8)
        self.key_norms = torch.zeros((capacity, self._row_count), device=target, dtype=torch.float32)
        self.value_codes = torch.zeros_like(self.key_codes)
        self.value_betas = torch.zeros_like(self.key_betas)
        self.value_norms = torch.zeros_like(self.key_norms)
        self.beta_table = torch.tensor(key_beta_values, device=target, dtype=torch.float32)
        self.value_beta_table = (
            self.beta_table if legacy_shared else
            torch.tensor(value_beta_values, device=target, dtype=torch.float32)
        )

        self._free = list(range(capacity - 1, -1, -1))
        self._generation = [0] * capacity
        self._allocated = [False] * capacity
        self._stored = [False] * capacity
        self._valid_tokens = [0] * capacity

    @property
    def free_pages(self) -> int:
        return len(self._free)

    @property
    def allocated_pages(self) -> int:
        return self.capacity - len(self._free)

    @property
    def resident_bytes(self) -> int:
        tensors = (
                self.key_codes, self.key_betas, self.key_norms,
                self.value_codes, self.value_betas, self.value_norms,
                self.beta_table, self.value_beta_table,
            )
        return sum(
            tensor.numel() * tensor.element_size()
            for index, tensor in enumerate(tensors)
            if tensor.data_ptr() not in {item.data_ptr() for item in tensors[:index]}
        )

    def allocate(self) -> NQKVPageHandle:
        if not self._free:
            raise RuntimeError("NQKV page pool is exhausted")
        page_id = self._free.pop()
        self._allocated[page_id] = True
        self._stored[page_id] = False
        self._valid_tokens[page_id] = 0
        return NQKVPageHandle(page_id, self._generation[page_id])

    def release(self, handle: NQKVPageHandle) -> None:
        page_id = self._validate_handle(handle)
        self._allocated[page_id] = False
        self._stored[page_id] = False
        self._valid_tokens[page_id] = 0
        self._generation[page_id] += 1
        self._free.append(page_id)

    def store(self, handle: NQKVPageHandle, page: PackedNQKVPage) -> None:
        page_id = self._validate_handle(handle)
        self._validate_page(page)
        self.key_codes[page_id].copy_(_host_bytes(page.key_indices.code_payload).to(self.device))
        self.key_betas[page_id].copy_(_host_bytes(page.key_indices.beta_payload).to(self.device))
        self.key_norms[page_id].copy_(page.key_norms.to(self.device))
        self.value_codes[page_id].copy_(_host_bytes(page.value_indices.code_payload).to(self.device))
        self.value_betas[page_id].copy_(_host_bytes(page.value_indices.beta_payload).to(self.device))
        self.value_norms[page_id].copy_(page.value_norms.to(self.device))
        self._valid_tokens[page_id] = page.valid_tokens
        self._stored[page_id] = True

    def store_prepared(self, handle: NQKVPageHandle, page: PreparedNQKVPage) -> None:
        """Copy a GPU-produced packed page into its fixed physical slot."""
        page_id = self._validate_handle(handle)
        if not isinstance(page, PreparedNQKVPage):
            raise TypeError("page must be a PreparedNQKVPage")
        expected = (
            self.page_size, self.num_kv_heads, self.head_dim, self.q,
            self.code_width, self.beta_width, self.output_dtype, self.device,
        )
        actual = (
            page.page_size, page.num_kv_heads, page.head_dim, page.q,
            page.code_width, page.beta_width, page.output_dtype, page.device,
        )
        if actual != expected:
            raise ValueError("prepared NQKV page configuration does not match the pool")
        expected_sizes = (
            self._code_bytes, self._beta_bytes, self._row_count,
            self._code_bytes, self._beta_bytes, self._row_count,
        )
        tensors = (
            page.key_code_payload, page.key_beta_payload, page.key_norms,
            page.value_code_payload, page.value_beta_payload, page.value_norms,
        )
        if tuple(tensor.numel() for tensor in tensors) != expected_sizes:
            raise ValueError("prepared NQKV payload sizes do not match the pool")
        if (
            not torch.equal(page.key_beta_table, self.beta_table)
            or not torch.equal(page.value_beta_table, self.value_beta_table)
        ):
            raise ValueError("prepared NQKV beta tables do not match the pool")
        for destination, source in zip(
            (
                self.key_codes[page_id], self.key_betas[page_id], self.key_norms[page_id],
                self.value_codes[page_id], self.value_betas[page_id], self.value_norms[page_id],
            ),
            tensors,
            strict=True,
        ):
            destination.copy_(source)
        self._valid_tokens[page_id] = page.valid_tokens
        self._stored[page_id] = True

    def mark_graph_stored(self, handle: NQKVPageHandle, *, valid_tokens: int) -> None:
        """Publish host bookkeeping after a Graph wrote the fixed page tensors."""
        page_id = self._validate_handle(handle)
        if not 1 <= valid_tokens <= self.page_size:
            raise ValueError("valid_tokens must be in [1, page_size]")
        self._valid_tokens[page_id] = valid_tokens
        self._stored[page_id] = True

    def prepared(self, handle: NQKVPageHandle) -> PreparedNQKVPage:
        page_id = self._validate_handle(handle)
        if not self._stored[page_id]:
            raise RuntimeError("NQKV page has not been stored")
        return PreparedNQKVPage(
            key_code_payload=self.key_codes[page_id],
            key_beta_payload=self.key_betas[page_id],
            key_norms=self.key_norms[page_id],
            value_code_payload=self.value_codes[page_id],
            value_beta_payload=self.value_betas[page_id],
            value_norms=self.value_norms[page_id],
            key_beta_table=self.beta_table,
            value_beta_table=self.value_beta_table,
            page_size=self.page_size,
            valid_tokens=self._valid_tokens[page_id],
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            q=self.q,
            code_width=self.code_width,
            beta_width=self.beta_width,
            output_dtype=self.output_dtype,
        )

    def _validate_handle(self, handle: NQKVPageHandle) -> int:
        if not isinstance(handle, NQKVPageHandle):
            raise TypeError("handle must be an NQKVPageHandle")
        if not 0 <= handle.page_id < self.capacity:
            raise ValueError("NQKV page ID is outside pool capacity")
        if (
            not self._allocated[handle.page_id]
            or self._generation[handle.page_id] != handle.generation
        ):
            raise RuntimeError("stale or released NQKV page handle")
        return handle.page_id

    def _validate_page(self, page: PackedNQKVPage) -> None:
        if not isinstance(page, PackedNQKVPage):
            raise TypeError("page must be a PackedNQKVPage")
        page._validate()
        expected = (
            self.page_size, self.num_kv_heads, self.head_dim, self.q,
            self.betas, self.value_beta_values, self.output_dtype,
        )
        actual = (
            page.page_size, page.num_kv_heads, page.head_dim, page.q,
            page.betas, page.resolved_value_betas, page.output_dtype,
        )
        if actual != expected:
            raise ValueError("NQKV page configuration does not match the pool")


def _host_bytes(payload: bytes) -> torch.Tensor:
    return torch.frombuffer(bytearray(payload), dtype=torch.uint8)
