"""Inference-only Linear backed by one lazily decoded NQMC weight record."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from threading import RLock
import time
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from ..core.model_checkpoint_file import NQMCFileReader
from ..core.rotation import hadamard
from ..core.tensor_checkpoint import PackedNestQuantTensor


CachePolicy = Literal["none", "persistent", "lru"]
CacheKey = tuple[str, str, torch.dtype]
RuntimeProfileCallback = Callable[[dict[str, object]], None]


class LazyNQMCWeightCache:
    """Capacity-bounded, process-local LRU cache for decoded weight tensors."""

    def __init__(self, capacity_bytes: int) -> None:
        if isinstance(capacity_bytes, bool) or not isinstance(capacity_bytes, int):
            raise TypeError("capacity_bytes must be an integer")
        if capacity_bytes < 0:
            raise ValueError("capacity_bytes must be nonnegative")
        self.capacity_bytes = capacity_bytes
        self._values: OrderedDict[CacheKey, torch.Tensor] = OrderedDict()
        self._resident_bytes = 0
        self._peak_resident_bytes = 0
        self._hits = 0
        self._misses = 0
        self._decodes = 0
        self._evictions = 0
        self._bypasses = 0
        self._lock = RLock()

    @staticmethod
    def _tensor_bytes(tensor: torch.Tensor) -> int:
        return tensor.numel() * tensor.element_size()

    def get_or_load(
        self,
        key: CacheKey,
        expected_bytes: int,
        loader: Callable[[], torch.Tensor],
    ) -> torch.Tensor:
        """Return a cached tensor or load it after making room under the budget."""
        if expected_bytes < 0:
            raise ValueError("expected_bytes must be nonnegative")
        with self._lock:
            cached = self._values.get(key)
            if cached is not None:
                self._values.move_to_end(key)
                self._hits += 1
                return cached
            self._misses += 1
            if expected_bytes > self.capacity_bytes:
                self._bypasses += 1
                self._decodes += 1
                return loader()
            while self._values and self._resident_bytes + expected_bytes > self.capacity_bytes:
                _, evicted = self._values.popitem(last=False)
                self._resident_bytes -= self._tensor_bytes(evicted)
                self._evictions += 1
            value = loader()
            self._decodes += 1
            actual_bytes = self._tensor_bytes(value)
            if actual_bytes != expected_bytes:
                raise ValueError("decoded tensor size disagrees with the cache estimate")
            self._values[key] = value
            self._resident_bytes += actual_bytes
            self._peak_resident_bytes = max(
                self._peak_resident_bytes, self._resident_bytes
            )
            return value

    def contains(self, key: CacheKey) -> bool:
        with self._lock:
            return key in self._values

    def bytes_for(self, key: CacheKey) -> int:
        with self._lock:
            value = self._values.get(key)
            return 0 if value is None else self._tensor_bytes(value)

    def remove(self, key: CacheKey) -> None:
        with self._lock:
            value = self._values.pop(key, None)
            if value is not None:
                self._resident_bytes -= self._tensor_bytes(value)

    def clear(self, *, reset_statistics: bool = False) -> None:
        with self._lock:
            self._values.clear()
            self._resident_bytes = 0
            if reset_statistics:
                self._peak_resident_bytes = 0
                self._hits = self._misses = self._decodes = 0
                self._evictions = self._bypasses = 0

    def snapshot(self) -> dict[str, int | float]:
        """Return immutable accounting suitable for logs and benchmark reports."""
        with self._lock:
            accesses = self._hits + self._misses
            return {
                "capacity_bytes": self.capacity_bytes,
                "resident_bytes": self._resident_bytes,
                "peak_resident_bytes": self._peak_resident_bytes,
                "resident_tensor_count": len(self._values),
                "hits": self._hits,
                "misses": self._misses,
                "decodes": self._decodes,
                "evictions": self._evictions,
                "bypasses": self._bypasses,
                "hit_rate": self._hits / accesses if accesses else 0.0,
            }


class LazyNQMCLinear(nn.Module):
    """Decode one packed NQMC weight on demand, then run a floating-point GEMM.

    This is a bounded-scope runtime prototype, not a packed low-bit kernel.  A
    ``none`` cache decodes the weight for every call and releases the temporary
    after the call.  A ``persistent`` cache retains the decoded weight after
    the first call.  Paper-coordinate checkpoints can request the matching
    online input Hadamard transform with ``input_hadamard=True``.
    """

    def __init__(
        self,
        reader: NQMCFileReader,
        weight_name: str,
        bias: torch.Tensor | None = None,
        *,
        input_hadamard: bool = False,
        compute_dtype: torch.dtype = torch.bfloat16,
        output_dtype: torch.dtype | None = None,
        rows_per_chunk: int | None = 128,
        cache_policy: CachePolicy = "none",
        weight_cache: LazyNQMCWeightCache | None = None,
        profile_callback: RuntimeProfileCallback | None = None,
        synchronize_profile: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(reader, NQMCFileReader):
            raise TypeError("reader must be an NQMCFileReader")
        if cache_policy not in {"none", "persistent", "lru"}:
            raise ValueError("cache_policy must be 'none', 'persistent', or 'lru'")
        if cache_policy == "lru" and not isinstance(weight_cache, LazyNQMCWeightCache):
            raise ValueError("lru cache_policy requires a LazyNQMCWeightCache")
        if cache_policy != "lru" and weight_cache is not None:
            raise ValueError("weight_cache is only valid with lru cache_policy")
        if profile_callback is not None and not callable(profile_callback):
            raise TypeError("profile_callback must be callable")
        if synchronize_profile and profile_callback is None:
            raise ValueError("synchronize_profile requires a profile_callback")
        if compute_dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
            raise ValueError("compute_dtype must be a floating-point dtype")
        if output_dtype is not None and output_dtype not in {
            torch.float16, torch.bfloat16, torch.float32, torch.float64,
        }:
            raise ValueError("output_dtype must be a floating-point dtype or None")
        if rows_per_chunk is not None and (
            isinstance(rows_per_chunk, bool)
            or not isinstance(rows_per_chunk, int)
            or rows_per_chunk < 1
        ):
            raise ValueError("rows_per_chunk must be a positive integer or None")

        entry = reader.get_entry(weight_name)
        if entry.kind != "nestquant":
            raise ValueError("LazyNQMCLinear requires a packed NestQuant weight record")
        if len(entry.shape) != 2 or not all(entry.shape):
            raise ValueError("weight record must be nonempty with shape [out_features, in_features]")
        out_features, in_features = entry.shape
        if input_hadamard and in_features & (in_features - 1):
            raise ValueError("Hadamard input transform requires power-of-two in_features")
        if bias is not None:
            if not isinstance(bias, torch.Tensor) or not bias.is_floating_point():
                raise TypeError("bias must be a floating-point tensor")
            if tuple(bias.shape) != (out_features,):
                raise ValueError("bias must match out_features")

        self.reader = reader
        self.weight_name = weight_name
        self.canonical_weight_name = entry.name
        self.in_features = in_features
        self.out_features = out_features
        self.input_hadamard = input_hadamard
        self.compute_dtype = compute_dtype
        self.output_dtype = output_dtype
        self.rows_per_chunk = rows_per_chunk
        self.cache_policy: CachePolicy = cache_policy
        self.weight_cache = weight_cache
        self.profile_callback = profile_callback
        self.synchronize_profile = synchronize_profile
        self._last_device = torch.device("cpu")
        self.register_buffer(
            "bias", None if bias is None else bias.detach().clone().to(compute_dtype)
        )
        self.register_buffer("_cached_weight", None, persistent=False)

    @property
    def cached(self) -> bool:
        """Whether a decoded weight is currently retained."""
        if self.cache_policy == "lru":
            assert self.weight_cache is not None
            return self.weight_cache.contains(self._cache_key(self._runtime_device()))
        return self._cached_weight is not None

    @property
    def cached_weight_bytes(self) -> int:
        """Device bytes retained by the persistent cache."""
        if self.cache_policy == "lru":
            assert self.weight_cache is not None
            return self.weight_cache.bytes_for(
                self._cache_key(self._runtime_device())
            )
        if self._cached_weight is None:
            return 0
        return self._cached_weight.numel() * self._cached_weight.element_size()

    def clear_cache(self) -> None:
        """Release a persistent decoded weight."""
        if self.cache_policy == "lru":
            assert self.weight_cache is not None
            self.weight_cache.remove(self._cache_key(self._runtime_device()))
        self._cached_weight = None

    def _runtime_device(self) -> torch.device:
        if self._cached_weight is not None:
            return self._cached_weight.device
        if self.bias is not None:
            return self.bias.device
        return self._last_device

    def _cache_key(self, device: torch.device) -> CacheKey:
        return self.canonical_weight_name, str(device), self.compute_dtype

    def _weight(self, device: torch.device) -> torch.Tensor:
        self._last_device = device

        def decode() -> torch.Tensor:
            entry = self.reader.get_entry(self.weight_name)
            blob = self._profile_phase(
                "record_read", device, lambda: self.reader.read_entry_blob(entry),
                synchronize=False,
            )
            record = self._profile_phase(
                "record_parse", device,
                lambda: self.reader.parse_entry_blob(entry, blob),
                synchronize=False,
            )
            if not isinstance(record, PackedNestQuantTensor):
                raise TypeError("LazyNQMCLinear weight record must remain quantized")

            def dequantize() -> torch.Tensor:
                if self.rows_per_chunk is None:
                    return record.dequantize(device=device).to(self.compute_dtype)
                return record.dequantize_chunked(
                    device=device,
                    output_dtype=self.compute_dtype,
                    rows_per_chunk=self.rows_per_chunk,
                )

            return self._profile_phase(
                "unpack_dequant", device, dequantize, synchronize=True
            )

        if self.cache_policy == "lru":
            assert self.weight_cache is not None
            element_size = torch.empty((), dtype=self.compute_dtype).element_size()
            expected_bytes = self.out_features * self.in_features * element_size
            return self.weight_cache.get_or_load(
                self._cache_key(device), expected_bytes, decode
            )
        cached = self._cached_weight
        if cached is not None and cached.device == device and cached.dtype == self.compute_dtype:
            return cached
        # A module moved after caching must not retain an obsolete device copy.
        if cached is not None:
            self.clear_cache()
        decoded = decode()
        if self.cache_policy == "persistent":
            self._cached_weight = decoded
        return decoded

    def _profile_phase(
        self,
        phase: str,
        device: torch.device,
        operation: Callable[[], object],
        *,
        synchronize: bool,
    ):
        callback = self.profile_callback
        if callback is None:
            return operation()
        if self.synchronize_profile and synchronize and device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        result = operation()
        if self.synchronize_profile and synchronize and device.type == "cuda":
            torch.cuda.synchronize(device)
        callback({
            "weight_name": self.weight_name,
            "phase": phase,
            "seconds": time.perf_counter() - started,
            "device": str(device),
        })
        return result

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not isinstance(inputs, torch.Tensor) or not inputs.is_floating_point():
            raise TypeError("inputs must be a floating-point tensor")
        if inputs.ndim == 0 or inputs.shape[-1] != self.in_features:
            actual = None if inputs.ndim == 0 else inputs.shape[-1]
            raise ValueError(
                f"input feature dimension {actual} does not match {self.in_features}"
            )
        original_dtype = inputs.dtype
        def transform_inputs() -> torch.Tensor:
            transformed = inputs.to(self.compute_dtype)
            return hadamard(transformed) if self.input_hadamard else transformed

        compute_inputs = self._profile_phase(
            "input_transform", inputs.device, transform_inputs, synchronize=True
        )
        weight = self._weight(inputs.device)
        bias = None if self.bias is None else self.bias.to(inputs.device)
        target_dtype = original_dtype if self.output_dtype is None else self.output_dtype
        return self._profile_phase(
            "gemm_output_cast",
            inputs.device,
            lambda: F.linear(compute_inputs, weight, bias).to(target_dtype),
            synchronize=True,
        )
