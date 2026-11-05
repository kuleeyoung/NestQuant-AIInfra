"""Tests for logical sequence ownership over physical NQKV pages."""

import unittest

import torch

from nestquant.core.kv_format import PackedNQKVPage
from nestquant.runtime.nqkv_pool import NQKVPagePool
from nestquant.runtime.paged_nqkv_cache import BatchedNQKVMetadata, PagedNQKVCache
from nestquant.runtime.triton_unpack import triton_available


@unittest.skipUnless(triton_available(), "Triton CUDA runtime is unavailable")
class TestPagedNQKVCache(unittest.TestCase):
    def setUp(self):
        self.betas = (0.5, 0.75, 1.0, 1.25)
        self.pool = NQKVPagePool(
            capacity=6, page_size=4, num_kv_heads=2, head_dim=64,
            q=14, betas=self.betas,
        )
        self.cache = PagedNQKVCache(
            self.pool, max_sequences=3, max_pages_per_sequence=3
        )

    def _page(self, seed: int, tokens: int) -> PackedNQKVPage:
        generator = torch.Generator().manual_seed(seed)
        key = torch.randn(tokens, 2, 64, generator=generator).to(torch.bfloat16)
        value = torch.randn(tokens, 2, 64, generator=generator).to(torch.bfloat16)
        return PackedNQKVPage.from_tensors(
            key, value, page_size=4, q=14, betas=self.betas
        )

    def test_sequence_page_table_tracks_full_and_partial_pages(self):
        slot = self.cache.register("request-a")
        pointers = self.cache.table_pointers()
        first = self.cache.append_page("request-a", self._page(1, 4))
        second = self.cache.append_page("request-a", self._page(2, 2))
        self.assertEqual(self.cache.total_tokens("request-a"), 6)
        self.assertEqual(self.cache.handles("request-a"), (first, second))
        self.assertEqual(self.cache.page_counts[slot].item(), 2)
        self.assertEqual(self.cache.last_page_len[slot].item(), 2)
        self.assertEqual(self.cache.sequence_lengths[slot].item(), 6)
        self.assertEqual(
            self.cache.page_indices[slot, :2].cpu().tolist(),
            [first.page_id, second.page_id],
        )
        self.assertEqual(self.cache.table_pointers(), pointers)

    def test_fixed_batch_metadata_matches_dense_page_table(self):
        metadata = self.cache.metadata
        self.assertEqual(metadata.slot_ids.cpu().tolist(), [0, 1, 2])
        self.assertEqual(metadata.page_indptr.cpu().tolist(), [0, 3, 6, 9])
        self.assertIs(self.cache.page_indices, metadata.page_indices)
        self.assertIs(self.cache.hot_lengths, metadata.hot_lengths)
        self.assertIs(self.cache.active, metadata.active_mask)
        self.assertEqual(len(set(metadata.pointers())), 8)

    def test_explicit_scheduler_slot_binding_and_reuse(self):
        slot = self.cache.register("request-a", slot=2)
        self.assertEqual(slot, 2)
        self.assertEqual(self.cache.sequence_id(2), "request-a")
        self.assertTrue(self.cache.active[2].item())
        with self.assertRaisesRegex(RuntimeError, "already active"):
            self.cache.register("request-b", slot=2)
        self.cache.release("request-a")
        with self.assertRaisesRegex(KeyError, "not active"):
            self.cache.sequence_id(2)
        reused = self.cache.register("request-b", slot=2)
        self.assertEqual(reused, 2)
        self.assertEqual(self.cache.sequence_lengths[2].item(), 0)

    def test_rejects_invalid_explicit_slot(self):
        with self.assertRaisesRegex(TypeError, "integer"):
            self.cache.register("request-a", slot=True)
        with self.assertRaisesRegex(ValueError, "outside"):
            self.cache.register("request-b", slot=3)

    def test_partial_page_can_grow_but_not_shrink(self):
        self.cache.register("request-a")
        handle = self.cache.append_page("request-a", self._page(3, 1))
        replaced = self.cache.replace_last_page("request-a", self._page(4, 3))
        self.assertEqual(replaced, handle)
        self.assertEqual(self.cache.total_tokens("request-a"), 3)
        with self.assertRaisesRegex(ValueError, "shrink"):
            self.cache.replace_last_page("request-a", self._page(5, 2))

    def test_release_reclaims_pages_and_slot_without_changing_tables(self):
        first_slot = self.cache.register("old")
        self.cache.append_page("old", self._page(6, 4))
        pointers = self.cache.table_pointers()
        free_before_release = self.pool.free_pages
        self.cache.release("old")
        self.assertEqual(self.pool.free_pages, free_before_release + 1)
        self.assertFalse(self.cache.active[first_slot].item())
        self.assertTrue(torch.all(self.cache.page_indices[first_slot] == -1).item())
        self.assertEqual(self.cache.sequence_lengths[first_slot].item(), 0)
        self.assertEqual(self.cache.hot_lengths[first_slot].item(), 0)
        new_slot = self.cache.register("new")
        self.assertEqual(new_slot, first_slot)
        self.assertEqual(self.cache.table_pointers(), pointers)

    def test_rejects_append_after_partial_page_and_rolls_back_bad_store(self):
        self.cache.register("request-a")
        self.cache.append_page("request-a", self._page(7, 2))
        with self.assertRaisesRegex(RuntimeError, "partial"):
            self.cache.append_page("request-a", self._page(8, 4))
        free_before = self.pool.free_pages
        wrong = torch.randn(4, 2, 32)
        wrong_page = PackedNQKVPage.from_tensors(
            wrong, wrong.clone(), page_size=4, q=14, betas=self.betas
        )
        self.cache.register("request-b")
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.cache.append_page("request-b", wrong_page)
        self.assertEqual(self.pool.free_pages, free_before)


class TestBatchedNQKVMetadata(unittest.TestCase):
    def test_cpu_contract_is_fixed_address_and_int32(self):
        metadata = BatchedNQKVMetadata.create(
            capacity=8, max_pages_per_sequence=4, device="cpu"
        )
        pointers = metadata.pointers()
        metadata.page_indices[3, 1] = 7
        metadata.page_counts[3] = 2
        metadata.last_page_lens[3] = 3
        metadata.hot_lengths[3] = 1
        metadata.sequence_lengths[3] = 8
        metadata.active_mask[3] = True
        self.assertEqual(metadata.pointers(), pointers)
        self.assertEqual(metadata.slot_ids.tolist(), list(range(8)))
        self.assertEqual(metadata.page_indptr.tolist(), list(range(0, 33, 4)))
        for tensor in (
            metadata.slot_ids,
            metadata.page_indptr,
            metadata.page_indices,
            metadata.page_counts,
            metadata.last_page_lens,
            metadata.hot_lengths,
            metadata.sequence_lengths,
        ):
            self.assertEqual(tensor.dtype, torch.int32)
        self.assertEqual(metadata.active_mask.dtype, torch.bool)

    def test_rejects_invalid_capacity_or_page_limit(self):
        for kwargs in (
            {"capacity": 0, "max_pages_per_sequence": 4},
            {"capacity": 8, "max_pages_per_sequence": True},
        ):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                BatchedNQKVMetadata.create(device="cpu", **kwargs)


if __name__ == "__main__":
    unittest.main()
