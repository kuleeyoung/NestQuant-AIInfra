"""Tokenization and next-token perplexity evaluation helpers."""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TokenizedCorpus:
    token_ids: tuple[int, ...]
    text_sha256: str
    token_ids_sha256: str
    token_count: int
    chunk_count: int
    predicted_token_count: int
    dropped_tail_tokens: int
    bos_token_id: int | None
    eos_token_id: int | None


def tokenize_corpus(texts: Sequence[str], tokenizer, context_length: int) -> TokenizedCorpus:
    """Join dataset text rows and tokenize once with explicit boundary tokens."""
    if context_length < 2:
        raise ValueError("context_length must be at least 2")
    text = "\n\n".join(texts)
    # We intentionally tokenize the full joined corpus before splitting it into
    # model-sized windows. Raise the tokenizer's warning threshold to the input
    # character length; this does not truncate or alter token IDs.
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and len(text) > tokenizer_limit:
        tokenizer.model_max_length = len(text)
    encoded = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
    token_ids = list(encoded["input_ids"])
    bos = getattr(tokenizer, "bos_token_id", None)
    eos = getattr(tokenizer, "eos_token_id", None)
    if bos is not None:
        token_ids.insert(0, int(bos))
    if eos is not None:
        token_ids.append(int(eos))
    chunks = len(token_ids) // context_length
    if chunks == 0:
        raise ValueError("tokenized corpus is shorter than one full context window")
    usable = chunks * context_length
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    digest = hashlib.sha256()
    for token in token_ids:
        digest.update(struct.pack("<q", int(token)))
    return TokenizedCorpus(
        token_ids=tuple(token_ids), text_sha256=text_hash,
        token_ids_sha256=digest.hexdigest(), token_count=len(token_ids),
        chunk_count=chunks, predicted_token_count=chunks * (context_length - 1),
        dropped_tail_tokens=len(token_ids) - usable,
        bos_token_id=bos, eos_token_id=eos,
    )


@torch.inference_mode()
def evaluate_perplexity(
    model,
    token_ids: Sequence[int],
    *,
    context_length: int,
    device: torch.device | str,
    logit_chunk_size: int = 32,
    cache_factory: Callable[[], object] | None = None,
    batch_size: int = 1,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    """Compute exact aggregated next-token NLL over complete non-overlap chunks.

    Logits are converted to float32 in small sequence slices to avoid keeping a
    second full float32 vocabulary tensor on the GPU.
    """
    if context_length < 2 or logit_chunk_size < 1 or batch_size < 1:
        raise ValueError("context length, logit chunk size and batch size must be positive")
    device = torch.device(device)
    chunks = len(token_ids) // context_length
    if chunks < 1:
        raise ValueError("token_ids must contain at least one full context window")
    token_tensor = torch.tensor(token_ids[:chunks * context_length], dtype=torch.long)
    token_tensor = token_tensor.reshape(chunks, context_length)
    model.eval()
    total_nll = torch.zeros((), dtype=torch.float64, device=device)
    predicted = 0
    for batch_start in range(0, chunks, batch_size):
        input_ids = token_tensor[batch_start:batch_start + batch_size].to(device)
        if cache_factory is None:
            output = model(input_ids=input_ids, use_cache=False)
            cache = None
        else:
            cache = cache_factory()
            output = model(
                input_ids=input_ids, use_cache=True, past_key_values=cache
            )
        logits = output.logits
        chunk_nll = torch.zeros((), dtype=torch.float64, device=device)
        for start in range(0, context_length - 1, logit_chunk_size):
            end = min(start + logit_chunk_size, context_length - 1)
            logits_slice = logits[:, start:end, :].float().reshape(-1, logits.shape[-1])
            labels = input_ids[:, start + 1:end + 1].reshape(-1)
            loss = F.cross_entropy(logits_slice, labels, reduction="sum")
            chunk_nll += loss.to(torch.float64)
            predicted += labels.numel()
        # Synchronize once per context batch rather than once per logit slice.
        total_nll += chunk_nll
        del output, logits, input_ids, cache
        if progress_callback is not None:
            progress_callback(min(batch_start + batch_size, chunks), chunks)
    total_nll_value = float(total_nll.item())
    mean_nll = total_nll_value / predicted
    ppl = math.exp(mean_nll)
    return {"total_nll": total_nll_value, "mean_nll_per_predicted_token": mean_nll,
            "perplexity": ppl, "predicted_token_count": predicted,
            "chunk_count": chunks, "context_length": context_length}
