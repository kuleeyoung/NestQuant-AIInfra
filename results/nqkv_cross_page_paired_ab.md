# NQKV one-boundary segmented prefill same-process paired A/B

Pairs: 20; paired bootstrap resamples: 10000.

| metric | transactional median | segmented median | paired median improvement | bootstrap 95% CI | improved pairs |
|---|---:|---:|---:|---:|---:|
| token throughput | 17.40 | 21.11 | 22.25% | [20.12%, 27.03%] | 20/20 |
| TTFT P95 | 460.01 | 367.21 | 20.24% | [19.40%, 24.45%] | 20/20 |
| TPOT P95 | 134.03 | 109.68 | 17.37% | [14.45%, 19.25%] | 20/20 |
| E2E P95 | 504.80 | 414.63 | 19.31% | [17.97%, 22.64%] | 20/20 |

## Protocol boundary

Both arms run in one process with one loaded model and one CUDA context. Each repetition shares identical requests/tokens; arm order alternates. The experiment excludes tokenizer/network I/O and is not a production SLA.

## Implementation boundary

The segmented path accepts prefix-contiguous ragged rows whose append crosses at
most one NQKV page boundary and leaves a strictly partial second hot page. For each
layer it writes the first per-row segment, packs and publishes boundary rows, then
writes the remainder using an independent source offset per row. Page reservations
are committed only after the complete model forward succeeds.

Direct Hugging Face Cache calls retain the transactional fallback because they do
not provide a post-forward commit boundary. The optimized path is explicitly owned
by `BatchedNQKVLlamaRunner`. If a cross-page model forward fails after mutating KV,
the runner cancels reservations and releases every affected request rather than
continuing with partially updated cache state.

This is not an arbitrary multi-page writer or a GPU allocator. Appends reaching two
or more page boundaries continue to use the transactional fallback.
