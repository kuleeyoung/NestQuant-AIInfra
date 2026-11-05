# NQKV full-page prefill fusion same-process paired A/B

Pairs: 20; paired bootstrap resamples: 10000.

| metric | transactional median | fused median | paired median improvement | bootstrap 95% CI | improved pairs |
|---|---:|---:|---:|---:|---:|
| token throughput | 18.61 | 21.60 | 16.22% | [14.41%, 20.45%] | 20/20 |
| TTFT P95 | 429.07 | 342.87 | 16.16% | [14.72%, 18.93%] | 20/20 |
| TPOT P95 | 127.47 | 106.35 | 13.02% | [11.47%, 15.33%] | 20/20 |
| E2E P95 | 469.13 | 389.37 | 15.02% | [13.42%, 18.21%] | 20/20 |

## Protocol boundary

Both arms run in one process with one loaded model and one CUDA context. Each repetition shares identical requests/tokens; arm order alternates. The experiment excludes tokenizer/network I/O and is not a production SLA.

## Implementation boundary

The optimized arm accepts only prefix-contiguous ragged rows whose append remains
inside one NQKV page and where at least one row lands exactly on the page boundary.
The host reserves one physical page per boundary row and layer before the forward;
the GPU writer fills the hot page, the existing graph-safe pack kernel publishes the
packed page, and the host commits all reserved handles after a successful forward.
The failure path can retract an already published page and restore the full hot page.

This is not a generic GPU page allocator. Multi-page or cross-page prefill continues
to use the transactional fallback. The raw run records explicitly show zero
`nqkv_ragged_full_page_layer_packs` in the baseline and positive dispatch counts in
every fused run.
