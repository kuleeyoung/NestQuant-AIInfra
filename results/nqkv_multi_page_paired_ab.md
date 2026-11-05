# NQKV four-page segmented prefill same-process paired A/B

Pairs: 20; paired bootstrap resamples: 10000.

| metric | transactional median | segmented median | paired median improvement | bootstrap 95% CI | improved pairs |
|---|---:|---:|---:|---:|---:|
| token throughput | 13.46 | 13.49 | -0.84% | [-1.72%, 4.95%] | 9/20 |
| TTFT P95 | 629.65 | 642.02 | -0.84% | [-1.95%, 4.91%] | 9/20 |
| TPOT P95 | 176.49 | 178.86 | -1.02% | [-3.30%, 1.49%] | 8/20 |
| E2E P95 | 680.50 | 693.66 | -0.89% | [-1.84%, 4.82%] | 9/20 |

## Protocol boundary

Both arms run in one process with one loaded model and one CUDA context. Each repetition shares identical requests/tokens; arm order alternates. The experiment excludes tokenizer/network I/O and is not a production SLA.

## Decision

The general multi-page segmented path is correctness-complete but remains opt-in.
Its prompt-64 paired median changes are -0.84% throughput, -0.84% TTFT P95,
-1.02% TPOT P95 and -0.89% E2E P95; only 8-9 of 20 pairs improve and every
bootstrap interval crosses zero. It therefore fails the predeclared >=5% promotion
gate. The transactional writer remains the default for appends producing two or
more complete pages.

Removing one GPU-to-CPU `flush_mask.any().item()` synchronization per layer/round
improved the earlier throughput result from -1.85% to -0.84%, but did not make the
path competitive. The next optimization target is batched multi-page packing and
fewer writer/pack launches, not additional planner complexity.

## Scope

The opt-in implementation supports prefix-contiguous ragged rows, an arbitrary
finite number of complete pages, mixed per-row segment counts, atomic cross-layer
reservation, post-forward commit and fail-closed release after an exception. Direct
Hugging Face Cache calls and the default runner policy retain the transactional
fallback.
