# NQKV4 dense single-page prefill optimization

## Outcome

Profiling localized the remaining BF16-KV versus NQKV4 service gap to prefill,
not ordinary K1 decode or the host page allocator. A narrow dense-empty fast
path now bulk-copies K/V and vector-updates metadata when all fixed-capacity
slots start empty, share one prompt width, and remain strictly inside one page.
Ragged, nonempty, partial-capacity and cross-page writes retain the validated
transactional writer.

## Paired stage attribution before the change

Five fixed-seed repetitions use identical requests, packed W4A4 kernels and
GPU. Runs are paired by profile/repetition; the two backends run in separate
processes rather than alternating within one process.

| profile and stage | NQKV4 | BF16 KV | NQKV4 overhead |
|---|---:|---:|---:|
| prompt 8 prefill-only | 119.89 ms | 96.32 ms | 25.48% |
| prompt 8 decode-only | 53.35 ms | 52.32 ms | 1.83% |
| prompt 15 prefill-only | 149.00 ms | 118.17 ms | 25.97% |
| prompt 15 decode-only | 58.67 ms | 52.68 ms | 12.02% |
| prompt 15 boundary step | 56.82 ms | 52.09 ms | 8.59% |

The prompt-8 throughput deficit was 7.64%; the boundary workload deficit was
16.01%. This rejected further control-plane allocator work and selected prefill
storage as the next bottleneck.

## Dense-empty fast path result

| metric | before | after | change |
|---|---:|---:|---:|
| prompt-15 prefill-only | 149.00 ms | 132.35 ms | -11.17% |
| prompt-15 workload throughput | 22.53 token/s | 24.20 token/s | +7.41% |
| prompt-15 throughput deficit vs BF16 | 16.01% | 9.79% | -6.22 points |
| prompt-8 prefill-only | 119.89 ms | 116.09 ms | -3.17% |

Prompt-8 total throughput varied downward in the second non-alternating run,
while its measured prefill improved. Because the fast path is inactive after
prefill, this is treated as timing variance that still requires an alternating
A/B before any general service-throughput claim.

## Profiler evidence

One warmed `batch=2, prompt=15` prefill was captured before and after:

| diagnostic | before | after | change |
|---|---:|---:|---:|
| profiler range | 165.31 ms | 152.69 ms | -7.64% |
| `aten::copy_` launches | 1,159 | 999 | -160 |
| host-to-device memcpy launches | 288 | 224 | -64 |
| device-to-host memcpy launches | 39 | 8 | -31 |

The profiler event named `_nqkv_encode_kernel` also appears in BF16-KV runs:
the W4A4 activation quantizer reuses that Triton primitive. It is therefore not
counted as KV4 page-encoding evidence.

## Correctness and evidence

- Dense writer output is compared elementwise with the original writer.
- Full Llama, graph/eager, active-row, writer and manager tests pass.
- NQKV runtime regression: 65 tests plus 16 parameterized subtests passed.
- Before A/B: `results/raw/kv_stage_ab_summary.json`.
- After A/B: `results/raw/kv_stage_ab_dense_prefill_summary.json`.
- Before profiler: `results/raw/kv_prefill_nqkv4_profile.json`.
- After profiler: `results/raw/kv_prefill_nqkv4_dense_profile.json`.
- BF16 profiler: `results/raw/kv_prefill_bf16_profile.json`.

## Claim boundary

The promoted path covers full-capacity, equal-width, empty-cache, sub-page
prefill. It does not accelerate ragged admission, partial-capacity prefill,
prompts spanning a packed page, or append-to-existing-cache. The fixed-shape
results are local kernel/runtime evidence, not a network-service SLA.
