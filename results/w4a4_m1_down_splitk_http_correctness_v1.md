# M1 down-projection split-K HTTP correctness gate

## Purpose

This follow-up adds an end-to-end autoregressive correctness gate to the earlier
performance A/B. Each policy ran in a fresh process with the same deterministic
HTTP workload. SHA-256 hashes cover both the submitted token IDs and the ordered
generated-token sequences.

## Result

| item | split-K off | split-K on | conclusion |
|---|---|---|---|
| requests | 12/12 completed | 12/12 completed | pass |
| workload SHA-256 | `4124ec9a23a99d971f8688e9323eaa5d5987d08c59b187dc97976fc191292fd2` | same | identical input |
| generated sequence | `482,15704,13` for all 12 requests | `482,220,679` for all 12 requests | **not equivalent** |
| TPOT P95 | 62.73 ms | 52.27 ms | split-K improved 16.69% |
| request throughput | baseline | +0.42% | small improvement |

The matching first token but divergent later tokens show that this is an
autoregressive model-output difference, not a workload-hash or ordering error.
The follow-up runner-level diagnosis reproduced the exact sequences without HTTP
or concurrency. At the first divergent decision, both paths had identical input,
the unsplit top-1/top-2 margin was 0.25 and the maximum split-K logit error was
1.71875. The supported mechanism is therefore a combination of changed
floating-point accumulation order and the small decision margin of the NQKV4 +
CUDA Graph serving path. After argmax flips, feeding the changed token back into
the next step amplifies the difference.

## Decision

The correctness gate overrides the latency gain. The exact
`M=1,K=8192,N=2048` split-8 path is retained as an experimental probe but is
**disabled by default**. It can only be selected explicitly with
`--enable-m1-down-splitk`; ordinary service execution uses the unsplit path.

This result also narrows the evidence boundary of the earlier engine A/B: one
mixed-length pair had matching engine-level output hashes, but that observation
does not establish full HTTP autoregressive equivalence.

Raw artifact: `results/raw/http_m1_splitk_hash_ab_v1/`

Detailed diagnosis: `results/w4a4_m1_down_splitk_logit_diagnosis_v1.md`
