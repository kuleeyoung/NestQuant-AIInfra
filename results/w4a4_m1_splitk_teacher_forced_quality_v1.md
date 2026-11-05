# M1 split-K teacher-forced quality screen

## Formal 2,048-context gate result (2026-11-05)

The scheduled production-runner evaluation completed successfully:

| metric | result |
|---|---:|
| contexts | 2,048 |
| paired decode decisions | 65,536 |
| baseline sampled PPL | 186.875878 |
| split-K sampled PPL | 186.701716 |
| relative PPL delta | -0.0932% |
| context-bootstrap 95% CI | [-0.5664%, +0.3658%] |
| predeclared non-inferiority margin | +1.0000% |
| argmax agreement | 77.7740% |
| mean KL | 0.138359 |
| deterministic replay max logit error | 0 |
| gate | **passed** |

The CI upper bound is below the predeclared +1% margin, so the sampled
teacher-forced M1 path passes the non-inferiority contract. The interval covers
zero, so this is not evidence that split-K improves model quality. Argmax is not
identical, so this result also does not claim identical autoregressive token
sequences. Combined with the existing 10-pair service A/B, the online service
now enables this exact split-K shape by default and retains
`--disable-m1-down-splitk` as the rollback switch.

Formal artifacts:

- `results/raw/m1_splitk_teacher_forced_2048x32_1pct_v1.json`
- `results/raw/m1_splitk_teacher_forced_2048x32_1pct_analysis_v1.json`
- `results/raw/m1_splitk_teacher_forced_2048x32_1pct_gate_v1.json`

## Why full prefill PPL is not the gate

The existing full WikiText-2 packed W4A4 evaluation uses `use_cache=False`
with 2048-token prefill. It does not dispatch the `M=1,K=8192,N=2048`
split-K kernel, so rerunning that protocol would not test this candidate.

This screen instead uses teacher-forced decode: baseline and split-K receive the
same prompt and the same prescribed continuation tokens at every step. This
keeps inputs paired after an argmax difference and directly measures the M1
path in the production NQKV CUDA Graph runner.

## Protocol

- model/checkpoint: Llama-3.2-1B NQRT4 packed W4A4KV4 checkpoint;
- dataset: cached WikiText-2 test, token SHA-256
  `54927bf19e6e64b4008eeb93370c370fb25db18cf68eac8015cfe094c464d567`;
- deterministic sample-plan SHA-256
  `8f6e63f72c7f6001487bc2dac1d2ddcb4dda13bc60b256d9b7c6877c4594c2a2`;
- 32 contexts, 32 prompt tokens and 32 teacher-forced target tokens each;
- 1,024 paired M1 decode decisions;
- AB/BA alternating execution order;
- early ABA/BAB repeatability checks;
- bootstrap unit: complete context, not correlated individual tokens;
- 20,000 deterministic bootstrap repetitions.

## Results

| metric | result |
|---|---:|
| baseline sampled PPL | 234.2724 |
| split-K sampled PPL | 233.4433 |
| relative PPL delta | -0.3539% |
| context-bootstrap 95% CI | [-5.4196%, +4.5418%] |
| better / worse contexts | 16 / 16 |
| argmax agreement | 77.5391% |
| mean KL | 0.14705 |
| maximum logit error | 9.5 |
| repeated-strategy maximum logit error | 0.0 |

The point estimate shows no average quality regression, and exact replay proves
the protocol is deterministic. The confidence interval is nevertheless too
wide for a strict non-inferiority claim. Argmax disagreement is diagnostic, not
an automatic failure: teacher forcing preserves identical subsequent inputs,
and the target-token NLL is the quality metric.

## Current decision boundary

The candidate remains explicit opt-in. Promoting it requires a predeclared
non-inferiority margin and enough contexts for the upper confidence bound to
fall below that margin. The present result neither proves regression nor proves
a tight quality bound. Full prefill PPL and existing downstream evaluations
remain valid model-quality evidence, but they do not exercise this M1 kernel.

Raw artifacts:

- `results/raw/m1_splitk_teacher_forced_32x32_v1.json`
- `results/raw/m1_splitk_teacher_forced_32x32_analysis_v1.json`
