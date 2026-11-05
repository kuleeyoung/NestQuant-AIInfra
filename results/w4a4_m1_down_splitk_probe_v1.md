# M1 down-projection split-K probe

## Design

The production M1 packed-word GEMV assigns one four-lane subgroup to one output
channel and serially traverses all E8 blocks in K. The probe adds a grid-y split
dimension. Each split processes a deterministic contiguous K slice and writes
an FP32 partial; the existing fixed-order reduction kernel combines partials.

This is an isolated probe only. Production dispatch is unchanged.

## Real packed down-projection result

Protocol: `M=1,K=8192,N=2048`, ten warmups, 100 iterations, seven rotated-order
groups, same Llama-3.2-1B NQRT4 checkpoint and packed A4 activation.

| path | median | speedup | max abs error | cosine | tolerance |
|---|---:|---:|---:|---:|---:|
| baseline packed-word | 985.46 us | 1.00x | 0 | 1.0 | pass |
| split-2 | 537.59 us | 1.83x | 0.000122 | 1.0 | pass |
| split-4 | 419.24 us | 2.35x | 0 | 1.0 | pass |
| split-8 | **375.08 us** | **2.63x** | 0 | 1.0 | pass |

Tolerance is `atol=0.05,rtol=0.01`. The probe also passes synthetic correctness
coverage for split-2/4/8 and rejects unsupported split values.

## Promotion gate

The result justifies the next stage but is insufficient for production
promotion by itself. Required gates are:

1. normal and page-boundary full-model CUDA Graph replay;
2. active-logit and KV-cache equivalence;
3. same-revision split-K on/off online A/B;
4. partial-workspace memory accounting;
5. fallback restricted to the verified `M1,K8192,N2048` shape.

Raw data: `results/raw/w4a4_m1_down_splitk_probe_v1.json`.
