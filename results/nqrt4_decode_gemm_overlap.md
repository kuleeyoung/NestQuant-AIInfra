# NQRT v4 decode/GEMM overlap feasibility

## Question

Can a second CUDA stream hide the next weight's tiled Triton decode behind the
current layer's BF16 GEMM? This isolated benchmark excludes record I/O and
parsing, keeps the current BF16 weight resident, and compares sequential versus
two-stream execution on an RTX 5070.

The benchmark covers three execution-order pairs from Llama layer 0:
`q_proj -> k_proj`, `gate_proj -> up_proj`, and `up_proj -> down_proj`.
Each result is warmed up and repeated 30 times for sequence lengths 1 and 8,
and 20 times for sequence length 2048.

## Results

| Sequence length | Pair | Current GEMM mean | Next decode mean | Sequential mean | Two-stream mean | Ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | q -> k | 0.063 ms | 0.234 ms | 0.275 ms | 0.306 ms | 0.90x |
| 1 | gate -> up | 0.052 ms | 1.885 ms | 2.183 ms | 1.938 ms | 1.13x |
| 1 | up -> down | 0.048 ms | 1.515 ms | 1.694 ms | 1.697 ms | 1.00x |
| 8 | q -> k | 0.068 ms | 0.336 ms | 0.442 ms | 0.339 ms | 1.30x |
| 8 | gate -> up | 0.087 ms | 1.840 ms | 1.525 ms | 1.874 ms | 0.81x |
| 8 | up -> down | 0.085 ms | 1.687 ms | 1.891 ms | 1.936 ms | 0.98x |
| 2048 | q -> k | 0.327 ms | 0.315 ms | 0.532 ms | 0.442 ms | 1.21x |
| 2048 | gate -> up | 1.106 ms | 1.696 ms | 2.438 ms | 1.802 ms | 1.35x |
| 2048 | up -> down | 1.101 ms | 1.948 ms | 2.182 ms | 1.977 ms | 1.10x |

`Ratio` is sequential mean divided by two-stream mean; values below 1 are
regressions. Short-sequence GEMMs are tens of microseconds while MLP weight
decode is roughly 1.5-1.9 ms, so there is too little useful computation to
hide most decode work. Variance also makes isolated short operations noisy;
the mixed sequence-1/8 results do not justify a general speedup claim.

At sequence length 2048, GEMM grows to 0.3-1.1 ms and all three pairs benefit.
The overlap path also keeps both the current operation's output and the next
materialized BF16 weight live. Representative peak increments are about
11.0 MiB for q-to-k and 76.1 MiB for the MLP pairs in this isolated setup.

## Decision

A full double-stream scheduler is justified only if the runtime target is
long-context prefill. It is not the next general solution for batch-1 token
generation, where materializing every BF16 weight remains dominant and cannot
be hidden behind a much shorter GEMM. For a general low-bit inference claim,
the next substantive step is decode/GEMM fusion or a packed-weight GEMM rather
than additional Python scheduling.

Machine-readable results are stored under `results/raw/` as
`llama32_1b_nqrt4_decode_gemm_overlap_s1.json`, `_s8.json`, and `_s2048.json`.
This benchmark is an isolated concurrency ceiling, not a full-model latency or
serving-throughput result.
