# Packed W4A4 dynamic-tile optimization

## Scope

This experiment compares BF16, packed W4 (`cuda_mma_j2`), and tiled packed
W4A4 on the same Llama-3.2-1B revision, NQRT v4 checkpoint, token tensors, GPU,
and benchmark harness. It measures batch-one prefill and steady decode; it is
not a serving-throughput result.

## Bottleneck decomposition

Representative q/k/gate/down projections were measured independently at 1,
17, and 128 tokens. A4 encoding costs roughly 0.27--0.70 ms per unique input.
After accounting for q/k/v and gate/up input reuse, the W4xA4 matrix kernel
accounts for approximately 88%, 92%, and 97% of projection time at 1, 17, and
128 tokens respectively. Kernel scheduling, rather than activation encoding,
is therefore the first optimization target.

## Bounded launch search

The search holds `BLOCK_K=32` fixed so the feature-axis accumulation order is
unchanged. It sweeps `BLOCK_M={16,32,64}`, `BLOCK_N={16,32,64}`, and
`num_warps={2,4,8}` for q, gate, and down projection shapes. Every selected
configuration has zero max/mean absolute error against the former default.

| Tokens | q_proj best | gate_proj best | down_proj best |
|---:|---|---|---|
| 1 | M16 N16 W8, 550.17 us | M16 N32 W8, 1879.33 us | M16 N16 W8, 2249.61 us |
| 17 | M32 N32 W8, 770.27 us | M32 N32 W8, 2143.56 us | M32 N16 W8, 3183.31 us |
| 128 | M64 N32 W8, 1708.60 us | M64 N32 W8, 5976.78 us | M64 N32 W8, 7789.05 us |

The runtime selector scales M with token count, uses N32 for wide-output or
long-prefill shapes, keeps N16 for short wide-input projections, and uses eight
warps. Raw sweep records are `results/raw/w4a4_sweep_{q,gate,down}.json`.

## End-to-end result

Five timed repeats follow warmup. Values below are medians for prefill and
means across six measured decode transitions for TPOT.

| Sequence | Former W4A4 prefill | Dynamic-tile W4A4 | Speedup |
|---:|---:|---:|---:|
| 1 | 208.72 ms | 191.52 ms | 1.09x |
| 17 | 332.14 ms | 235.29 ms | 1.41x |
| 128 | 1025.41 ms | 427.29 ms | **2.40x** |

Steady decode TPOT improves from 214.37 ms to 177.03 ms for the one-token
case (1.21x). The packed model/runtime allocation remains about 1.00 GiB versus
2.33 GiB for BF16, a 57.0% reduction in this harness.

The optimized 128-token W4A4 prefill is still about 35.5x slower than BF16 and
6.7x slower than the packed-W4 path. This is a real optimization, not evidence
of low-bit speedup over BF16. The remaining dominant cost is reconstructing E8
activation/weight tiles and stateless dither inside every matrix tile.

Raw paired evidence:

- before: `results/raw/llama32_1b_bf16_w4_w4a4_paired_benchmark.json`
- after: `results/raw/llama32_1b_bf16_w4_w4a4_autotiled_benchmark.json`

## Validation

- selected launch shapes match former default outputs exactly in the bounded
  sweep;
- full-model BF16-relative logit error is unchanged before and after;
- 296 project tests and 37 parameterized subtests pass on WSL/CUDA.

The earlier full WikiText-2 PPL 10.683118 remains applicable because the
selected schedules retain `BLOCK_K=32` and the measured outputs are identical.

## Weight-stationary boundary probes

Two follow-up probes tested whether the current Triton kernel could obtain
another large gain merely by increasing tile sizes:

- `BLOCK_M=128` reconstructs each weight tile for more tokens, but q/gate/down
  regress from 1.71/5.98/7.79 ms at M64 to 2.54/7.27/9.64 ms. The larger
  accumulator raises register pressure and reduces occupancy.
- Holding M64/N32/W8, increasing `BLOCK_K` from 32 to 64 or 128 also regresses
  all three shapes. For q_proj the times are 1.75/2.34/2.82 ms; for gate_proj
  5.99/8.86/10.32 ms; for down_proj 7.42/10.50/11.53 ms.

All variants remain numerically identical in these probes. The negative result
shows that the current Triton parameter space is bounded by register pressure,
not by an insufficiently large tile. The next pure-packed optimization requires
an architectural kernel change: staged packed-weight decode, shared-memory
reuse, and Tensor-Core compute in a custom pipeline rather than another launch
parameter sweep.

