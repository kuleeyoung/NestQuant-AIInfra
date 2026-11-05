# M1 split-K numerical-stability baseline

## Isolated performance

Fresh RTX 5070 measurements for `M=1,K=8192,N=2048`:

| path | median | speedup |
|---|---:|---:|
| unsplit | 979.44 us | 1.00x |
| split-2 | 536.10 us | 1.83x |
| split-4 | 412.19 us | 2.38x |
| split-8 | 377.08 us | 2.60x |

## Raw-FP32 error distribution

Thirty-two deterministic BF16 activation samples were quantized to packed A4.
The new raw probes compared CUDA FP32 accumulators before output-dtype rounding.

| split | FP32 max-error mean | FP32 max-error worst | relative-L2 mean | BF16 samples with any differing element |
|---:|---:|---:|---:|---:|
| 2 | 2.99e-6 | 4.05e-6 | 3.29e-7 | 13/32 |
| 4 | 3.02e-6 | 4.77e-6 | 3.57e-7 | 12/32 |
| 8 | 3.03e-6 | 4.29e-6 | 3.60e-7 | 11/32 |

For every split, the median number of different BF16 outputs among 2048 channels
was zero, the P95 count was two and the worst BF16 absolute difference was
0.00390625.

## Interpretation

Reducing split-8 to split-2 did not materially reduce the isolated raw error,
while speedup fell from 2.60x to 1.83x. Split count alone is therefore not a
promising numerical fix. The direct layer error is tiny compared with the
runner-level final-logit error, indicating amplification through later residual,
normalization, attention/KV4 and argmax-margin interactions.

## Partial-versus-reduction attribution

A CUDA diagnostic subsequently exposed the raw `[split_k,N]` FP32 partials.
The same partials were recombined with sequential FP32, pairwise FP32, Kahan
FP32 and FP64 summation.

- split-2: every method was bitwise equal to the CUDA reducer;
- split-4/8: alternative reducers differed from CUDA by at most about 9.54e-7;
- all methods retained roughly 2.9e-6 mean maximum error versus unsplit;
- FP64 did not materially improve error versus baseline.

Therefore the final 2/4/8-value reduction is not the dominant source. Error is
already introduced by independent per-partition FP32 accumulation and the loss
of the unsplit kernel's single continuous rounding state. Replacing only the
final reducer with FP64, pairwise or Kahan summation is rejected as ineffective.
The next Route-A experiment must change accumulation inside each partition.
No production dispatch changed.

## Per-partition compensated accumulation

A diagnostic Kahan kernel maintained a compensation term inside every split.
It intentionally replaced each FP32 FMA with a separate multiply and
compensated add, and remained outside production dispatch.

| path | median | speedup vs unsplit | mean BF16 differences / 2048 outputs |
|---|---:|---:|---:|
| unsplit | 976.53 us | 1.00x | 0 |
| split-2 | 534.01 us | 1.83x | 0.50 |
| split-2 compensated | 551.71 us | 1.77x | 0.59 |
| split-4 | 412.57 us | 2.37x | 0.50 |
| split-4 compensated | 418.70 us | 2.33x | 0.56 |
| split-8 | 373.75 us | 2.61x | 0.47 |
| split-8 compensated | 379.92 us | 2.57x | 0.66 |

Compensation did not improve agreement with the unsplit FP32/BF16 result and
cost 1.6% to 3.3% latency relative to the corresponding ordinary split. It is
therefore rejected as a production candidate.

## FP64 decoded-value oracle

The stronger test decoded both packed A4 activations and packed W4 weights to
their actual BF16 runtime values, then computed the dot product in FP64. Over
32 deterministic activations, mean absolute error versus this oracle was:

| path | split-2 context | split-4 context | split-8 context |
|---|---:|---:|---:|
| unsplit | 2.414e-6 | 2.414e-6 | 2.414e-6 |
| ordinary split | 2.278e-6 | 2.196e-6 | 2.155e-6 |
| compensated split | 2.083e-6 | 2.093e-6 | 2.105e-6 |

Ordinary split-8 reduced mean absolute accumulation error versus unsplit by
about 10.7%. Consequently, non-bitwise equality with the old unsplit path is
not evidence that split-K is less numerically accurate. The full-model token
flip remains real, but it is an amplification of a legal floating-point
rounding-path change at a small decision margin, rather than a demonstrated
kernel arithmetic defect. Compensated split is still rejected: its small
oracle improvement did not reduce BF16 divergence and it was slower.

Raw artifacts:

- `results/raw/m1_splitk_numeric_baseline_v1.json`
- `results/raw/m1_splitk_numeric_stability_v1.json`
- `results/raw/m1_splitk_reduction_stability_v1.json`
- `results/raw/m1_splitk_compensated_stability_v1.json`
- `results/raw/m1_splitk_compensated_benchmark_v1.json`
- `results/raw/m1_splitk_fp64_oracle_v1.json`
