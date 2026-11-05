# M1 split-K mixed-length engine A/B

4 fresh-process pairs replayed identical deterministic requests with mixed prompt and
output lengths. Positive throughput delta is better; negative latency delta is better.

## Pair results

| Pair | token/s off | token/s on | TPOT P95 off ms | TPOT P95 on ms | TPOT delta ms | TTFT delta ms | E2E delta ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.474 | 5.048 | 340.79 | 194.51 | -146.28 | -2586.03 | -2630.10 |
| 2 | 5.343 | 5.374 | 201.17 | 179.47 | -21.70 | +1988.14 | +1965.93 |
| 3 | 5.343 | 5.378 | 204.40 | 188.97 | -15.42 | -39.87 | -80.51 |
| 4 | 5.334 | 5.352 | 199.70 | 212.53 | +12.83 | +33.33 | +105.69 |

## Paired-median result

| Metric | median relative change | improving pairs |
|---|---:|---:|
| token throughput | +0.62% | 4/4 |
| TTFT P95 | -0.57% | 2/4 |
| TPOT P95 | -9.17% | 3/4 |
| E2E P95 | +1.90% | 2/4 |

## Step-phase attribution

| Pair | decode mean off ms | decode mean on ms | decode P95 off ms | decode P95 on ms | mixed mean off ms | mixed mean on ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 65.78 | 50.81 | 87.86 | 79.08 | 1149.96 | 228.23 |
| 2 | 55.38 | 50.60 | 62.20 | 64.20 | 209.77 | 640.43 |
| 3 | 55.52 | 46.86 | 65.08 | 60.06 | 211.83 | 209.33 |
| 4 | 61.14 | 51.24 | 69.50 | 65.33 | 241.92 | 221.78 |

Decode-only mean step latency improved in 4/4 pairs; the paired-median relative change was -15.90%.
Request-level TPOT P95 can still move in the opposite direction because changing step
duration changes which arrivals coalesce and which token intervals land in mixed steps.

All pairs completed every request and generated the same token count from the same
seeded workload. M histograms are retained per side as scheduling outcomes: faster or
slower steps can change which arrivals coalesce into one batch. NQKV boundary Graph
fallback count was zero in every run.

1/4 pairs were recorded after
identity hashes were added; those pairs have matching workload and generated-token SHA-256
between split-K off/on. Earlier pairs retain count/config checks but cannot be retroactively
promoted to token-sequence identity evidence.

The first pair had a large cross-process state outlier in both TTFT and TPOT. It is
retained in the table; paired medians are reported instead of deleting it or relying on
an outlier-sensitive arithmetic mean.

## Evidence boundary

4 paired fresh-process, deterministic synthetic-token engine runs. The paired median is robust to the observed first-pair process-state outlier, but n=4 remains a small local sample and is not a production SLA.

Machine-readable summary: `results/raw/m1_splitk_mixed_lowload_ab_v1_summary.json`
