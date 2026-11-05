# M1 down-projection split-K variance analysis (10 pairs)

This report diagnoses 10 4 req/s paired runs. A positive delta means the split-K
candidate is larger than baseline; for latency, positive is a regression.

## Aggregate result

| Metric | split-K off mean | split-K on mean | mean change | paired median delta | improving pairs |
|---|---:|---:|---:|---:|---:|
| request throughput (req/s) | 3.833 | 3.862 | +0.77% | +0.026 | 8/10 |
| TTFT P95 (ms) | 496.598 | 434.627 | -12.48% | -75.995 ms | 9/10 |
| TPOT P95 (ms) | 69.693 | 65.858 | -5.50% | -3.649 ms | 7/10 |
| E2E P95 (ms) | 642.786 | 545.145 | -15.19% | -124.319 ms | 9/10 |

Candidate-second odd pairs: TPOT mean delta -5.740 ms. Candidate-first even pairs: TPOT mean delta -1.928 ms.

## Pair-aligned evidence

| Pair | TPOT P95 delta ms | TTFT P95 delta ms | M1 replay delta | M1 share delta | command queue delta ms | prefill delta ms | GPU util delta pp | power delta W | SM clock delta MHz |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | -4.527 | -93.052 | +12 | +0.1418 | -0.682 | -25.390 | +1.73 | +4.61 | -5.2 |
| 2 | -6.109 | -95.272 | +12 | +0.1418 | +3.743 | -26.719 | +1.00 | +3.24 | +0.6 |
| 3 | +3.158 | +1.697 | +8 | +0.0985 | +6.472 | -9.002 | -0.95 | +1.96 | +0.0 |
| 4 | +2.181 | -20.034 | +4 | +0.0471 | -2.692 | -3.860 | -2.14 | +2.05 | +0.0 |
| 5 | -3.778 | -74.106 | +8 | +0.0904 | -0.552 | -9.390 | -2.32 | +2.73 | +0.0 |
| 6 | -2.964 | -92.691 | +8 | +0.0904 | -1.618 | -11.213 | -1.91 | +2.36 | +0.0 |
| 7 | -5.469 | -77.885 | +8 | +0.0904 | -8.317 | -10.973 | +0.55 | +2.76 | +0.0 |
| 8 | -3.519 | -59.184 | +4 | +0.0433 | -12.611 | -1.348 | -0.82 | +5.96 | +0.0 |
| 9 | -18.085 | -93.274 | +8 | +0.0833 | -16.375 | -3.018 | -1.64 | +1.92 | +0.0 |
| 10 | +0.772 | -15.915 | +0 | +0.0000 | +2.390 | +12.288 | -6.51 | -1.25 | +0.0 |

## Descriptive correlations with TPOT P95 delta

Pearson r ranges from -1 to +1. Because there are only 10 pairs, these values are
used to rank follow-up hypotheses, not to prove causality.

| Candidate signal delta | Pearson r |
|---|---:|
| `waiting_queue_mean_ms` | +0.786 |
| `command_queue_mean_ms` | +0.702 |
| `scheduled_lag_mean_ms` | +0.440 |
| `graph_m1_replays` | -0.376 |
| `graph_m1_share` | -0.295 |
| `temperature_c_mean` | -0.280 |
| `gpu_utilization_percent_mean` | -0.265 |
| `power_w_mean` | -0.180 |
| `prefill_compute_mean_ms` | +0.163 |
| `sm_clock_mhz_mean` | +0.024 |

## Evidence boundary

This is a descriptive diagnosis over 10 fresh-process pairs. The sample is too small
for a statistical or causal claim. Any production-default change still requires a larger
paired run or a deterministic replay that controls the scheduling trace.

Machine-readable evidence: `results/raw/http_m1_splitk_policy_ab_v2_analysis.json`
