# M1 down-projection split-K variance analysis v1

This report diagnoses the five 4 req/s paired runs. A positive delta means the split-K
candidate is larger than baseline; for latency, positive is a regression.

## Pair-aligned evidence

| Pair | TPOT P95 delta ms | TTFT P95 delta ms | M1 replay delta | M1 share delta | command queue delta ms | prefill delta ms | GPU util delta pp | power delta W | SM clock delta MHz |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | -18.415 | -136.661 | +8 | +0.0833 | -31.289 | -4.088 | -1.59 | +2.58 | -6.8 |
| 2 | +5.749 | -6.923 | +0 | +0.0000 | -3.783 | +11.370 | -2.64 | +2.56 | -3.2 |
| 3 | -2.344 | -91.841 | +8 | +0.0904 | -18.124 | -10.811 | -2.17 | +2.61 | -2.1 |
| 4 | -21.500 | -116.542 | +20 | +0.2381 | +9.789 | -38.822 | -0.50 | +4.36 | -3.5 |
| 5 | +6.008 | +43.219 | -4 | -0.0471 | +14.474 | +23.429 | -0.09 | +1.77 | -1.3 |

## Descriptive correlations with TPOT P95 delta

Pearson r ranges from -1 to +1. Because there are only five pairs, these values are
used to rank follow-up hypotheses, not to prove causality.

| Candidate signal delta | Pearson r |
|---|---:|
| `graph_m1_replays` | -0.878 |
| `graph_m1_share` | -0.861 |
| `prefill_compute_mean_ms` | +0.824 |
| `power_w_mean` | -0.747 |
| `sm_clock_mhz_mean` | +0.697 |
| `temperature_c_mean` | -0.675 |
| `scheduled_lag_mean_ms` | +0.585 |
| `waiting_queue_mean_ms` | +0.467 |
| `command_queue_mean_ms` | +0.296 |
| `gpu_utilization_percent_mean` | -0.232 |

## Evidence boundary

This is a descriptive diagnosis over five fresh-process pairs. The sample is too small
for a statistical or causal claim. Any production-default change still requires a larger
paired run or a deterministic replay that controls the scheduling trace.

Machine-readable evidence: `results/raw/http_m1_splitk_policy_ab_v1_analysis.json`
