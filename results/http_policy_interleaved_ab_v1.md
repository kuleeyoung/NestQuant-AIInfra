# Interleaved fixed versus ragged execution-policy A/B

## Question

Does the capacity-4 M1/M2/M4 Graph pool plus compact prefill improve the
service when compared against fixed-M4 execution in the same code revision?

## Controlled policies

- `fixed`: capture only M4; prefill and decode both execute four model rows.
- `full`: capture M1/M2/M4; compact decode and prefill rows while preserving
  the same four logical NQKV slots.

Both policies use the same W4A4 kernels, NQKV4 format, scheduler, admission
policy, model checkpoint, prompts, output length, client concurrency, and
offered load. Runs were interleaved to reduce ordering bias:

```text
fixed-1, full-1, fixed-2, full-2, fixed-3, full-3
```

Each fresh process received two warmup requests followed by 24 measured
requests at four offered requests/s. All 144 measured requests completed; none
failed or received overload rejection.

## Per-run tail latency

| pair | fixed TPOT P95 | full TPOT P95 | fixed TTFT P95 | full TTFT P95 |
|---:|---:|---:|---:|---:|
| 1 | 282.96 ms | 68.50 ms | 788.04 ms | 482.39 ms |
| 2 | 249.73 ms | 74.82 ms | 672.67 ms | 504.13 ms |
| 3 | 270.57 ms | 69.76 ms | 734.41 ms | 523.28 ms |

Every pair favors `full`; the earlier isolated 187.79 ms compact-prefill TPOT
outlier did not recur in this interleaved experiment.

## Three-run means

| metric | fixed M4 | full ragged | relative change |
|---|---:|---:|---:|
| request throughput | 3.670 req/s | 3.851 req/s | +4.9% |
| TTFT P95 | 731.70 ms | 503.26 ms | -31.2% |
| TPOT P95 | 267.75 ms | 71.02 ms | -73.5% |
| E2E P95 | 1255.00 ms | 644.17 ms | -48.7% |
| command-queue mean | 110.17 ms | 65.42 ms | -40.6% |
| scheduler-wait mean | 63.47 ms | 85.25 ms | +34.3% |
| prefill-compute mean | 310.95 ms | 190.89 ms | -38.6% |
| post-prefill mean | 63.30 ms | 0.05 ms | -99.9% |

The scheduler-wait increase is real but smaller than the reductions in command
queue, prefill, and post-prefill delay. Percentile columns are compared across
runs but not added together, because different requests can define each tail.

## Work selection

The fixed runs replayed M4 27, 29, and 29 times. The full runs replayed only M1
or M2: M1 counts were 40, 36, 36 and M2 counts were 6, 8, 8. No full-policy run
needed M4 under this workload. Compact-prefill call counts were 23, 22, and 22.

This is direct path evidence that the full policy avoided fixed-width work; it
does not imply M4 can be removed for higher-concurrency workloads.

## Resource trade-off

| benchmark telemetry | fixed M4 | full ragged | difference |
|---|---:|---:|---:|
| mean GPU utilization | 86.9% | 73.9% | -13.0 points |
| mean GPU power | 110.5 W | 104.4 W | -6.1 W |
| peak observed VRAM | 3170 MiB | 3425 MiB | +255 MiB |

Higher fixed-policy utilization did not produce higher throughput; it reflected
more padded work. The full policy pays approximately 255 MiB in this process for
additional captured Graphs and shape-specific workspaces. This is a deliberate
memory-for-latency trade-off, not a free optimization.

## Claim boundary

This is a controlled, interleaved, single-host experiment on one GPU and one
finite workload. It supports a causal engineering claim for this service
configuration, not a production SLA, universal workload result, or claim that
all 255 MiB is irreducible Graph-pool storage.

Raw artifacts are under `results/raw/http_policy_ab_v1/`.
