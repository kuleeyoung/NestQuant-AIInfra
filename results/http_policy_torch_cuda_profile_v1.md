# Fixed versus ragged PyTorch CUDA profile

## Protocol

The online server gained a controlled PyTorch profiler window driven by the
same local start/stop file as the HTTP benchmark. Two fresh capacity-4 processes
were captured with four measured requests, four client workers, four offered
requests/s, a 29-token prompt, and three output tokens after two warmups:

- `fixed`: fixed-capacity prefill and M4 decode;
- `full`: compact prefill plus M1/M2/M4 decode Graph pool.

Both completed 4/4 requests. The smoke gate required at least one CUDA event
with non-zero `self_device_time_total_us`; the full trace contained 82 such
event groups. Unlike the WSL Nsight reports, these traces contain device-side
kernel time.

## Device-time families

Family totals use self device time, so parent/child profiler events are not
intentionally double-counted. Packed W4A4 includes the M128 staged, small-M, and
M1 packed-word kernels.

| CUDA family | fixed | full | change | full share |
|---|---:|---:|---:|---:|
| packed W4A4 Linear | 1050.40 ms | 722.23 ms | -31.2% | 83.6% |
| NQKV attention | 121.37 ms | 45.03 ms | -62.9% | 5.2% |
| NQKV encode | 31.72 ms | 14.48 ms | -54.4% | 1.7% |
| copy / cat | 23.89 ms | 25.78 ms | +7.9% | 3.0% |
| other CUDA kernels | 50.00 ms | 56.85 ms | +13.7% | 6.6% |
| **total self CUDA time** | **1277.38 ms** | **864.37 ms** | **-32.3%** | **100%** |

The earlier coarse grouping initially placed the fixed small-M W4A4 kernel in
`other`; the table above corrects that classification.

## Principal kernels

| policy | kernel | calls | self device time |
|---|---|---:|---:|
| fixed | W4A4 M128 staged | 336 | 667.72 ms |
| fixed | W4A4 small-M | 672 | 382.69 ms |
| full | W4A4 M128 staged | 448 | 363.48 ms |
| full | W4A4 M1 packed-word | 896 | 358.74 ms |
| fixed | NQKV page attention | 96 | 120.49 ms |
| full | NQKV page attention | 128 | 43.84 ms |
| fixed | NQKV encode | 960 | 31.72 ms |
| full | NQKV encode | 896 | 14.48 ms |

Call counts are not directly comparable without shape context: a smaller Graph
can replay more, shorter kernels. Device-time totals are the relevant quantity.

## Decision

The primary remaining device bottleneck is packed W4A4 Linear, which consumes
83.6% of full-policy self CUDA time. NQKV attention is secondary at 5.2%.
Compact input/output copying has measurable overhead but is only about 3.0%, so
writing a custom copy/scatter kernel is not the next priority.

The next optimization should therefore profile the packed W4A4 M1 and staged
paths by projection shape, then improve the dominant small-M shape. NQKV fusion
or copy optimization should be revisited only if that ranking changes under a
longer-context or higher-concurrency workload.

## Limitations

- One short fresh process per policy; use this for bottleneck ranking, not a
  production latency claim.
- PyTorch profiler perturbs execution, so the unprofiled three-pair interleaved
  A/B remains the primary service-performance evidence.
- The profiler emitted a cross-thread external-init warning because control and
  model execution are owned by different threads. CUDA device events and
  non-zero device times were nevertheless recorded; CPU-event completeness is
  not assumed.

Artifacts:

- `results/raw/http_policy_torch_profiler_fixed_v1/trial_01.torch-summary.json`
- `results/raw/http_policy_torch_profiler_fixed_v1/trial_01.torch-trace.json`
- `results/raw/http_policy_torch_profiler_smoke_v1/trial_01.torch-summary.json`
- `results/raw/http_policy_torch_profiler_smoke_v1/trial_01.torch-trace.json`
