# Fixed versus ragged policy Nsight API trace

## Scope and limitation

One `fixed` and one `full` capacity-4 service process were profiled with Nsight
Systems using a CUDA Profiler API capture range. Both completed 24/24 requests.
The reports contain CUDA API events, including Graph launches, but their SQLite
exports contain no CUDA GPU activity/kernel table in the current WSL/Nsight
environment. Therefore this artifact is **not** device-kernel timing evidence.

The trace-latency comparison remains directional because both policies carry
the same profiler mechanism, but the unprofiled interleaved A/B is the primary
service-performance result.

## Trace-mode service result

| metric | fixed | full | change |
|---|---:|---:|---:|
| request throughput | 3.156 req/s | 3.726 req/s | +18.1% |
| TTFT P95 | 1471.79 ms | 739.67 ms | -49.7% |
| TPOT P95 | 326.87 ms | 284.49 ms | -13.0% |
| E2E P95 | 1960.37 ms | 1267.45 ms | -35.3% |

Absolute trace-mode latency must not be compared directly with unprofiled
latency because Nsight substantially perturbs this workload.

## CUDA API summary

| CUDA API | fixed calls / total | full calls / total |
|---|---:|---:|
| `cudaGraphLaunch` | 21 / 66.64 ms | 27 / 89.85 ms |
| `cudaLaunchKernel` | 103,888 / 2185.48 ms | 95,532 / 756.70 ms |
| `cudaStreamSynchronize` | 7,967 / 3137.77 ms | 14,006 / 2423.32 ms |
| `cudaMemcpyAsync` | 11,233 / 112.56 ms | 20,353 / 1158.02 ms |

The full policy launches fewer ordinary kernels and spends less host API time
inside `cudaLaunchKernel` and stream synchronization. It performs more async
copies because compact rows, shape-specific Graph inputs, and scatter-back
require extra transfers/copies. These are host-side API observations; without
GPU activity records they do not identify device execution time or prove a
particular kernel is the causal bottleneck.

## Decision

Keep the trace and its explicit limitation. Do not claim Nsight kernel
attribution from it. The next device-side attribution attempt should use the
already functioning PyTorch CUDA profiler path or a native Linux Nsight setup.

The PyTorch CUDA follow-up succeeded and is documented in
`results/http_policy_torch_cuda_profile_v1.md`; it supersedes this API-only
trace for device bottleneck ranking.

Artifacts:

- `results/raw/http_policy_nsys_v1/fixed/trial_01.nsys-rep`
- `results/raw/http_policy_nsys_v1/full/trial_01.nsys-rep`
- `results/raw/http_policy_nsys_v1/fixed_api_cuda_api_sum.csv`
- `results/raw/http_policy_nsys_v1/full_api_cuda_api_sum.csv`
