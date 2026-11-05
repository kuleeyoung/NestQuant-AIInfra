# W4A4KV4 HTTP offered-load Nsight A/B

| metric | baseline | candidate | candidate / baseline |
|---|---:|---:|---:|
| offered load | 2.0 req/s | 4.0 req/s | 2.00x |
| completed / overloaded | 24 / 0 | 24 / 0 | - |
| achieved throughput | 2.033 | 3.908 | 1.92x |
| TTFT P95 (ms) | 182.97 | 458.67 | 2.51x |
| TPOT P95 (ms) | 69.40 | 70.80 | 1.02x |
| E2E P95 (ms) | 318.56 | 600.73 | 1.89x |
| GPU utilization mean | 47.1% | 70.3% | 1.49x |
| CUDA Graph launches/request | 2.00 | 1.67 | 0.83x |
| CUDA kernel launches/request | 6795.0 | 5679.0 | 0.84x |

The candidate doubled offered load and nearly doubled completed throughput. TPOT changed only slightly, while TTFT increased because requests waited longer for admission/slots. Fewer launches per request show that continuous batching amortized model execution across more requests.

Claim boundary: Nsight Systems under this WSL runtime recorded CUDA API and OS runtime summaries but no GPU kernel timeline. CUDA API attribution is valid; kernel-duration or stall attribution requires a supported GPU trace or NCU.
