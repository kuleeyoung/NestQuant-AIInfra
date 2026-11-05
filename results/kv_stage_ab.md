# NQKV4 versus BF16 KV stage A/B

Five fixed-seed repetitions use identical requests and the same packed W4A4 path.
Only the KV backend changes. Runs are paired by profile and repetition.

## boundary_fixed_p15_o3

- NQKV4 throughput median: 22.53 token/s.
- BF16 KV throughput median: 26.88 token/s.
- Paired NQKV4 throughput deficit: 16.01%.

| stage | NQKV4 median | BF16 median | paired NQKV4 overhead |
|---|---:|---:|---:|
| prefill_only | 149.00 ms | 118.17 ms | 25.97% |
| decode_only | 58.67 ms | 52.68 ms | 12.02% |
| boundary | 56.82 ms | 52.09 ms | 8.59% |

## decode_fixed_p8_o8

- NQKV4 throughput median: 32.02 token/s.
- BF16 KV throughput median: 34.56 token/s.
- Paired NQKV4 throughput deficit: 7.64%.

| stage | NQKV4 median | BF16 median | paired NQKV4 overhead |
|---|---:|---:|---:|
| prefill_only | 119.89 ms | 96.32 ms | 25.48% |
| decode_only | 53.35 ms | 52.32 ms | 1.83% |

## Claim boundary

The requests, random seeds, fixed shapes, GPU and W4A4 implementation are identical. The two KV backends were executed in separate processes, not alternated inside one process, so this is paired-by-seed stage evidence rather than a randomized timing trial.
