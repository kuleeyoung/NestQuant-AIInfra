# NQKV4 versus BF16 KV stage A/B

Five fixed-seed repetitions use identical requests and the same packed W4A4 path.
Only the KV backend changes. Runs are paired by profile and repetition.

## boundary_fixed_p15_o3

- NQKV4 throughput median: 24.20 token/s.
- BF16 KV throughput median: 26.88 token/s.
- Paired NQKV4 throughput deficit: 9.79%.

| stage | NQKV4 median | BF16 median | paired NQKV4 overhead |
|---|---:|---:|---:|
| prefill_only | 132.35 ms | 118.17 ms | 11.90% |
| decode_only | 57.96 ms | 52.68 ms | 10.67% |
| boundary | 56.89 ms | 52.09 ms | 9.29% |

## decode_fixed_p8_o8

- NQKV4 throughput median: 30.40 token/s.
- BF16 KV throughput median: 34.56 token/s.
- Paired NQKV4 throughput deficit: 11.81%.

| stage | NQKV4 median | BF16 median | paired NQKV4 overhead |
|---|---:|---:|---:|
| prefill_only | 116.09 ms | 96.32 ms | 17.11% |
| decode_only | 57.85 ms | 52.32 ms | 10.31% |

## Claim boundary

The requests, random seeds, fixed shapes, GPU and W4A4 implementation are identical. The two KV backends were executed in separate processes, not alternated inside one process, so this is paired-by-seed stage evidence rather than a randomized timing trial.
