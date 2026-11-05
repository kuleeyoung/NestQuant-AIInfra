# Packed W4A4 projection-shape profile

## Protocol

The first real packed projection of each Llama-3.2-1B suffix was loaded from the
same NQRT4 checkpoint used by the online service. M1, M2, and M128 were measured
with three warmups and ten timed iterations. The end-to-end measurement clears
Hadamard/A4 input caches before every call. The kernel-only measurement packs A4
once and times only the selected packed-W4A4 CUDA path.

Use this experiment to rank matrix shapes, not as service latency.

## Kernel-only median

| projection | shape `[N,K]` | M1 | M2 | M128 |
|---|---:|---:|---:|---:|
| q_proj | 2048×2048 | 0.251 ms | 0.389 ms | 1.043 ms |
| k_proj | 512×2048 | 0.240 ms | 0.388 ms | 0.494 ms |
| v_proj | 512×2048 | 0.239 ms | 0.387 ms | 0.491 ms |
| o_proj | 2048×2048 | 0.254 ms | 0.410 ms | 1.064 ms |
| gate_proj | 8192×2048 | 0.406 ms | 0.470 ms | 3.682 ms |
| up_proj | 8192×2048 | 0.406 ms | 0.473 ms | 3.670 ms |
| down_proj | 2048×8192 | **0.946 ms** | 0.386 ms | **4.300 ms** |

The M1 kernel medians sum to about 2.742 ms per layer. `down_proj` alone is
34.5%; gate/up/down together are about 64.1%. At M128 the three MLP projections
are about 79% of the seven-projection kernel sum.

## Decision

The first shape-specific target is M1 `down_proj [N=2048,K=8192]`. Its long K
dimension makes the direct packed-word GEMV about 3.7x slower than q/o. M2 is
already covered by the promoted split-K path and should not be rewritten.

Raw data: `results/raw/w4a4_projection_shapes_v2.json`.
