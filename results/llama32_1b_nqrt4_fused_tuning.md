# NQRT v4 packed fused Linear tuning

## Scope

This experiment tunes the packed-resident Triton Linear prototype on the real
`model.layers.0.self_attn.q_proj.weight` tensor from Llama-3.2-1B. The weight is
2048 x 2048, inputs are BF16, and the device is an NVIDIA GeForce RTX 5070.

The sweep covers:

- output-row tiles: 8, 16, 32;
- token tiles: 1, 2, 4, 8, 16;
- warps per Triton program: 2, 4, 8;
- sequence lengths: 1, 8, 16;
- 20 timed repetitions for every one of the 135 configurations.

All configurations completed successfully. The same configuration won all
three sequence lengths: 16 output rows, 16 tokens, and 2 warps.

## Independent 100-repeat confirmation

| Sequence length | Previous default median | Tuned median | Median speedup | Previous default mean | Tuned mean | Mean speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.4328 ms | 0.3468 ms | 1.248x | 0.4399 ms | 0.3640 ms | 1.209x |
| 8 | 0.4849 ms | 0.3430 ms | 1.414x | 0.5065 ms | 0.3532 ms | 1.434x |
| 16 | 0.5335 ms | 0.3480 ms | 1.533x | 0.5526 ms | 0.3618 ms | 1.527x |

The previous default used 16 output rows and 4 warps, with a token tile equal
to the next power of two of the input length. The new default fixes the token
tile at 16 and uses 2 warps. For short inputs, inactive token lanes are masked,
so this changes scheduling and layout but not numerical semantics.

## Claim boundary

This is local autotuning evidence, not a universal kernel configuration. The
winner depends on the GPU, weight shape, token count, Triton/CUDA versions, and
the searched parameter space. The script keeps every configuration explicit so
the sweep can be repeated on another machine instead of assuming that the RTX
5070 result transfers unchanged.

Raw records:

- `results/raw/llama32_1b_nqrt4_qproj_fused_tuning.json`
- `results/raw/llama32_1b_nqrt4_qproj_fused_tuning_confirm.json`

Reproduction entry point: `scripts/tune_triton_fused_linear.py`.
