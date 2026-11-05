# M1 split-K layer sensitivity sweep

## Protocol

The capacity-4 NQKV4 CUDA Graph runner replayed the deterministic prompt from the
HTTP correctness failure. The unsplit baseline was captured first. Sixteen
candidate runs then enabled split-K in exactly one `8192 -> 2048` MLP down
projection, followed by an all-layers positive control. Each candidate received
its own newly captured CUDA Graph.

## Result

The baseline generated `482,15704`. Layer 10 alone generated `482,220` and
therefore reproduced the known second-token divergence. Layer 14 changed logits
but retained `482,15704`. The other fourteen single-layer trials were bitwise
equal to the baseline at the measured second decision.

| enabled module | step-1 token | max abs logit error | relative L2 error | argmax |
|---|---:|---:|---:|---|
| `model.layers.10.mlp.down_proj` | 220 | 1.53125 | 0.07011 | **diverged** |
| `model.layers.14.mlp.down_proj` | 15704 | 1.328125 | 0.07307 | equal |
| each of the other 14 modules | 15704 | 0 | 0 | equal |
| all 16 modules | 220 | — | — | **diverged** |

The all-layers positive control reproduced `482,220`, confirming that the sweep
still exercised the known failure rather than silently falling back to the
unsplit path.

## Fixed-prompt hybrid probe

A second candidate enabled split-K in 15 layers and kept only
`model.layers.10.mlp.down_proj` unsplit. Its three-token output matched the
baseline exactly:

```text
baseline:                 482, 15704, 13
split-K except layer 10:  482, 15704, 13
```

The logits were not equal. Step 1 and step 2 still had maximum absolute errors
of 1.328125 and 1.625. This is a fixed-prompt token result, not proof that layer
10 is the only sensitive layer for arbitrary prompts or sequence states.

## Decision

The evidence justifies developing a per-module experimental fallback and a
multi-prompt sensitivity protocol. It does **not** justify a production layer-10
blacklist or re-enabling split-K by default. A layer can be safe for one decision
margin and unsafe for another; combinations can also behave differently from
single-layer trials.

The subsequent eight-prompt screen confirmed that sensitivity is input-dependent:
layers 11/14 flipped `short_greeting`, while layer 1 flipped `python_code` and
`long_context`. A `{1,10,11,14}` fallback matched baseline tokens for all 8
screening prompts, but remains an experimental small-sample result. See
`results/w4a4_m1_down_splitk_prompt_suite_v1.md`.

Machine-readable artifacts:

- `results/raw/m1_splitk_layer_sweep_v1.json`
- `results/raw/m1_splitk_hybrid_exclude_layer10_v1.json`
