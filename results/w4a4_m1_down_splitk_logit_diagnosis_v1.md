# M1 split-K logit divergence diagnosis

## Question

The HTTP correctness A/B produced identical first tokens but different later
greedy sequences. This experiment removes layers of the serving stack to locate
the first divergence and measure whether the split-K logit perturbation is large
enough to cross the greedy decision margin.

## Eager isolation

One converted model ran the same prompt twice in one process. The first run used
the unsplit M1 kernel and the second enabled split-8 for all 16
`8192 -> 2048` MLP down projections. This path used the ordinary Hugging Face
cache and no HTTP, scheduler, NQKV cache, or CUDA Graph.

| decode step | unsplit token | split-K token | max abs logit error | relative L2 error | result |
|---:|---:|---:|---:|---:|---|
| 0 | 578 | 578 | 0 | 0 | equal |
| 1 | 6864 | 6864 | 0 | 0 | equal |
| 2 | 315 | 315 | 1.1328125 | 0.06925 | equal argmax |

The kernel can change logits, but this eager prompt retained the same greedy
tokens because the winning margins remained 3.5, 5.25 and 5.0.

## Production-runner isolation

The second run retained the same model and prompt but added the production
capacity-4 NQKV4 cache, scheduler and ragged CUDA Graph runner. HTTP, tokenizer
I/O and request concurrency remained excluded. It exactly reproduced the prior
HTTP token sequences.

| decode step | same input off/on | unsplit token | split-K token | unsplit margin | split-K margin | max abs error | relative L2 error |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 0 | yes | 482 | 482 | 0.5 | 0.5 | 0 | 0 |
| 1 | yes | 15704 | 220 | 0.25 | 0.125 | 1.71875 | 0.07363 |
| 2 | no | 13 | 679 | 0.125 | 0.75 | 15.21094 | 0.56415 |

Step 1 is the causal comparison: both policies receive token 482 and identical
positions/cache history, but the baseline top-1/top-2 margin is only 0.25 while
the split-K perturbation reaches 1.71875, so greedy argmax flips. Step 2 is no
longer a controlled same-input comparison because each policy feeds back its own
different token.

## Conclusion

The divergence is not caused by HTTP ordering or request concurrency. It is also
not universal to every cache/execution path. The evidence supports a combined
mechanism:

1. split-K changes the FP32 accumulation/reduction order and perturbs layer
   outputs;
2. the NQKV4 + CUDA Graph serving path produces a much smaller final logit
   decision margin for this prompt than the ordinary eager-cache path;
3. the perturbation crosses that margin at the second greedy decision and the
   autoregressive feedback then amplifies the difference.

The production default therefore remains the unsplit kernel. The split-K path is
an explicit performance experiment, not an output-equivalent service kernel.

Machine-readable artifacts:

- `results/raw/m1_splitk_logit_diagnosis_v1.json`
- `results/raw/m1_splitk_nqkv_graph_logit_diagnosis_v1.json`

The follow-up single-layer sweep localized this prompt's token flip to
`model.layers.10.mlp.down_proj`. Keeping that module unsplit while enabling the
other 15 restored the three baseline tokens, although logits remained different.
See `results/w4a4_m1_down_splitk_layer_sensitivity_v1.md`; the result is not a
general production-safe layer blacklist.
