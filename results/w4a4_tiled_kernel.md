# Packed W4A4 tiled-kernel checkpoint

The direct packed W4xA4 path now has two implementations: a scalar
correctness oracle and a tiled `tl.dot` candidate. Both consume packed A4 and
packed W4 streams directly; neither materializes a complete BF16 activation or
weight matrix before the dot product.

## Isolated real q_proj result

The record is `model.layers.0.self_attn.q_proj.weight` from the real NQRT4
Llama-3.2-1B checkpoint. Timings use five warmups and 20 CUDA-event iterations.

| shape `[tokens, in, out]` | scalar | tiled | speedup | max abs difference |
|---|---:|---:|---:|---:|
| `[1, 2048, 2048]` | 2185.47 us | 699.40 us | 3.12x | 0.0625 |
| `[17, 2048, 2048]` | 34008.60 us | 1593.58 us | 21.34x | 0.0625 |

For 17 tokens, online A4 encoding takes 464.92 us and produces an 18,580-byte
payload.

## Full-model paired smoke result

The same 17-token prompt and three-token `model.generate()` procedure is run
twice per process; the second run excludes lazy weight preparation and kernel
compilation.

| A4 kernel | hot prefill + generation |
|---|---:|
| scalar | 8502.18 ms |
| tiled | 784.70 ms |
| tiled speedup | 10.83x |

Both paths complete all 112 packed-W4/A4 projections and leave all 16 NQKV
layer caches at the expected length. Their generated token sequences differ,
however. The tiled path casts decoded tiles to BF16 for `tl.dot`, while the
scalar oracle performs per-block FP32 products and accumulation. Small rounding
differences can change an argmax and then compound autoregressively.

The tiled path therefore remains a performance candidate, not yet a
quality-equivalent default. Logit-error and perplexity regression are required
before promotion.

## Numerical regression

For the same 17-token prompt, the final prefill logits have cosine similarity
0.95463 and relative L2 error 0.31252 between scalar and tiled execution; the
argmax differs. This is an autoregressive-sensitivity warning, not by itself a
perplexity result.

On one deterministic 128-token text window, scalar and tiled packed W4A4 give
PPL 3.58780 and 3.58422 respectively. The 0.10% relative change does not show a
quality regression on this bounded sample, while the first forward including
lazy preparation drops from 63.99 s to 12.00 s. This micro-PPL is not a
WikiText-2 result and cannot replace the full 288,627-token evaluation.

A 2048-token feasibility probe also completes without compilation or memory
failure. Its second hot forward is 15.70 s. Multiplying by the 141 full
WikiText-2 windows gives a lower-order estimate of roughly 36.9 minutes for a
batch-one tiled run, before dataset/evaluator overhead. The repeated local text
has artificial PPL and is used only for runtime feasibility.
