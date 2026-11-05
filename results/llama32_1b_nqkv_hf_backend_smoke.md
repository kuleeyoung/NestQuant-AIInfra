# Llama-3.2-1B NQKV Hugging Face backend smoke test

## Scope

This is a local integration test of the custom Transformers `Cache` and
registered `AttentionInterface`. It is not a throughput benchmark or a model
quality evaluation. The run uses the complete 16-layer Llama-3.2-1B model in
BF16, while its attention Key/Value history uses NQKV v2 cold pages and a BF16
hot tail.

## Protocol

- GPU: local RTX 5070
- model: locally cached `meta-llama/Llama-3.2-1B`
- cache page size: 16 tokens
- prompt: 17 tokens, forcing one full packed page plus a one-token hot tail
- continuation: two manual single-token decode steps, then a reset and a
  standard `model.generate()` run for three new tokens
- batch size: 1
- raw record: `results/raw/llama32_1b_nqkv_backend_smoke.json`

## Result

| Check | Observed |
|---|---:|
| model layers using the backend | 16 |
| final cache length in every layer | 19 tokens |
| logits finite | yes |
| model load | 1.633 s |
| prefill including first-call compilation | 1923.0 ms |
| first decode including K1 compilation | 394.0 ms |
| second, hot decode | 19.48 ms |
| standard three-token `model.generate()` | 248.38 ms |
| CUDA allocated memory after run | 2,512,475,648 bytes |
| CUDA peak allocated memory | 2,516,456,448 bytes |

The result proves that the complete Hugging Face model path executes
projection, RoPE, NQKV cache update, packed-page reconstruction for initial
causal prefill, K1 packed-cache decode, output projection, and LM head without
falling back to a separate fake-quant model.
The standard Hugging Face generation loop also advances every layer cache to
the expected length and returns three generated token IDs.

## Evidence boundary

- The hot decode value is one observation, not a statistically stable latency
  distribution; a dedicated warmup/iteration benchmark is still required.
- The allocated-memory number includes model weights and runtime buffers. It is
  not a KV-cache-only memory measurement.
- Initial prefill currently supports only an unpadded batch-one prompt.
- Beam reordering, continuous batching, and network serving are outside this
  backend smoke test.
