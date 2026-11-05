# Llama-3.2-1B packed W4A4KV4 runtime smoke test

## What this run proves

This is the first correctness-first full-model run in the repository where all
three named low-bit components execute as packed runtime paths:

- **W4:** all 112 Llama projection weights remain NQRT4 packed on CUDA and are
  decoded inside the projection kernel;
- **A4:** every projection input is encoded on CUDA into 4-bit E8 residues,
  packed beta indices, and FP32 row norms; the W4xA4 kernel consumes that packed
  representation directly rather than materializing the complete activation as
  BF16 before the dot product;
- **KV4:** all 16 layers use NQKV v2 packed cold pages, a BF16 hot tail, and K1
  decode attention.

## Protocol

- model: `meta-llama/Llama-3.2-1B`, complete 16-layer model
- weight artifact: `llama32_1b_paper_hadamard_w4_shared128_ldlq_nqrt4.nqmc`
- prompt: 17 tokens, forcing a 16-token packed KV page and one-token hot tail
- generation: Hugging Face `model.generate()`, three new tokens
- activation q: 14, requiring four residue bits
- activation beta values: 0.24785714, 0.33857143, 0.49285714, 1.29357143
- GPU: local RTX 5070
- raw evidence: `results/raw/llama32_1b_packed_w4a4kv4_smoke.json`

## Result

| Check | Result |
|---|---:|
| packed W4 projection modules | 112 / 112 |
| packed A4 projection modules | 112 / 112 |
| observed A4 kernel calls | 336 |
| actual A4 encode operations after shared-input reuse | 192 |
| NQKV layers at expected final length | 16 / 16 |
| final cache length per layer | 19 tokens |
| generated token count | 3 |
| packed weight CUDA bytes | 518,457,088 |
| largest one-call packed A4 payload | 74,068 bytes |
| same activation shape in BF16 | 278,528 bytes |
| largest activation payload reduction | 73.41% |
| cumulative packed A4 bytes processed | 3,321,408 bytes |
| unique packed A4 bytes actually encoded after reuse | 2,323,200 bytes |
| CUDA allocated after run | 1,079,957,504 bytes |
| CUDA peak allocated | 1,083,938,304 bytes |
| elapsed, including lazy weight preparation | 12.264 s |

The 336 A4 calls equal 112 projections multiplied by the initial prefill and
two model-executed decode transitions. Hugging Face samples the third generated
token from the second transition's logits without running that final sampled
token through the model.
The q/k/v projections share one packed encoding, as do gate/up, reducing the
number of activation encodes from 336 to 192 without changing kernel outputs.

## Full WikiText-2 packed W4A4 quality validation

After the end-to-end W4A4KV4 smoke test, the tiled packed W4A4 backend was
evaluated over the complete WikiText-2 test protocol used elsewhere in this
repository. All 112 projection weights stayed packed and every projection
input used the packed A4 path.

| Check | Result |
|---|---:|
| complete context windows | 141 / 141 |
| predicted tokens | 288,627 |
| packed W4A4 projection modules | 112 / 112 |
| total NLL | 683,660.600720 |
| mean NLL per predicted token | 2.368664750 |
| WikiText-2 PPL | **10.683118** |
| elapsed | 2,094.98 s (34m 54.98s) |
| peak CUDA allocated | 1,638,561,280 bytes |
| peak CUDA reserved | 3,454,009,344 bytes |

The token stream SHA-256 is
`54927bf19e6e64b4008eeb93370c370fb25db18cf68eac8015cfe094c464d567`.
The NQRT v4 checkpoint SHA-256 is
`dcecdf797ca08ad435612c2bef4376a098ba1a6107e9a390dc8e9e9e0cbb9aa4`.
Raw evidence is stored in
`results/raw/llama32_1b_packed_w4a4_tiled_wikitext2_full.json`; the resumable
evaluator is `scripts/evaluate_packed_w4a4_wikitext.py`.

This full-corpus run uses `use_cache=False`, as required by the established
non-overlapping-window PPL protocol. It therefore validates the packed W4A4
quality path, not the packed KV4 cache path. The latter is covered by the
end-to-end generation smoke test above and must not be folded into the
10.683118 PPL claim.

## Evidence boundary and current bottleneck

- This establishes real packed storage and direct packed W4xA4 computation; it
  does **not** yet establish a speedup over BF16 or the existing W4 runtime.
- The original scalar W4xA4 correctness kernel has been superseded for this
  evaluation by a tiled Triton kernel. The tiled implementation is materially
  faster, but this report still does not claim a BF16 end-to-end speedup.
- Packed A4 tensors are transient per projection. The cumulative byte count is
  traffic processed across calls, not simultaneous resident memory.
- FP32 row norms and beta metadata make the effective activation rate slightly
  greater than four bits per entry, especially for one-token decode.
- Batch size, padding, beam reorder, continuous batching, and network serving
  remain outside this smoke test.

## Subsequent Tensor Core result

The former next stage is now implemented for the validated M16 case. The
direct-packed CUDA MMA kernel exactly matches upstream packed activation
decoding, covers all 112 projection shapes, and reduces complete-model
sequence-16 prefill median latency across three independent trials from 178.93
ms to 117.17 ms versus tiled Triton (median speedup 1.527x), with 1,232 audited
custom-M16 dispatches per trial. It remains explicit opt-in because accumulation-order
differences amplify across repeated A4 requantization; no quality-equivalence
claim is made yet. See `results/llama32_1b_w4a4_cuda_mma_m16.md`.
