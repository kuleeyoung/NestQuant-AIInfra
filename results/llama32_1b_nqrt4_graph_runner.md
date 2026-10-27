# Reusable NQRT4 CUDA Graph decode runner

## Outcome

`NQRTGraphDecodeRunner` turns the previously isolated CUDA Graph experiment into a reusable
fixed-batch autoregressive decode component. It owns a fixed-address `StaticCache`, captures
the one-token model forward once, and accepts new token IDs and cache positions on every
replay. The graph can be reused after a new prompt prefill without recapturing.

The production J2 CUDA kernels were also corrected to launch on PyTorch's current CUDA
stream. Without that change, custom kernels launched on the default stream and were omitted
from non-default-stream CUDA Graph capture, producing deceptively fast but invalid output.

## Environment and protocol

- GPU: NVIDIA GeForce RTX 5070
- Model: Llama-3.2-1B packed NQRT4 checkpoint
- Backend: `cuda_mma_j2`, BF16 output
- Prompt length: 128 tokens
- Generation validation: 32 new tokens per request, greedy decoding
- Numerical validation: prefill plus 8 consecutive decode positions
- Timing: 5 repeats, 30 calls per repeat; table reports the median wall-clock time
- Raw evidence: `results/raw/llama32_1b_nqrt4_graph_runner_b1_b16.json`

## Results

| Batch | Eager decode | Graph replay | Speedup | Graph aggregate throughput |
|---:|---:|---:|---:|---:|
| 1 | 40.432 ms/token | 16.610 ms/token | 2.434x | 60.20 token/s |
| 16 | 40.340 ms/step | 20.899 ms/step | 1.930x | 765.57 token/s |

For both batch sizes:

- prefill maximum absolute logit error: `0.0`;
- all eight decode-step maximum absolute logit errors: `0.0`;
- validation argmax tokens: exactly equal;
- complete 32-token greedy generations: exactly equal.

## Implementation boundary

The runner currently targets fixed-batch offline or controlled-service decoding:

- model identity, batch size, maximum cache length, tensor shapes, and tensor addresses are
  fixed for the runner lifetime;
- token values, prompt contents, prompt length, and decode cache positions may change within
  the allocated cache;
- greedy `argmax` and the Python generation loop remain outside the captured graph;
- continuous batching, request admission/eviction, mixed batch sizes, beam search, and sampling
  are not implemented;
- the measured speedup is a decode-forward latency result, not an end-to-end serving-QPS claim.

## Defensible project claim

Implemented a reusable CUDA Graph decode runner for a packed 4-bit Llama-3.2-1B runtime,
including fixed-address KV-cache replay and current-stream-safe custom CUDA kernels. On an
RTX 5070 it preserved exact BF16 logits and 32-token greedy outputs while reducing median
one-token forward latency from 40.43 ms to 16.61 ms at batch 1 (2.43x), and from 40.34 ms to
20.90 ms at batch 16 (1.93x, 765.57 aggregate token/s).
