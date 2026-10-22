# QuaRot compatibility patch

This directory contains the compatibility patch used to run the official
[QuaRot](https://github.com/spcl/QuaRot) fake-quantization pipeline on
Llama-3.2-1B and an NVIDIA RTX 5070 (sm_120).

## Provenance

- Upstream repository: `https://github.com/spcl/QuaRot.git`
- Upstream revision: `5008669b08c1f11f9b64d52d16fddd47ca754c5a`
- Patch: `quarot_llama32_pytorch_fallback.patch`
- Patch SHA-256: `E22326EDE8DE286120B390F3B6EADAA3223A7077CA4FC34F159C9CB7B10FE75A`
- Upstream license: Apache License 2.0; see `LICENSE.QuaRot`

The patch changes the upstream files named in the diff. Its main changes are:

1. add a pure-PyTorch Hadamard fallback when the CUDA extension is unavailable;
2. support Llama-3.2 model-level RoPE and replay the layer arguments required by
   current Transformers versions;
3. handle grouped-query-attention key-cache dimensions;
4. avoid corrupting tied input/output embeddings during rotation;
5. record model revisions and add focused compatibility tests.

## Apply

Clone QuaRot at the revision above, then run:

```bash
git apply /path/to/quarot_llama32_pytorch_fallback.patch
```

The formal comparison protocol and limitations are documented in
[`../docs/experiment_protocol.md`](../docs/experiment_protocol.md) and
[`../results/llama32_1b_quarot_comparison.md`](../results/llama32_1b_quarot_comparison.md).
