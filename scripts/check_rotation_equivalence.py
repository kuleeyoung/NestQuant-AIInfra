"""Check real-model logits after weight rotation/inversion without quantization."""

import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from nestquant.core.rotation import signed_hadamard


@torch.inference_mode()
def main():
    config = json.loads(Path("configs/project.json").read_text(encoding="utf-8"))
    exp = config["llm_ppl_experiment"]
    model = AutoModelForCausalLM.from_pretrained(
        exp["model_id"], revision=exp["model_revision"], local_files_only=True,
        torch_dtype=torch.bfloat16).to("cuda").eval()
    tokens = torch.arange(1, 33, device="cuda").unsqueeze(0)
    before = model(tokens, use_cache=False).logits.float()
    generator = torch.Generator().manual_seed(123)
    maximum_error = 0.
    changed_values = 0
    layers = 0
    for module in model.modules():
        if not isinstance(module, nn.Linear):
            continue
        weight = module.weight
        signs = (torch.randint(0, 2, (weight.shape[1],), generator=generator) * 2 - 1).to(
            device=weight.device, dtype=torch.float32)
        for start in range(0, weight.shape[0], 512):
            original = weight[start:start + 512].float()
            rotated = signed_hadamard(original, signs)
            direct = signed_hadamard(rotated, signs, inverse=True)
            maximum_error = max(maximum_error, (direct - original).abs().max().item())
            # Identity quantizer, using the same residual reconstruction as the adapter.
            restored = original + signed_hadamard(rotated - rotated, signs, inverse=True)
            cast = restored.to(weight.dtype)
            changed_values += (cast != weight[start:start + 512]).sum().item()
            weight[start:start + 512].copy_(cast)
        layers += 1
    after = model(tokens, use_cache=False).logits.float()
    error = (before - after).abs().max().item()
    result = {"model_id": exp["model_id"], "revision": exp["model_revision"],
              "rotation_seed": 123, "layers": layers, "test_token_ids": tokens.cpu().tolist(),
              "max_fp32_weight_roundtrip_abs_error": maximum_error,
              "bf16_changed_weight_values": changed_values,
              "max_logit_abs_error": error,
              "logits_exactly_equal": torch.equal(before, after),
              "reconstruction": "W + (Q(WR)-WR) R^T with identity Q",
              "note": "One fixed 32-token input; numerical equivalence check, not PPL."}
    path = Path("results/raw/rotation_model_equivalence.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)
    torch.testing.assert_close(after, before, rtol=0, atol=0)


if __name__ == "__main__":
    main()
