"""Time one cached Llama Linear layer using the project W4 settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from nestquant.models.quantization import quantize_nestquant_w4_


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", default="model.layers.0.self_attn.q_proj")
    parser.add_argument("--output", type=Path, default=Path("results/raw/layer_profile.json"))
    parser.add_argument("--export", action="store_true", help="include NQRT packing time")
    args = parser.parse_args()
    config = json.loads(Path("configs/project.json").read_text(encoding="utf-8"))
    experiment = config["llm_ppl_experiment"]
    model = AutoModelForCausalLM.from_pretrained(
        experiment["model_id"], revision=experiment["model_revision"],
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True,
    ).to("cuda")
    layer = model.get_submodule(args.layer)
    if not isinstance(layer, nn.Linear):
        raise TypeError(f"{args.layer} is not a Linear layer")
    artifacts = {} if args.export else None
    stats = quantize_nestquant_w4_(
        nn.Sequential(layer), q=config["initial_algorithm_config"]["q"],
        beta_coefficients=experiment["beta_coefficients_project_scale"],
        selected_beta_count=config["initial_algorithm_config"]["selected_beta_count"],
        seed=config["initial_algorithm_config"]["seed"],
        rows_per_chunk=experiment["rows_per_chunk"], output_artifacts=artifacts,
        progress_callback=lambda record: print(json.dumps(record), flush=True),
    )
    output = {"model_id": experiment["model_id"], "layer": args.layer,
              "shape": list(layer.weight.shape), "export": args.export,
              "stats": stats}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Profile saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
