"""Run the Llama-3.2 QuaRot compatibility port under the unified PPL protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


MODEL_ID = "meta-llama/Llama-3.2-1B"
MODEL_REVISION = "4e20de362430cd3b72f300e6b0f18e50e7166e08"
DATASET_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quarot-root", type=Path, default=Path.home() / "external/QuaRot")
    parser.add_argument("--project-root", type=Path, default=Path.home() / "code/NestQuant-AIInfra")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--w-bits", type=int, choices=(4, 16), default=4)
    parser.add_argument("--seed", type=int, default=58)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--limit-chunks", type=int)
    parser.add_argument("--a4", action="store_true")
    parser.add_argument("--kv4", action="store_true")
    parser.add_argument("--w-clip", action=argparse.BooleanOptionalAction, default=True,
                        help="enable QuaRot/GPTQ weight MSE clipping (official example enables it)")
    parser.add_argument("--fp32-had", action=argparse.BooleanOptionalAction, default=False,
                        help="run online Hadamard in FP32; official example leaves this disabled")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def tensor_digest(tensors) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.to(dtype=torch.int64, device="cpu").numpy().tobytes())
    return digest.hexdigest()


args = parse_args()
if args.nsamples < 1 or args.context_length < 2:
    raise ValueError("nsamples and context length must be positive")
sys.path[:0] = [str(args.quarot_root / "fake_quant"), str(args.project_root)]

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from nestquant.evaluation.perplexity import evaluate_perplexity, tokenize_corpus
import gptq_utils
import hadamard_utils
import model_utils
import quant_utils
import rotation_utils
import utils


started = time.perf_counter()
phase_seconds: dict[str, float] = {}


def phase(name: str, since: float) -> float:
    now = time.perf_counter()
    phase_seconds[name] = now - since
    return now


test_data = load_dataset(
    DATASET_ID, DATASET_CONFIG, split="test", revision=DATASET_REVISION,
    download_mode="reuse_dataset_if_exists",
)
train_data = load_dataset(
    DATASET_ID, DATASET_CONFIG, split="train", revision=DATASET_REVISION,
    download_mode="reuse_dataset_if_exists",
)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, revision=MODEL_REVISION, local_files_only=True, use_fast=True,
)
corpus = tokenize_corpus(list(test_data["text"]), tokenizer, args.context_length)
train_text = "\n\n".join(train_data["text"])
old_limit = tokenizer.model_max_length
tokenizer.model_max_length = max(old_limit, len(train_text))
train_ids = tokenizer(
    train_text, add_special_tokens=False, return_attention_mask=False,
    return_tensors="pt",
).input_ids

rng = random.Random(args.seed)
starts = [
    rng.randint(0, train_ids.shape[1] - args.context_length - 1)
    for _ in range(args.nsamples)
]
calibration = []
calibration_inputs = []
for start in starts:
    inp = train_ids[:, start:start + args.context_length]
    target = inp.clone()
    target[:, :-1] = -100
    calibration.append((inp, target))
    calibration_inputs.append(inp)
stamp = phase("data_and_tokenization", started)

torch.manual_seed(args.seed)
model = model_utils.get_model(
    MODEL_ID, revision=MODEL_REVISION, local_files_only=True,
).eval()
stamp = phase("model_load", stamp)
rotation_utils.fuse_layer_norms(model)
rotation_utils.rotate_model(model, SimpleNamespace(rotate_mode="hadamard"))
quant_utils.add_actquant(model)
for name, layer in quant_utils.find_qlayers(model).items():
    if "down_proj" in name:
        had_k, k = hadamard_utils.get_hadK(model.config.intermediate_size)
        layer.online_full_had = True
        layer.had_K, layer.K, layer.fp32_had = had_k, k, args.fp32_had
    if "o_proj" in name:
        had_k, k = hadamard_utils.get_hadK(model.config.num_attention_heads)
        layer.online_partial_had = True
        layer.had_K, layer.K = had_k, k
        layer.had_dim = model.config.hidden_size // model.config.num_attention_heads
        layer.fp32_had = args.fp32_had
stamp = phase("rotation_and_wrapping", stamp)

qargs = SimpleNamespace(
    nsamples=args.nsamples, w_groupsize=-1, w_bits=args.w_bits,
    int8_down_proj=False, w_asym=False, w_clip=args.w_clip,
    percdamp=0.01, act_order=False,
)
if args.w_bits < 16:
    gptq_utils.gptq_fwrd(model, calibration, utils.DEV, qargs)
stamp = phase("gptq_w4", stamp)

for name, layer in quant_utils.find_qlayers(
    model, layers=[quant_utils.ActQuantWrapper]
).items():
    layer.quantizer.configure(
        bits=4 if args.a4 and "lm_head" not in name else 16,
        groupsize=-1, sym=True, clip_ratio=1.0,
    )
    if args.kv4 and "v_proj" in name:
        layer.out_quantizer.configure(bits=4, groupsize=-1, sym=True, clip_ratio=1.0)
if args.kv4:
    for layer in model_utils.get_layers(model):
        rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
            layer.self_attn, model_utils.get_rope_function_name(model),
            config=model.config, k_bits=4, k_groupsize=-1,
            k_sym=True, k_clip_ratio=1.0,
        )
stamp = phase("runtime_quantizer_setup", stamp)

tokens = corpus.token_ids
if args.limit_chunks is not None:
    tokens = tokens[:args.limit_chunks * args.context_length]
model.to("cuda")
result = evaluate_perplexity(
    model, tokens, context_length=args.context_length, device="cuda",
)
phase("evaluation", stamp)

# Persist the expensive numerical result before collecting optional source
# metadata, so a provenance-only failure cannot discard a completed run.
args.output.parent.mkdir(parents=True, exist_ok=True)
partial_payload = {
    "schema_version": 1,
    "status": "result_captured_provenance_pending",
    "result": result,
    "timing_seconds": phase_seconds,
}
args.output.write_text(json.dumps(partial_payload, indent=2), encoding="utf-8")

commit = subprocess.run(
    ["git", "-C", str(args.quarot_root), "rev-parse", "HEAD"],
    check=True, capture_output=True, text=True,
).stdout.strip()
diff = subprocess.run(
    ["git", "-C", str(args.quarot_root), "diff", "--binary"],
    check=True, capture_output=True,
).stdout
payload = {
    "schema_version": 1,
    "status": "smoke_completed" if args.limit_chunks else "completed",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "method": ("QuaRot compatibility port GPTQ W4" if args.w_bits == 4
               else "QuaRot compatibility port rotated BF16")
              + ("A4" if args.a4 else "") + ("KV4" if args.kv4 else ""),
    "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "dtype": "bfloat16"},
    "dataset": {"id": DATASET_ID, "config": DATASET_CONFIG, "revision": DATASET_REVISION},
    "protocol": {
        "seed": args.seed, "context_length": args.context_length,
        "calibration_samples": args.nsamples,
        "calibration_start_indices": starts,
        "calibration_token_ids_sha256": tensor_digest(calibration_inputs),
        "test_token_ids_sha256": corpus.token_ids_sha256,
        "w_bits": args.w_bits,
        "w_method": "GPTQ" if args.w_bits == 4 else "none",
        "w_groupsize": -1,
        "w_clip": args.w_clip, "percdamp": 0.01, "act_order": False,
        "fp32_hadamard": args.fp32_had,
        "a_bits": 4 if args.a4 else 16,
        "k_bits": 4 if args.kv4 else 16,
        "v_bits": 4 if args.kv4 else 16,
        "limit_chunks": args.limit_chunks,
    },
    "result": result,
    "timing_seconds": phase_seconds,
    "elapsed_seconds": time.perf_counter() - started,
    "environment": {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    },
    "source": {
        "quarot_commit": commit,
        "quarot_dirty_patch_sha256": hashlib.sha256(diff).hexdigest(),
        "implementation_boundary": "compatibility port with pure-PyTorch Hadamard; no low-bit kernel claim",
    },
}
args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps({"output": str(args.output), "result": result,
                  "timing_seconds": phase_seconds}, indent=2), flush=True)
