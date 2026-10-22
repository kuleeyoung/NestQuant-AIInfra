"""Evaluate BF16, RTN-W4 and NestQuant-W4 fake-quant quality on WikiText-2."""

from __future__ import annotations

import argparse
from functools import partial
import gc
import hashlib
import json
import math
import platform
import importlib.metadata
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from ..models.quantization import quantize_nestquant_w4_, quantize_rtn_w4_
from ..evaluation.perplexity import evaluate_perplexity, tokenize_corpus
from ..core.model_checkpoint import PackedNestQuantModel
from ..models.hessian_store import LlamaHessianStore
from ..models.qa_calibration import ActivationVarianceStore


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "project.json"
UPSTREAM_ACTIVATION_BETA_COEFFICIENTS = (3.47, 4.74, 6.90, 18.11)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=("bf16", "rtn_w4", "nestquant_w4"),
                        default=["bf16", "rtn_w4", "nestquant_w4"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=58)
    parser.add_argument("--q", type=int, default=14)
    parser.add_argument("--beta-count", type=int, default=4)
    parser.add_argument("--rows-per-chunk", type=int, default=512)
    parser.add_argument("--exclude-linear", nargs="*", default=(),
                        help="exact nn.Linear names to retain in floating point")
    parser.add_argument("--rotation-seed", type=int, default=None,
                        help="enable signed-Hadamard weight-space ablation for NestQuant only")
    parser.add_argument("--ldlq-hessian-dir", type=Path, default=None,
                        help="enable block-LDLQ using grouped calibration Hessians")
    parser.add_argument("--ldlq-calibration-chunks", type=int, default=16)
    parser.add_argument("--ldlq-relative-damp", type=float, default=0.01)
    parser.add_argument("--qa-variance-file", type=Path, default=None,
                        help="20-sequence activation-variance JSON enabling QA-LDLQ")
    parser.add_argument("--qa-calibration-chunks", type=int, default=20)
    parser.add_argument("--qa-epsilon-constant", type=float, default=155.0)
    parser.add_argument("--kv4", action="store_true",
                        help="fake-quantize post-RoPE keys and values before cache insertion")
    parser.add_argument("--a4", action="store_true",
                        help="fake-quantize Llama attention/MLP projection inputs")
    parser.add_argument("--act-rows-per-chunk", type=int, default=4096)
    parser.add_argument("--use-cache", action="store_true",
                        help="evaluate through an unquantized DynamicCache control path")
    parser.add_argument("--kv-rotation-control", action="store_true",
                        help="matched post-RoPE Q/K Hadamard with no KV quantization")
    parser.add_argument("--kv-beta-config", type=Path, default=None,
                        help="per-layer key/value beta JSON produced by calibration")
    parser.add_argument("--kv-rows-per-chunk", type=int, default=4096)
    parser.add_argument("--limit-chunks", type=int, default=None,
                        help="debug smoke limit; omit for full WikiText-2 test split")
    parser.add_argument("--eval-batch-size", type=int, default=1,
                        help="independent context windows evaluated per forward pass")
    parser.add_argument("--offline", action="store_true", help="load model, tokenizer and dataset only from local cache")
    parser.add_argument("--export-checkpoint", type=Path, default=None,
                        help="write and reload an NQMC artifact after NestQuant W4, before PPL evaluation")
    parser.add_argument("--load-checkpoint", type=Path, default=None,
                        help="load an existing NQMC artifact for PPL without requantizing")
    parser.add_argument("--output", type=Path, default=None,
                        help="manifest path; default is a unique timestamped file under results/raw/")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="project JSON config (CLI flags override experiment defaults)")
    args = parser.parse_args()
    if args.offline:
        # datasets and huggingface_hub read these flags during import. Setting
        # them here also prevents a cached dataset load from making HEAD calls.
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    project_config = json.loads(args.config.read_text(encoding="utf-8"))
    experiment = project_config["llm_ppl_experiment"]
    target = project_config["reproduction_target"]
    model_id = experiment["model_id"]
    model_revision = experiment["model_revision"]
    dataset_id = experiment["dataset_id"]
    dataset_config = experiment["dataset_config"]
    dataset_revision = experiment["dataset_revision"]
    methods = args.methods if "--methods" in sys.argv else experiment["methods"]
    context_length = args.context_length if "--context-length" in sys.argv else target["context_length"]
    seed = args.seed if "--seed" in sys.argv else project_config["initial_algorithm_config"]["seed"]
    q = args.q if "--q" in sys.argv else project_config["initial_algorithm_config"]["q"]
    beta_count = args.beta_count if "--beta-count" in sys.argv else project_config["initial_algorithm_config"]["selected_beta_count"]
    rows_per_chunk = args.rows_per_chunk if "--rows-per-chunk" in sys.argv else experiment["rows_per_chunk"]
    beta_coefficients = tuple(experiment["beta_coefficients_project_scale"])
    rtn_group_size = experiment["rtn_group_size"]
    model_dtype_name = experiment["model_dtype"]
    model_dtype = getattr(torch, model_dtype_name, None)
    if not isinstance(model_dtype, torch.dtype) or not model_dtype.is_floating_point:
        raise ValueError(f"unsupported floating-point model dtype: {model_dtype_name}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device")
    if context_length < 2 or beta_count > len(beta_coefficients):
        raise ValueError("invalid context length or beta count")
    if args.export_checkpoint is not None and "nestquant_w4" not in methods:
        raise ValueError("--export-checkpoint requires nestquant_w4 in --methods")
    if args.load_checkpoint is not None:
        if args.export_checkpoint is not None or methods != ["nestquant_w4"]:
            raise ValueError("--load-checkpoint requires only nestquant_w4 and cannot be combined with --export-checkpoint")
    if args.rotation_seed is not None and (
            methods != ["nestquant_w4"] or args.export_checkpoint or args.load_checkpoint):
        raise ValueError("--rotation-seed requires only nestquant_w4 and no checkpoint import/export")
    if args.ldlq_hessian_dir is not None and (
            methods != ["nestquant_w4"] or args.load_checkpoint):
        raise ValueError(
            "--ldlq-hessian-dir requires only nestquant_w4 and no checkpoint loading"
        )
    if args.exclude_linear and methods != ["nestquant_w4"]:
        raise ValueError("--exclude-linear requires only nestquant_w4")
    if args.ldlq_calibration_chunks < 1 or args.ldlq_relative_damp < 0:
        raise ValueError("invalid LDLQ calibration chunk count or damping")
    if args.qa_variance_file is not None:
        if args.ldlq_hessian_dir is None or args.load_checkpoint is not None:
            raise ValueError("--qa-variance-file requires fresh LDLQ quantization with --ldlq-hessian-dir")
        if args.qa_calibration_chunks < 1 or args.qa_epsilon_constant <= 0:
            raise ValueError("invalid QA-LDLQ calibration count or epsilon constant")
    if args.kv4 and methods != ["nestquant_w4"]:
        raise ValueError("--kv4 requires only nestquant_w4 in --methods")
    if args.a4 and methods != ["nestquant_w4"]:
        raise ValueError("--a4 requires only nestquant_w4 in --methods")
    if sum(bool(flag) for flag in (args.kv4, args.use_cache, args.kv_rotation_control)) > 1:
        raise ValueError("choose only one cache evaluation mode")
    if args.kv_rotation_control and methods != ["nestquant_w4"]:
        raise ValueError("--kv-rotation-control requires only nestquant_w4")
    if args.kv_beta_config is not None and not args.kv4:
        raise ValueError("--kv-beta-config requires --kv4")
    if args.kv_rows_per_chunk < 1 or args.act_rows_per_chunk < 1 or args.eval_batch_size < 1:
        raise ValueError("runtime quantization chunk sizes must be positive")

    provisional_kv_betas = tuple(value / q for value in beta_coefficients[:beta_count])
    activation_betas = tuple(value / q for value in UPSTREAM_ACTIVATION_BETA_COEFFICIENTS)
    kv_layer_betas = None
    kv_beta_metadata = None
    if args.kv4 and args.kv_beta_config is not None:
        kv_layer_betas, kv_beta_metadata = _load_kv_beta_config(
            args.kv_beta_config, model_id=model_id, model_revision=model_revision,
            q=q, selected_beta_count=beta_count,
        )

    config = {"config_file": str(args.config.resolve()),
              "project_config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
              "model_id": model_id,
              "model_revision": model_revision,
              "dataset_id": dataset_id, "dataset_config": dataset_config,
              "dataset_revision": dataset_revision, "methods": methods,
              "context_length": context_length, "seed": seed,
              "q": q, "beta_coefficients_project_scale": list(beta_coefficients),
              "beta_grid_note": experiment["beta_grid_note"],
              "selected_beta_count": beta_count, "rtn_group_size": rtn_group_size,
              "rows_per_chunk": rows_per_chunk,
              "excluded_linear_names": sorted(set(args.exclude_linear)),
              "dither_rng_device": "cpu",
              "weight_rotation": "none" if args.rotation_seed is None else "signed_hadamard_inverse_weight",
              "rotation_seed": args.rotation_seed,
              "ldlq": args.ldlq_hessian_dir is not None,
              "ldlq_hessian_dir": str(args.ldlq_hessian_dir.resolve()) if args.ldlq_hessian_dir else None,
              "ldlq_calibration_chunks": args.ldlq_calibration_chunks if args.ldlq_hessian_dir else None,
              "ldlq_relative_damp": args.ldlq_relative_damp if args.ldlq_hessian_dir else None,
              "qa_ldlq": args.qa_variance_file is not None,
              "qa_variance_file": str(args.qa_variance_file.resolve()) if args.qa_variance_file else None,
              "qa_variance_file_sha256": None if args.qa_variance_file is None else hashlib.sha256(
                  args.qa_variance_file.read_bytes()).hexdigest(),
              "qa_calibration_chunks": args.qa_calibration_chunks if args.qa_variance_file else None,
              "qa_epsilon_constant": args.qa_epsilon_constant if args.qa_variance_file else None,
              "kv_cache_quantization": "nestquant_e8_fake_quant" if args.kv4 else "none",
              "evaluation_cache_mode": (
                  "nestquant_kv4" if args.kv4 else
                  "matched_qk_hadamard_control" if args.kv_rotation_control else
                  "floating_dynamic_cache" if args.use_cache else "disabled"
              ),
              "kv_beta_config": str(args.kv_beta_config.resolve()) if args.kv_beta_config else None,
              "kv_beta_config_sha256": None if args.kv_beta_config is None else hashlib.sha256(
                  args.kv_beta_config.read_bytes()).hexdigest(),
              "kv_default_beta_values": list(provisional_kv_betas) if args.kv4 else None,
              "kv_rows_per_chunk": args.kv_rows_per_chunk if args.kv4 else None,
              "activation_quantization": "nestquant_e8_fake_quant" if args.a4 else "none",
              "activation_beta_source": "public_upstream_parallel/ppl.py" if args.a4 else None,
              "activation_beta_coefficients_project_scale": list(
                  UPSTREAM_ACTIVATION_BETA_COEFFICIENTS) if args.a4 else None,
              "activation_beta_values": list(activation_betas) if args.a4 else None,
              "activation_rows_per_chunk": args.act_rows_per_chunk if args.a4 else None,
              "device": str(device), "model_dtype": model_dtype_name,
              "logit_loss_dtype": experiment["logit_loss_dtype"], "limit_chunks": args.limit_chunks,
              "evaluation_batch_size": args.eval_batch_size,
              "offline": args.offline,
              "export_checkpoint": str(args.export_checkpoint.resolve()) if args.export_checkpoint else None,
              "load_checkpoint": str(args.load_checkpoint.resolve()) if args.load_checkpoint else None,
              "quantization_scope": experiment["quantization_scope"]}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True,
                                             separators=(",", ":")).encode()).hexdigest()
    created_at = datetime.now(timezone.utc)
    run_id = created_at.strftime("%Y%m%dT%H%M%S%fZ")
    output_path = args.output or Path("results/raw") / f"llama32_1b_wikitext2_{run_id}.json"
    payload = {"schema_version": 1, "run_id": run_id,
               "created_at_utc": created_at.isoformat(),
               "status": "preparing", "config": config, "config_sha256": config_hash,
               "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                               "packages": _package_versions(("torch", "transformers", "datasets", "huggingface-hub", "safetensors")),
                               "torch_cuda": torch.version.cuda,
                               "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"},
               "source": _source_provenance(PROJECT_ROOT),
               "protocol_notes": [
                   "Project protocol: join test text rows with two newlines, add BOS/EOS once when available, then use complete non-overlapping chunks.",
                   "Each chunk contributes context_length-1 next-token targets; the incomplete tail is dropped.",
                   "RTN W4 uses symmetric group-wise quantization with group size 128.",
                   "NestQuant W4 uses per-layer beta subset DP and Opt-beta; optional weight-space signed-Hadamard rotation and 8-column block-LDLQ are recorded in config and may be combined. Combined runs transform each calibration Hessian as R.T @ H @ R before LDLQ. This is not global model rotation or QA-LDLQ.",
                   "Optional KV4 quality simulation applies a matched normalized Hadamard to post-RoPE queries and keys, then intercepts key/value states in DynamicCache.update and reconstructs per-head Algorithm-3 E8 vectors before attention. It stores floating reconstructed tensors and therefore does not prove packed KV-cache memory reduction or speedup.",
                   "Optional A4 quality simulation applies normalized Hadamard, Algorithm-3 E8 fake quantization, and inverse Hadamard to inputs of Llama q/k/v/o and gate/up/down projections. It follows the public upstream hook scope, reconstructs floating activations, and does not prove low-bit GEMM speedup.",
                   "Optional QA-LDLQ uses a separately hashed 20-sequence activation-variance artifact and the public eps=sqrt(x_var/155) weight/Hessian adjustment before this project's block-LDLQ factorization.",
                   "The --use-cache control follows the same DynamicCache attention path without quantizing K/V, separating cache-path numerical effects from KV4 error.",
                   "NestQuant subtractive dither is generated from a CPU RNG stream so per-layer generator states can be serialized and regenerated independent of the model execution device.",
                   "Model beta candidates are [3.5, 4.5, 6.0, 14.5, 25.0, 40.0, 50.0]. Paper Appendix G reports a search universe through 40.0/q and a selected list through 25.0/q; this project adds 40.0 to the selected candidates and extends the reported universe with 50.0.",
                   "Linear quantization scope is recorded in config.excluded_linear_names. When lm_head is quantized and shares parameter storage with input embeddings, both names refer to the same quantized matrix.",
                   "NQMC stores packed W4 codes and dense non-Linear tensors. Evaluation dequantizes weights into BF16 model storage; no low-bit inference acceleration is measured.",
               ], "results": {}, "current_method": None}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, payload)

    try:
        from datasets import load_dataset
        from huggingface_hub import get_token
        from transformers import AutoModelForCausalLM, AutoTokenizer

        token = get_token()
        print(f"Loading WikiText-2 test split ({dataset_revision})...", flush=True)
        dataset = load_dataset(dataset_id, dataset_config, split="test",
                               revision=dataset_revision, download_mode="reuse_dataset_if_exists")
        texts = [text if isinstance(text, str) else "" for text in dataset["text"]]
        print(f"Loading tokenizer for {model_id} at pinned revision...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=model_revision, token=token,
            local_files_only=args.offline, use_fast=True)
        corpus = tokenize_corpus(texts, tokenizer, context_length)
        payload["data"] = {"text_sha256": corpus.text_sha256,
                           "token_ids_sha256": corpus.token_ids_sha256,
                           "token_count_with_boundary_tokens": corpus.token_count,
                           "chunk_count": corpus.chunk_count,
                           "predicted_token_count": corpus.predicted_token_count,
                           "dropped_tail_tokens": corpus.dropped_tail_tokens,
                           "bos_token_id": corpus.bos_token_id,
                           "eos_token_id": corpus.eos_token_id,
                           "dataset_rows": len(texts)}
        if args.limit_chunks is not None:
            if args.limit_chunks < 1:
                raise ValueError("limit_chunks must be positive")
            corpus_tokens = corpus.token_ids[:args.limit_chunks * context_length]
            payload["config"]["effective_limit_chunks"] = min(args.limit_chunks, corpus.chunk_count)
        else:
            corpus_tokens = corpus.token_ids[:corpus.chunk_count * context_length]
        del tokenizer, dataset, texts
        gc.collect()
        print(f"Tokenized evaluation input: {payload['data']['chunk_count']} full chunks, "
              f"{payload['data']['predicted_token_count']} predicted tokens.", flush=True)

        for method in methods:
            method_started = time.perf_counter()
            payload["status"] = "running"
            payload["current_method"] = method
            _write_json(output_path, payload)
            print(f"[{method}] loading BF16 checkpoint...", flush=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, revision=model_revision, token=token,
                torch_dtype=model_dtype, low_cpu_mem_usage=True,
                local_files_only=args.offline)
            model.to(device)
            model.eval()
            input_embedding = model.get_input_embeddings()
            output_embedding = model.get_output_embeddings()
            input_weight = getattr(input_embedding, "weight", None)
            output_weight = getattr(output_embedding, "weight", None)
            payload["model_architecture"] = {
                "tie_word_embeddings_config": bool(getattr(model.config, "tie_word_embeddings", False)),
                "input_output_embedding_share_storage": bool(
                    input_weight is not None and output_weight is not None
                    and input_weight.data_ptr() == output_weight.data_ptr()),
            }
            method_stats = {"method": method, "quality_stage": "bf16_reference" if method == "bf16" else "fake_quant_quality_simulation"}
            if method == "rtn_w4":
                print(f"[{method}] applying group-wise fake quantization...", flush=True)
                method_stats.update(quantize_rtn_w4_(model, group_size=rtn_group_size))
            elif method == "nestquant_w4":
                if args.load_checkpoint is not None:
                    print(f"[{method}] loading existing NQMC checkpoint...", flush=True)
                    method_stats["quality_stage"] = "packed_checkpoint_dequantized_quality"
                    method_stats["checkpoint_round_trip"] = _load_nqmc_into_model(
                        args.load_checkpoint, model, device, model_id, model_revision,
                        q, seed, beta_coefficients, beta_count,
                    )
                else:
                    print(f"[{method}] applying E8/beta-subset fake-quant adapter...", flush=True)
                    hessian_store = None
                    variance_store = None
                    try:
                        ldlq_checkpoint_metadata = None
                        if args.ldlq_hessian_dir is not None:
                            hessian_store = LlamaHessianStore(
                                args.ldlq_hessian_dir,
                                calibration_chunks=args.ldlq_calibration_chunks,
                                expected_model_id=model_id,
                                expected_model_revision=model_revision,
                            )
                            ldlq_checkpoint_metadata = _ldlq_calibration_metadata(
                                args.ldlq_hessian_dir,
                                args.ldlq_calibration_chunks,
                                model_id,
                                model_revision,
                                args.ldlq_relative_damp,
                            )
                        if args.qa_variance_file is not None:
                            variance_store = ActivationVarianceStore(
                                args.qa_variance_file,
                                expected_model_id=model_id,
                                expected_model_revision=model_revision,
                                expected_chunks=args.qa_calibration_chunks,
                            )
                            ldlq_checkpoint_metadata = dict(ldlq_checkpoint_metadata or {})
                            ldlq_checkpoint_metadata["qa_ldlq"] = {
                                "activation_variance_file": str(args.qa_variance_file.resolve()),
                                "activation_variance_sha256": hashlib.sha256(
                                    args.qa_variance_file.read_bytes()).hexdigest(),
                                "calibration_chunks": args.qa_calibration_chunks,
                                "epsilon_constant": args.qa_epsilon_constant,
                                "formula": "eps=sqrt(x_var/epsilon_constant)",
                            }
                        method_stats.update(_quantize_and_optionally_export(
                            model, q, beta_coefficients, beta_count, seed, rows_per_chunk,
                            args.export_checkpoint, payload, output_path, model_id, model_revision, device,
                            rotation_seed=args.rotation_seed,
                            hessian_provider=None if hessian_store is None else hessian_store.resolve,
                            activation_variance_provider=(
                                None if variance_store is None else variance_store.resolve
                            ),
                            qa_epsilon_constant=args.qa_epsilon_constant,
                            ldlq_relative_damp=args.ldlq_relative_damp,
                            ldlq_checkpoint_metadata=ldlq_checkpoint_metadata,
                            excluded_linear_names=args.exclude_linear,
                        ))
                        if hessian_store is not None:
                            method_stats["ldlq_calibration"] = ldlq_checkpoint_metadata
                    finally:
                        if hessian_store is not None:
                            hessian_store.close()
            print(f"[{method}] evaluating perplexity...", flush=True)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                # Large calibration/quantization workspaces can otherwise keep
                # cached allocations resident under Windows GPU memory pressure.
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            cache_stats = None
            cache_factory = None
            activation_stats = None
            activation_handles = []
            if args.a4:
                from ..evaluation.activation_quantization import (
                    ActivationQuantizationStats, enable_activation_quantization,
                )

                activation_stats = ActivationQuantizationStats()
                activation_stats, activation_handles = enable_activation_quantization(
                    model, q=q, betas=activation_betas,
                    rows_per_chunk=args.act_rows_per_chunk, stats=activation_stats,
                )
                method_stats["activation_beta_configuration"] = {
                    "status": "public_upstream_global_defaults",
                    "coefficient_values": list(UPSTREAM_ACTIVATION_BETA_COEFFICIENTS),
                    "beta_values": list(activation_betas),
                    "upstream_file": "parallel/ppl.py",
                }
            if args.kv4:
                from ..evaluation.kv_cache import (
                    KVQuantizationStats, NestQuantDynamicCache,
                    enable_kv_attention_rotation,
                )

                enable_kv_attention_rotation(model)
                cache_stats = KVQuantizationStats()
                cache_factory = partial(
                    NestQuantDynamicCache,
                    config=model.config,
                    q=q,
                    default_betas=provisional_kv_betas,
                    layer_betas=kv_layer_betas,
                    rows_per_chunk=args.kv_rows_per_chunk,
                    stats=cache_stats,
                )
                method_stats["quality_stage"] = "w4_kv4_fake_quant_quality_simulation"
                method_stats["kv_beta_calibration"] = kv_beta_metadata or {
                    "status": "provisional_shared_beta_subset",
                    "beta_values": list(provisional_kv_betas),
                    "note": "Use --kv-beta-config for paper-style per-layer key/value beta subsets.",
                }
            elif args.kv_rotation_control:
                from ..evaluation.kv_cache import (
                    RotatedKeyDynamicCache, enable_kv_attention_rotation,
                )

                enable_kv_attention_rotation(model)
                cache_factory = partial(RotatedKeyDynamicCache, config=model.config)
                method_stats["evaluation_cache_control"] = "matched_post_rope_qk_hadamard"
            elif args.use_cache:
                from transformers.cache_utils import DynamicCache

                cache_factory = partial(DynamicCache, config=model.config)
                method_stats["evaluation_cache_control"] = "floating_dynamic_cache"
            last_reported_chunk = 0

            def report_evaluation_progress(completed: int, total: int) -> None:
                nonlocal last_reported_chunk
                if completed == total or completed - last_reported_chunk >= 10:
                    print(
                        f"[{method}] evaluation progress: {completed}/{total} chunks",
                        flush=True,
                    )
                    last_reported_chunk = completed

            measurement = evaluate_perplexity(
                model, corpus_tokens, context_length=context_length, device=device,
                cache_factory=cache_factory, batch_size=args.eval_batch_size,
                progress_callback=report_evaluation_progress,
            )
            if cache_stats is not None:
                method_stats["kv_cache_quantization"] = cache_stats.summary()
            if activation_stats is not None:
                method_stats["activation_quantization"] = activation_stats.summary()
                method_stats["quality_stage"] = (
                    "w4_a4_kv4_fake_quant_quality_simulation"
                    if args.kv4 else "w4_a4_fake_quant_quality_simulation"
                )
            for handle in activation_handles:
                handle.remove()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                measurement["peak_cuda_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                measurement["peak_cuda_memory_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
            method_stats["evaluation"] = measurement
            method_stats["elapsed_seconds"] = time.perf_counter() - method_started
            payload["results"][method] = method_stats
            payload["current_method"] = None
            payload["status"] = "partial" if args.limit_chunks else "running"
            _write_json(output_path, payload)
            print(f"[{method}] PPL={measurement['perplexity']:.6f}; "
                  f"elapsed={method_stats['elapsed_seconds']:.1f}s", flush=True)
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        payload["status"] = "completed" if args.limit_chunks is None else "smoke_completed"
    except Exception as exc:
        payload["status"] = "failed"
        payload["failed_method"] = payload.get("current_method")
        payload["error"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_json(output_path, payload)
        if "gated repo" in str(exc).lower() or "restricted" in str(exc).lower():
            print("The selected model is gated. Request/accept access on Hugging Face, then run `hf auth login` locally. Do not paste the token into this chat.", file=sys.stderr)
        raise
    finally:
        payload["current_method"] = None
        payload["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(output_path, payload)
    print(f"Saved run manifest: {output_path}")
    for method, result in payload["results"].items():
        print(f"{method}: PPL={result['evaluation']['perplexity']:.6f} "
              f"over {result['evaluation']['predicted_token_count']} predicted tokens")


def _quantize_and_optionally_export(
    model, q, beta_coefficients, beta_count, seed, rows_per_chunk,
    export_path, payload, output_path, model_id, model_revision, device,
    rotation_seed=None,
    hessian_provider=None,
    activation_variance_provider=None,
    qa_epsilon_constant=155.0,
    ldlq_relative_damp=0.01,
    ldlq_checkpoint_metadata=None,
    excluded_linear_names=(),
) -> dict:
    packed_tensors: dict = {}
    payload["quantization_progress"] = {"completed_layers": 0, "layer_timings": {}}

    def record_layer_progress(record: dict) -> None:
        progress = payload["quantization_progress"]
        progress["completed_layers"] = record["completed_layers"]
        progress["total_layers"] = record["total_layers"]
        progress["layer_timings"][record["layer"]] = record["timing_seconds"]
        _write_json(output_path, payload)
        times = record["timing_seconds"]
        print(f"[nestquant_w4] {record['completed_layers']}/{record['total_layers']} "
              f"{record['layer']} {record['shape']}: total={times['total_seconds']:.1f}s "
              f"candidates={times['candidate_seconds']:.1f}s "
              f"search={times['beta_search_seconds']:.1f}s "
              f"quantize={times['quantize_reconstruct_seconds']:.1f}s "
              f"packing={times.get('packing_seconds', 0.0):.1f}s", flush=True)

    method_stats = quantize_nestquant_w4_(
        model, q=q, beta_coefficients=beta_coefficients,
        selected_beta_count=beta_count, seed=seed, rows_per_chunk=rows_per_chunk,
        output_artifacts=packed_tensors if export_path else None,
        progress_callback=record_layer_progress,
        rotation_seed=rotation_seed,
        hessian_provider=hessian_provider,
        activation_variance_provider=activation_variance_provider,
        qa_epsilon_constant=qa_epsilon_constant,
        ldlq_relative_damp=ldlq_relative_damp,
        excluded_linear_names=excluded_linear_names,
    )
    payload.pop("quantization_progress", None)
    if export_path is None:
        return method_stats
    print("[nestquant_w4] serializing model tensors into NQMC...", flush=True)
    encoder = (
        "qa_block_ldlq" if activation_variance_provider is not None else
        "block_ldlq" if hessian_provider is not None else
        "independent_e8_blocks"
    )
    model_container = PackedNestQuantModel.from_state_dict(
        model.state_dict(), packed_tensors,
        model_metadata={"model_id": model_id, "model_revision": model_revision,
                        "model_config": json.loads(model.config.to_json_string()),
                        "method": "nestquant_w4_quality_simulation", "q": q,
                        "seed": seed, "dither_rng_device": "cpu",
                        "encoder": encoder,
                        "encoder_metadata": ldlq_checkpoint_metadata,
                        "torch_version": torch.__version__},
    )
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_path.write_bytes(model_container.to_bytes())
    del model_container, packed_tensors
    gc.collect()
    print(f"[nestquant_w4] reloading serialized state from {export_path}...", flush=True)
    method_stats["checkpoint_round_trip"] = _load_nqmc_into_model(
        export_path, model, device, model_id, model_revision,
        q, seed, beta_coefficients, beta_count,
    )
    method_stats["quality_stage"] = "packed_checkpoint_dequantized_quality"
    return method_stats


def _load_nqmc_into_model(path: Path, model, device, model_id: str, model_revision: str,
                          q: int, seed: int, beta_coefficients, beta_count: int) -> dict:
    blob = path.read_bytes()
    blob_sha256 = hashlib.sha256(blob).hexdigest()
    container = PackedNestQuantModel.from_bytes(blob)
    metadata = container.model_metadata
    if metadata.get("model_id") != model_id or metadata.get("model_revision") != model_revision:
        raise ValueError("NQMC model ID or revision does not match the evaluation config")
    if metadata.get("method") != "nestquant_w4_quality_simulation":
        raise ValueError("NQMC method is not NestQuant W4 quality simulation")
    encoder = metadata.get("encoder", "independent_e8_blocks")
    if encoder not in {"independent_e8_blocks", "block_ldlq", "qa_block_ldlq"}:
        raise ValueError(f"unsupported NQMC encoder: {encoder}")
    if encoder in {"block_ldlq", "qa_block_ldlq"} and not isinstance(
        metadata.get("encoder_metadata"), dict
    ):
        raise ValueError(f"{encoder} NQMC is missing encoder metadata")
    if metadata.get("q") != q or metadata.get("seed") != seed:
        raise ValueError("NQMC q or dither seed does not match the evaluation config")
    allowed_betas = {value / q for value in beta_coefficients}
    if any(len(record.betas) != beta_count or any(beta not in allowed_betas for beta in record.betas)
           for record in container.quantized_tensors.values()):
        raise ValueError("NQMC selected beta values do not match the evaluation candidate grid")
    restored = container.state_dict(device=device)
    load_result = model.load_state_dict(restored, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(f"state-dict mismatch; missing={load_result.missing_keys}, "
                           f"unexpected={load_result.unexpected_keys}")
    result = {"format": "NQMC v1", "path": str(path.resolve()),
              "file_bytes": len(blob), "file_sha256": blob_sha256,
              "tensor_count": len(restored),
              "unique_quantized_tensors": len(container.quantized_tensors),
              "encoder": encoder,
              "encoder_metadata": metadata.get("encoder_metadata"),
              "reloaded_before_perplexity": True,
              "execution": "floating-point dequantized weights loaded into model dtype"}
    del restored, container, blob
    gc.collect()
    return result


def _ldlq_calibration_metadata(
    directory: Path,
    chunks: int,
    model_id: str,
    model_revision: str,
    relative_damp: float,
) -> dict:
    """Validate lightweight Hessian manifests and return checkpoint metadata."""
    expected_names = [
        *(f"hessian_layers{start:02d}_{start + 3:02d}_train{chunks}.pt"
          for start in range(0, 16, 4)),
        f"hessian_lm_head_train{chunks}.pt",
    ]
    artifacts = []
    selected_hash = None
    corpus_hash = None
    dataset = None
    dataset_revision = None
    context_length = None
    module_count = 0
    for name in expected_names:
        tensor_path = directory / name
        manifest_path = tensor_path.with_suffix(tensor_path.suffix + ".json")
        if not tensor_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"incomplete LDLQ calibration artifact: {tensor_path}")
        manifest_blob = manifest_path.read_bytes()
        manifest = json.loads(manifest_blob.decode("utf-8"))
        if manifest.get("model_id") != model_id or manifest.get("model_revision") != model_revision:
            raise ValueError(f"{manifest_path}: model identity mismatch")
        if manifest.get("chunk_count") != chunks:
            raise ValueError(f"{manifest_path}: calibration chunk mismatch")
        current_context = manifest.get("context_length")
        if not isinstance(current_context, int) or current_context < 1:
            raise ValueError(f"{manifest_path}: invalid context length")
        current_dataset = manifest.get("dataset")
        current_dataset_revision = manifest.get("dataset_revision")
        if not isinstance(current_dataset, str) or not isinstance(current_dataset_revision, str):
            raise ValueError(f"{manifest_path}: invalid dataset identity")
        if manifest.get("output_bytes") != tensor_path.stat().st_size:
            raise ValueError(f"{manifest_path}: Hessian byte size mismatch")
        current_selected = manifest.get(
            "calibration_token_ids_sha256", manifest.get("train_token_ids_sha256")
        )
        current_corpus = manifest.get("train_corpus_token_ids_sha256")
        if not isinstance(current_selected, str) or len(current_selected) != 64:
            raise ValueError(f"{manifest_path}: invalid calibration token hash")
        if selected_hash is None:
            selected_hash, corpus_hash = current_selected, current_corpus
            dataset, dataset_revision = current_dataset, current_dataset_revision
            context_length = current_context
        elif (current_selected != selected_hash or current_corpus != corpus_hash
              or current_dataset != dataset or current_dataset_revision != dataset_revision
              or current_context != context_length):
            raise ValueError(f"{manifest_path}: calibration protocol metadata disagrees")
        modules = manifest.get("modules")
        if not isinstance(modules, dict):
            raise ValueError(f"{manifest_path}: invalid module metadata")
        module_count += len(modules)
        artifacts.append({
            "file": name,
            "bytes": manifest["output_bytes"],
            "sha256": manifest["output_sha256"],
            "manifest_sha256": hashlib.sha256(manifest_blob).hexdigest(),
            "module_count": len(modules),
        })
    if module_count != 65:
        raise ValueError(f"expected 65 unique Hessians, found {module_count}")
    return {
        "algorithm": "8-column E8 block-LDLQ",
        "block_size": 8,
        "relative_damp": relative_damp,
        "chunks": chunks,
        "dataset": dataset,
        "dataset_revision": dataset_revision,
        "context_length": context_length,
        "calibration_tokens": chunks * context_length,
        "calibration_token_ids_sha256": selected_hash,
        "train_corpus_token_ids_sha256": corpus_hash,
        "hessian_artifacts": artifacts,
        "hessian_bytes_total": sum(item["bytes"] for item in artifacts),
    }


def _load_kv_beta_config(
    path: Path,
    *,
    model_id: str,
    model_revision: str,
    q: int,
    selected_beta_count: int,
) -> tuple[dict[int, dict[str, tuple[float, ...]]], dict]:
    """Validate a calibrated per-layer K/V beta configuration."""
    blob = path.read_bytes()
    payload = json.loads(blob.decode("utf-8"))
    if payload.get("status") != "completed":
        raise ValueError("KV beta config is not marked completed")
    if payload.get("model_id") != model_id or payload.get("model_revision") != model_revision:
        raise ValueError("KV beta config model identity mismatch")
    if payload.get("q") != q or payload.get("selected_beta_count") != selected_beta_count:
        raise ValueError("KV beta config q or beta count mismatch")
    raw_layers = payload.get("layers")
    if not isinstance(raw_layers, dict) or not raw_layers:
        raise ValueError("KV beta config must contain layer entries")
    layers: dict[int, dict[str, tuple[float, ...]]] = {}
    for raw_index, raw_layer in raw_layers.items():
        try:
            layer_index = int(raw_index)
        except (TypeError, ValueError) as error:
            raise ValueError("KV beta layer indices must be integers") from error
        if layer_index < 0 or not isinstance(raw_layer, dict):
            raise ValueError("invalid KV beta layer entry")
        layers[layer_index] = {}
        for kind in ("key", "value"):
            raw_kind = raw_layer.get(kind)
            if not isinstance(raw_kind, dict):
                raise ValueError(f"layer {layer_index} is missing {kind} beta metadata")
            try:
                betas = tuple(float(value) for value in raw_kind["beta_values"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"layer {layer_index} has invalid {kind} beta values") from error
            if (len(betas) != selected_beta_count
                    or any(not math.isfinite(value) or value <= 0 for value in betas)
                    or any(left >= right for left, right in zip(betas, betas[1:]))):
                raise ValueError(f"layer {layer_index} has invalid {kind} beta subset")
            layers[layer_index][kind] = betas
    metadata = {
        key: payload.get(key) for key in (
            "dataset", "dataset_config", "dataset_revision", "split",
            "context_length", "calibration_chunks", "token_ids_sha256",
            "candidate_beta_values", "selected_beta_count", "sample_limit_per_tensor",
        )
    } | {
        "status": "calibrated_per_layer_key_value_betas",
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "layer_count": len(layers),
    }
    return layers, metadata


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _package_versions(distributions: tuple[str, ...]) -> dict[str, str | None]:
    versions = {}
    for name in distributions:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _source_provenance(root: Path) -> dict:
    def git_value(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *arguments], check=True,
                capture_output=True, text=True, timeout=5)
            return result.stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None

    revision = git_value("rev-parse", "HEAD")
    status = git_value("status", "--porcelain")
    digest = hashlib.sha256()
    for path in sorted((root / "nestquant").rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {"git_commit": revision,
            "nestquant_python_source_sha256": digest.hexdigest(),
            "working_tree_dirty": None if status is None else bool(status),
            "note": "Commit is null when this checkout has no commit history; dirty includes staged, unstaged, and untracked files."}


if __name__ == "__main__":
    main()
