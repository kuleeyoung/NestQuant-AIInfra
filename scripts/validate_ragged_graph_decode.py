"""Validate divergent per-request positions on the packed 1B CUDA Graph path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nestquant.core.model_checkpoint_file import NQMCFileReader
from nestquant.models.fused_linear import convert_linears_to_fused_nqrt_
from nestquant.runtime.hybrid_residency import (
    residency_groups_from_payload,
    select_residency_plan,
)
from nestquant.runtime.ragged_graph_decode import NQRTRaggedGraphDecodeRunner
from scripts.benchmark_full_nqrt_model import _load_model, _projection_names


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--max-cache-len", type=int, default=64)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--hybrid-profile", type=Path)
    parser.add_argument("--hybrid-config", type=Path)
    parser.add_argument("--budget-mib", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    hybrid_values = (args.hybrid_profile, args.hybrid_config, args.budget_mib)
    if any(value is not None for value in hybrid_values) and not all(
        value is not None for value in hybrid_values
    ):
        parser.error("hybrid-profile, hybrid-config and budget-mib must be supplied together")
    if args.budget_mib is not None and args.budget_mib < 0:
        parser.error("budget-mib must be nonnegative")

    reader = NQMCFileReader(args.checkpoint, verify_checksum=False)
    model = _load_model(reader.model_metadata["model_id"],
                        reader.model_metadata["model_revision"], args.offline)
    names = _projection_names(model, reader)
    resident_names: tuple[str, ...] = ()
    packed_names = names
    if args.hybrid_profile is not None:
        profile = json.loads(args.hybrid_profile.read_text(encoding="utf-8"))
        config = json.loads(args.hybrid_config.read_text(encoding="utf-8"))
        if args.budget_mib not in config["budget_mib"]:
            raise ValueError("budget-mib is not present in the hybrid config")
        groups = residency_groups_from_payload(profile["groups_ranked"])
        plan = select_residency_plan(
            groups,
            budget_bytes=args.budget_mib * 1024 * 1024,
            all_projection_names=names,
        )
        resident_names = plan.resident_projection_names
        packed_names = plan.packed_projection_names
    if packed_names:
        convert_linears_to_fused_nqrt_(
            model, reader, packed_names, output_dtype=torch.bfloat16,
            group_qkv=False, compute_backend="cuda_mma_j2",
        )
    runner = NQRTRaggedGraphDecodeRunner(
        model, batch_size=2, max_cache_len=args.max_cache_len
    )
    generator = torch.Generator(device="cuda").manual_seed(20261026)
    prompt = torch.randint(
        0, model.config.vocab_size, (2, args.sequence_length),
        device="cuda", generator=generator,
    )
    tokens = torch.randint(
        0, model.config.vocab_size, (2, 2), device="cuda", generator=generator
    )

    runner.prefill(prompt)
    capture_positions = torch.full(
        (2, 1), args.sequence_length, dtype=torch.long, device="cuda"
    )
    runner.capture(
        tokens[:, :1], positions=capture_positions,
        active_mask=torch.ones(2, 1, dtype=torch.bool, device="cuda"),
        valid_lengths=torch.full(
            (2,), args.sequence_length + 1, dtype=torch.long, device="cuda"
        ),
    )

    # Reset capture side effects. Row 0 then advances alone; row 1 remains paused.
    runner.prefill(prompt)
    paused_active = torch.tensor([[True], [False]], device="cuda")
    paused_lengths = torch.tensor(
        [args.sequence_length + 1, args.sequence_length], device="cuda"
    )
    runner.decode(
        tokens[:, :1], positions=capture_positions,
        active_mask=paused_active, valid_lengths=paused_lengths,
    )

    # Next replay uses divergent positions: row 0 at L+1, row 1 at L.
    divergent_positions = torch.tensor(
        [[args.sequence_length + 1], [args.sequence_length]], device="cuda"
    )
    divergent_lengths = divergent_positions[:, 0] + 1
    graph_logits = runner.decode(
        tokens[:, 1:2], positions=divergent_positions,
        active_mask=torch.ones(2, 1, dtype=torch.bool, device="cuda"),
        valid_lengths=divergent_lengths,
    )[:, -1].clone()

    reference_0 = model(
        input_ids=torch.cat((prompt[0:1], tokens[0:1, :1], tokens[0:1, 1:2]), dim=1)
    ).logits[0, -1]
    reference_1 = model(
        input_ids=torch.cat((prompt[1:2], tokens[1:2, 1:2]), dim=1)
    ).logits[0, -1]
    references = torch.stack((reference_0, reference_1))

    # Repeat the same state transition without Graph replay to localize failures.
    runner.prefill(prompt)
    runner.decode_eager(
        tokens[:, :1], positions=capture_positions,
        active_mask=paused_active, valid_lengths=paused_lengths,
    )
    eager_logits = runner.decode_eager(
        tokens[:, 1:2], positions=divergent_positions,
        active_mask=torch.ones(2, 1, dtype=torch.bool, device="cuda"),
        valid_lengths=divergent_lengths,
    )[:, -1].clone()

    # Slot reuse: A advances, B is released, and a shorter new request C is
    # prefetched into B's physical row without modifying A's cache.
    runner.prefill(prompt)
    both_active = torch.ones(2, 1, dtype=torch.bool, device="cuda")
    first_lengths = torch.full(
        (2,), args.sequence_length + 1, dtype=torch.long, device="cuda"
    )
    runner.decode(
        tokens[:, :1], positions=capture_positions,
        active_mask=both_active, valid_lengths=first_lengths,
    )
    runner.clear_slots(torch.tensor([False, True], device="cuda"))
    new_prompt = torch.randint(
        0, model.config.vocab_size, (1, 3), device="cuda", generator=generator
    )
    slot_prefill_ids = torch.cat((torch.zeros_like(new_prompt), new_prompt), dim=0)
    slot_prefill_positions = torch.tensor([[0, 0, 0], [0, 1, 2]], device="cuda")
    slot_prefill_mask = torch.tensor(
        [[False, False, False], [True, True, True]], device="cuda"
    )
    runner.prefill_slots(
        slot_prefill_ids,
        positions=slot_prefill_positions,
        write_mask=slot_prefill_mask,
        valid_lengths=torch.tensor([args.sequence_length + 1, 3], device="cuda"),
    )
    new_token = torch.randint(
        0, model.config.vocab_size, (1, 1), device="cuda", generator=generator
    )
    reused_tokens = torch.cat((tokens[0:1, 1:2], new_token), dim=0)
    reused_positions = torch.tensor(
        [[args.sequence_length + 1], [3]], device="cuda"
    )
    reused_lengths = reused_positions[:, 0] + 1
    reused_logits = runner.decode(
        reused_tokens, positions=reused_positions,
        active_mask=both_active, valid_lengths=reused_lengths,
    )[:, -1].clone()
    reused_reference_0 = model(
        input_ids=torch.cat((prompt[0:1], tokens[0:1, :2]), dim=1)
    ).logits[0, -1]
    reused_reference_1 = model(
        input_ids=torch.cat((new_prompt, new_token), dim=1)
    ).logits[0, -1]
    reused_references = torch.stack((reused_reference_0, reused_reference_1))

    runner.prefill(prompt)
    runner.decode_eager(
        tokens[:, :1], positions=capture_positions,
        active_mask=both_active, valid_lengths=first_lengths,
    )
    runner.clear_slots(torch.tensor([False, True], device="cuda"))
    runner.prefill_slots(
        slot_prefill_ids,
        positions=slot_prefill_positions,
        write_mask=slot_prefill_mask,
        valid_lengths=torch.tensor([args.sequence_length + 1, 3], device="cuda"),
    )
    reused_eager_logits = runner.decode_eager(
        reused_tokens, positions=reused_positions,
        active_mask=both_active, valid_lengths=reused_lengths,
    )[:, -1].clone()
    torch.cuda.synchronize()
    errors = (graph_logits - references).abs().amax(dim=1)
    eager_errors = (eager_logits - references).abs().amax(dim=1)
    graph_eager_errors = (graph_logits - eager_logits).abs().amax(dim=1)
    reused_errors = (reused_logits - reused_references).abs().amax(dim=1)
    reused_graph_eager_errors = (reused_logits - reused_eager_logits).abs().amax(dim=1)
    reused_argmax_equal = reused_logits.argmax(dim=1) == reused_references.argmax(dim=1)
    graph_argmax = graph_logits.argmax(dim=1)
    reference_argmax = references.argmax(dim=1)
    result = {
        "status": "passed",
        "device": torch.cuda.get_device_name(),
        "sequence_length": args.sequence_length,
        "hybrid_budget_mib": args.budget_mib,
        "resident_projection_count": len(resident_names),
        "packed_projection_count": len(packed_names),
        "positions": divergent_positions[:, 0].tolist(),
        "valid_lengths": divergent_lengths.tolist(),
        "per_request_max_abs_error": errors.tolist(),
        "eager_per_request_max_abs_error": eager_errors.tolist(),
        "graph_vs_ragged_eager_max_abs_error": graph_eager_errors.tolist(),
        "per_request_argmax_equal": (graph_argmax == reference_argmax).tolist(),
        "slot_reuse": {
            "positions": reused_positions[:, 0].tolist(),
            "valid_lengths": reused_lengths.tolist(),
            "per_request_max_abs_error": reused_errors.tolist(),
            "graph_vs_ragged_eager_max_abs_error": reused_graph_eager_errors.tolist(),
            "per_request_argmax_equal": reused_argmax_equal.tolist(),
        },
        "claim_boundary": (
            "Two-request divergent-position and slot-reuse correctness probe for the selected "
            "packed/BF16 residency plan. It is not a latency benchmark or production service test."
        ),
    }
    if (float(graph_eager_errors.max().item()) != 0.0
            or float(errors.max().item()) > 0.25
            or not bool(torch.all(graph_argmax == reference_argmax))
            or float(reused_graph_eager_errors.max().item()) != 0.0
            or float(reused_errors.max().item()) > 1.0
            or not bool(torch.all(reused_argmax_equal))):
        result["status"] = "failed"
        raise AssertionError(json.dumps(result))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
