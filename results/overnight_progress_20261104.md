# Overnight progress — 2026-11-04

## Completed without route decisions

- Added same-model eager and NQKV Graph per-token logit diagnosis.
- Reproduced the HTTP divergence at runner level and localized the first causal
  argmax flip.
- Swept all 16 eligible MLP down projections for the original prompt.
- Added an eight-prompt, three-token policy screen.
- Identified input-dependent sensitive layers: 1, 10, 11 and 14 across the
  observed failures.
- Verified a four-layer fallback with split-K active in the other 12 layers:
  8/8 prompt sequences matched the unsplit baseline in the small screen.
- Added explicit per-module fallback to conversion, service and benchmark APIs.
- Completed a real HTTP smoke: 8/8 requests, zero failures, identical
  `482,15704,13` output for every request.
- Kept the production default fully unsplit.
- Full WSL/CUDA regression: 413 tests and 64 subtests passed.
- Added a deterministic 12-prompt token-ID screen across lengths 2 to 64.
- Found a counterexample to static fallback generalization: all-layer split-K
  matched one input while the four-layer fallback diverged.
- Rejected static layer blacklisting as the primary correctness solution.

## Evidence boundary

The four-layer fallback is not proven universally correct. The screen contains
eight hand-selected prompts and three greedy tokens per prompt. Logits may differ
even when tokens match, and sensitive layers can vary with input, position,
batching and cache state. The HTTP run is an integration smoke, not a paired
performance experiment.

## Decisions reserved for the user

1. Whether the target contract should require bitwise/logit equivalence, greedy
   token equivalence on a defined suite, or bounded quality degradation. These
   lead to different admissible optimizations.
2. Whether to return to a numerically stable split-K reduction that aims to
   eliminate prompt-dependent fallbacks, or explicitly adopt a bounded-quality
   contract that permits different greedy token paths. The static blacklist is
   no longer recommended as the primary solution.
3. Only after a broader correctness gate: whether to spend GPU time on paired
   HTTP performance A/B for the 12-layer split-K hybrid.

Recommended current position: keep the default unsplit, retain the hybrid only
as a diagnostic branch, and do not spend more optimization effort on expanding
a static blacklist before the correctness contract is selected.

## Route A started

After the user selected numerical stability, raw-FP32 probes and a 32-activation
split-2/4/8 screen were added. All splits showed roughly 3e-6 mean maximum layer
error; reducing split count did not materially improve stability and reduced
isolated speedup from 2.60x to 1.83x. The next diagnostic separates partial-slice
accumulation from final reduction.

The partial probe then showed that FP64, Kahan and pairwise final reductions do
not materially reduce the roughly 2.9e-6 error. The dominant difference is
already present inside independently accumulated K partitions; final-reducer
replacement is rejected, and the next probe must modify partition accumulation.
