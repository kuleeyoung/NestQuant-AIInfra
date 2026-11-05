# M1 split-K multi-prompt correctness screen

## Protocol

Eight deterministic prompts covered short text, factual completion, list
continuation, arithmetic, Python code, reasoning, longer context and Chinese.
Each prompt generated three greedy tokens through the production capacity-4
NQKV4 CUDA Graph runner under three policies:

1. unsplit baseline;
2. split-K in all 16 eligible MLP down projections;
3. a static fallback policy that keeps selected sensitive layers unsplit.

Every policy/prompt pair received a newly captured CUDA Graph. Logit error is
only attributed while both policies still receive identical token input.

## Layer-10-only fallback screen

The first hybrid kept only `model.layers.10.mlp.down_proj` unsplit. It improved
token-equal prompts from 3/8 to 5/8, repairing `capital_list` and `reasoning`, but
failed on `short_greeting`, `python_code` and `long_context`.

Targeted single-layer sweeps then found input-dependent sensitive layers:

| prompt | single layers that independently flipped a token |
|---|---|
| `capital_list` | 10 |
| `short_greeting` | 11, 14 |
| `python_code` | 1 |
| `long_context` | 1 |

This falsifies the hypothesis that layer 10 is a universal static fallback.

## Four-layer fallback screen

The second hybrid kept layers `{1,10,11,14}` unsplit and enabled split-K in the
other 12 eligible layers.

| prompt | baseline tokens | all-layer split-K | four-layer fallback |
|---|---|---|---|
| short greeting | `11,358,2846` | `11,358,1097` | `11,358,2846` |
| short factual | `755,7627,539` | equal | equal |
| capital list | `482,15704,13` | `482,220,679` | `482,15704,13` |
| arithmetic | `755,7627,539` | equal | equal |
| Python code | `457,791,2768` | `457,791,3485` | `457,791,2768` |
| reasoning | `11,323,430` | `11,323,279` | `11,323,430` |
| long context | `264,220,16` | `264,220,679` | `264,220,16` |
| Chinese | `320,16,8` | equal | equal |

All-layer split-K matched 3/8 prompts. The four-layer fallback matched 8/8.

## Evidence boundary and decision

The 8/8 result is a development gate, not a universal correctness proof. The
prompts are hand-selected, only three greedy tokens were generated, logits can
still differ, and sensitivity can change with prompt, decode position, batch
composition and cache state. The production default remains fully unsplit.

The next non-default implementation step is an explicit per-module fallback API
plus a larger deterministic token-ID screen. Default promotion remains a future
decision requiring broader correctness and paired performance evidence.

A subsequent deterministic 12-prompt token-ID screen found only 8/12 equality
for the four-layer fallback and one case where all-layer split-K matched while
the fallback diverged. This falsifies the static-blacklist generalization from
the hand-selected suite. See
`results/w4a4_m1_down_splitk_token_screen_v1.md`.

The explicit API was subsequently threaded through model conversion, the HTTP
server and fresh-process benchmark. A one-process HTTP smoke with fallback layers
`{1,10,11,14}` completed 8/8 requests with zero failures; every request generated
`482,15704,13`. Its TPOT/TTFT values are integration observations only, not an A/B
performance claim.

Machine-readable artifacts:

- `results/raw/m1_splitk_prompt_suite_v1.json`
- `results/raw/m1_splitk_prompt_suite_fallback_1_10_11_14_v1.json`
- `results/raw/m1_splitk_layer_sweep_short_greeting_v1.json`
- `results/raw/m1_splitk_layer_sweep_python_code_v1.json`
- `results/raw/m1_splitk_layer_sweep_long_context_v1.json`
- `results/raw/http_m1_splitk_hybrid_smoke_v1_summary.json`
- `results/http_m1_splitk_hybrid_smoke_v1.md`
