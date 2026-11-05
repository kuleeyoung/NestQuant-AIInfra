# M1 split-K deterministic token-ID screen

## Purpose

The hand-selected eight-prompt screen found a four-layer fallback that matched
all observed greedy tokens. This follow-up checks whether that result generalizes
to broader activation patterns without claiming natural-language quality.

## Protocol

A fixed RNG seed (`20261104`) generated 12 legal token-ID prompts: two samples
at each length `{2,4,8,16,32,64}`. Every prompt generated three greedy tokens in
the capacity-4 NQKV4 CUDA Graph runner under:

1. fully unsplit baseline;
2. split-K in all 16 eligible down projections;
3. split-K in 12 layers with `{1,10,11,14}` kept unsplit.

Each policy/prompt pair received a new Graph capture. Direct logit-error summaries
stop when off/on token inputs diverge.

## Results

| policy | token-equal prompts | failures |
|---|---:|---:|
| all-layer split-K | 6/12 | 6 |
| four-layer fallback | 8/12 | 4 |

Hybrid equality by prompt length was 1/2, 1/2, 1/2, 2/2, 2/2 and 1/2 for
lengths 2, 4, 8, 16, 32 and 64 respectively. The small sample does not support a
length trend.

The decisive counterexample is `len2_sample1`: all-layer split-K matched the
baseline tokens `755,275,560`, while the four-layer fallback diverged at the
third token. Therefore removing more split-K layers is not monotonically safer.
Layer perturbations can cancel as well as compound; a static blacklist selected
from one prompt suite can break an input that the all-layer policy happened to
preserve.

The four hybrid failures were:

- `len2_sample1`;
- `len4_sample1`;
- `len8_sample0`;
- `len64_sample0`.

## Decision impact

The four-layer fallback remains useful as a diagnostic control but is rejected
as the primary correctness solution. Expanding a static blacklist risks a
combinatorial, prompt-dependent search and cannot establish universal token
equivalence. The production default remains fully unsplit.

Future promotion requires one of two explicit contracts:

- redesign the split-K reduction for stronger numerical stability and rerun the
  same screens; or
- accept non-identical token paths under a bounded model-quality contract rather
  than presenting the kernel as output-equivalent.

Raw artifact: `results/raw/m1_splitk_token_screen_v1.json`

Evidence boundary: deterministic synthetic token IDs broaden length and
activation coverage but are not natural-language evaluation, statistical
confidence, or a universal proof.
