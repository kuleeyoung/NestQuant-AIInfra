# Llama-3.2-1B QA-LDLQ ablation

## Result

Under the local covariance-scaled Hessian and relative damping 0.01, adding the public QA-LDLQ formula with `epsilon_constant=155` made perplexity worse. The degradation already appears in W4-only, so A4/KV4 is not the root cause.

| Scope | Ordinary block-LDLQ PPL | QA block-LDLQ PPL | Absolute change | Relative change |
|---|---:|---:|---:|---:|
| W4 | 10.276601 | 11.053594 | +0.776993 | +7.56% |
| W4A4KV4 | 11.578171 | 12.386133 | +0.807962 | +6.98% |

All four results use Llama-3.2-1B Base revision `4e20de362430cd3b72f300e6b0f18e50e7166e08`, WikiText-2 test revision `b08601e04326c79dfdd32d625aee71d232d685c3`, context 2048, 141 complete non-overlapping windows, 288,627 predicted tokens, and token-id SHA-256 `54927bf19e6e64b4008eeb93370c370fb25db18cf68eac8015cfe094c464d567`.

## QA configuration

- Hessian calibration: 64 WikiText-2 train chunks, covariance-scale statistics, block size 8, relative damping 0.01.
- Activation variance calibration: 20 train chunks / 40,960 tokens / 112 Transformer projections.
- QA formula: `epsilon^2=x_var/155`, followed by the adjusted weight and `H+epsilon^2 I` block-LDLQ path.
- QA NQMC: 659,281,687 bytes, SHA-256 `23dcebae6d2874499f831b0244f1ac3b51ed3264b18ee63e84ab6f3b54fd9955`.
- `lm_head` uses ordinary block-LDLQ, consistent with the scope of the public weight script.

## Comparison caveat

The baseline checkpoint used candidate coefficients `[3.5, 4.5, 6.0, 14.5, 25.0, 40.0, 50.0]`. QA adjustment created one block in `model.layers.1.mlp.down_proj` still overloaded at `50/q`, so the QA run used a disclosed `100/q` extension. Only that layer selected 100. The comparison therefore is not a perfectly isolated one-factor experiment, although the extension was required to produce a valid no-overload QA checkpoint. Source hashes also differ because QA support was added after the baseline run. Model, evaluation data, tokens, A4/KV4 settings, and calibration Hessians remain fixed.

## Interpretation

The result does not refute the paper's QA-LDLQ result. It rejects a narrower local hypothesis: the upstream constant 155 can be transferred directly into this project's covariance-scaled, additionally damped block-LDLQ implementation. The next scientifically useful experiment is an epsilon/scale sweep with a cheap layer-level proxy and short PPL screening before any new full checkpoint export.

Raw evidence:

- Ordinary W4: `results/raw/llama32_1b_w4_ldlq_cache_control_batch4_full_20261021.json`
- Ordinary W4A4KV4: `results/raw/llama32_1b_w4a4kv4_upstream_betas_batch4_full_20261021.json`
- QA W4: `results/raw/llama32_1b_w4_qa_ldlq_batch4_full_20261022.json`
- QA W4A4KV4: `results/raw/llama32_1b_w4a4kv4_qa_ldlq_batch4_full_20261022.json`
- QA export smoke: `results/raw/llama32_1b_w4a4kv4_qa_ldlq_export_smoke.json`

## Scale-corrected follow-up

Source audit found a concrete normalization mismatch. The public Hessian is `sum(X.T @ X) / sequence_count`, while the local Hessian is `sum(X.T @ X) / activation_row_count`. With context length 2048, the public Hessian is 2048 times the local matrix. Because both paths compute `x_var` per activation element, the upstream constant 155 corresponds to local constant 317440.

Four-layer proxy screening tested ordinary LDLQ and constants 155, 158720, 317440, and 634880. Constant 155 produced a 73.7% relative weight adjustment and 9.07x proxy loss on `model.layers.1.mlp.down_proj`. Corrected candidates removed this outlier. Constant 634880 minimized the worst proxy regression and was the only corrected candidate promoted to a full checkpoint, respecting the bounded-search protocol.

| Scope | Ordinary block-LDLQ PPL | C=155 PPL | C=634880 PPL | C=634880 vs ordinary |
|---|---:|---:|---:|---:|
| W4 | 10.276601 | 11.053594 | 10.283140 | +0.006539 / +0.064% |
| W4A4KV4 | 11.578171 | 12.386133 | 11.584234 | +0.006063 / +0.052% |

The corrected checkpoint selected no beta-100 entries. It is 659,281,690 bytes and has SHA-256 `8854812a4fc097890590bc38e19b5aea530685e12a9a685c1acce8a0d9421a8d`.

Additional evidence:

- Scale audit: `results/raw/qa_ldlq_scale_audit.json`
- Layer sweep: `results/raw/qa_ldlq_layer_sweep.json`
- Corrected export smoke: `results/raw/llama32_1b_w4_qa_ldlq_c634880_export_smoke4.json`
- Corrected W4 full: `results/raw/llama32_1b_w4_qa_ldlq_c634880_batch4_full_20261022.json`
- Corrected W4A4KV4 full: `results/raw/llama32_1b_w4a4kv4_qa_ldlq_c634880_batch4_full_20261022.json`

The QA route is closed under the predefined stopping rule: scale correction explained and removed the large failure, but it did not improve full PPL. No wider post-hoc constant search is reported.
