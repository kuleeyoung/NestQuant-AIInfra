# QA-LDLQ implementation and local adaptation

## Implemented path

The project now implements the public NestQuant QA-LDLQ weight adjustment

`epsilon^2 = x_var / epsilon_constant`

`W_qa = W @ (I - epsilon^2 * inverse(H + epsilon^2 I))`

and then runs the existing 8-column E8 block-LDLQ recursion with `H + epsilon^2 I`. The default `epsilon_constant` is 155. Activation second moments are collected from 20 complete WikiText-2 train chunks. Q/K/V share the attention input statistic and gate/up share the MLP input statistic, matching the public implementation's input grouping. The public weight script applies QA to Transformer layer projections, so this project leaves `lm_head` on ordinary block-LDLQ.

The formula has a unit test against the direct dense inverse expression. Model-level tests cover activation-variance lookup, QA quantization, NQMC export, and reconstruction without the calibration tensors.

## Local adaptation boundary

This is an adaptation of the public QA formula to the project's existing block-LDLQ pipeline. It is not byte-for-byte execution of the upstream repository:

- The local Hessians are stored as covariance-scale `E[x x^T]` matrices.
- The local block factorization additionally uses relative damping 0.01.
- The upstream scripts and the local pipeline differ in calibration organization, beta search, dither handling, and model execution code.
- QA produced one normalized outlier block in `model.layers.1.mlp.down_proj` that was not covered by the previous largest candidate `50/q`. The project candidate grid therefore includes the disclosed engineering extension `100/q`; only that layer selected it. The paper Appendix G search range does not justify calling this a paper-default beta.

These differences matter because the upstream constant 155 implicitly assumes the upstream Hessian/noise scale. Reusing it with a differently scaled and damped Hessian is a hypothesis that must be tested, not an identity guaranteed by the formula.

## Reproducible artifacts

- Activation variance: `results/raw/calibration/activation_variance_train20.json`
  - 20 chunks / 40,960 tokens / 112 projection modules
  - SHA-256 `64db1197350e4c11d30469438619e487e34cd18ac3c714aef3ce1db7a44eb904`
- QA-LDLQ NQMC: `results/raw/llama32_1b_nestquant_w4_qa_ldlq_train64_var20.nqmc`
  - 659,281,687 bytes / 113 quantized tensors
  - SHA-256 `23dcebae6d2874499f831b0244f1ac3b51ed3264b18ee63e84ab6f3b54fd9955`
  - encoder `qa_block_ldlq`; evaluation reloads this file before PPL
- Full evaluation manifests:
  - `results/raw/llama32_1b_w4_qa_ldlq_batch4_full_20261022.json`
  - `results/raw/llama32_1b_w4a4kv4_qa_ldlq_batch4_full_20261022.json`

The current result is a useful negative ablation: default upstream `epsilon_constant=155` degrades both W4 and W4A4KV4 PPL in this local pipeline. See `results/llama32_1b_qa_ldlq_ablation.md`. It does not show that QA-LDLQ is ineffective in the paper implementation; it shows that transferring the constant without matching Hessian scaling is not valid for this adaptation.

## Hessian-scale audit and corrected experiment

The public `parallel/hessian.py` accumulates token rows but divides the Gram matrix only by the number of calibration sequences. The local collector divides by the number of activation rows, which is sequence count times context length. At context length 2048:

`H_public = 2048 * H_local`

Both paths compute `x_var` per activation element. Preserving the ratio between `epsilon^2` and `H` therefore requires:

`C_local = C_public * 2048 = 155 * 2048 = 317440`

This is recorded by `scripts/audit_qa_ldlq_scale.py` in `results/raw/qa_ldlq_scale_audit.json`. A representative-layer sweep compared ordinary LDLQ with constants 155, 158720, 317440, and 634880. Constant 155 moved the layer-1 down-projection weight by 73.7%, raised its Hessian proxy loss to 9.07 times the ordinary control, and required the project-only beta 100. The scale-corrected candidates kept proxy loss near the control and did not require beta 100. Constant 634880 had the smallest worst-case proxy regression across the four audited layers and was selected for one bounded full-model test.

The corrected full results were:

| Scope | Ordinary block-LDLQ | QA, C=155 | Scale-corrected QA, C=634880 |
|---|---:|---:|---:|
| W4 PPL | 10.276601 | 11.053594 | 10.283140 |
| W4A4KV4 PPL | 11.578171 | 12.386133 | 11.584234 |

Scale correction removed the severe failure but did not beat ordinary block-LDLQ: corrected QA was 0.064% worse for W4 and 0.052% worse for W4A4KV4. Following the predefined stopping rule, the project does not expand the parameter search. The defensible conclusion is that the original regression came from a 2048-fold normalization mismatch, while this local QA adaptation did not reproduce a quality gain after correction.

The corrected NQMC is `results/raw/llama32_1b_nestquant_w4_qa_ldlq_c634880.nqmc`, 659,281,690 bytes with SHA-256 `8854812a4fc097890590bc38e19b5aea530685e12a9a685c1acce8a0d9421a8d`.
