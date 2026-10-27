# 固定作者代码与局部 Hadamard A4 语义对齐

## 对照范围

- 固定作者仓库：`cookiedoth/nestquant`；
- 固定 commit：`6b269f84264afad990a09248d8eeb45d5f6248ff`；
- 作者文件：`parallel/quant_utils.py`、`parallel/model.py`、`hadamard.py`；
- 本项目文件：`nestquant/evaluation/activation_quantization.py`、`nestquant/models/rotated_linear.py`。

这里核对的是 2 次幂输入维度下的激活旋转和量化坐标。Llama-3.2-1B 的投影输入维度为 2048 或 8192，均属于该范围。固定作者代码针对 Llama-3-8B 14336 等非 2 次幂维度的 Kronecker Hadamard 不在本次验收范围内。

## 作者路径

固定作者代码在七类投影前注册 `get_quant_act_hook`。hook 调用 `quantsim`：

1. 将输入展平为最后一维向量；
2. 计算每行 L2 范数并归一化到半径 `sqrt(N)`；
3. 执行 `kron_h_ip(X)`；
4. 用 `use_dither=False, try_all=True` 执行 E8 multi-beta 编解码；
5. 执行逆旋转；
6. 恢复行范数；
7. 返回 BF16 重构激活。

对 2 次幂 N，`kron_h_ip` 使用归一化 Hadamard。记量化后的旋转坐标激活为 `Q_A(XH)`，作者 hook 返回：

\[
\widehat X_{\mathrm{up}}=Q_A(XH)H.
\]

权重 `w_quant.py` 使用同一 `quantsim` 结构。记量化后的旋转坐标权重为 `Q_W(WH)`，保存回模型的权重是：

\[
\widehat W_{\mathrm{up}}=Q_W(WH)H.
\]

因此作者普通坐标点积为：

\[
\widehat X_{\mathrm{up}}\widehat W_{\mathrm{up}}^T
=Q_A(XH)HH^TQ_W(WH)^T
=Q_A(XH)Q_W(WH)^T.
\]

## 本项目局部坐标路径

`HadamardRotatedLinear` 直接保存 `Q_W(WH)`，并在模块内部计算：

\[
X_H=XH,
\qquad
\widehat X_H=Q_A(X_H),
\qquad
Y=\widehat X_HQ_W(WH)^T.
\]

该点积与作者路径数学相同，同时省略作者重构后激活/权重的两个逆变换。A4 通过 `register_rotated_input_hook` 挂在模块完成 `XH` 之后、Linear 之前，不再使用普通 Linear 的 `H -> A4 -> H` pre-hook。

## 自动验收

`tests/unit/evaluation/test_activation_quantization.py` 使用相同输入、权重、q 和 beta，对照：

- 作者式普通坐标：`H -> A4 -> H -> Linear(W)`；
- 本项目局部坐标：`H -> A4 -> Linear(WH)`。

两条路径在 FP32 量化内部计算下最大差异处于约 `1e-7` 量级；差异来自作者式路径额外的逆/正 Hadamard浮点舍入，因此使用 `2e-6` 绝对/相对容差，而不要求逐位一致。当前全仓 246 项主项目测试及 25 个参数化 subtests 通过。

真实 Llama-3.2-1B 首窗口进一步验证：

- 112 个投影全部使用 `local_hadamard_input`；
- W4 控制 PPL：7.462738；
- W4A4 PPL：7.860542；
- A4 相对平方误差：0.00645637；
- overload 比例：5.4576%；
- 没有 shape、dtype、NaN、Inf 或显存错误。

## 尚未对齐的项目

- 作者 `try_all=True` 的内部候选实现没有公开逐层 beta 优化脚本；本项目使用固定全局 A beta 做首轮基线；
- 作者激活路径明确关闭 dither，本项目运行时 A4 同样关闭 dither；但其他论文表格路径是否一致，正文无法确认；
- 作者使用 FairScale 行/列并行层，单 GPU Hugging Face 模型不需要 `re_glue`；若做多 GPU 张量并行，必须重新验证 row-parallel 语义；
- 非 2 次幂 Kronecker Hadamard 尚未实现；
- 当前 A4 仍重构为浮点，不是 packed 激活 buffer 或低比特 kernel。

因此当前准确结论是：2 次幂、单 GPU Llama-3.2-1B 下的 A4 点积坐标已经与固定作者代码的公开公式对齐；逐层 beta、非 2 次幂和分布式列重组仍未完成。
