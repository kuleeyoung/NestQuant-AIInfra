# QuaRot 兼容性审查与适配路线

审查日期：2026-10-22

## 当前同协议基线

WSL2 环境已经固定为 Ubuntu 24.04、Python 3.12.3、PyTorch 2.14.1+cu130、Transformers 5.20.2、CUDA 13.0 和 RTX 5070（compute capability 12.0）。模型与数据固定为：

- `meta-llama/Llama-3.2-1B`，revision `4e20de362430cd3b72f300e6b0f18e50e7166e08`
- `Salesforce/wikitext/wikitext-2-raw-v1`，revision `b08601e04326c79dfdd32d625aee71d232d685c3`
- context length 2048，141 个窗口，288,627 个预测 token

本环境完整结果：

| 方法 | PPL | 相对 BF16 增量 |
|---|---:|---:|
| BF16 | 9.758111861928 | 0 |
| RTN W4 | 13.023721577561 | +3.265609715634 |

原始 manifest：`results/raw/wsl_bf16_rtn_w4_full_20261022.json`。

## 官方 QuaRot 仓库快照

- 仓库：`https://github.com/spcl/QuaRot.git`
- 主仓库 commit：`5008669b08c1f11f9b64d52d16fddd47ca754c5a`
- CUTLASS submodule：`ffa34e70756b0bc744e1dfcc115b5a991a68f132`
- fast-hadamard-transform submodule：`4ea722e434e3d4f2a14522341959ebdbe62be2de`
- nvbench submodule：`d8dced8a64d9ce305add92fa6d274fd49b569b7e`

## 已确认的兼容性问题

1. `fake_quant/README.md` 明确写着当前只支持 Llama-2；代码白名单虽加入 Llama-3 8B/70B，但没有 Llama-3.2-1B。
2. 官方依赖固定为 PyTorch 2.2.1、Transformers 4.38.0、Hugging Face Hub 0.20.3 和 Datasets 2.17.1。该版本组合早于 Llama-3.2 发布，不能直接视为 Llama-3.2-1B 的受支持环境。
3. 官方 CUDA extension 只生成 `sm_75`、`sm_80`、`sm_86`，没有 RTX 5070 所需的 `sm_120`。
4. fake-quant 路径仍直接导入 `fast_hadamard_transform`，旋转、在线 down-proj/o-proj Hadamard 和 Q/K 旋转会调用该扩展；不能简单地认为 fake quant 与 CUDA extension 完全无关。
5. Llama-3.2-1B 的 GQA、RoPE/attention 接口和 Transformers 实现需要逐项验证；直接得到一个 PPL 数字不等于协议已经正确。
6. 官方 evaluator 自己拼接和截断 WikiText-2，并采用自己的逐层执行逻辑。最终公平结果必须接入本项目统一 token 流和 PPL 聚合器，或证明两者 token/NLL 完全一致。

## 路线 A：现代 PyTorch + 纯 PyTorch Hadamard 的质量仿真适配（推荐）

在独立环境中使用支持 RTX 5070 和 Llama-3.2 的现代 PyTorch/Transformers；保留 QuaRot 的旋转、权重/激活/KV fake-quant 语义，把 fast Hadamard CUDA 调用替换为数学等价的 PyTorch 参考实现。先验证无量化旋转前后 logits/PPL 等价，再做 W4A4KV4。

优点：本机可完成；最贴合当前“同协议质量对照”目标；不把 kernel 兼容问题混入算法结论；预计 1～2 周。

缺点：速度慢于官方 CUDA 扩展；属于清楚披露的兼容性移植，不是官方仓库零修改复现；必须增加等价性测试。

验收标准：BF16 统一基线一致；旋转但不量化时 logits 最大误差和 PPL 漂移在预设容差内；W4A4KV4 的量化位置、粒度、排除层有 manifest；完整 PPL 使用统一 288,627 token。

## 路线 B：租用 Ampere 云 GPU，运行官方固定环境

在支持 `sm_80` 或 `sm_86` 的云 GPU 上安装官方 PyTorch 2.2.1/CUDA 环境，先用官方支持的 Llama-2 复核仓库，再另外处理 Llama-3.2-1B 适配。

优点：最容易验证官方代码原始行为；可以区分“上游本身能否复现”和“本机新架构兼容问题”。

缺点：仍不能自动解决 Llama-3.2-1B 不受支持以及统一协议问题；增加云 GPU 费用与环境迁移；预计官方复核 2～5 天，Llama-3.2 适配另需 1～2 周。

## 路线 C：为 RTX 5070 移植官方 CUDA kernel

升级 PyTorch/CUDA/CUTLASS/fast-hadamard-transform，加入 `sm_120` 编译目标并修复 CUDA API 与模板兼容问题，再验证 kernel correctness 和性能。

优点：系统工程含量最高；若完成，可继续走真实低比特 kernel 与吞吐测试。

缺点：范围从质量对照扩大到 CUDA kernel 移植；预计 1～3 个月且结果不确定；会推迟当前论文复现项目收口。

## 建议

当前项目选择路线 A。它直接服务于“QuaRot 等外部方法的同协议质量对照”，同时保留清楚的实现边界。路线 B 可作为路线 A 遇到上游语义疑问时的短期验证手段；路线 C 留给独立推理系统项目。
