# 工程故障复盘

本页记录项目中能被测试和实验验证的故障。复盘关注“现象—根因—修复—防复发”，不是把正常的
试错包装成全部成功。

## 1. CUDA current stream 导致 Graph 漏捕获

### 现象

J2 自定义 CUDA kernel 在 eager 模式可以运行，但接入完整模型 CUDA Graph 后，捕获行为与预期
不一致，存在自定义 kernel 在 Graph 外执行的风险。仅看最终输出或“Graph capture 没报错”不足以
证明自定义操作已被捕获。

### 根因

扩展使用了错误的 CUDA stream 语义。PyTorch operator 应在调用线程的 current stream 上执行；
如果扩展直接使用默认 stream，PyTorch 在非默认 stream 或 CUDA Graph capture 中提交的前后算子
与自定义 kernel 可能不在同一执行序列。

### 修复

- 在 C++/CUDA binding 中获取 PyTorch current CUDA stream；
- partial kernel 与 Split-K reduction 都显式发射到该 stream；
- 不通过额外全局同步掩盖依赖错误；
- 重新捕获完整模型 Graph。

### 验证

- batch 1/16 连续生成 32 token；
- Graph 与 packed eager 比较逐步 logits 和 argmax；
- ragged batch-2 比较 position `17/16`；
- slot 释放复用后比较 position `17/3`；
- 混合 BF16/packed 512 MiB 方案重新通过 Graph replay。

### 防复发

- runtime smoke 把 current-stream 与 Graph capture 作为硬断言；
- 扩展新增 kernel 时必须复用 current stream；
- 不接受只验证单次 eager 输出的 CUDA 扩展修改。

## 2. QA-LDLQ Hessian 归一化相差 2048 倍

### 现象

按作者默认 `epsilon_constant=155` 接入 QA 权重调整后，W4 与 W4A4KV4 PPL 明显退化；代表层
出现最高 73.7% 权重调整和 9.07 倍 proxy loss。这与预期的轻量修正不符。

### 根因

上游 Hessian 用 token Gram matrix 除以 sequence 数，本项目 collector 用 activation row 数归一化。
在 context length 2048 下，两种定义相差 2048 倍；直接复用同一 epsilon 常数改变了正则项的实际
尺度。

### 修复

- 审计 Hessian 收集与公式中的归一化单位；
- 将常数按 2048 倍换算，做有界候选扫描；
- 用 representative layer 的 adjustment、proxy loss 和数值有限性先筛选；
- 只对通过代理检查的候选运行完整模型 PPL。

### 结果

修正后的 `C=634880` 消除了严重退化，但 W4/W4A4KV4 PPL 10.283140/11.584234 仍未优于
普通 LDLQ 的 10.276601/11.578171。因此 QA 路线以负结果关闭，没有继续用 test set 做事后调参。

### 防复发

- checkpoint 保存 calibration provenance 和归一化语义；
- 比较外部常数前先做量纲与尺度审计；
- 完整评测前设置代表层异常阈值。

## 3. 隔离 grouped kernel 加速没有转化为 block 加速

### 现象

Grouped QKV 把三次 launch 合为一次，孤立 kernel 得到 1.065×～1.121× 加速，但接入真实
Transformer block 后，seq 1/8/16 的收益为 -1.4%/约 0%/+1.0%。Grouped gate/up 甚至慢
0.5%～3.0%。

### 根因

隔离 microbenchmark 只覆盖三个 Linear 的 launch；真实 block 还包含输入变换、attention、
coordinator、张量领取/清理和其他 projection。被优化部分占 block 总时间不足，额外协调成本抵消
了 launch 收益。

### 修复与决策

- 同时保留 isolated kernel 与 block-level benchmark；
- QKV coordinator 只作为有边界的实验实现；
- gate/up 在负结果后不进入生产 dispatch；
- 优化方向转向 Tensor Core 数据流和 E8 解码热点。

### 防复发

任何 microbenchmark 优化必须经过“单 kernel—真实 block—完整模型”三级验证。简历只引用与声明
范围相匹配的结果。

## 4. Packed 显存下降但端到端延迟变差

### 现象

完整 1B 模型的 warm CUDA allocation 从 BF16 2390.1 MiB 降至 packed 1028.6 MiB，但全 packed
TPOT 约 40.42 ms，BF16 为 10.45 ms。

### 根因

packed 路径每次 projection 都要重放 hash dither、执行两次 E8 最近点恢复、生成 BF16 fragment
并做 Split-K；BF16 路径直接使用已经常驻的 dense 权重和高度优化的 GEMM。profiling 表明 J2
partial kernel 是最大 GPU 热点，且 112 个 projection 的 launch 与 Hadamard 小 kernel 形成共同瓶颈。

### 处理

- 用 Tensor Core、M=1 specialization、shuffle/ballot 优化和 CUDA Graph 降低不同层级开销；
- 不将 packed 路径表述为超过 BF16 的加速；
- 实现 packed/BF16 混合驻留，把负结果转化为可配置的显存—延迟 Pareto。

### 防复发

分别测量质量、文件大小、CUDA allocation、TTFT、TPOT 和吞吐；任何“压缩”结论不得自动推导为
“加速”结论。

## 5. 发布环境可能被本机未跟踪文件掩盖

### 风险

在长期开发目录直接运行测试，可能意外依赖未提交模块、编译缓存、本机绝对路径或已经安装的
editable package。这样的“本机通过”不能证明 GitHub 用户可以安装。

### 处理

`scripts/smoke_publishable_tree.sh` 只复制 Git 已跟踪和未忽略的新文件到 `/tmp`，随后：

1. 构建 source wheel；
2. 运行完整测试；
3. 构建 sdist，再从 sdist 构建 wheel；
4. 安装到独立 target；
5. 检查 console entry points；
6. 检查 wheel 内 CUDA/C++ 源码；
7. 可选 fresh compile CUDA E8 smoke。

仓库审计器另行检查本机绝对路径、敏感文件名、疑似凭据、失效 Markdown 链接和大文件。

## 复盘原则

- 先建立最小复现，再修改实现；
- 数值正确与性能提升分别验收；
- microbenchmark 与端到端结论分开；
- 负结果保留原始配置和停止条件；
- 不用同步、复制或放宽容差掩盖并发与数值错误；
- 修复后增加自动化检查，使同类问题不能静默复发。
