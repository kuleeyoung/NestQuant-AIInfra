# NQKV GPU 页环路线调研

更新时间：2026-11-02

## 目标

本文调研 `docs/next_route_decisions.md` 中的路线 B：为每个服务 slot 预留物理页，
让页边界 CUDA Graph 直接取得下一页，减少逐 token CPU/GPU 同步和边界控制面工作。

需要特别区分两个概念：

1. **GPU 消费页表**：CPU 在 batch 执行前分配页，GPU kernel 按页表读写。这是
   vLLM、SGLang、TensorRT-LLM 和 FlashInfer 的常见做法。
2. **GPU 自主分配物理页**：GPU 在 decode Graph 内从全局空闲池取页、处理失败、
   发布所有权并负责回收。公开主流 serving runtime 并没有提供可直接移植到 NQKV 的
   完整实现。

因此，本项目不应把“GPU 读取预留页 ID”误写成“通用 GPU allocator 已完成”。

## 调研结论

### 1. vLLM / PagedAttention

来源：

- PagedAttention 论文：<https://arxiv.org/abs/2309.06180>
- vLLM KV manager：<https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py>
- vLLM prefix-cache allocator 设计：<https://github.com/vllm-project/vllm/blob/main/docs/design/prefix_caching.md>

可借鉴：

- 初始化时创建完整 block pool，运行期只移动 block ID，不动态创建 GPU buffer；
- request ID 到 block ID list 的明确所有权映射；
- block 引用计数；
- intrusive doubly-linked free queue，支持 O(1) 取页、归还和从中间移除；
- admission 前计算本轮需要的 block 数，空间不足时不启动不完整的执行；
- watermark/reserved-block 机制，为已经在途的请求保留完成空间。

不能直接复用：

- vLLM allocator 位于 scheduler/control plane，仍在模型执行前由主机分配，不是
  Graph 内 GPU 自主分配；
- vLLM 标准 K/V block 布局不包含 NQKV 的 E8 code、Key/Value beta ID、row norm
  和 BF16 hot tail；
- 直接引入 vLLM runtime 会迫使本项目重写模型接入和 kernel ABI，成本远高于收益。

### 2. FlashInfer

来源：

- paged KV API：<https://github.com/flashinfer-ai/flashinfer/blob/main/docs/api/page.rst>
- CUDA Graph attention API：<https://docs.flashinfer.ai/api/attention.html>

可借鉴：

- CUDA Graph 模式由调用方预留固定地址的 `paged_kv_indptr`、`paged_kv_indices`、
  `last_page_len` 和 workspace；Graph 生命周期内 batch shape 和指针不变；
- page-index buffer 按最大计划页数预留，实际长度通过 GPU 元数据表达；
- `append_paged_kv_cache` 把“页表管理”和“向物理页写入 K/V”分开；
- `*_with_slot_mapping` 的接口形式适合本项目：scheduler 准备目标 slot/page ID，
  kernel 只负责按映射写入，不在计算 kernel 内实现复杂 allocator；
- caller-owned scratch buffer 可避免 Graph replay 时重新分配临时内存。

不能直接复用：

- FlashInfer 当前公开 append 路径面向 BF16/FP16/FP8/NVFP4 等既定布局，不能解释
  NQKV E8 packed page；
- 其 quantize-append kernel 不能替换 `nqkv_graph_pack.py`，但元数据 ABI 和测试方式
  可以借鉴。

### 3. SGLang

来源：

- KV cache 模块分层：<https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/README.md>
- allocator 架构说明：<https://github.com/sgl-project/sglang-jax/blob/main/docs/architecture/07-kv-cache.md>

可借鉴：

- pool 只管理物理存储，allocator 管理空闲页，allocation policy 位于更上层；
- `backup_state()/restore_state()` 支持“乐观分配，任何子分配失败则整体回滚”；
- `free_group_begin()/free_group_end()` 将多个结束请求的页面合并释放，避免反复整理
  free list；
- padding row 写入保留的 dummy slot，避免 inactive row 污染真实页面；
- request slot 与物理 KV slot 通过显式 mapping 分离，便于 slot 复用。

不能直接复用：

- allocator 仍由 scheduler 驱动；
- SGLang 的 RadixAttention/HiCache 解决前缀复用和分层存储，不是本项目当前页边界
  同步问题的必要条件；
- 引入 radix tree 会扩大项目范围，当前不应实施。

### 4. TensorRT-LLM

来源：

- KV Cache System：<https://nvidia.github.io/TensorRT-LLM/features/kvcache.html>
- memory model：<https://nvidia.github.io/TensorRT-LLM/reference/memory.html>

可借鉴：

- 初始化阶段按模型结构建立一个或多个固定 block pool；
- block 按需分给请求，满 block 才进入可复用/索引状态；
- 不同 attention window/head 组合使用独立 pool；
- 物理池容量在 runtime 初始化时确定，执行期间不改变底层地址。

不能直接复用：

- KV manager 是 TensorRT-LLM C++ runtime 的组成部分，接口、engine 和数据布局耦合；
- 直接移植的成本接近重写现有 NestQuant runtime；
- 它同样不能证明 GPU Graph 内自主分配物理 block。

### 5. vAttention

来源：

- 论文：<https://arxiv.org/abs/2405.04437>
- 官方仓库：<https://github.com/microsoft/vattention>

思路：使用 CUDA Virtual Memory Management，将大块连续虚拟地址与按需提交的物理
显存解耦，从而让普通连续 KV attention kernel 无需理解用户态页表。

对本项目的判断：暂不采用。

- NQKV attention 本来就需要读取非连续 packed page 和自定义 beta/norm 元数据，
  vAttention 的“复用连续 KV kernel”优势不成立；
- 官方实现集成 Sarathi-Serve，并包含修改后的 NVIDIA UVM driver，环境和发布复杂度高；
- CUDA VMM 映射/提交仍是主机 API，不等同于 Graph 内设备自主换页；
- 对单张 RTX 5070 简历项目而言，这会扩大部署风险而没有直接解决 E8 packed decode。

## 推荐的可复用组合

不直接引入大型 serving framework，采用以下组合：

| 本项目组件 | 借鉴来源 | 采用内容 |
|---|---|---|
| `NQKVPagePool` | vLLM | 预创建全部 page、稳定 page ID、free queue、watermark |
| reservation transaction | SGLang | snapshot/rollback、跨 16 层全有或全无、批量释放 |
| GPU metadata ABI | FlashInfer | 固定容量 indices/length/target buffers、slot mapping、caller-owned workspace |
| boundary Graph | 现有 NQKV | 保留 E8 quantize/pack/publish kernel，不替换 codec |
| admission policy | vLLM/TensorRT-LLM | 执行 batch 前保证在途请求有足够页面，不在 Graph 内处理 OOM |

这里的关键取舍是：**不要在第一版实现通用 device-side malloc**。页面不足、请求取消、
异常回滚和跨层一致性属于控制面事务；把它们全部塞进一个 GPU atomic free stack 会让
正确性证明和调试成本显著高于潜在的几十微秒收益。

## 建议实现：两级预留窗口，而不是无限 page ring

### 数据结构

为每个 `[layer, slot]` 增加一个小的固定深度 reserve window，推荐先从深度 2 开始：

```text
reserved_page_ids[layer, slot, 2]  int32 CUDA tensor
reserve_head[slot]                  int32 CUDA tensor
reserve_count[slot]                 int32 CUDA tensor
```

控制面在 admission 或 refill 时一次性为全部层预留下一到两个 page ID，并把 ID 批量写入
固定地址 tensor。boundary Graph 只做：

1. 读取 `reserved_page_ids[..., reserve_head]`；
2. 将 hot page 编码到该物理页；
3. 发布 page table/page count；
4. 推进 `reserve_head` 并减少 `reserve_count`。

CPU 不再在每个页边界同步分配，但仍异步补充低水位 reserve window。因为页面不会在同一
请求结束前循环覆盖，严格来说它是“预留窗口/队列”，不是会覆盖历史 KV 的 ring buffer。

### 为什么深度 2 比每槽预留最大上下文更合理

- page size 16 时，一个 decode 请求每 16 token 才消耗一页，控制面有充足时间补页；
- capacity 8、16 层、深度 2 仅预留 `8 * 16 * 2 = 256` 个 NQKV 页面；
- 避免按 `max_cache_len` 给每个槽位静态占满全部页面；
- 请求提前结束时最多浪费两个未使用页，释放逻辑可控；
- refill 失败可以停止下一轮 admission，而不是让 Graph 内处理复杂 OOM。

### 必须保持的事务边界

1. 一次 refill 必须同时覆盖所有 Transformer 层；任一层不足则全部回滚。
2. 未被 Graph 消费的 reserved page 不计入 request page table。
3. Graph 消费后，host mirror 通过完成事件批量提交，不逐层同步。
4. cancellation 同时释放 committed pages 和 unconsumed reserved pages。
5. generation ID 必须进入 reservation ownership，防止 slot 复用后旧页误提交。
6. pool exhaustion 在调度前拒绝 admission；Graph 中只保留 debug assert/counter。

## 分阶段实验与停止条件

### Probe 0：先测同步成本

在当前 runner 中分别计时：

- GPU hot-length 读取和 `.item()`；
- boundary slot 列表 D2H；
- 16 层页面 reservation/commit；
- 两个 Graph replay 本体。

如果非边界步的主机同步 P50 小于总 TPOT 的 2%，不进入完整预留窗口实现，优先 GitHub
交付。若大于 5%，继续 Probe 1。

### Probe 1：单层、单 slot、深度 2

实现固定 reserve queue，只验证连续 4 个页面、取消、复用和耗尽，不接完整模型。

验收：与当前 eager payload byte-for-byte 一致；无重复 page ID；取消后 allocated page
恢复到基线；generation mismatch 必须硬失败。

### Probe 2：16 层、capacity 8 事务 refill

实现跨层 reserve/refill/rollback，使用随机 mixed-length workload 验证 10,000 次状态转换。

验收：任一层注入分配失败时 16 层状态全部不变；无部分提交；页池计数守恒。

### Probe 3：接入双 Graph

boundary Graph 从 reserve queue 取目标页；普通 Graph 不再读取 GPU hot length。用 scheduler
的 host request length 选择 Graph，并保留抽样 GPU assertion。

验收：连续至少 8 个页面，动态 active mask、slot 释放/复用、取消、页耗尽均与 eager
逐步 logits 一致；0 fallback；终态 0 page leak。

### Probe 4：真实 1B A/B

同一请求 trace、同一 warmup、至少 30 次重复，对比：

- 当前双 Graph；
- scheduler boundary hint；
- depth-2 reserve window。

报告 TPOT P50/P95/P99、boundary/non-boundary latency、CPU wall time、CUDA allocated/
reserved、预留页利用率和失败率。

停止条件：reserve window 相对 scheduler hint 的 TPOT 中位收益小于 3%，或 P95 没有改善，
则不把复杂 allocator 合入默认路径，仅保留 probe 和负结果报告。

## 最终建议

调研不支持“直接复制一个成熟 GPU 自主 allocator”。最合理路线是：

1. 先完成同步成本 profiling；
2. 以 FlashInfer 固定元数据 ABI 为形态；
3. 以 vLLM 预分配 pool/watermark 和 SGLang transaction/rollback 为控制面；
4. 实现深度 2 的跨层预留窗口；
5. 保留 NQKV 自有 E8 pack/attention kernel；
6. 只有相对简单 scheduler-hint 路线获得可重复的额外收益，才合入默认 runtime。

这比“每个 slot 预留整个 max context”更节省显存，也比在 CUDA Graph 内实现通用
lock-free allocator 更容易证明正确。

## Probe 0 实测更新（2026-11-02）

真实 Llama-3.2-1B、capacity 2、page size 16 上完成 30 轮同步分段测量：

- normal Graph replay 中位 72.755 ms，控制面 0.623 ms，占 0.85%；
- boundary Graph replay 中位 74.736 ms，控制面 3.498 ms，占 4.46%；
- boundary page reservation 中位 2.717 ms，但每 16 token 才发生一次；
- 将边界额外成本摊销后，全部可见控制面的乐观消除上界约 0.803 ms/token，约占
  当前 decode 的 1.1%。

该结果触发 Probe 0 的 `<2%` 停止条件。当前不实施 reserve window；保留设计作为
未来 decode kernel 降到 20 ms 以下、capacity 扩大或控制面比例超过 5% 时的候选。
完整协议与原始数据见 `results/nqkv_graph_control_plane_probe0.md` 和
`results/raw/nqkv_graph_control_plane_probe0.json`。
