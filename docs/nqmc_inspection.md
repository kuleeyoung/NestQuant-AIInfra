# NQMC checkpoint inspection

`scripts/inspect_nqmc.py` 提供只读、bounded-memory 的 NQMC 审计，不加载 Transformers 模型，也不把量化权重反量化为完整浮点 state dict。

## 常用命令

快速查看可信文件的 manifest 和存储账目，不扫描 payload：

```bash
python scripts/inspect_nqmc.py model.nqmc --skip-payload-checksum
```

校验 NQMC payload checksum（默认行为）：

```bash
python scripts/inspect_nqmc.py model.nqmc
```

完整格式审计并保存 JSON：

```bash
python scripts/inspect_nqmc.py model.nqmc \
  --records --file-sha256 --output inspection.json
```

安装项目后也可使用：

```bash
nestquant-inspect-nqmc model.nqmc --records --file-sha256
```

## 成本与语义

- 默认读取并验证外层 NQMC manifest、entry 边界和 payload SHA-256；内存上只保留 checksum 分块；
- `--skip-payload-checksum` 适合已经由可信渠道校验过、只需快速查看目录的文件，但输出会明确标记 `payload_checksum_verified=false`；
- `--records` 每次只读取一个 tensor record，解析 NQRT 版本、codec semantics、q、beta count 和 dither 存储方式，不执行 E8 解码或反量化；
- `--file-sha256` 增加一次全文件扫描，用于得到分发层面的文件身份；payload checksum 与整文件 SHA-256 不是同一个值；
- “BF16 equivalent” 是把相同唯一状态值按 BF16 计算的逻辑载荷，不是另行导出的 BF16 checkpoint 文件；
- 码率和容器检查只证明存储格式，不证明低比特权重常驻 GPU、INT4 GEMM 或推理加速。

真实 Llama-3.2-1B NQRT v3 checkpoint 的审计见[结果报告](../results/llama32_1b_nqrt3_checkpoint_inspection.md)。

## 按名称选择性读取

`NQMCFileReader` 为 manifest record 建立名称索引，并提供：

- `canonical_name(name)`：把 tied alias 解析到唯一存储名；
- `get_entry(name)`：O(1) 获取 shape、dtype、offset、length 和 record kind；
- `read_named_record(name)`：只 seek/read 一个 record，不解码量化值；
- `decode_tensor(name, device=...)`：只解码一个 tensor，dense 与 NQRT 使用同一入口。

单 tensor benchmark：

```bash
python scripts/benchmark_nqmc_selective_decode.py model.nqmc \
  model.layers.0.self_attn.q_proj.weight \
  model.layers.0.mlp.down_proj.weight \
  --device cuda --warmup 1 --repeats 3 --output selective_decode.json
```

benchmark 默认跳过全 payload checksum，以免把一次 O(file size) 扫描混入单 record 延迟；需要端到端可信文件校验时显式添加 `--verify-checksum`。该 benchmark 只测 seek/read 和独立 tensor dequantization，不含 GEMM、层调度或推理。

分块 BF16 解码：

```python
reader.decode_tensor(
    name, device="cuda", output_dtype=torch.bfloat16, rows_per_chunk=128
)
```

分块路径只解包目标行的 residue/beta ID，并把 float32 codec 结果立即写入 BF16 输出。generator-state dither 为保持历史 NQRT bit-exact 语义，仍按原始完整 shape 在 CPU 重放，再按行传输；详见[benchmark 与正确性报告](../results/llama32_1b_nqrt3_chunked_bf16_decode.md)。
