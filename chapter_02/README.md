# 第 02 期：KV Cache——第一个推理优化

## 1. 对应书籍章节

对应《超轻量AI Infra教程—推理篇》第 02 章“KV Cache——第一个推理优化”。

## 2. 本节目的与实现概览

本目录使用基础 Python 和 PyTorch 独立实现 Qwen3-0.6B 的单请求推理，同时提供 no-cache baseline 和 KV Cache optimized 两条路径。代码不引用第 01 期目录，也不使用 Transformers 的模型实现。

本期只改变一个核心变量：Decode 是否复用历史 Token 已经计算出的 Key 和 Value。KV Cache 使用普通 Tensor 并通过 `torch.cat` 增长；预分配、分页管理、批处理和请求调度不属于本期范围。

粗略步骤：保留每层历史 K/V；Decode 时只计算新 Token 的 Q/K/V；把新 K/V 追加进 Cache；在相同输入下比较 no-cache 与 cached 两条路径的正确性和耗时。

## 3. 代码使用方法

### 验证环境

- GPU：NVIDIA GeForce RTX 3080 Ti 12GB
- CPU：Intel Xeon Gold 6226R
- 内存：31 GiB
- 操作系统：Ubuntu 22.04
- Python：3.10
- GPU 驱动：595.80
- CUDA Runtime：12.6
- PyTorch：2.7.1+cu126
- 模型：`Qwen/Qwen3-0.6B`
- 模型 revision：`c1899de289a04d12100db370d81485cdf75e47ca`
- dtype：bfloat16

### 安装依赖

从项目根目录创建独立环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r chapter_02/requirements-lock.txt
```

如果只安装直接依赖，也可以使用：

```bash
python -m pip install -r chapter_02/requirements.txt
```

首次运行会下载固定 revision 的模型。也可以通过 `--model` 传入包含模型文件的本地目录。

### CPU 快速自检

随机小模型测试不下载真实权重，也不要求 GPU：

```bash
source .venv/bin/activate
python chapter_02/smoke_test.py
```

测试内容包括 Prefill 一致性、KV Cache 形状与数据量、单步 Decode 一致性以及 Cache 参数检查。

### 运行推理

默认依次运行 no-cache 和 KV Cache，并比较生成结果：

```bash
source .venv/bin/activate
python chapter_02/run_inference.py \
  --prompt "请用一句话介绍 KV Cache。" \
  --max-new-tokens 32 \
  --mode both
```

计时范围只包含模型前向、greedy Token 选择以及必要的 CUDA 同步，不包含 tokenizer、模型下载和模型加载。第一轮完整 Prompt 前向记为 Prefill，后续生成步骤记为 Decode。

### 逐 Token 正确性验证

下面的程序在每一步使用相同完整上下文运行 no-cache baseline，同时使用最新 Token 和历史 Cache 运行 optimized 路径，然后比较 Logits 与 greedy Token：

```bash
source .venv/bin/activate
python chapter_02/compare_cache.py \
  --prompt "请用一句话介绍 KV Cache。" \
  --max-new-tokens 32 \
  --dtype float32
```

正确性对照默认使用 float32，使测试主要检查 Cache、RoPE 位置和 Attention Mask 逻辑，不让低精度矩阵计算路径的舍入差异干扰判断。Token 不一致时返回退出码 `1`；Token 一致但最大 Logits 误差超过阈值时返回退出码 `2`。

也可以额外执行 `--dtype bfloat16 --logits-atol 1.0`，观察 no-cache 与 KV Cache 因矩阵形状和计算次序不同而产生的低精度数值差异。bfloat16 下可能在候选 Logits 非常接近时生成不同 Token；这不等同于 KV Cache 的位置或内容错误。性能实验仍使用模型配置声明的 bfloat16。

### 固定长度性能实验

性能实验直接构造长度精确的合成 Token IDs，避免自然语言文本经过 tokenizer 后长度不确定。两组均使用同一输入、同一权重、greedy decoding 和固定输出长度；测试强制生成指定数量的 Token，不因 EOS 提前退出。

```bash
source .venv/bin/activate
python chapter_02/benchmark.py \
  --prompt-lengths 16 64 256 512 1024 \
  --max-new-tokens 32 \
  --warmup 1 \
  --repeats 3 \
  --output chapter_02/benchmark-results.local.json
```

本次 3080 Ti 实验的聚合结果和全部逐 Token 延迟样本保存在 `benchmark-results.json`。建议复现时使用不同文件名，保留参考记录以便对照。

输出指标包括：

- Prefill 平均延迟；
- Decode 平均、P50 和 P95 ITL；
- 端到端模型计算延迟；
- Prefill 后与全程峰值已分配显存；
- KV Cache Tensor 的实际数据量。

### KV Cache 数据结构

每层 Cache 由一对 Tensor 组成：

```text
key/value: [batch, num_key_value_heads, sequence_length, head_dim]
```

Key 在完成 Q/K Norm 和 RoPE 后写入 Cache。Cache 保留原始 KV heads，只在计算 Attention 前按照 GQA 分组展开，避免存储重复数据。Decode 的位置编号从历史 Cache 长度继续，而不是从 0 重新开始。

理论数据量为：

```text
2 × layers × num_kv_heads × head_dim × sequence_length × batch × dtype_bytes
```

该数据量不等同于 CUDA allocator 报告的进程显存，也不包含模型权重、临时 Attention Tensor 和内存分配器保留空间。

### 3080 Ti 实测记录

测试日期：2026-08-24。测试使用 bfloat16、batch size 1、固定合成 Token IDs、greedy decoding、32 个固定输出 Token、1 次 warm-up 和 3 次正式重复。计时包含模型前向、Token 选择和 CUDA 同步，不包含 tokenizer、模型下载、模型加载、排队或网络时间。P50/P95 由三次运行合计 93 个 Decode ITL 样本使用 nearest-rank 方法计算。

| 模式 | Prompt Token | Prefill 均值 (ms) | Decode 均值 (ms/token) | P50 ITL (ms) | P95 ITL (ms) | 端到端均值 (ms) | 峰值显存 (MiB) | 最终 Cache (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-cache | 16 | 32.71 | 31.58 | 31.47 | 32.57 | 1011.79 | 1158.73 | 0.00 |
| KV Cache | 16 | 39.88 | 37.14 | 41.54 | 44.40 | 1191.07 | 1155.57 | 5.14 |
| no-cache | 64 | 33.98 | 34.66 | 34.84 | 36.10 | 1108.47 | 1173.20 | 0.00 |
| KV Cache | 64 | 32.16 | 30.41 | 30.31 | 31.28 | 975.00 | 1170.69 | 10.39 |
| no-cache | 256 | 33.11 | 33.57 | 33.65 | 34.82 | 1073.66 | 1229.58 | 0.00 |
| KV Cache | 256 | 33.83 | 32.49 | 32.45 | 33.17 | 1041.09 | 1247.71 | 31.39 |
| no-cache | 512 | 34.95 | 34.97 | 34.83 | 35.94 | 1118.87 | 1305.03 | 0.00 |
| KV Cache | 512 | 35.45 | 32.41 | 32.25 | 33.11 | 1040.10 | 1350.40 | 59.39 |
| no-cache | 1024 | 56.63 | 60.43 | 61.36 | 63.06 | 1930.00 | 1453.11 | 0.00 |
| KV Cache | 1024 | 55.95 | 32.64 | 32.59 | 33.49 | 1067.75 | 1555.79 | 115.39 |

在 1024 Token Prompt 下，KV Cache 将平均 Decode 延迟降低约 46%，对应约 `1.85×` 加速；但在 16 Token Prompt 下出现了负优化。Qwen3-0.6B 的单 Token Decode 计算量很小，短上下文中 Python 调度、Kernel Launch 以及教学实现的 `torch.cat` 开销足以覆盖节省的计算，因此不能概括成“开启 KV Cache 必然更快”。

KV Cache 数据量随序列长度线性增长。表中的最终 Cache 长度等于 `Prompt 长度 + 31`，因为第 32 个生成 Token 尚未作为下一轮输入写入 Cache。以 1024 Token Prompt 为例，最终 Cache 长度为 1055，实测 Tensor 数据量为 115.39 MiB，与理论公式一致。

正确性实验另行使用 float32，三组自然语言 Prompt 的结果如下：

| Prompt | 最大生成长度 | 实际对比 Token | Token 是否一致 | 最大 Logits 绝对误差 |
| --- | ---: | ---: | --- | ---: |
| `请用一句话介绍 KV Cache。` | 32 | 26（遇到 EOS） | 是 | 0.000055 |
| `Explain prefill in one short sentence.` | 16 | 16 | 是 | 0.000031 |
| `计算 17 乘以 23，只给出结果。` | 16 | 12（遇到 EOS） | 是 | 0.000042 |

第一组实验改用 bfloat16 时，最大 Logits 绝对误差为 `0.625`，并在第 24 步因候选 Logits 接近而出现生成分叉；这是不同矩阵形状和计算次序下的低精度数值差异，性能实验没有将“bfloat16 输出逐字相同”作为成立前提。

#### 可以推广的机制性结论

- KV Cache 通过保存各层历史 K/V，使 Decode 不再对所有历史 Token 重复执行模型层计算。
- KV Cache 不会减少 Prefill 对完整 Prompt 的计算，主要收益发生在 Decode。
- Cache 显存随层数、KV heads、head dimension、batch 和序列长度线性增长。
- 短上下文和小模型中，Cache 管理开销可能覆盖节省的计算；上下文变长后收益才更明显。

#### 不能推广的测量性结论

- `1.85×` 只属于本文记录的软件、硬件、模型和参数组合，不能外推到 7B、32B 或其他 GPU。
- 单请求 batch size 1 的结果不能代表并发服务吞吐或排队延迟。
- 动态 `torch.cat` 的教学实现不能代表 vLLM 等成熟推理引擎的 Cache 分配效率。
- 三次重复足以形成课程中的初步受控对照，但不应当视为长期稳定的生产基准。

### 已知边界

- 仅支持固定 revision 的 Qwen3-0.6B dense 模型。
- 仅支持 batch size 为 1、无 padding 的单请求实验。
- 仅实现 greedy decoding，默认聊天模板关闭 thinking。
- Cache 使用 `torch.cat` 逐步增长，会产生重新分配与复制开销。
- 没有实现 Cache 预分配、Block、分页、回收或并发请求管理。
- KV Cache 避免历史 Token 的模型层重复计算，但当前 Query 仍需读取全部历史 K/V，因此 Decode 并非与上下文长度无关的常数复杂度。
- Qwen3-0.6B 较小，Python、CUDA Kernel Launch 和 Tensor 拼接开销占比较高；绝对性能和加速倍数不能直接外推到更大的模型或其他硬件。
