# 第 07 期：Chunked Prefill

## 1. 对应书籍章节

对应《超轻量AI Infra教程—推理篇》第 07 章“Chunked Prefill”。

## 2. 本节目的与实现概览

本目录使用基础 Python 和 PyTorch 独立实现 Qwen3-0.6B 的 Chunked Prefill。
代码完整包含模型结构、Tokenizer、Paged KV Cache、请求状态机、完整 Prefill baseline、
Chunked Prefill、正确性检查和性能实验，不 import `chapter_01`～`chapter_06` 的任何文件。

本期只改变一个核心变量：单个 Prompt 是否允许跨多个 Prefill iteration 推进。

- `full`：保留第 06 期的软 Token Budget；单个长 Prompt 必须完整执行，可以超预算。
- `chunked`：保存部分 Prompt 的 K/V 和逻辑位置，每轮 Prefill 严格遵守硬 Token Budget。

两条路径使用相同模型、Paged KV Cache、Block Size、FCFS、最大运行请求数、Token
Budget、Decode 保护规则、请求顺序、输入 Token、输出预算、greedy decoding 和计时方法。

粗略步骤：把长 Prompt 按 Token Budget 切成多轮；每轮保存已完成部分的 KV 与位置；在 Prefill Chunk 之间穿插 Decode；最后一块完成后生成首 Token，并与完整 Prefill 对照。

## 3. 代码使用方法

### 实现边界

本期 Chunked Prefill 是推理引擎内部对同一条 Token 序列的增量计算，不是文本语义
切分、RAG chunking 或流式接收 Prompt。中间 Chunk 只写入 KV Cache，最后一个
Chunk 才生成第一个输出 Token。

本期也不实现：

- 请求优先级、抢占、换出或重计算；
- 动态显存准入；
- Prefix Cache、Block 共享、引用计数或 Copy-on-Write；
- Prefill/Decode 分离；
- CUDA/Triton fused Chunked Prefill Kernel；
- 多机多卡调度。

多请求 Chunk 按相同历史前缀长度分组执行。普通 PyTorch reference 路径以正确性和
教学清晰度为优先，不能代表 vLLM、SGLang 或生产 Kernel 的绝对性能。

### 验证环境

- GPU：NVIDIA GeForce RTX 3080 Ti，11.63 GiB 可见显存
- CPU：Intel Xeon Gold 6248
- 操作系统：Ubuntu 22.04.5 LTS，Linux 5.15.0-113-generic
- Python：3.10.12
- GPU 驱动：595.80
- CUDA Runtime：12.6
- PyTorch：2.7.1+cu126
- 模型：`Qwen/Qwen3-0.6B`
- 模型 revision：`c1899de289a04d12100db370d81485cdf75e47ca`
- 性能 dtype：bfloat16

验证日期：2026-08-25。

### 独立安装

从项目根目录创建环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r chapter_07/requirements-lock.txt
```

只安装直接依赖时可以使用：

```bash
python -m pip install -r chapter_07/requirements.txt
```

首次运行会下载固定 revision 的模型。也可以通过 `--model` 传入已经下载的模型目录。

### 代码结构

```text
chapter_07/
├── qwen3_model.py                         # 完整 Qwen3 模型
├── qwen3_tokenizer.py                     # BPE 与 non-thinking 模板
├── paged_cache.py                         # 支持 Prompt Chunk 追加的 Paged KV Cache
├── scheduler.py                           # full/chunked 两种 Prefill 计划
├── engine.py                              # 增量 Prefill、Decode 与指标
├── run_inference.py                       # 自然语言推理
├── compare_prefill.py                     # 真实权重正确性实验
├── benchmark_chunked_prefill.py           # 长 Prompt 干扰与预算扫描
├── smoke_test.py                          # CPU 快速自检
├── compare-prefill-float32-results.json
├── compare-prefill-bfloat16-results.json
├── benchmark-chunked-prefill-results.json
├── benchmark-short-control-results.json
└── benchmark-no-decode-control-results.json
```

### Chunked Prefill 语义

请求新增 `prefilling` 状态和 `prefill_cursor`：

```text
waiting -> prefilling -> running -> finished
```

调度器为每个请求返回连续区间 `[start, end)`。Chunk 内 Position ID 使用 Prompt 的
全局逻辑位置；当前 Token 可以看到历史 K/V 和本 Chunk 中更早的 Token，不能看到未来
Token。Chunk 结束后只把本段 K/V 追加到 Block Table：

```text
完整 Prefill
[ prompt 0 ................................ prompt N ] -> first token

Chunked Prefill
[ chunk 0 ] -> Decode -> [ chunk 1 ] -> ... -> [ final chunk ] -> first token
```

多请求组成同一轮计划时，预算沿用实际矩形输入的 padded cost：

```text
prefill_cost = batch_size × max(chunk_length_in_batch)
```

`chunked` 路径始终检查：

```text
prefill_cost <= token_budget
```

### CPU 快速自检

```bash
source .venv/bin/activate
python chapter_07/smoke_test.py
```

自检覆盖：

- 完整 Prefill 的单请求超预算规则；
- Chunk 切分、cursor 单调推进和硬预算；
- 增量 Attention 与完整 Prefill 的 Logits/Token 一致性；
- Chunk 与 Block 边界错开时的追加；
- 请求完成后的 Block 释放与复用。

参考环境输出：

```text
Smoke test 通过
Prefill cursor、跨 Block 追加、硬 Token Budget 检查通过
full 与 chunked 逐请求输出一致
Prefill iteration: full=4, chunked=6
```

### 自然语言推理

```bash
source .venv/bin/activate
python chapter_07/run_inference.py \
  --max-new-tokens 16 \
  --max-running-requests 3 \
  --token-budget 32 \
  --block-size 16 \
  --mode both
```

聊天模板关闭 Qwen3 thinking，解码使用 greedy sampling。

### 真实权重正确性实验

```bash
source .venv/bin/activate
python chapter_07/compare_prefill.py \
  --dtype float32 \
  --token-budget 32 \
  --output chapter_07/compare-prefill-float32-results.local.json

python chapter_07/compare_prefill.py \
  --dtype bfloat16 \
  --token-budget 32 \
  --output chapter_07/compare-prefill-bfloat16-results.local.json
```

float32 正确性检查要求逐请求 Token 完全一致，并限制最后 Prompt 位置的最大 Logits
误差。bfloat16 单独记录因矩阵 Shape 改变产生的数值差异，不把低精度分叉解释为
请求串扰或模型质量变化。

3080 Ti 实测中，两种 dtype 的 full/chunked 均逐请求 Token 完全一致：

| dtype | Prompt Token 长度 | 首 Token Logits 最大误差 | full 超预算 | chunked 超预算 |
| --- | --- | ---: | ---: | ---: |
| float32 | 20 / 140 / 23 / 30 | 0.00003624 | 1 | 0 |
| bfloat16 | 20 / 140 / 23 / 30 | 0.37500000 | 1 | 0 |

bfloat16 的误差明显更大，但当前 Prompt 和输出预算没有发生 Token 分叉。这不能证明
所有低精度输入、Chunk Size 和 GPU Kernel 都必然产生相同 Token。

### 长 Prompt 干扰与 Token Budget 实验

```bash
source .venv/bin/activate
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python chapter_07/benchmark_chunked_prefill.py \
  --token-budgets 256 512 1024 \
  --warmup 1 \
  --repeats 3 \
  --output chapter_07/benchmark-chunked-prefill-results.local.json
```

默认工作负载先让 2 个 128 Token Prompt 持续 Decode，在逻辑时间 200 ms 加入一个
2048 Token Prompt。三个请求分别固定生成 48、48 和 8 个 Token。两条路径使用相同
Token Budget；由于初始两个 Prompt 的 padded cost 为 256，三个预算档位的初始执行
方式一致，主对照只在长 Prompt 到达后产生差异。

计时包含调度、模型前向、argmax、CUDA 同步、Paged KV Cache 重建与追加；不包含
模型下载、权重加载、Tokenizer、网络或真实 sleep。各配置先 warm-up，再按正序/逆序
交错重复。

#### 3080 Ti 实测结果

以下为 1 次 warm-up、3 次正式重复的均值：

| 路径 | Budget | 输出吞吐 (token/s) | 既有请求 ITL p95 (ms) | 既有请求最大 ITL (ms) | 长请求 TTFT (ms) | 最大单轮 Prefill (ms) | 峰值显存 (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 256 | 40.07 | 57.61 | 374.78 | 349.66 | 316.28 | 2318.30 |
| chunked | 256 | 37.59 | 108.76 | 120.60 | 844.64 | 65.39 | 1928.90 |
| full | 512 | 40.83 | 56.06 | 367.56 | 349.07 | 311.00 | 2318.30 |
| chunked | 512 | 40.60 | 128.11 | 145.45 | 504.38 | 88.89 | 1984.52 |
| full | 1024 | 40.86 | 63.06 | 369.26 | 346.66 | 310.80 | 2318.30 |
| chunked | 1024 | 40.93 | 60.28 | 217.09 | 368.82 | 161.26 | 2095.78 |

三组 full 都需要一次处理完整 2048 Token Prompt，因此各有 1 次软预算超限；三组
chunked 的超预算和硬预算违规均为 0。

Budget 256 将最大单轮 Prefill 缩短 79.3%、既有请求最大 ITL 降低 67.8%、峰值显存
降低 16.8%，代价是长请求 TTFT 增加 141.6%、输出吞吐下降 6.2%，ITL p95 反而增加
88.8%。它把一次长停顿拆成多次中等停顿，并不是延迟全面改善。

Budget 512 的最大 ITL 降低 60.4%，吞吐仅下降 0.6%，但长请求 TTFT 增加 44.5%，
ITL p95 增加 128.5%。Budget 1024 的最大 ITL 降低 41.2%，吞吐基本不变，长请求
TTFT 只增加 6.4%；它是本次离散扫描中更平衡的档位，但不是其他工作负载的通用最优值。

#### 短 Prompt 负对照

将延迟到达的 Prompt 从 2048 Token 改为 128 Token，并固定 Budget 256 后，两条路径
都只用一个 Prefill iteration 处理该请求，不存在可拆分工作：

| 路径 | 输出吞吐 (token/s) | 既有请求 ITL p95 (ms) | 最大 ITL (ms) | 长请求 TTFT (ms) | 最大单轮 Prefill (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 46.19 | 55.65 | 101.06 | 81.58 | 59.85 |
| chunked | 43.25 | 58.25 | 102.31 | 84.57 | 60.66 |

Chunked 没有机制性收益，输出吞吐低 6.4%，其他延迟接近。这项负对照支持“收益来自
拆分超预算长 Prompt”，而不是来自新增状态机本身。

#### 无并行 Decode 负对照

下面的实验移除两个初始请求，只运行单个 2048 Token Prompt。此时系统中没有既有
请求需要保护，ITL 隔离收益按定义为 0：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python chapter_07/benchmark_chunked_prefill.py \
  --initial-requests 0 \
  --long-prompt-length 2048 \
  --token-budgets 256 512 1024 \
  --warmup 1 \
  --repeats 3 \
  --output chapter_07/benchmark-no-decode-control-results.local.json
```

| 路径 | Budget | 输出吞吐 (token/s) | 长请求 TTFT (ms) | 最大单轮 Prefill (ms) | 峰值显存 (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 256 | 12.86 | 308.09 | 308.09 | 2279.80 |
| chunked | 256 | 9.75 | 506.11 | 66.19 | 1890.40 |
| full | 512 | 12.88 | 306.80 | 306.80 | 2279.80 |
| chunked | 512 | 12.03 | 348.85 | 90.57 | 1946.02 |
| full | 1024 | 12.79 | 308.80 | 308.80 | 2279.80 |
| chunked | 1024 | 13.02 | 297.67 | 162.42 | 2057.28 |

Budget 256 和 512 分别使吞吐下降约 24.2% 和 6.6%，同时拉长请求自身 TTFT；没有
其他 Decode 请求从更短的单轮停顿中受益。Budget 1024 的吞吐和 TTFT 略好于 full，
说明 Chunking 的 Shape 变化也可能形成局部正优化，不能把“无并行 Decode”简单写成
必然更慢。可以推广的结论只是：此时不存在需要改善的既有请求 ITL。

### 结果适用范围

Chunked Prefill 的主要目标是限制单轮 Prefill 停顿和硬化 Token Budget，不是无条件
提高吞吐。更小的 Chunk 可能降低既有请求最大 ITL 和单轮临时显存，但会增加调度、
Kernel Launch、历史 K/V 读取和 Python 状态管理开销，长请求 TTFT、ITL p95 与总吞吐
不保证同时改善。

具体性能数字只适用于本目录代码、固定 Qwen3-0.6B revision、3080 Ti 和所记录的合成
工作负载，不能直接外推到更大模型、其他 GPU、真实在线服务或成熟推理框架。

### 可以推广的机制性结论

- Chunked Prefill 把不可中断的长 Prompt 变成多个迭代级工作单元。
- 只有单请求 Prefill 可以拆分时，Token Budget 才能成为真正的硬约束。
- 更小的 Chunk 可以降低最大停顿和临时显存，但会增加迭代与执行开销。
- 最大 ITL 改善不代表 ITL p95、长请求 TTFT 和吞吐同时改善。

### 不能推广的测量性结论

- 256、512、1024 的相对结果依赖本期 Prompt 长度、到达轨迹和 reference 实现。
- 2048 Token Prompt 下的具体毫秒数、显存和吞吐不能外推到其他模型或 GPU。
- Qwen3-0.6B 较小，Python、dense Cache 重建和 Kernel Launch 占比较高。
- 单卡逻辑到达实验不能替代真实在线服务或 Prefill/Decode 分离测试。
