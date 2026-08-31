# 第 05 期：Paged KV Cache

## 1. 对应书籍章节

对应《超轻量AI Infra教程—推理篇》第 05 章“Paged KV Cache”。

## 2. 本节目的与实现概览

本目录使用基础 Python 和 PyTorch 独立实现 Qwen3-0.6B 的 Paged KV Cache，
并提供按最大物理长度保存 K/V 的 dense cache baseline。

本期代码不 import `chapter_01`、`chapter_02`、`chapter_03`、`chapter_04` 或其他期
文件。模型结构、权重加载、Tokenizer、Continuous Batching、Block Pool、Block
Table、两条 Attention 路径、正确性检查和性能实验全部位于 `chapter_05` 内。

本期只改变一个核心变量：请求的逻辑 KV 序列是绑定到按全局最长请求扩张的 dense
Tensor，还是映射到可以独立分配、释放和复用的固定大小物理 Block。两条路径都使用
FCFS、相同最大运行请求数、Prefill 优先、greedy decoding 和相同请求到达轨迹。

粗略步骤：把 KV 显存切成固定大小物理 Block；用 Block Table 建立逻辑 Token 到物理块的映射；按需分配、写入和释放 Block；比较 dense cache 与 paged cache 的正确性、容量和碎片。

## 3. 代码使用方法

### 验证环境

- GPU：NVIDIA GeForce RTX 3080 Ti，11.63 GiB 可见显存
- CPU：Intel Xeon Gold 6226R
- 内存：31.34 GiB
- 操作系统：Ubuntu 22.04.5 LTS，Linux 5.15.0-113-generic
- Python：3.10.12
- GPU 驱动：595.80
- CUDA Runtime：12.6
- PyTorch：2.7.1+cu126
- 模型：`Qwen/Qwen3-0.6B`
- 模型 revision：`c1899de289a04d12100db370d81485cdf75e47ca`
- 性能实验 dtype：bfloat16

验证日期：2026-08-25。

### 独立安装

从项目根目录创建环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r chapter_05/requirements-lock.txt
```

如果只安装直接依赖，也可以使用：

```bash
python -m pip install -r chapter_05/requirements.txt
```

首次运行会下载固定 revision 的模型。也可以通过 `--model` 传入已经下载的模型目录。

### 代码结构

```text
chapter_05/
├── qwen3_model.py                    # 完整 Qwen3 模型与 dense Attention
├── qwen3_tokenizer.py                # 独立 BPE 与 non-thinking 聊天模板
├── paged_cache.py                    # Block Pool、Block Table 和分页 Attention
├── cache_engine.py                   # dense/paged 两条 Continuous Batching 路径
├── run_inference.py                  # 自然语言推理对照
├── compare_logits.py                 # 逐 Token Logits 数值对照
├── compare_outputs.py                # 真实权重正确性与块复用实验
├── benchmark.py                      # 长短比例和 Block Size 实验
├── benchmark_capacity.py             # 固定显存下的并发容量实验
├── smoke_test.py                     # 不下载权重、不要求 GPU 的快速自检
├── benchmark-results.json
├── benchmark-block-size-results.json
├── benchmark-capacity-results.json
├── compare-logits-float32-results.json
├── compare-logits-bfloat16-results.json
├── compare-float32-results.json
└── compare-bfloat16-results.json
```

### Dense baseline

Dense 路径延续第 04 期暴露的问题，但本期提供完整独立实现。每层 Cache 形状为：

```text
[max_running_requests, num_kv_heads, physical_length, head_dim]
```

`physical_length` 等于该运行集合历史上出现过的最大 Cache 长度。新长 Prompt 加入时，
所有层和所有槽位一起左扩容，已有 K/V 被复制到新 Tensor。长请求完成后，只清空其
Mask；只要还有请求运行，物理长度不会自动缩短。

因此，一个 4096 Token 请求可以让其他 128 Token 请求也在 4096 级别的物理宽度上
执行 masked Attention。

### Paged KV Cache

Paged 路径使用预分配物理 Block Pool。单个物理 Block 同时保存模型全部层的 K/V：

```text
[num_layers, 2, num_kv_heads, block_size, head_dim]
```

池本身使用连续 Tensor，以避免教学实现为每个物理块创建独立 CUDA Tensor。每个
请求只保存 Block Table 和逻辑序列长度：

```text
request A -> [7, 2, 11]
request B -> [5]
```

执行过程为：

```text
Prefill：ceil(prompt_length / block_size) 个物理块
Decode：写入尾块；尾块满时再取一个空闲块
Attention：按逻辑长度分组，通过 Block Table 读取有效块
Finish：物理块 ID 返回空闲堆，后续请求复用
```

Block ID 对所有模型层使用同一映射。换槽或释放请求不会移动其他请求已经保存的 K/V。

本期根据输入工作负载中可能同时运行的最大 Cache 长度，保守计算物理 Block Pool
容量并一次预分配，以形成可重复的离线实验。本期不实现根据实时空闲 Block 进行
动态准入；生产系统中的过载保护与安全容量仍需结合实际请求分布、延迟目标和显存
余量确定，不属于本期实验范围。

### Paged Attention 的实现边界

本期没有实现 CUDA/Triton fused PagedAttention Kernel。参考路径使用普通 PyTorch：

1. 相同有效上下文长度的请求组成一个 Attention 分组；
2. 使用二维 Block ID Tensor 一次读取该组物理块；
3. 在 `[batch, blocks, heads, block_size, head_dim]` 布局上计算 Attention；
4. 尾块超过逻辑长度的 Token 槽位被 Mask。

短请求不会补齐到全局最长 Cache，因此能够减少无效 K/V 读取和 Attention Score
计算。但分组、Block gather 和额外 Kernel Launch 会降低矩阵批量效率。本期保留该
开销，并通过实测区分“存储与计算量减少”和“端到端一定变快”这两个不同命题。

### CPU 快速自检

```bash
source .venv/bin/activate
python chapter_05/smoke_test.py
```

Smoke test 检查：

- Prefill 跨越 Block 边界后的 K/V 内容；
- Decode 写入尾块；
- 请求完成后的 Block 释放和确定性复用；
- FCFS 动态加入与槽位复用；
- dense 与 paged 路径的逐请求输出一致性。

### 自然语言推理

```bash
source .venv/bin/activate
python chapter_05/run_inference.py \
  --prompts \
    "用一句话解释 Paged KV Cache。" \
    "用一句话解释 Block Table。" \
  --max-new-tokens 16 \
  --max-running-requests 2 \
  --block-size 16 \
  --mode both
```

聊天模板默认关闭 Qwen3 thinking，解码使用 greedy sampling。

### 真实权重正确性实验

首先在 Batch Size 1 下逐步比较 dense 和 paged Logits：

```bash
source .venv/bin/activate
python chapter_05/compare_logits.py \
  --dtype float32 \
  --block-size 16 \
  --max-new-tokens 8 \
  --output chapter_05/compare-logits-float32-results.local.json
```

然后使用动态请求轨迹检查换槽和 Block 复用：

```bash
source .venv/bin/activate
python chapter_05/compare_outputs.py \
  --dtype float32 \
  --block-sizes 4 16 \
  --max-new-tokens 8 \
  --output chapter_05/compare-float32-results.local.json

python chapter_05/compare_outputs.py \
  --dtype bfloat16 \
  --block-sizes 4 16 \
  --max-new-tokens 8 \
  --output chapter_05/compare-bfloat16-results.local.json
```

逐 Token 数值对照使用两个自然语言 Prompt、Block Size 16 和 8 个输出 Token。
float32 的最大 Logits 绝对误差为 `0.0000172`，bfloat16 为 `0.4375`；两种 dtype
的全部 Top-1 Token 都一致。float32 阈值为 `1e-4`，bfloat16 阈值为 `1.0`。

四个请求使用输出预算 `[2, 8, 3, 8]` 和最大运行请求数 2，执行中会发生请求完成、
新请求入槽和物理块复用。

3080 Ti 实测中，float32 和 bfloat16 下，Block Size 4、16 的 paged 路径都与 dense
路径逐请求 Token 完全一致。Block Size 4 复用 13 个物理块，Block Size 16 复用
4 个物理块；FCFS 加入轨迹均为：

```text
[0, 1] -> [2] -> [3]
```

这组结果验证了当前 Prompt、Decode、跨块位置和复用轨迹，没有观察到换槽或 Block
复用造成的请求串扰。它不代表所有 bfloat16 Batch Shape 都会逐 Token 相同。

### 长短请求比例实验

```bash
source .venv/bin/activate
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python chapter_05/benchmark.py \
  --suite ratios \
  --long-prompt-lengths 128 512 2048 3072 \
  --short-prompt-length 128 \
  --short-output 64 \
  --followup-output 48 \
  --long-output 8 \
  --max-running-requests 8 \
  --block-size 16 \
  --warmup 1 \
  --repeats 3 \
  --output chapter_05/benchmark-results.local.json
```

工作负载首先让 7 个短请求运行，在逻辑时间 200 ms 加入一个长 Prompt，并在
201 ms 后让 9 个后续短请求进入等待队列。长请求只生成 8 Token，完成后 paged
路径立即释放其 Block；dense 路径保留已经扩张的物理宽度。

计时包含模型前向、argmax、CUDA 同步、dense 扩容写入、Block 分配与写入；不包含
模型下载、权重加载、Tokenizer、网络和真实 sleep。请求到达使用逻辑时间线。三次
正式重复交错执行两条路径。

| 长 Prompt | Paged KV 读取减少 | Cache Pool 减少 | Dense 吞吐 | Paged 吞吐 | Paged 同并发吞吐变化 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 15.9% | 19.8% | 227.66 token/s | 132.55 token/s | -41.8% |
| 512 | 71.1% | 62.0% | 226.79 token/s | 125.09 token/s | -44.8% |
| 2048 | 91.7% | 80.2% | 188.15 token/s | 120.24 token/s | -36.1% |
| 3072 | 94.1% | 82.6% | 201.43 token/s | 124.80 token/s | -38.0% |

长度越不均匀，Paged KV Cache 减少的物理存储和 KV 访问越多。但当前普通 PyTorch
参考路径没有把计算量节省转化为同并发延迟收益。Qwen3-0.6B 的 Decode 较小，dense
路径的一次大批量矩阵运算效率很高；paged 路径的 Block gather、分组 Attention 和
额外 Kernel Launch 占据了更高比例。

因此，这组结果不能写成“Paged KV Cache 在相同并发下一定更快”。成熟引擎通常使用
专门的 PagedAttention Kernel 直接读取 Block Table，本期结果不能代表这些 Kernel。

### Block Size 实验

```bash
source .venv/bin/activate
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python chapter_05/benchmark.py \
  --suite blocks \
  --long-prompt-length 2048 \
  --block-sizes 8 16 32 64 \
  --short-output 64 \
  --followup-output 48 \
  --warmup 1 \
  --repeats 3 \
  --output chapter_05/benchmark-block-size-results.local.json
```

| Block Size | Cache Pool 减少 | Paged 吞吐 | 同并发吞吐变化 |
| ---: | ---: | ---: | ---: |
| 8 | 80.3% | 120.07 token/s | -42.0% |
| 16 | 80.2% | 128.51 token/s | -40.5% |
| 32 | 80.1% | 131.35 token/s | -39.6% |
| 64 | 79.9% | 131.47 token/s | -38.4% |

本工作负载中，Block Size 增大降低了部分索引开销，速度略有改善，而尾块碎片只使
Cache Pool 节省下降 0.4 个百分点。这个结果依赖长度分布；不能据此得出 64 Token
是其他负载的最佳 Block Size。本期主配置仍使用 16 Token。

### 固定显存容量实验

```bash
source .venv/bin/activate
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python chapter_05/benchmark_capacity.py \
  --concurrencies 4 8 12 16 \
  --total-requests 24 \
  --short-prompt-length 128 \
  --long-prompt-length 4096 \
  --short-output 32 \
  --followup-output 32 \
  --long-output 8 \
  --block-size 16 \
  --repeats 3 \
  --output chapter_05/benchmark-capacity-results.local.json
```

| 路径 | 最大运行请求数 | 状态 | 输出吞吐 | 峰值已分配显存 |
| --- | ---: | --- | ---: | ---: |
| Dense | 4 | 成功 | 99.20 token/s | 5027.87 MiB |
| Paged | 4 | 成功 | 68.47 token/s | 4769.33 MiB |
| Dense | 8 | 成功 | 156.35 token/s | 8734.79 MiB |
| Paged | 8 | 成功 | 119.13 token/s | 4839.33 MiB |
| Dense | 12 | OOM | — | — |
| Paged | 12 | 成功 | 155.63 token/s | 4909.33 MiB |
| Dense | 16 | OOM | — | — |
| Paged | 16 | 成功 | 157.80 token/s | 4979.33 MiB |

在这组固定工作负载和 12GB GPU 上，dense 最大成功并发为 8，paged 至少成功运行到
16。Paged 并发 16 的吞吐比 dense 最大成功配置高约 0.9%，峰值已分配显存低约
43.0%。主要性能收益来自更高的可运行并发抵消了 reference Attention 的单轮开销，
而不是相同并发下每轮 Decode 更快。

`12/16` 是本次测试的离散档位，不表示精确整数 OOM 边界。实验没有逐个测试 9～11
或 13～15，也没有实现运行时准入控制。

### 可以推广的机制性结论

- Block Table 解耦逻辑 Token 位置和物理 K/V 地址；逻辑连续不要求物理连续。
- 请求只需要为实际长度申请 Block，内部浪费被限制在每个活跃请求的最后一个 Block。
- 请求完成后可以直接归还物理块，不需要移动其他请求的 K/V。
- 请求长度越不均匀，dense 的全局最大长度浪费通常越大，分页的存储收益越明显。
- 减少 K/V 访问和理论计算量不保证教学实现端到端更快；Kernel、gather 和批量效率同样重要。
- 固定显存下，更高的可运行并发可以转化为更高的系统可实现吞吐。

### 不能推广的测量性结论

- `157.80 token/s`、约 0.9% 最大吞吐提升和并发 16 只属于本文的软件、硬件和负载。
- Dense 并发 12 OOM 不代表所有 Qwen3-0.6B 实现或所有 3080 Ti 都以 8 为上限。
- PyTorch reference paged Attention 的 36%～45% 同并发负优化不能代表 vLLM、SGLang
  或专用 CUDA/Triton PagedAttention Kernel。
- Block Size 64 在本次扫描中稍快，不代表它是其他长度分布或模型的最佳配置。
- Qwen3-0.6B 很小，Python、Block gather 和 Kernel Launch 的占比不能外推到大模型。
- 逻辑到达时间不包含网络、Tokenizer、真实并发、请求取消和超时，不是服务 SLO。

### 已知边界

- 仅支持固定 revision 的 Qwen3-0.6B dense 模型。
- 仅实现 greedy decoding，聊天模板默认关闭 thinking。
- Block Pool 容量在单次实验开始前根据已知请求上界预分配。
- 没有实现动态准入、抢占、换出、重计算或 CPU offload。
- 没有实现 Chunked Prefill、Prefix Cache、Block 共享、引用计数或 Copy-on-Write。
- 没有实现 fused PagedAttention Kernel；普通 PyTorch 分组路径以教学清晰度为优先。
- 三次重复足以形成课程中的初步受控对照，不是生产性能基准。
