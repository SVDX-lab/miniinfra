# 第 08 期：Prefix Cache

## 1. 对应书籍章节

对应《超轻量AI Infra教程—推理篇》第 08 章“Prefix Cache”。

## 2. 本节目的与实现概览

本目录使用基础 Python 和 PyTorch 独立实现 Qwen3-0.6B 的块级 Prefix Cache。代码
完整包含模型结构、Tokenizer、Paged KV Cache、Chunked Prefill 调度器、链式哈希、
完整块匹配、引用计数、LRU 安全回收、正确性检查和共享长前缀主实验，不 import
`chapter_01`～`chapter_07` 的任何文件。

本期只改变一个核心变量：是否复用前序请求已经计算出的完整 Prompt KV Block。

- `disabled`：每条请求重新计算完整 Prompt。
- `enabled`：匹配最长连续完整前缀，只计算未命中的 Prompt 后缀。

两条路径使用相同模型、Paged KV Cache、Block Size、Chunked Prefill、硬 Token
Budget、FCFS、最大运行请求数、Decode、输入 Token、输出预算、greedy decoding 和
计时方法。

粗略步骤：对完整 Token Block 建立带父摘要的链式哈希；查找最长连续命中前缀；复用不可变 KV Block 并只计算后缀；通过引用计数和 leaf-LRU 安全释放或淘汰缓存。

## 3. 代码使用方法

### 实现边界

缓存键不是单独对当前 Token Block 求哈希，而是包含父块摘要：

```text
block_hash = SHA256(model_namespace || parent_hash || block_token_ids)
```

这是因为相同 Token Block 位于不同历史上下文时会产生不同 K/V。哈希命中后还会
确定性比较父摘要和 Token IDs，不能把摘要相同直接当作内容必然相同。

本期只共享已经填满并发布为不可变状态的 KV Block。未填满的 Prompt 尾块和 Decode
Block 始终归请求私有，因此不需要 Copy-on-Write。Prefix Cache 只保存 K/V，不保存
首 Token Logits，所以最多复用 `prompt_length - 1` 个 Token 对应的完整块，至少重新
计算最后一个 Prompt Token。

请求引用和缓存所有权分别管理：请求结束只减少请求引用；缓存索引仍持有的块不会
回到空闲池。容量不足时按固定 LRU 顺序从叶子缓存块开始淘汰；只有缓存所有权已经
移除且请求引用计数为零的物理块才能安全复用。

本期不实现：

- 部分 Block 共享或 Copy-on-Write；
- 并发冷请求的 in-flight Prefix 去重；
- 跨模型、跨 revision 或跨 KV dtype 共享缓存；
- 持久化、跨进程或分布式 Prefix Cache；
- Cache-aware 调度、动态缓存容量或淘汰策略对比；
- CUDA/Triton Prefix Cache Kernel。

### 验证环境

- GPU：NVIDIA GeForce RTX 3080 Ti，11.63 GiB 可见显存
- CPU：Intel Xeon Gold 6248
- 内存：31.34 GiB
- 操作系统：Ubuntu 22.04.5 LTS，Linux 5.15.0-113-generic
- Python：3.10.12
- GPU 驱动：595.80
- CUDA Runtime：12.6
- PyTorch：2.7.1+cu126
- 模型：`Qwen/Qwen3-0.6B`
- 模型 revision：`c1899de289a04d12100db370d81485cdf75e47ca`
- 性能 dtype：bfloat16

验证日期：2026-08-26。

### 独立安装

从项目根目录创建环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r chapter_08/requirements-lock.txt
```

首次运行会下载固定 revision 的 `Qwen/Qwen3-0.6B`。也可以通过 `--model` 传入已经
下载的模型目录。

### 代码结构

```text
chapter_08/
├── qwen3_model.py                 # 完整 Qwen3 模型
├── qwen3_tokenizer.py             # BPE 与 non-thinking 聊天模板
├── paged_cache.py                 # Paged KV Cache、Prefix 索引和引用计数
├── scheduler.py                   # 固定 Chunked Prefill 调度器
├── engine.py                      # Prefix Cache disabled/enabled 执行路径
├── run_inference.py               # 自然语言推理入口
├── compare_prefix_cache.py        # 真实权重正确性实验
├── benchmark_prefix_cache.py      # 共享 2048 Token 长前缀主实验
└── smoke_test.py                  # CPU 快速自检
```

### CPU 快速自检

```bash
source .venv/bin/activate
python chapter_08/smoke_test.py
```

自检覆盖完整块命中、不同历史上下文中的相同 Token Block、输出与 Logits 一致性、
请求释放后的引用计数，以及小容量下的 LRU 淘汰。

### 自然语言推理

```bash
source .venv/bin/activate
python chapter_08/run_inference.py \
  --max-new-tokens 16 \
  --token-budget 64 \
  --block-size 16 \
  --mode both
```

Tokenizer 使用 Qwen3 non-thinking 聊天模板，解码使用 greedy sampling。

### 正确性实验

```bash
source .venv/bin/activate
python chapter_08/compare_prefix_cache.py \
  --dtype float32 \
  --output chapter_08/compare-prefix-cache-float32-results.local.json

python chapter_08/compare_prefix_cache.py \
  --dtype bfloat16 \
  --output chapter_08/compare-prefix-cache-bfloat16-results.local.json
```

实验比较 disabled/enabled 的逐请求输出 Token 和首 Token Logits，并验证：

- 32 Token 对齐公共前缀命中 32 Token；
- 34 Token 非对齐公共前缀只命中前 32 Token；
- 相同第二个 Token Block 位于不同历史上下文时不能误命中；
- 所有请求结束后请求引用计数归零。

3080 Ti 实测结果：

| dtype | 输出 Token 一致 | 首 Token Logits 最大误差 | enabled 命中 Token | 请求引用归零 |
| --- | --- | ---: | --- | --- |
| float32 | 是 | 0.00002098 | 0 / 32 / 0 / 32 / 0 / 0 | 是 |
| bfloat16 | 是 | 0.25000000 | 0 / 32 / 0 / 32 / 0 / 0 | 是 |

bfloat16 的矩阵形状变化带来更大的数值误差，但当前固定输入和输出预算没有发生 Token
分叉。这不能证明任意 Prompt、Chunk Size 或 Kernel 都必然输出相同 Token。

### 共享长前缀主实验

```bash
source .venv/bin/activate
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python chapter_08/benchmark_prefix_cache.py \
  --requests 8 \
  --common-prefix-length 2048 \
  --private-suffix-length 64 \
  --max-new-tokens 32 \
  --token-budget 512 \
  --block-size 16 \
  --cache-capacity-blocks 160 \
  --warmup 1 \
  --repeats 3 \
  --output chapter_08/benchmark-prefix-cache-results.local.json
```

八条请求在同一引擎中按 FCFS 顺序执行，`max_running_requests=1`。第 1 条请求建立
冷缓存，第 2～8 条请求复用 2048 Token 公共前缀，不研究并发冷启动去重。

主实验报告：

- 冷请求 service TTFT；
- 热请求 service TTFT p50/p95；
- 实际执行和命中的 Prompt Token；
- 全部请求 Makespan 与输出吞吐；
- 峰值显存和物理 Block 峰值。

`service_ttft_ms` 从引擎接纳请求开始计时，包含 Prefix 查找、Prefill 和首 Token
选择，但不包含前面请求造成的 FCFS 队列等待。`makespan_ms` 包含八条顺序请求的完整
执行时间。计时不包含模型下载、权重加载、Tokenizer、网络或真实 sleep。

#### 3080 Ti 实测结果

以下为 1 次 warm-up、3 次正式交错重复的统计结果：

| 路径 | 冷请求 service TTFT (ms) | 热请求 service TTFT p50 (ms) | 实际 Prefill Token | 命中 Token | Makespan (ms) | 输出吞吐 (token/s) | 峰值显存 (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| disabled | 405.78 | 384.11 | 16896 | 0 | 13697.33 | 18.69 | 1954.77 |
| enabled | 397.25 | 52.64 | 2560 | 14336 | 12145.11 | 21.13 | 2234.77 |

第 1 条冷请求没有缓存可用，两条路径的 service TTFT 接近。第 2～8 条请求分别命中
2048 Token，热请求 service TTFT p50 降低 86.30%，实际 Prefill Token 减少
84.85%。全部请求 Makespan 降低 11.33%，输出吞吐提高 13.04%。三次 enabled
Makespan 为 11.59～13.01 秒，说明普通 PyTorch 教学路径仍有可见运行波动。

端到端改善小于 Prefill Token 和热请求 TTFT 的改善幅度：本期普通 PyTorch 教学实现
的 Paged Decode/Attention 路径占据了大量总执行时间，Prefix Cache 不会减少 Decode
计算。enabled 还为缓存池保留更多物理块，本次峰值显存增加 280 MiB。该显存差异同时
受固定池容量和 PyTorch 分配方式影响，不能当作任意 Prefix Cache 实现的固定开销。

### 结论边界

Prefix Cache 的机制性收益是减少重复前缀的 Prefill 计算；它会额外保留 KV Block，
并增加哈希查找和生命周期管理开销。3080 Ti 上 Qwen3-0.6B 的绝对 TTFT、吞吐和
显存数据不能直接外推到 7B、32B、生产 Kernel 或真实在线命中率。
