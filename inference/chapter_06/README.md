# 第 06 期：迭代级推理请求调度器

## 1. 对应书籍章节

对应《超轻量AI Infra教程—推理篇》第 06 章“推理请求调度器”。

## 2. 本节目的与实现概览

本目录使用基础 Python 和 PyTorch 独立实现 Qwen3-0.6B 的迭代级请求调度器。
代码包含模型结构、Tokenizer、Paged KV Cache、请求状态机、两种调度策略、
正确性检查和性能实验，不 import `chapter_01`～`chapter_05` 的任何文件。

本期只改变调度策略：

- `baseline`：FCFS，有空闲槽位时一次 Prefill 所有可加入请求；
- `budgeted`：FCFS，按 padded Prefill Token 扣软 Token Budget；已有 Decode
  请求时，两个 Prefill iteration 之间至少执行一次 Decode iteration。

模型、Paged KV Cache、Block Size、最大运行请求数、请求顺序、输入 Token、输出预算、
greedy decoding 和计时方法保持一致。

粗略步骤：维护等待、运行和完成队列；每轮根据 FCFS 与 Token Budget 选择请求；调度 Prefill/Decode 并更新状态；比较 baseline 与 budgeted 策略的延迟、公平性和吞吐。

## 3. 代码使用方法

### 实现边界

本期没有实现 Chunked Prefill。单个 Prompt 大于 Token Budget 时无法被拆分，调度器
允许队头请求单独超预算执行，避免 FCFS 队列永久阻塞。因此这里的 Token Budget 是
软预算。

本期也不实现：

- 根据空闲 KV Block 做显存准入；
- 请求抢占、换出和重计算；
- 优先级调度或截止时间调度；
- Prefill/Decode 分离；
- 多机多卡调度。

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
- 性能 dtype：bfloat16

验证日期：2026-08-25。

### 独立安装

从项目根目录创建环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r chapter_06/requirements-lock.txt
```

只安装直接依赖时可以使用：

```bash
python -m pip install -r chapter_06/requirements.txt
```

首次运行会下载固定 revision 的模型。也可以通过 `--model` 传入已经下载的模型目录。

### 代码结构

```text
chapter_06/
├── qwen3_model.py                              # 完整 Qwen3 模型
├── qwen3_tokenizer.py                          # BPE 与 non-thinking 模板
├── paged_cache.py                              # 独立 Paged KV Cache
├── scheduler.py                                # 两种迭代级调度策略
├── engine.py                                   # 调度输出与模型执行器
├── run_inference.py                            # 自然语言推理
├── compare_schedulers.py                       # 真实权重正确性实验
├── benchmark_scheduler.py                      # Prefill burst 与预算扫描
├── smoke_test.py                               # CPU 快速自检
├── compare-schedulers-float32-results.json
├── compare-schedulers-bfloat16-results.json
└── benchmark-scheduler-results.json
```

### 调度规则

调度器每轮只返回一个 `SchedulerOutput`，其中记录阶段、请求、Token 数和是否发生
单请求超预算。执行器根据输出运行一次 Batched Prefill 或一次 Decode。

当前 Prefill 使用左 Padding，因此预算按实际输入模型的 padded token 数计算：

```text
prefill_cost = batch_size × max(prompt_length_in_batch)
decode_cost  = running_request_count
```

`budgeted` 策略保持 FCFS。队头请求加入后若没有足够预算继续加入下一请求，本轮选择
立即停止，不跳过队头去寻找更短 Prompt。

### CPU 快速自检

```bash
source .venv/bin/activate
python chapter_06/smoke_test.py
```

自检覆盖：

- FCFS Prefill 选择；
- padded Token Budget 扣减；
- Prefill 后的 Decode 保护；
- 超预算单请求规则；
- 两种策略的逐请求输出一致性；
- Block 释放与复用。

参考环境输出：

```text
Smoke test 通过
FCFS、Token Budget、Decode 保护和超预算单请求检查通过
baseline 与 budgeted 逐请求输出一致
最大连续 Prefill iteration: baseline=3, budgeted=1
```

### 自然语言推理

```bash
source .venv/bin/activate
python chapter_06/run_inference.py \
  --max-new-tokens 16 \
  --max-running-requests 3 \
  --token-budget 128 \
  --block-size 16 \
  --policy both
```

聊天模板关闭 Qwen3 thinking，解码使用 greedy sampling。

### 真实权重正确性实验

```bash
source .venv/bin/activate
python chapter_06/compare_schedulers.py \
  --dtype float32 \
  --token-budget 64 \
  --output chapter_06/compare-schedulers-float32-results.local.json

python chapter_06/compare_schedulers.py \
  --dtype bfloat16 \
  --token-budget 64 \
  --output chapter_06/compare-schedulers-bfloat16-results.local.json
```

3080 Ti 实测中，float32 和 bfloat16 的两种调度策略均逐请求 Token 完全一致。
调度轨迹不同：baseline 的最大连续 Prefill iteration 为 2，budgeted 为 1。

这项检查验证当前 Prompt、请求状态与调度轨迹，没有证明所有 bfloat16 Batch Shape
都必然逐 Token 相同。

### Prefill burst 与 Token Budget 实验

```bash
source .venv/bin/activate
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python chapter_06/benchmark_scheduler.py \
  --token-budgets 256 512 1024 2048 4096 \
  --warmup 1 \
  --repeats 3 \
  --output chapter_06/benchmark-scheduler-results.local.json
```

工作负载首先让 4 个 128 Token Prompt 持续 Decode。在逻辑时间 200 ms，一次到达
12 个长短混合 Prompt，长度为：

```text
128 / 512 / 128 / 1024 / 128 / 2048 /
128 / 512 / 256 / 1024 / 256 / 2048
```

最大运行请求数为 8。初始请求生成 64 Token，突发请求生成 8 Token。计时包含调度、
模型前向、argmax、CUDA 同步、Block 写入和释放；不包含模型下载、权重加载、
Tokenizer、网络和真实 sleep。三次正式重复交错执行各策略。

| 策略 | 输出吞吐 (token/s) | 既有请求 ITL p95 (ms) | 既有请求最大 ITL (ms) | 突发请求 TTFT p95 (ms) | Prefill Padding |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 63.01 | 98.52 | 906.84 | 3061.44 | 58.5% |
| budget 256 | 58.71 | 223.42 | 394.88 | 2936.85 | 0.0% |
| budget 512 | 68.93 | 210.00 | 381.11 | 2787.28 | 0.0% |
| budget 1024 | 70.87 | 211.28 | 378.67 | 2614.65 | 8.1% |
| budget 2048 | 71.26 | 207.37 | 376.79 | 2578.44 | 11.7% |
| budget 4096 | 66.60 | 288.98 | 541.19 | 2795.30 | 45.2% |

本次工作负载中，budget 2048 相对 baseline：

- 输出吞吐提高 13.1%；
- 既有请求最大 ITL 降低 58.5%；
- 突发请求 TTFT p95 降低 15.8%；
- Prefill padded token 数减少 53.0%。

ITL p95 同时从 98.52 ms 升至 207.37 ms。baseline 把大量 Prefill 集中成较少的长
停顿，budgeted 将它拆成更多次中等停顿，因此最大值下降而 p95 上升。这个结果不能
写成延迟全面改善。

budget 256 还出现 6 次单请求超预算，吞吐比 baseline 低 6.8%。budget 2048 是当前
离散扫描中的最佳吞吐档位，不代表其他长度分布、模型或 GPU 的通用最优值。

### 结果适用范围

性能数字只适用于上述代码、Qwen3-0.6B、3080 Ti 和合成工作负载。当前 Paged
Attention 是普通 PyTorch reference 实现，绝对吞吐不能外推到 vLLM、SGLang 或
专用 CUDA/Triton Kernel。单卡实验也不能替代真实 Prefill/Decode 分离测试。
