# 第 04 期：Continuous Batching

## 1. 对应书籍章节

对应《超轻量AI Infra教程—推理篇》第 04 章“Continuous Batching”。

## 2. 本节目的与实现概览

本目录使用基础 Python 和 PyTorch 独立实现 Qwen3-0.6B 的 Continuous
Batching，并提供成员不可动态变化的固定批次 baseline。

本期代码不 import `chapter_01`、`chapter_02`、`chapter_03` 或其他期文件。
模型结构、权重加载、Tokenizer、请求状态机、dense KV Cache 槽位、两条执行路径、
正确性检查和性能实验全部位于 `chapter_04` 内。

本期只改变一个核心变量：已完成请求留下的运行名额能否在下一次迭代中被等待请求
复用。两条路径都使用 FCFS、相同的最大运行请求数、Batched Prefill、Batched
Decode 和 KV Cache。本期不实现 Paged KV Cache、Token Budget、优先级、抢占、
Chunked Prefill 或通用请求调度器。

粗略步骤：为请求维护 waiting/running/finished 状态；按迭代执行 Prefill 或 Decode；请求完成后立即释放槽位；在下一轮把等待请求补入运行批次，并与固定批处理对照。

## 3. 代码使用方法

### 验证环境

- GPU：NVIDIA GeForce RTX 3080 Ti 12GB
- CPU：Intel Xeon Gold 6226R
- 内存：31 GiB
- 操作系统：Ubuntu 22.04.5 LTS，Linux 5.15.0-113-generic
- Python：3.10.12
- GPU 驱动：595.80
- CUDA Runtime：12.6
- PyTorch：2.7.1+cu126
- huggingface-hub：0.31.4
- safetensors：0.5.3
- regex：2026.7.19
- 模型：`Qwen/Qwen3-0.6B`
- 模型 revision：`c1899de289a04d12100db370d81485cdf75e47ca`
- 性能实验 dtype：bfloat16

验证日期：2026-08-24。

### 独立安装

从项目根目录创建环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r chapter_04/requirements-lock.txt
```

如果只安装直接依赖，可以使用：

```bash
python -m pip install -r chapter_04/requirements.txt
```

首次运行会下载固定 revision 的模型。也可以通过 `--model` 传入包含模型文件的
本地目录。

### 代码结构

```text
chapter_04/
├── qwen3_model.py                 # 支持 Batched KV Cache 的完整手写模型
├── qwen3_tokenizer.py             # 独立 BPE 与 non-thinking 聊天模板
├── continuous_batching.py         # 请求状态、固定 baseline 和动态批处理循环
├── run_inference.py               # 自然语言推理与两条路径对照
├── compare_outputs.py             # 真实权重请求归属与输出一致性实验
├── benchmark.py                   # 槽位、到达轨迹和 Prefill 干扰实验
├── smoke_test.py                  # 不下载权重、不要求 GPU 的随机小模型自检
├── benchmark-results.json
├── compare-float32-results.json
└── compare-bfloat16-results.json
```

### 本期的执行语义

每个请求依次经历：

```text
waiting → running → finished
```

固定批次 baseline 按 FCFS 选出至多 `max_running_requests` 个请求。一批中已经
结束的请求不再产生有效 Token，但其行仍参与模型执行；只有整批结束后，下一批
请求才会进入。

Continuous Batching 在每次模型迭代前检查空槽位：

```text
接收已经到达的请求
  → 回收完成槽位
  → FCFS 填充空槽位
  → 若有新请求，执行一次 Batched Prefill
  → 否则对 dense 槽位执行一轮 Batched Decode
```

当空位与已到达请求同时存在时，本期固定为 Prefill 优先。该规则是为了得到唯一、
可复现的执行路径，不代表生产调度器的通用最优策略。

### Dense KV Cache 槽位

本期尚未引入 Paged KV Cache。每层 Cache 使用固定 batch 容量：

```text
key/value: [max_running_requests, num_kv_heads, physical_length, head_dim]
```

每个运行请求占用一个槽位，并独立保存逻辑长度、Position ID 和 Attention Mask。
请求结束后清空该槽位的有效 Mask；新请求 Prefill 完成后，其 Cache 被右对齐写入
空槽位。如果新 Prompt 比当前物理 Cache 更长，所有槽位需要在左侧扩容并搬移。

该实现不会在每轮 Decode 拆装每个请求的 Cache，但请求加入时仍可能发生扩容和
写入。它用于讲清动态槽位，不应视作高性能 Cache 管理方案；第 05 期再使用 Paged
KV Cache 解耦逻辑序列和物理存储。

### CPU 快速自检

```bash
source .venv/bin/activate
python chapter_04/smoke_test.py
```

Smoke test 检查：

- 左 Padding、Attention Mask 和 Position IDs；
- waiting、running、finished 状态变化；
- FCFS 动态补位顺序；
- Cache 槽位复用后，请求输出仍映射到正确请求；
- 固定批次、Continuous Batching 与逐请求输出一致；
- Continuous Batching 对异长输出的执行槽位有效率高于固定 baseline。

### 运行自然语言推理

```bash
source .venv/bin/activate
python chapter_04/run_inference.py \
  --prompts \
    "用一句话解释 Prefill。" \
    "用一句话解释 Decode。" \
    "用一句话解释 KV Cache。" \
    "用一句话解释固定批处理。" \
  --max-new-tokens 8 24 8 24 \
  --max-running-requests 2 \
  --mode both
```

聊天模板默认关闭 Qwen3 thinking，解码使用 greedy sampling。

### 正确性实验

```bash
source .venv/bin/activate
python chapter_04/compare_outputs.py \
  --dtype float32 \
  --max-new-tokens 8 \
  --output chapter_04/compare-float32-results.local.json

python chapter_04/compare_outputs.py \
  --dtype bfloat16 \
  --max-new-tokens 8 \
  --output chapter_04/compare-bfloat16-results.local.json
```

实验让四个请求以输出预算 `[2, 8, 3, 8]` 运行，最大运行请求数为 2。实测动态
加入轨迹为：

```text
[0, 1] → [2] → [3]
```

float32 下，固定批次和 Continuous Batching 的每个请求都与单请求独立执行产生
相同 Token。bfloat16 下，两种 Batch 路径彼此一致，但请求 0 的第二个 Token 与
Batch Size 1 路径不同。其原因是 Batch Shape 改变可能改变低精度矩阵 Kernel 和
浮点归约顺序；float32 对照与两条 Batch 路径的一致性表明本次实验未观察到动态
换槽造成的请求串扰。不能据此得出“批处理降低模型准确率”的结论。

### 性能实验

```bash
source .venv/bin/activate
python chapter_04/benchmark.py \
  --suite all \
  --max-running-requests 4 \
  --request-count 8 \
  --prompt-length 128 \
  --short-output 8 \
  --long-output 32 \
  --arrival-interval-ms 150 \
  --interference-arrival-ms 250 \
  --interference-prompt-length 512 \
  --warmup 1 \
  --repeats 4 \
  --output chapter_04/benchmark-results.local.json
```

输入为精确长度的合成 Token IDs，强制生成指定数量的 Token，不因 EOS 提前停止。
两条路径使用相同输入、权重、KV Cache、greedy decoding 和最大运行请求数。

请求到达使用逻辑时间线，不执行真实 `sleep`。Makespan、排队时间、TTFT 和 ITL
包含模型前向及 Continuous Batching 显式统计的 Cache 写入、扩容和输入准备；不
包含模型下载、权重加载、Tokenizer、网络以及模型调用之间的 Python 状态更新。

#### 3080 Ti 实测结果

以下为 1 次 warm-up、4 次交错重复的均值：

| 场景 | 路径 | Makespan (ms) | 输出吞吐 (token/s) | 执行槽位有效率 | TTFT p95 (ms) | ITL p95 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 异长输出 `[8,32,…]` | 固定 | 2187.66 | 73.21 | 62.50% | 1108.93 | 38.91 |
| 异长输出 `[8,32,…]` | Continuous | 1871.39 | 85.51 | 74.07% | 845.00 | 67.92 |
| 等长输出负对照 | 固定 | 1634.36 | 99.48 | 100.00% | 842.24 | 43.34 |
| 等长输出负对照 | Continuous | 1628.60 | 99.13 | 100.00% | 892.17 | 42.39 |
| 每 150 ms 到达一个请求 | 固定 | 3456.24 | 46.31 | 68.97% | 1511.07 | 36.50 |
| 每 150 ms 到达一个请求 | Continuous | 2326.61 | 68.82 | 65.58% | 205.39 | 71.48 |

异长输出场景中，Continuous Batching 的 Makespan 加速为 `1.169×`，输出吞吐
提高约 `16.8%`。等长负对照没有可提前复用的槽位，两条路径的平均 Makespan 只差
约 `0.4%`，四次样本区间也明显重叠；本次不能认为动态路径获得了稳定性能收益。
Continuous 路径仍有约 `2.3%` 的 Cache 管理耗时。

固定到达轨迹中，Continuous Batching 的 Makespan 加速为 `1.486×`，TTFT p95
明显降低，但 ITL p95 从 `36.50 ms` 增至 `71.48 ms`。吞吐、TTFT 和已有请求的
Token 间隔之间存在权衡。

#### 新请求 Prefill 对 Decode 的干扰

干扰实验让两个 128 Token、32 Token 输出的请求在 0 ms 到达，再让一个 512
Token Prompt 在 250 ms 到达。Continuous Batching 的 ITL 最大值为 `72.71 ms`，
固定 baseline 为 `35.07 ms`。Continuous 路径的 p95 只有 `34.89 ms`，会掩盖
这次局部尖峰。

本期只观察并解释完整 Prefill 对已有 Decode 的阻塞，不在这里加入 Chunked
Prefill。后续相应章节再研究如何拆分长 Prompt。

### 可以推广的机制性结论

- 固定 Batch 的完成槽位无法服务等待请求；Continuous Batching 可以在迭代边界复用槽位。
- 输出长度或到达时间越不整齐，可复用空位通常越多，但收益仍取决于实现开销和硬件。
- 新请求的完整 Prefill 可能降低 TTFT，同时制造已有请求的 ITL 尖峰。
- Dense Cache 可以实现动态槽位，但扩容和写入仍有搬移成本，Paged KV Cache 有独立价值。

### 不能推广的测量性结论

- 表中的吞吐、延迟、加速比和显存只适用于记录的软件、硬件、模型和参数。
- Qwen3-0.6B 较小，Python、Kernel Launch 和状态管理开销占比较高，不能直接外推到 7B、32B 或更大模型。
- 逻辑到达时间实验不是网络服务压测，不包含 HTTP、Tokenizer、真实并发和请求取消。
- 本期固定的 FCFS、Prefill 优先策略不代表所有工作负载下的最优调度策略。
