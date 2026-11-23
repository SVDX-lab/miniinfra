# 第 10 期：CUDA Graph——降低 Decode 启动开销

## 1. 对应书籍章节

对应《超轻量AI Infra教程—推理篇》第 10 章“CUDA Graph 优化 Decode”。

## 2. 本节目的与实现概览

本目录使用基础 Python、PyTorch 和 Triton 独立实现 Qwen3-0.6B 的静态 Paged
Decode 与 CUDA Graph Replay。代码完整包含模型结构、Tokenizer、FlashAttention、
静态物理 KV Block Pool、固定宽度 Block Table、正确性实验、性能实验、Profiler
实验和自然语言入口，不 import `chapter_01`～`chapter_09` 的任何文件。

本期只改变一个核心变量：一轮静态 Decode 的 CUDA Kernel 提交方式。

- `static_eager`：Python 每轮按普通 Eager 路径逐个提交 Kernel；
- `cuda_graph`：重放提前捕获的同一组 Kernel。

两条路径调用完全相同的 Decode Tensor 程序，使用相同模型、权重、Batch Capacity、
Context Bucket、Block Table、KV Pool、FlashAttention、输入 Token、greedy argmax 和
EOS 设置。静态化不是实验变量，而是两条路径共有的控制条件。

粗略步骤：为固定 Batch Capacity 与 Context Bucket 预分配静态 Tensor；先运行 Eager Decode baseline；捕获相同 Tensor 程序为 CUDA Graph；更新输入缓冲并 Replay，比较启动事件与 TPOT。

## 3. 代码使用方法

### 实现边界

CUDA Graph 要求捕获和重放时使用稳定的 Tensor 地址与形状。本期因此为每个
`(batch_capacity, context_bucket)` 建立独立 Runner，并预分配：

- `input_ids`、`position_ids`、`sequence_lengths` 和 `active_mask`；
- 固定宽度 Block Table；
- 覆盖整个 Context Bucket 的物理 KV Block；
- Graph 输出 Token 和 Logits Buffer；
- 每个非活跃 Slot 独占的安全 Sink Block。

Block Table 中的物理 Block ID、有效长度和输入 Token 可以在 Replay 前更新，但
Tensor 本身的地址和形状不变。Graph 内先按 Block Table 读取固定数量的物理块，再用
`sequence_lengths` 和 `active_mask` 屏蔽无效位置。

这仍然展示了 Paged KV Cache 的间接寻址，但不是生产级 PagedAttention：

- 没有新增直接遍历 Block Table 的 Attention Kernel；
- 每个 Slot 提前预留完整 Context Bucket，而不是按需申请物理块；
- Bucket 尾部会发生无效 K/V 读取和 Mask 计算；
- Block 分配、请求准入、取消和 Slot 生命周期仍应在 Graph 外处理。

本期也不实现 `torch.compile`、Graph Update、条件节点、动态输出 Shape、多 GPU
Graph、量化或生产级 Graph Cache 淘汰策略。Prefill 保持 Eager，不属于本期优化对象。

### 验证环境

验证日期：2026-08-26。

| 项目 | 版本或配置 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3080 Ti 12GB（可见 11.63 GiB） |
| Compute Capability | 8.6 |
| CPU | Intel Xeon Gold 6226R |
| 内存 | 31.34 GiB |
| 操作系统 | Ubuntu 22.04，Linux 5.15.0-113-generic |
| Python | 3.10.12 |
| PyTorch | 2.7.1+cu126 |
| Triton | 3.3.1 |
| CUDA Runtime | 12.6 |
| NVIDIA Driver | 595.80 |
| 模型 | `Qwen/Qwen3-0.6B` |
| Revision | `c1899de289a04d12100db370d81485cdf75e47ca` |
| dtype | bfloat16 |
| Attention | 本期独立包含的 Triton FlashAttention |

### 独立安装

从项目根目录创建环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r chapter_10/requirements.txt
```

首次运行会下载固定 revision 的 `Qwen/Qwen3-0.6B`。也可以通过 `--model` 指向已经
下载的模型目录。

### 文件说明

```text
chapter_10/
├── qwen3_model.py             # 完整 Qwen3-0.6B 模型
├── qwen3_tokenizer.py         # 独立 BPE 与 non-thinking 模板
├── flash_attention.py         # 独立 Triton FlashAttention 前向 Kernel
├── static_decode.py           # 静态 Paged KV、Eager 与 CUDA Graph Runner
├── experiment_utils.py        # 模型加载、环境记录和结果输出
├── smoke_test.py              # 无权重、无 GPU 的快速自检
├── compare_cuda_graph.py      # 真实权重正确性实验
├── benchmark_cuda_graph.py    # Batch/上下文扫描主实验
├── benchmark_end_to_end.py    # 单请求 Prefill + Decode 完整指标
├── profile_launches.py        # Kernel 提交事件 Profiler 实验
└── run_inference.py           # 自然语言推理入口
```

### CPU 快速自检

```bash
source .venv/bin/activate
python chapter_10/smoke_test.py
```

自检覆盖静态 Buffer、Block Table 唯一性、Prompt KV 写入、Decode KV 追加、非活跃
Slot Mask、跨 Block 读取和重复运行确定性。

### 真实权重正确性实验

```bash
source .venv/bin/activate
python chapter_10/compare_cuda_graph.py \
  --dtype bfloat16 \
  --attention-backend flash \
  --capacity 4 \
  --requests 2 \
  --prompt-length 32 \
  --decode-steps 4 \
  --context-bucket 64 \
  --output chapter_10/compare-cuda-graph-results.local.json
```

该实验比较两条路径的完整生成 Token、每一步 Logits、新追加的 Decode K/V，以及非
活跃 Slot 输出。Graph 在所有 Slot inactive 的安全状态下完成预热和 Capture，之后再
加载真实 Prompt，避免 Capture 本身消耗正式请求状态。

3080 Ti 实测中，两条请求均生成 5 个 Token，`static_eager` 与 `cuda_graph` 的 Token
完全一致，逐步 Logits 最大绝对误差为 0，新追加 Decode K/V 最大绝对误差为 0，两个
非活跃 Slot 均保持 Pad。该结果验证了当前固定输入和 Graph 实例，不证明所有模型、
dtype 和 Kernel 都必然逐位一致。

### Decode 性能主实验

```bash
source .venv/bin/activate
python chapter_10/benchmark_cuda_graph.py \
  --capacities 1 2 4 \
  --prompt-lengths 128 512 2048 \
  --decode-steps 32 \
  --warmup 1 \
  --repeats 3 \
  --output chapter_10/benchmark-cuda-graph-results.local.json
```

`context_bucket` 自动取能够容纳 `prompt_length + decode_steps` 的最小 Block Size
整数倍。每次正式样本都会重新执行相同 Prompt 的 Prefill 以恢复 KV 和输入状态，但
Prefill 不计入 Decode 计时。

主实验记录：

- CUDA Event 测得的设备侧 Decode TPOT；
- 包含最终同步的 Decode Wall TPOT；
- Host 提交整段 Decode 所用时间；
- 输出吞吐；
- 峰值及增量峰值显存；
- Capture 时间、Capture 后新增显存和估算摊销 Token 数。

正式实验关闭 EOS，argmax 和下一 Token 回填均在 Graph 内完成，不在每一步执行
`.item()`。否则 CPU 同步会掩盖 Graph Replay 的稳态收益。

#### 3080 Ti 主实验结果

以下为 1 次 warm-up、3 次正式交错重复的均值。TPOT 使用 CUDA Event 测量；Capture
已经在稳态样本前完成，不计入 TPOT。

| Capacity | Prompt | Bucket | Eager TPOT | Graph TPOT | 加速比 | Capture | Eager 吞吐 | Graph 吞吐 | Eager 峰值 | Graph 峰值 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 160 | 37.90 ms | 5.22 ms | 7.26x | 115.82 ms | 26.4 token/s | 192.1 token/s | 1183.8 MiB | 1181.6 MiB |
| 1 | 512 | 544 | 39.14 ms | 6.13 ms | 6.38x | 95.52 ms | 25.6 token/s | 163.6 token/s | 1241.1 MiB | 1231.7 MiB |
| 1 | 2048 | 2080 | 41.08 ms | 9.69 ms | 4.24x | 99.94 ms | 24.4 token/s | 103.2 token/s | 1436.3 MiB | 1407.8 MiB |
| 2 | 128 | 160 | 41.18 ms | 5.81 ms | 7.09x | 139.69 ms | 48.6 token/s | 344.6 token/s | 1229.4 MiB | 1224.7 MiB |
| 2 | 512 | 544 | 40.48 ms | 7.46 ms | 5.43x | 109.95 ms | 49.4 token/s | 268.4 token/s | 1331.8 MiB | 1316.9 MiB |
| 2 | 2048 | 2080 | 39.98 ms | 12.76 ms | 3.13x | 101.16 ms | 50.0 token/s | 156.8 token/s | 1717.9 MiB | 1661.0 MiB |
| 4 | 128 | 160 | 39.86 ms | 6.38 ms | 6.25x | 101.45 ms | 100.4 token/s | 629.6 token/s | 1299.1 MiB | 1289.3 MiB |
| 4 | 512 | 544 | 40.87 ms | 8.79 ms | 4.65x | 100.22 ms | 97.9 token/s | 455.1 token/s | 1495.2 MiB | 1465.4 MiB |
| 4 | 2048 | 2080 | 40.87 ms | 18.15 ms | 2.25x | 105.10 ms | 97.9 token/s | 220.4 token/s | 2259.4 MiB | 2145.5 MiB |

当前教学实现的 Eager 单步包含约 1900 次 Kernel 提交，所以 Qwen3-0.6B 上的 Host
Launch 开销占比很高。随着上下文和 Batch 增长，Graph TPOT 从 5.22 ms 增至 18.15
ms，加速比从 7.26x 降至 2.25x：Graph 没有减少 K/V 读取和矩阵计算，GPU 工作变重
后，能够消除的 Host 开销占比自然下降。

九组 Capture 时间为 95.52～139.69 ms。用“Capture 时间 ÷ 每步节省时间”做简单
估算，约 2.89～4.63 个 Decode 步可以摊销 Capture。这个估算假定 Graph 实例可以
复用，不包含生产系统中的 Graph Cache miss、并发 Capture 或驱逐成本。

静态 KV Pool 随 Capacity 和 Context Bucket 线性增大：本实验从 19.2 MiB 增至
917.0 MiB。表中 Graph 稳态峰值略低于 Eager，是因为 Replay 复用了 Capture 阶段
建立的内存；不能据此声称 CUDA Graph 普遍降低显存。Capture 后 CUDA Allocator 的
新增 reserved memory 在各组为 2～22 MiB，且会受到前序实验的内存池复用影响。

### Profiler 启动事件实验

```bash
source .venv/bin/activate
python chapter_10/profile_launches.py \
  --capacity 1 \
  --prompt-length 128 \
  --context-bucket 160 \
  --output chapter_10/profile-launches-results.local.json
```

Profiler 会扰动延迟，因此这里只使用它观察 `cudaLaunchKernel`、`cudaGraphLaunch`
等提交事件数量。性能结论以不启用 Profiler 的主实验为准。

3080 Ti 的单步 Profile 中，`static_eager` 记录到 1903 次 `cudaLaunchKernel` 和 4
次 `cudaMemcpyAsync`；`cuda_graph` 的 Host 提交侧记录到 1 次
`cudaGraphLaunch`。Profiler 仍能看到 Graph 内部的 GPU Activity，这不等于 Graph
只执行了一个 GPU Kernel；减少的是 Host API 提交次数。

### 单请求端到端实验

```bash
source .venv/bin/activate
python chapter_10/benchmark_end_to_end.py \
  --prompt-length 2048 \
  --max-new-tokens 32 \
  --warmup 1 \
  --repeats 3 \
  --output chapter_10/benchmark-end-to-end-results.local.json
```

该实验补充 TTFT、TPOT、端到端延迟和输出吞吐。TTFT 从 Prefill 开始计时，到首
Token argmax 和 Prompt KV 写入完成；不包含模型加载、Tokenizer 或网络。Capture
成本仍单独报告，不混入稳态请求。

#### 3080 Ti 端到端结果

工作负载为单请求、2048 Prompt Token、固定生成 32 Token、EOS disabled。以下为
1 次 warm-up 和 3 次正式交错重复的均值：

| 路径 | TTFT | Decode TPOT | 端到端延迟 | 输出吞吐 |
| --- | ---: | ---: | ---: | ---: |
| static_eager | 291.34 ms | 38.85 ms | 1495.84 ms | 21.39 token/s |
| cuda_graph | 291.96 ms | 9.63 ms | 590.75 ms | 54.17 token/s |

两条路径的 TTFT 接近，因为 Prefill 没有进入 CUDA Graph。Decode TPOT 提升 4.03x，
但完整请求端到端只提升 2.53x，说明不受本期机制影响的 Prefill 会稀释 Decode 优化。

### 自然语言推理

```bash
source .venv/bin/activate
python chapter_10/run_inference.py \
  --mode cuda_graph \
  --prompt "请用一句话解释 CUDA Graph。" \
  --max-new-tokens 32 \
  --context-bucket 512
```

自然语言入口为了逐 Token 检查 EOS，会把输出 Token 同步回 CPU，因此不代表主实验
的无逐步同步路径。可传入 `--mode static_eager` 使用相同静态计算路径。

### 结论边界

可以推广的机制性结论：CUDA Graph 可以把一组稳定的 CUDA 工作重放为一次 Host
提交，但不会减少模型算术量；它需要用静态 Buffer、Bucket 或 Padding 换取可捕获性，
并产生 Capture 时间和额外显存。

不能推广的测量性结论：Qwen3-0.6B、手写 FlashAttention、固定输出长度和 3080 Ti
上的加速比不能直接外推到更大模型、其他 GPU、动态在线流量或生产推理框架。请求
频繁进出、逐 Token EOS 检查、取消和 Bucket 命中率都会改变实际收益。
