# 第 09 期：FlashAttention

## 1. 对应书籍章节

对应《超轻量AI Infra教程—推理篇》第 09 章“高性能 Attention”。

## 2. 本节目的与实现概览

本目录使用 PyTorch 和 Triton 独立实现 Qwen3-0.6B 推理以及一版
inference-only FlashAttention 前向 Kernel。代码包含模型权重加载、Tokenizer、
Paged KV Cache、Prefix Cache、Chunked Prefill 调度器、Eager Attention、Triton
FlashAttention、正确性检查、长序列微基准和端到端实验，不 import
`chapter_01`～`chapter_08` 的任何文件。

本期只改变一个核心变量：Attention 核心是否物化完整 Score 和 Probability 矩阵。

- `eager`：显式执行 `QK^T -> mask -> softmax -> PV`。
- `flash`：在 Triton Kernel 中分块读取 K/V，使用 Online Softmax 累加输出。

两条路径使用相同的 Q/K/V Projection、QK Norm、RoPE、GQA Head 展开、模型权重、
Paged KV Cache、Prefix Cache、Chunked Prefill、调度策略、输入 Token、输出预算和
greedy decoding。正式实现不调用 PyTorch SDPA，也不依赖第三方 `flash-attn` 包。

粗略步骤：保留显式物化 Score 的 Eager baseline；用 Triton 分块加载 Q/K/V；通过 Online Softmax 累积稳定归一化结果；验证 Kernel 数值后再接入完整模型比较显存与性能。

## 3. 代码使用方法

### 实现边界

本期手写 Kernel 支持：

- NVIDIA CUDA GPU；
- bfloat16 输入和 float32 Online Softmax 状态；
- `head_dim=128`；
- forward-only 推理；
- Full Prefill；
- `Q < K` 的 Chunked Prefill；
- Decode；
- Causal Mask、尾 Tile Mask 和每条序列的 Padding Mask；
- Batch 和多头 Attention。

本期不实现：

- backward、训练、dropout、FP8 或任意稠密 Attention Mask；
- 通用 Head Dim 和生产级 Autotuning；
- Triton Kernel 内的原生 GQA Head 映射；
- 直接根据 Block Table 读取非连续物理块的 Paged Attention；
- CUDA Graph、`torch.compile` 或第三方高性能 Attention 库。

Qwen3 的 K/V 在进入 Eager 或 FlashAttention 前都按 Query Head 数展开。Paged
Decode 也先根据相同 Block Table 读取并整理出稠密 K/V，再进入两条 Attention
路径。因此主实验没有把 GQA 优化或 Paged Attention 混入 FlashAttention 对照。

### 核心算法

Eager Attention 会完整物化：

```text
scores        = Q @ K^T / sqrt(d)
probabilities = softmax(mask(scores))
output        = probabilities @ V
```

手写 Kernel 每次只处理一个 `BLOCK_M x BLOCK_N` Score Tile，并为每个 Query 行
维护最大值 `m`、归一化和 `l` 与 Value 累加结果 `acc`：

```text
m_new = max(m, rowmax(scores_tile))
alpha = exp(m - m_new)
p     = exp(scores_tile - m_new)
l     = alpha * l + rowsum(p)
acc   = alpha * acc + p @ value_tile
m     = m_new

output = acc / l
```

完整 Score 和 Probability 不会写入 GPU 全局显存。计算量仍是二次方；降低的是
中间显存规模和 GPU 全局显存读写量。

Chunked Prefill 使用全局因果偏移：历史长度为 `P` 时，本轮局部 Query `q` 只能
读取满足 `k <= P + q` 的 Key。不能直接按局部位置使用 `k <= q`。

### 文件说明

- `flash_attention.py`：Triton FlashAttention 前向 Kernel 与输入检查。
- `qwen3_model.py`：完整 Qwen3-0.6B 模型、Eager/Flash 后端切换和权重加载。
- `paged_cache.py`：Paged KV Cache、Prefix Cache 和 Paged Decode 前向。
- `scheduler.py`：FCFS、硬 Token Budget 和 Chunked Prefill 调度器。
- `engine.py`：独立的 Prefill/Decode 推理主循环和指标统计。
- `qwen3_tokenizer.py`：不依赖 Transformers 的 Byte-level BPE Tokenizer。
- `smoke_test.py`：不下载权重、不要求 GPU 的 Eager 基础设施测试。
- `compare_attention.py`：FlashAttention Kernel CUDA 数值测试。
- `compare_model.py`：真实 Qwen3-0.6B 端到端正确性比较。
- `benchmark_attention.py`：独立进程长序列 Attention 微基准。
- `benchmark_engine.py`：真实模型 Chunked Prefill + Decode 端到端实验。
- `run_inference.py`：自然语言推理入口。

### 安装

建议使用独立虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r chapter_09/requirements.txt
```

参考服务器已经存在环境时：

```bash
cd /root/mini_infra/chapter_09
source ../.venv/bin/activate
```

首次运行真实模型会从 Hugging Face 下载固定 revision 的权重文件。也可以通过
`--model /path/to/local/model` 使用已经下载的模型目录。

### 运行与验证

所有命令均从 `chapter_09` 目录运行。

#### 1. CPU 基础设施 Smoke Test

```bash
python smoke_test.py
```

该测试不运行 Triton Kernel，检查完整独立引擎、Chunked Prefill、Paged KV Cache、
Prefix Cache、引用计数和输出归属。

#### 2. Kernel 数值正确性

```bash
python compare_attention.py \
  --output compare-attention-results.json
```

覆盖 Full Prefill、非整 Tile 长度、Chunked Prefill、左 Padding Batch 和 Decode。
Reference 是显式 Eager Attention，没有通过 SDPA 间接调用其他融合 Kernel。

#### 3. 真实模型正确性

```bash
python compare_model.py \
  --output compare-model-results.json
```

使用两个不同长度的合成 Prompt、Chunked Prefill、Paged Decode 和固定四 Token 输出，
比较首 Token Logits 与完整生成 Token。

#### 4. 长序列微基准

```bash
python benchmark_attention.py \
  --lengths 256 1024 4096 8192 \
  --warmup 5 \
  --repeats 20 \
  --output benchmark-attention-results.json
```

每个长度和后端在独立进程中运行，避免前一个 OOM 或 CUDA Allocator 状态污染后续
样本。输入创建和 Kernel 首次编译不计入延迟。增量峰值显存是在 Q/K/V、Mask 和一个
预热输出已经存在后，执行下一次 Attention 所增加的 `max_memory_allocated`。

#### 5. 端到端引擎实验

```bash
python benchmark_engine.py \
  --prompt-length 2048 \
  --max-new-tokens 16 \
  --token-budget 256 \
  --warmup 1 \
  --repeats 3 \
  --output benchmark-engine-results.json
```

该实验固定使用单请求、8 次 Chunked Prefill、15 次后续 Decode、Prefix Cache 关闭、
greedy decoding 和 EOS disabled。计时包含模型前向、Paged KV Cache 与调度，不包含
模型加载、Tokenizer、网络或真实 sleep。

#### 6. 自然语言推理

```bash
python run_inference.py \
  --attention-backend flash \
  --max-new-tokens 16
```

也可以传入 `--attention-backend eager` 运行 Baseline。

### 3080 Ti 验证环境

验证日期：2026-08-26。

| 项目 | 版本或配置 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3080 Ti 12GB（可用 11.63 GiB） |
| Compute Capability | 8.6 |
| 操作系统 | Ubuntu 内核 5.15.0-113-generic |
| Python | 3.10.12 |
| PyTorch | 2.7.1+cu126 |
| Triton | 3.3.1 |
| CUDA Runtime | 12.6 |
| NVIDIA Driver | 595.80 |
| 模型 | `Qwen/Qwen3-0.6B` |
| Revision | `c1899de289a04d12100db370d81485cdf75e47ca` |
| dtype | bfloat16 |

### 实测结果

#### Kernel 正确性

| 用例 | Q | K | 最大绝对误差 | 平均绝对误差 |
| --- | ---: | ---: | ---: | ---: |
| Full Prefill | 128 | 128 | 0.003906 | 0.000064 |
| 非整 Tile Full Prefill | 257 | 257 | 0.003906 | 0.000048 |
| Chunked Prefill | 64 | 320 | 0.000244 | 0.000023 |
| 左 Padding Chunk Batch | 64 | 320 | 0.000244 | 0.000023 |
| Decode | 1 | 1025 | 0.000122 | 0.000013 |

真实 Qwen3-0.6B 中，两条 Prompt 的首 Token Logits 最大绝对误差分别为
`0.189453` 和 `0.289062`，平均绝对误差为 `0.026927` 和 `0.046690`。28 层中的
bfloat16 归约顺序差异会累积，但本次四 Token greedy 输出完全一致。

#### Full Prefill Attention 微基准

固定 `batch=1`、`heads=16`、`head_dim=128`、bfloat16、Causal Attention。

| 长度 | Eager 延迟 | Flash 延迟 | 加速比 | Eager 增量峰值 | Flash 增量峰值 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 0.346 ms | 0.133 ms | 2.60x | 10.1 MiB | 1.0 MiB |
| 1024 | 1.070 ms | 0.316 ms | 3.38x | 161.0 MiB | 4.0 MiB |
| 4096 | 15.937 ms | 4.309 ms | 3.70x | 2576.1 MiB | 16.0 MiB |
| 8192 | OOM | 15.791 ms | 不计算 | OOM | 32.0 MiB |

8192 Token 结果只说明本微基准中 Eager Attention 无法完成，而手写 Kernel 可以
完成；不能把 OOM 写成无限加速，也不能据此外推完整模型的最大上下文长度。

#### 真实模型端到端实验

工作负载为 2048 Prompt Token、256 Token 硬预算、16 个固定输出 Token。

| 指标 | Eager | Flash | 对比 |
| --- | ---: | ---: | ---: |
| 模型时间 | 987.12 ms | 895.50 ms | 1.10x |
| Makespan | 1166.86 ms | 1075.65 ms | 1.08x |
| Service TTFT | 507.00 ms | 472.80 ms | 1.07x |
| 输出吞吐 | 13.72 token/s | 14.88 token/s | +8.5% |
| 峰值显存 | 1890.7 MiB | 1865.7 MiB | -1.3% |

Chunked Prefill 把 Query 长度限制为 256，完整引擎还包含 Projection、MLP、Paged
KV 读取、Cache 管理、调度和 Decode，因此端到端收益远小于 Full Prefill Attention
微基准。这是预期结果，不应把 4096 Token Kernel 的 3.70x 直接写成模型加速比。

### 结论边界

可以推广的机制性结论：

- Online Softmax 可以在不保存完整 Attention 矩阵的情况下得到相同数学结果。
- Eager Attention 的中间显存随 Full Prefill 长度平方增长。
- FlashAttention 的显存优势随 Full Prefill 序列增长更加明显。
- Kernel 级加速会受到其他模型层和引擎开销的限制。

不能推广的测量性结论：

- 这里的加速比只对本代码、版本、输入形状和 RTX 3080 Ti 负责。
- Qwen3-0.6B 的端到端结果不能外推到 7B、32B 或其他模型。
- 手写教学 Kernel 不代表 PyTorch SDPA、官方 flash-attn 或生产推理框架性能。
- 8192 Token 的微基准成功不代表完整 Qwen3 推理能在相同显存下使用该长度。
- 本实现没有直接读取 Paged KV Block，不能代表生产 Paged Attention 性能。
