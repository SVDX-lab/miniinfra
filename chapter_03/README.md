# 第 03 期：固定批处理

## 1. 对应书籍章节

对应《超轻量AI Infra教程—推理篇》第 03 章“固定批处理”。

## 2. 本节目的与实现概览

本目录使用基础 Python 和 PyTorch 独立实现 Qwen3-0.6B 的固定批处理推理，包含 Batched Prefill、Batched Decode 和 Batched KV Cache，并提供多个请求逐个使用 KV Cache 的串行 baseline。

本期代码是完整的独立实现，不 import `chapter_01`、`chapter_02` 或其他期文件。模型结构、权重加载、Tokenizer、生成循环、正确性检查和性能实验都位于 `chapter_03` 内。

本期只改变一个核心变量：多个请求逐个执行，或者合并成一个成员固定的 Batch 执行。Batch 开始后不加入新请求；已经完成的槽位不接收新请求，直到最长请求完成。Continuous Batching、Paged KV Cache 和请求调度不属于本期范围。

粗略步骤：对不同长度 Prompt 做 Padding 和 Mask；一次完成 Batched Prefill；维护按 Batch 槽位组织的 KV Cache；同步执行 Batched Decode，并与串行 baseline 对照。

## 3. 代码使用方法

### 验证环境

- GPU：NVIDIA GeForce RTX 3080 Ti 12GB
- CPU：Intel Xeon Gold 6226R
- 内存：31 GiB
- 操作系统：Ubuntu 22.04.5 LTS
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
python -m pip install -r chapter_03/requirements-lock.txt
```

如果只安装直接依赖，可以使用：

```bash
python -m pip install -r chapter_03/requirements.txt
```

首次运行会下载固定 revision 的模型。也可以通过 `--model` 传入包含模型文件的本地目录。

### 代码结构

```text
chapter_03/
├── qwen3_model.py          # 支持 Padding 与 Batched KV Cache 的完整手写模型
├── qwen3_tokenizer.py      # 独立 Byte-level BPE 与 non-thinking 聊天模板
├── batch_generation.py     # 左 Padding、串行 baseline 和固定 Batch 生成循环
├── run_inference.py        # 自然语言推理与两条路径对照
├── compare_batch.py        # 数值差异、请求隔离和 Decode 分叉实验
├── benchmark.py            # 吞吐、Padding、槽位利用率和显存实验
├── smoke_test.py           # 不下载权重、不要求 GPU 的随机小模型自检
├── benchmark-results.json
├── benchmark-capacity-results.json
├── compare-float32-results.json
└── compare-bfloat16-results.json
```

### CPU 快速自检

```bash
source .venv/bin/activate
python chapter_03/smoke_test.py
```

Smoke test 检查：

- 左 Padding、Attention Mask 和 Position IDs；
- 单请求与 Batched 路径的数值和 Token 一致性；
- 替换同批其他请求内容时的请求隔离性；
- Batched KV Cache 形状、长度和数据量；
- 不同输出预算下的完成状态、活跃槽位和利用率；
- 错误长度的 Attention Mask 能否被拒绝。

### 运行自然语言推理

```bash
source .venv/bin/activate
python chapter_03/run_inference.py \
  --prompts "用一句话解释 Prefill。" "用一句话解释 Decode。" \
  --max-new-tokens 32 \
  --mode both
```

默认使用 greedy decoding，并通过手写聊天模板关闭 Qwen3 thinking。`serial` 和 `batch` 两条路径都使用 KV Cache；前者逐个完成请求，后者将所有请求组成固定 Batch。

### Batch 的数据表示

不同长度的 Prompt 使用左 Padding：

```text
input_ids:      [batch, padded_prompt_length]
attention_mask: [batch, total_physical_cache_length]
position_ids:   [batch, current_query_length]

key/value:      [batch, num_key_value_heads, cache_length, head_dim]
```

左 Padding 只是物理存储位置。每个请求的有效 Token 仍从逻辑位置 0 开始计算 RoPE。Decode 时，新 Token 的 Position ID 根据该请求自己的有效长度增长，而不是直接使用 Padding 后的物理列号。

已完成请求仍占据原 Batch 槽位。后续占位 Token 的 Attention Mask 为 false，不会进入该请求的有效历史，但底层矩形 Tensor 和模型前向仍保留整个槽位。

### 数值结果与 Decode 分叉实验

float32 用于检查批处理逻辑，bfloat16 用于观察真实性能配置下的低精度路径差异：

```bash
source .venv/bin/activate
python chapter_03/compare_batch.py \
  --dtype float32 \
  --max-new-tokens 8 \
  --output chapter_03/compare-float32-results.local.json

python chapter_03/compare_batch.py \
  --dtype bfloat16 \
  --prompt "用一句话解释 Prefill。" \
  --max-new-tokens 8 \
  --output chapter_03/compare-bfloat16-results.local.json
```

程序固定目标请求 A，检查以下变化：

- Batch Size 1、2、4；
- A 位于 Batch 的不同位置；
- 保持 Batch Shape 不变，只替换其他请求的内容；
- 改变同批请求长度和 Padding 后长度；
- 同一 Batch Shape 下直接比较换同伴和换排列前后的 A。

记录最大与平均 Logits 绝对误差、Top-1 Token 一致率、首个分叉步骤、分叉时 Top-1/Top-2 间隔，以及分叉前最大误差。

#### 3080 Ti 正确性结果

float32 下，全部 8 步的 Top-1 Token 均一致。Batch Size、请求位置和 Padding 长度变化对应的最大 Logits 绝对误差不超过 `0.000032`。在 Batch Shape 不变时，只替换其他请求内容或调整请求排列，目标请求的 Logits 误差为 `0`，未观察到请求串扰。

bfloat16 下，目标 Prompt `用一句话解释 Prefill。` 在 Batch Size 2 时于第 4 步分叉：

- 分叉前最大 Logits 绝对误差：`0.587891`；
- 分叉步骤的 Batch Top-1/Top-2 间隔：`0`；
- Batch Size 2 的 8 步 Top-1 一致率：`37.5%`；
- Batch Size 4 在本次 8 步实验中没有分叉，最大误差为 `0.593750`；
- 同一 Batch Shape 下替换其他请求内容或调整排列，目标请求误差仍为 `0`。

分叉后两条生成路径的输入历史已经不同，因此后续最大 Logits 误差增至约 `18.39`，不能再解释为同一输入下的舍入误差。判断实现正确性的主要依据是 float32 对照与同 Shape 请求隔离实验。bfloat16 的 Batch Shape 会改变矩阵计算路径；当候选 Logits 接近时，小幅舍入差异可能改变 greedy Token，并使后续序列分叉。这不等于 Batch 降低了模型任务准确率。

### 性能、Padding 与槽位实验

```bash
source .venv/bin/activate
python chapter_03/benchmark.py \
  --batch-sizes 1 2 4 8 16 \
  --prompt-length 128 \
  --max-new-tokens 32 \
  --warmup 1 \
  --repeats 3 \
  --suite all \
  --output chapter_03/benchmark-results.local.json
```

程序使用长度精确的合成 Token IDs，强制生成指定数量的 Token，不因 EOS 提前停止。主实验的串行和 Batch 两组使用相同输入、权重、KV Cache、greedy decoding 和输出长度。

计时包含模型前向、动态 Cache 追加、Attention Mask 构造、LM Head、argmax 和必要的 CUDA 同步；不包含模型下载、权重加载、Tokenizer、请求排队、网络，以及两次模型前向之间的 Python 请求状态整理。端到端模型计算延迟为各步计时之和。

#### Batch Size 主实验

测试条件为 128 Token Prompt、固定生成 32 Token、bfloat16、1 次 warm-up 和 3 次正式重复。

| Batch Size | 串行延迟 (ms) | Batch 延迟 (ms) | Batch 输出吞吐 (token/s) | Batch 峰值显存 (MiB) | Batch Cache (MiB) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1061.87 | 977.65 | 32.73 | 1197.27 | 17.39 |
| 2 | 2105.63 | 1047.62 | 61.09 | 1247.71 | 34.78 |
| 4 | 4332.65 | 1072.87 | 119.44 | 1350.40 | 69.56 |
| 8 | 8662.40 | 1050.37 | 243.73 | 1555.78 | 139.12 |
| 16 | 16174.13 | 1157.13 | 446.13 | 1967.05 | 278.25 |

Batch Size 16 相对串行 makespan 约为 `13.98×` 加速，但它只属于本次小模型、短输出和单卡环境。Batch Size 1 两次独立测量存在约 8% 波动，不应解释为批处理本身带来的收益。

#### 大 Batch 容量实验

容量实验只运行固定 Batch 路径，避免为 Batch Size 32 以上的配置重复执行耗时很长的串行 baseline：

```bash
source .venv/bin/activate
python chapter_03/benchmark.py \
  --batch-sizes 32 64 128 192 196 \
  --prompt-length 128 \
  --max-new-tokens 32 \
  --warmup 1 \
  --repeats 3 \
  --suite throughput \
  --modes batch \
  --output chapter_03/benchmark-capacity-results.local.json
```

成功配置的结果如下：

| Batch Size | Prefill (ms) | Decode 迭代 (ms) | 端到端 (ms) | 输出吞吐 (token/s) | 峰值显存 (MiB) | Cache (MiB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 134.78 | 33.75 | 1181.01 | 867.08 | 2789.08 | 556.50 |
| 64 | 258.81 | 35.33 | 1354.03 | 1520.18 | 4431.15 | 1113.00 |
| 128 | 514.42 | 33.74 | 1560.51 | 2626.03 | 7717.28 | 2226.00 |
| 192 | 763.06 | 41.85 | 2060.48 | 2981.83 | 11003.42 | 3339.00 |
| 196 | 775.39 | 43.99 | 2139.18 | 2931.97 | 11208.80 | 3408.56 |

在相同进程启动方式下继续逐档测试，Batch Size 197、198、200、208 和 256 均在 Prefill LM Head 处 OOM。因此，对本期记录的模型、代码、128 Token Prompt、32 Token 输出和 bfloat16 环境，可以把实测整数边界写为：

```text
最大成功 Batch Size：196
最小失败 Batch Size：197
```

Batch Size 192 的输出吞吐为 `2981.83 token/s`，高于 Batch Size 196 的 `2931.97 token/s`。这说明最大可运行 Batch 并不是吞吐最优 Batch；接近显存边界后，Prefill 和 Decode 延迟均开始上升。

本期模型接口会对 Prefill 的全部隐藏状态执行 LM Head，产生 `[batch, prompt_length, vocab_size]` Logits。Batch Size 197 需要为这个大 Tensor 尝试单次分配约 `7.14 GiB`，最终在 12GB GPU 上 OOM。因此，`196` 是当前教学实现与本次参数组合的边界，不是 Qwen3-0.6B、3080 Ti 或成熟推理引擎的通用上限。只对最后一个 Prefill 位置计算 LM Head 可以显著改变该边界，但它属于另一项实现优化，本期没有将其混入固定批处理实验。

#### Padding 实验

固定 Batch Size 4、Padding 后长度 128、输出长度 32：

| Prompt 长度 | Padding 有效率 | Prefill (ms) | 有效 Prompt 吞吐 (token/s) | 峰值显存 (MiB) |
| --- | ---: | ---: | ---: | ---: |
| `[128, 128, 128, 128]` | 100.00% | 34.97 | 14641.75 | 1350.40 |
| `[16, 32, 64, 128]` | 46.88% | 34.88 | 6880.90 | 1350.40 |

两组物理 Tensor Shape 相同，所以 Prefill 延迟和峰值显存几乎相同；混合长度组只承载 46.88% 的有效 Prompt Token，有效 Prompt 吞吐下降约一半。这正是 Padding 浪费，而不是“短 Prompt 让同 Shape Batch 更快”。

#### 完成槽位实验

固定 Batch Size 4、Prompt 长度 128，分别设置输出预算 `[8, 16, 32, 64]`：

- 有效槽位利用率：`46.88%`；
- Batch 总模型计算延迟：`2082.66 ms`；
- 有效输出吞吐：`57.63 token/s`；
- 峰值显存：`1350.40 MiB`；
- 最终 Batched KV Cache：`83.56 MiB`。

四个请求分别在第 8、16、32、64 轮完成，但固定 Batch 必须运行到第 64 轮。较短请求完成后的槽位被 Mask，却没有接纳新请求；这正是下一期 Continuous Batching 要解决的问题。

### 可以推广的机制性结论

- 固定 Batch 通过共享模型执行提高 GPU 并行度，但吞吐、单请求延迟和显存之间存在权衡。
- Batched KV Cache 数据量随 Batch Size 线性增长。
- 矩形 Tensor 按最长 Prompt 执行；Padding 不会因为无效 Token 而自动省去模型计算。
- 已完成槽位虽然可以被 Mask，固定 Batch 仍不会自动回收并接纳新请求。
- Batch 中请求在数学上相互独立，但 Batch Shape 可能改变低精度 Kernel 的数值结果。

### 不能推广的测量性结论

- `13.98×`、最大成功 Batch Size `196` 和具体吞吐、显存数据只适用于本文记录的软硬件、模型和参数。
- Qwen3-0.6B 较小，绝对结果不能外推到 7B、32B 或其他 GPU。
- 本实现使用普通 Tensor 和 `torch.cat` 增长 Cache，不能代表成熟推理引擎的 Cache 管理效率。
- 三次重复适合作为本期受控教学实验，不应视为长期生产基准。
- 离线固定 Batch 吞吐不包含排队和网络时间，不能代表在线服务的 TTFT、延迟分位数或 SLO。
- 本期没有使用任务数据集测量模型准确率，不能把 bfloat16 Token 分叉直接表述为准确率下降。

### 已知边界

- 只支持固定 revision 的 Qwen3-0.6B dense 模型。
- 只实现 greedy decoding，默认关闭 thinking。
- 使用左 Padding 和矩形 Batched KV Cache。
- Batch 成员在运行期间固定，不加入新请求，也不复用完成槽位。
- Cache 通过 `torch.cat` 增长，包含重新分配和复制开销。
- Prefill 会为全部 Prompt 位置计算完整词表 Logits，大 Batch 的实测容量边界主要受该临时 Tensor 限制。
- 不实现采样、Continuous Batching、Paged KV Cache、Prefix Cache、调度或服务接口。
