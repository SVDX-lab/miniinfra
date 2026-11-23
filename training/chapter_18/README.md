# 第 18 期：固定 Mini-batch

## 对应篇章

对应《超轻量AI Infra教程—训练篇》第 03 章；合并后的连续编号为第 18 章。

## 显卡要求

最低与参考配置均为 1 张 GPU。

第 17 期已经明确了每个样本中哪些 Token 参与 Assistant-only Loss。本期只改变样本进入模型的方式：Baseline 在一次参数更新内逐个样本执行 Forward/Backward 并累加梯度；Variant 把同一组样本右侧 Padding 到组内最大长度，再用一个二维 Batch 一次执行。两条路径使用相同样本、Token、Labels、有效监督 Token、Loss reduction 和一次 AdamW Step。

## 问题、假设与边界

- 问题：一次更新中的多个变长样本，合并执行后能否保持训练目标，并提高单卡吞吐？
- 假设：正确屏蔽 Padding 后，两条路径的 Loss、梯度和参数更新在 FP32 容差内一致；批量矩阵计算可能更快，但会增加物理 Token 和激活显存。
- 唯一变量：多个 `[1, T_i]` 模型调用，还是一个 `[B, max(T_i)]` 模型调用。
- 两种归约：`token_mean` 把所有有效监督 Token 等权平均；`sample_mean` 先对每个样本求平均，再让每个样本等权。变长样本下二者不是同一个目标。
- 固定项：模型与 Tokenizer revision、固定 MiniMind 样本及顺序、Assistant-only Labels、FP32、TF32 关闭、Eager Attention、AdamW、全局梯度裁剪阈值 1、单卡与每组一次 Optimizer Step。
- 排除项：训练版 Batch Sampler、长度分桶、Packing、BF16、跨 Step 梯度累积、预取、编译和分布式训练。

## 文件

- `assistant_loss.py`：沿用第 17 期定义、在本目录独立实现的 Qwen3 Chat Template 与 Assistant-only Labels。
- `mini_batch.py`：本期核心实现，包括去除外部 Padding、固定组内右 Padding、两种 Loss reduction，以及逐样本和批量反向路径。
- `qwen3_model.py`、`qwen3_tokenizer.py`：手写 Qwen3 模型与 Byte-level BPE Tokenizer；不依赖 Transformers。
- `training.py`：单卡 FP32 训练 Step、CUDA Event 分段计时、显存与遥测。
- `smoke_test.py`：CPU 极小模型的 Shape、Padding、两种 reduction、梯度和更新对照。
- `compare_correctness.py`：真实 Qwen3 FP32 的 Loss、有效 Logits、梯度和一次 AdamW 更新对照。
- `benchmark.py`：Batch Size 1/2/4/8 的短 Benchmark 和 OOM 记录。
- `train.py`：固定样本组上的短程轨迹入口。
- `prepare_data.py`：固定数据下载、大小和 SHA256 校验、32 条样本筛选。

## 环境和数据

参考环境为 Python 3.12、PyTorch `2.12.1+cu130`、CUDA 13.0。不同 CUDA 主版本需要重新解析 PyTorch 与 NVIDIA wheel。

```bash
cd chapter_18
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python prepare_data.py --samples 32
```

无法直连 Hugging Face 主站时，可显式设置兼容镜像；模型、数据 revision、文件大小和 SHA256 不变。

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  python prepare_data.py --samples 32
```

## 运行顺序

```bash
CUDA_VISIBLE_DEVICES="" python smoke_test.py
python compare_correctness.py --reduction token_mean
python compare_correctness.py --reduction sample_mean \
  --output correctness-sample-mean-results.local.json
python train.py --steps 5
python benchmark.py --batch-sizes 1 2 4 8 --warmup 1 --steps 3
```

默认结果写入 `*.local.json`。Smoke Test 不下载模型，不占用 GPU。真实正确性和 Benchmark 使用一张 GPU。

## Loss reduction

设第 `i` 个样本有 `n_i` 个有效监督 Token，逐 Token Loss 为 `ell_{ij}`。按有效 Token 加权时：

```text
L_token = sum_i sum_j ell_ij / sum_i n_i
```

逐样本路径必须让第 `i` 个样本的 mean Loss 乘 `n_i / sum_i n_i` 后再 Backward。平均样本 Loss 则是：

```text
L_sample = (1 / B) sum_i [(1 / n_i) sum_j ell_ij]
```

此时每个样本的 mean Loss 乘 `1 / B`。不能把两种权重混用，否则 Batch 化前后会暗中改变目标。

## 指标与计时

`physical_tokens` 是实际送入模型的槽位数：Baseline 为 `sum(T_i)`，Variant 为 `B × max(T_i)`。`non_padding_tokens` 为 `sum(T_i)`；`supervised_tokens` 按 shift 后的 `labels[:, 1:] != -100` 计数。主吞吐是有效监督 Token/s，同时报告非 Padding Token/s、物理 Token/s 和 Samples/s。

Step Time 包含内存取组、同步 H2D、Forward、Backward、全局梯度裁剪和 AdamW；不包含 JSONL 读取、Chat Template、Tokenize、Checkpoint 和 `nvidia-smi` 查询。CUDA 区间使用 Event，端到端使用同步后的 Wall Clock。通信和 Checkpoint 时间均为 0。输入没有使用 Pinned Memory，数据已在测量前编码到 Host 内存，因此本章不把磁盘读取带宽当作热路径指标。


## 适用边界

- 主线模型为 `Qwen/Qwen3-0.6B-Base`，训练目标为全参数 SFT。
- `smoke_test.py` 不依赖真实模型下载或 GPU；真实权重正确性检查与 Benchmark 需要上文注明的 GPU 配置。
- 本目录只提供代码、依赖和运行说明。正确性与 Benchmark 命令会在本地生成 `*.local.json`，仓库不附带实验结果。
