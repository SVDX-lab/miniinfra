# 第 20 期：Sequence Packing

## 对应篇章

对应《超轻量AI Infra教程—训练篇》第 05 章；合并后的连续编号为第 20 章。

## 显卡要求

最低与参考配置均为 1 张 GPU。

第 19 期在有限 Shuffle Window 内按长度分桶，减少了 Batch 内的长度差异。但只要一个样本独占一行，短样本就仍以 Padding 填满自己的行。本期只改变样本的物理排列：Baseline 沿用 Padding Batch（Batch Size 4，右 Padding 到组内最大长度）；Variant 把多个不切分的样本装入一个固定容量的物理序列（Pack），每个样本的 Position ID 从 0 重置，Attention 使用 Block-diagonal 加性因果 Mask 禁止跨样本读取，Loss Mask 在样本边界显式截断。简单的 EOS 拼接加跨样本 Attention 只作为负对照。

## 问题、假设与边界

- 问题：把短样本打包进同一物理序列，能否在严格隔离样本 Attention 的前提下提高物理槽位利用率与有效监督 Token/s？Mask 构造与逐步调度的固定开销会不会吃掉收益？
- 假设：打包收益由长度分布决定。长度多样时贪心装包把多个短样本塞进同一物理序列，减少 Padding 物理槽位；本章自然语料（26% 样本短于 512 Token）能产生自然的多样本 Pack。但每步固定成本（Optimizer、Kernel 启动）随 Pack 数线性放大，单 Pack 每步的设计可能仍慢于 4 行 Padding Batch。
- 唯一变量：样本的物理排列方式——独立成行加 Padding，与多样本共居一个 Pack。
- 固定项：与第 19 期相同（模型/Tokenizer/数据 revision、固定 512 条样本、Assistant-only Labels、`token_mean`、FP32、TF32 关闭、Eager Attention、AdamW、学习率 2e-5、梯度裁剪 1、种子 20260912）。两条路径每个 Epoch 处理完全相同的样本集合与监督 Token 总量；Step 数不同（Baseline 128，Packing 由装包决定）是 Packing 的固有属性。
- 排除项：FlashAttention（第 26 期）、跨样本 Attention 的正式使用（仅负对照）、样本切分与 best-fit 装包、动态容量、多 Pack 组批（[G, P] 二维 Batch，属工程后续）、离线 Tokenize、BF16、梯度累积、分布式。

## 文件

- `sequence_packing.py`：本期核心实现。`greedy_packing_plan`（固定种子 Shuffle 后 First-Fit 装包，样本不切分）、`build_pack`（拼接 Token、Position ID 重置、边界 Label 截断）、`build_block_diagonal_mask`（样本内下三角、跨样本与右 Padding 全屏蔽的加性 Mask）。
- `qwen3_model.py`：`forward` 增加可选 `position_ids` 与 `attention_mask_4d`；两者缺省时行为与第 33/34 期逐位一致。
- `mini_batch.py`、`training.py`：Padding 路径与第 33/34 期相同；`train_pack_step` 复用同一 Optimizer、裁剪与计时骨架，Mask 构造计入 H2D 段计时。
- `smoke_test.py`：CPU 极小模型的装包性质、Mask 结构、等价、隔离与负对照检查。
- `compare_correctness.py`：真实 Qwen3 FP32 的 Pack 等价、内容替换隔离证明、负对照与梯度锚定。
- `benchmark.py`：Regime A（自然语料）与 Regime B（合成形状机制放大）两条路径对照。
- `prepare_data.py`：数据下载、校验与 512 条固定样本筛选（数据流水章块语料 `sft_mini_512 @ 9b06bee2`）。

## 环境和数据

参考依赖环境为 Python 3.12、PyTorch `2.12.1+cu130`、CUDA 13.0。语料固定为 `sft_mini_512.jsonl @ 9b06bee29c0d5685647ac11b0dcea9f572c5064a` 的前 512 条。

```bash
cd chapter_20
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 python prepare_data.py --samples 512
```

## 运行顺序

```bash
CUDA_VISIBLE_DEVICES="" python smoke_test.py
python compare_correctness.py --output correctness-results.local.json
python benchmark.py --output benchmark-results.local.json
```

Smoke Test 不下载模型、不占用 GPU。样本编码在线执行；pack plan 是 `(lengths, capacity, seed, epoch)` 的纯函数，结果 JSON 保存完整 plan。

## Packing 的三个正确性约束

把多个样本放进一个物理序列，必须同时满足三件事，缺一不可：

1. Position ID 重置。RoPE 以相对位置编码，每个样本的 Position ID 必须从 0 重新开始，否则后续样本的全部位置相对独立执行都发生平移。
2. Block-diagonal Causal Mask。查询只允许看到同一样本中不晚于自己的 Key；跨样本与右侧 Padding 全部以 `finfo(dtype).min` 屏蔽。加性最小值使被屏蔽位置的 Softmax 权重精确为 0，这是隔离可以逐位验证的原因。
3. 边界 Loss Mask。Causal Shift 让位置 t 预测 labels[t+1]；每个样本的首个 Label 必须显式置为 `ignore_index`，否则前一个样本的末 Token 会把下一个样本的首 Token 当作目标。


## 适用边界

- 主线模型为 `Qwen/Qwen3-0.6B-Base`，训练目标为全参数 SFT。
- `smoke_test.py` 不依赖真实模型下载或 GPU；真实权重正确性检查与 Benchmark 需要上文注明的 GPU 配置。
- 本目录只提供代码、依赖和运行说明。正确性与 Benchmark 命令会在本地生成 `*.local.json`，仓库不附带实验结果。
