# 第 24 期：梯度累积

## 对应篇章

对应《超轻量AI Infra教程—训练篇》第 09 章；合并后的连续编号为第 24 章。

## 显卡要求

最低与参考配置均为 1 张 GPU。

单步能容纳的 Batch 仍然受单卡显存约束。本期只改变一件事：一个 Optimizer Step 不再由一次大 Forward/Backward 构成，而是由 K 个 Micro-batch 累积而成。正确性的关键在一个加权规则——微批 i 的损失必须按 `token_sum_i / total_supervised` 缩放，累积出的梯度才与大批的 `token_mean` 严格同口径。

## 问题、假设与边界

- 问题：一个 Optimizer Step 由 K 个 Micro-step 累积而成时，梯度与参数更新能否与一次大 Batch 数学等价？累积以多少时间换取多大的容量与 Padding 收益？单次大 Batch 的 OOM 边界在哪里？
- 假设：加权正确时累积梯度等于大批梯度（FP32 下仅剩求和顺序噪声）；代价是 K 次小 GEMM 的时间加总；收益是微批级显存上限带来的容量。
- 唯一变量：一个 Optimizer Step 的构成方式（1 个大 Batch 对照 K 个 Micro-step 累积）。
- 固定项：与第 19～23 期相同（模型/Tokenizer/数据、Assistant-only、token_mean、FP32、TF32 关、Eager Attention、AdamW loop、lr 2e-5、裁剪 1、种子 20260912、512 Token）。
- 排除项：BF16 叠加、no_sync 与梯度同步时机（第 30/31 期）、分桶与 Packing、DP 维度。

## 文件

- `gradient_accumulation.py`：本期核心。`accumulated_step`（K 个微批 Forward/Backward，损失按监督 Token 占比缩放，最后一次裁剪 + step）、`token_sum_loss`、`supervised_token_count`。
- `training.py`：大批臂的第 22 期稳定步体，以及 `make_step_plan`/`split_effective_batch`（同一 shuffle 按 16 分组、按微批尺寸切片）。
- `smoke_test.py`：CPU 上的加权数学直查、累积对照大批梯度、**反例验证**（微批均值平均在监督数不均时必须偏离联合目标）、计划覆盖性。
- `compare_correctness.py`：梯度/一步参数/8 步轨迹对照、大批递增 OOM 扫描、B_eff=16 累积容量、Padding 记账。
- `benchmark.py`：3 次重复 × 4 臂（big4 / acc_2x2 / acc_4x1 / acc_4x4）吞吐与显存。

## 环境和数据

参考依赖环境为 Python 3.12、PyTorch `2.12.1+cu130`、CUDA 13.0。语料固定为 `sft_mini_512.jsonl @ 9b06bee2…` 的前 512 条。

```bash
cd chapter_24
python3.12 -m venv --system-site-packages .venv
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


## 适用边界

- 主线模型为 `Qwen/Qwen3-0.6B-Base`，训练目标为全参数 SFT。
- `smoke_test.py` 不依赖真实模型下载或 GPU；真实权重正确性检查与 Benchmark 需要上文注明的 GPU 配置。
- 本目录只提供代码、依赖和运行说明。正确性与 Benchmark 命令会在本地生成 `*.local.json`，仓库不附带实验结果。
