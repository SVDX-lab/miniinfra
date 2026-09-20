# 第 23 期：BF16 混合精度

## 对应篇章

对应《超轻量AI Infra教程—训练篇》第 08 章；合并后的连续编号为第 23 章。

## 显卡要求

最低与参考配置均为 1 张 GPU。

第 22 期完成了数据与计算的重叠。本期只改变一件事：Forward 与 Loss 改在 `torch.autocast(bfloat16)` 下执行，主参数、梯度与 AdamW 状态保持 FP32。BF16 的动态范围使它不需要 GradScaler；dtype 普查会记录各类 Tensor 的实际 dtype。

## 问题、假设与边界

- 问题：FP32 → BF16 混合精度后，单卡时间花在哪里发生了变化，显存省了多少，数值分叉有多大、按什么速度累积？
- 假设：前向/反向由 GEMM 主导，Tensor Core 上的 BF16 应显著缩短 Forward 与 Backward；激活减半但 FP32 主状态不动，显存峰值只部分下降；Loss 与梯度出现 BF16 舍入级分叉并随步数放大。
- 唯一变量：计算与激活的 dtype 策略（FP32 对照 BF16 autocast）。
- 固定项：模型/Tokenizer/数据 revision、512 条样本、Assistant-only Labels、`token_mean`、Eager Attention、AdamW(loop)、学习率 2e-5、裁剪 1、种子 20260912、批次计划与第 19～22 期相同；TF32 显式关闭并记录。
- 排除项：GradScaler 与 FP16、纯 BF16 参数/优化器状态、TF32 作为变量、FlashAttention/Fused Optimizer/torch.compile（第 41、42、43 期）。

## 文件

- `mixed_precision.py`：本期核心。autocast 策略下的 dtype 普查（forward hook 逐模块记录输出 dtype）、参数/梯度/优化器状态 dtype 直方图、NaN/Inf 扫描、轨迹误差汇总。
- `training.py`：`train_batch_step` 以 `mode` 区分 `fp32` / `bf16_autocast` 两臂，CUDA Event 分段计时。
- `smoke_test.py`：CPU 上的 dtype 普查、FP32 模式与第 22 期步体逐位一致、Mask 记账、批次计划覆盖性。
- `compare_correctness.py`：FP32 锚点、梯度/一步参数对照、整 Epoch 轨迹分叉、真实模型 dtype 普查。
- `benchmark.py`：3 次重复 × 双臂的整 Epoch 分段计时与吞吐。

## 环境和数据

参考依赖环境为 Python 3.12、PyTorch `2.12.1+cu130`、CUDA 13.0。语料固定为 `sft_mini_512.jsonl @ 9b06bee2…` 的前 512 条。

```bash
cd chapter_23
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

远端服务器在 HF 缓存完整时应设置 `HF_HUB_OFFLINE=1`，否则 `snapshot_download` 直连 huggingface.co 校验元数据会被网络阻断挂起。


## 适用边界

- 主线模型为 `Qwen/Qwen3-0.6B-Base`，训练目标为全参数 SFT。
- `smoke_test.py` 不依赖真实模型下载或 GPU；真实权重正确性检查与 Benchmark 需要上文注明的 GPU 配置。
- 本目录只提供代码、依赖和运行说明。正确性与 Benchmark 命令会在本地生成 `*.local.json`，仓库不附带实验结果。
