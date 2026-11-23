# 第 27 期：Fused AdamW

## 对应篇章

对应《超轻量AI Infra教程—训练篇》第 12 章；合并后的连续编号为第 27 章。

## 显卡要求

最低与参考配置均为 1 张 GPU。

本期保持 AdamW 数学公式不变，只把参数更新的 Kernel 从逐 Tensor 循环改为 foreach 多 Tensor 打包或 fused 单 Kernel，用来比较不同 Kernel 组织的开销。

## 问题、假设与边界

- 问题：AdamW 的数学不变，只改更新的 Kernel 组织，0.6B 模型（310 个参数 Tensor）上 Optimizer 份额能压掉多少？三种 Kernel 的数值差异和 Kernel 数量差异各有多大？
- 假设：Kernel 发射次数可能是主要开销；fused 与 foreach 是否缩短 Step 由正确性检查和 Benchmark 判断。
- 唯一变量：Optimizer 更新的 Kernel 组织。
- 固定项：与第 19～26 期相同（模型/Tokenizer/数据、Assistant-only、token_mean、FP32、TF32 关、Eager Attention、lr 2e-5、wd 0.1、裁剪 1、种子 20260912、batch 4、512 Token）；`clip_grad_norm_` 三臂同路径并单独计时作锚点。
- 排除项：优化器状态 dtype 与分片（第 32 期）、学习率调度、CUDA Graph、多参数组。

## 文件

- `fused_optimizer.py`：本期核心。`build_optimizer(arm)`（loop/foreach/fused 三臂，同一 AdamW 数学）、`reference_adamw_step`（手写单 Tensor 参照，含解耦权重衰减与偏置校正）、`count_step_kernels`（profiler 统计一次 step 的 CUDA Kernel 数与设备时间）、`step_peak_memory`。
- `training.py`：稳定步体，CUDA Event 把 clip 与 step 分段计时。
- `smoke_test.py`：手写参照复现 torch.optim.AdamW、三臂构造标志、loop 与 foreach 的 CPU 对照。
- `compare_correctness.py`：一步参数对照、32 步轨迹、确定性、Kernel 普查。
- `benchmark.py`：3 次重复 × 3 臂，clip 与 step 分段。

## 环境和数据

参考依赖环境为 Python 3.12、PyTorch `2.12.1+cu130`、CUDA 13.0。语料固定为 `sft_mini_512.jsonl @ 9b06bee2…` 的前 512 条。

```bash
cd chapter_27
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
