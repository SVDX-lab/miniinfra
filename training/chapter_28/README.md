# 第 28 期：torch.compile

## 对应篇章

对应《超轻量AI Infra教程—训练篇》第 13 章；合并后的连续编号为第 28 章。

## 显卡要求

最低与参考配置均为 1 张 GPU。

本期只改变训练步的执行方式：Forward/Backward 交给 `torch.compile`（Inductor 默认配置），Loss、裁剪与优化器留在 Eager。代码分别记录稳态开销、首次编译成本和变长序列的重编译行为。

## 问题、假设与边界

- 问题：编译训练步后稳态能快多少？首次编译代价如何摊销（Break-even）？动态长度造成多少重编译，`dynamic=True` 能否消化？
- 假设：手写前向是干净静态图，应得到单图零 break；Inductor 只能融合逐元素链，收益受 GEMM 份额限制；重编译是变长训练的主要风险。
- 唯一变量：训练步的执行方式（Eager 对照 Inductor 编译）。
- 固定项：与第 19～27 期相同（模型/Tokenizer/数据、Assistant-only、token_mean、FP32、TF32 关、Eager Attention、AdamW loop 区外一致、lr 2e-5、裁剪 1、种子 20260912、batch 4、512 Token）。
- 排除项：BF16 叠加、编译优化器/裁剪、手动 CUDA Graph、max-autotune。

## 文件

- `compiled_step.py`：本期核心。`compile_model`、`timed_first_step`（首步与稳态分离）、`dynamo_stats`/`RecompileTracker`（图数、重编译、break 计数）、`explain_graph_breaks`。
- `training.py`：稳定步体 + `pad_batch_to`（自然批次 → 固定 512 形状，不触碰真实 Token 与 Labels）。
- `smoke_test.py`：CPU 编译对照（Loss/梯度 <1e-5）、pad 助手、dynamo 统计管道。
- `compare_correctness.py`：Padding 不变性、固定 Shape 20 步对照、编译成本、重编译风暴（dynamic=False 对照 dynamic=True）。
- `benchmark.py`：3 次重复 × 3 臂（eager_natural / eager_fixed / compiled_fixed）。

## 环境和数据

参考依赖环境为 Python 3.12、PyTorch `2.12.1+cu130`、CUDA 13.0。语料固定为 `sft_mini_512.jsonl @ 9b06bee2…` 的前 512 条。

```bash
cd chapter_28
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
