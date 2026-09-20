# 第 26 期：训练版 FlashAttention

## 对应篇章

对应《超轻量AI Infra教程—训练篇》第 11 章；合并后的连续编号为第 26 章。

## 显卡要求

最低与参考配置均为 1 张 GPU。

第 25 期用重算减少了保存的激活，但 Eager Attention 仍然会物化完整的 S² score/probability 矩阵。本期只改变 Attention 的计算 Kernel，并检查 `scaled_dot_product_attention` 各后端的可用性、数值差异和适用边界。

## 问题、假设与边界

- 问题：把物化 S² 的 Eager Attention 换成分块计算的 SDPA 后端后，训练（Forward 与 Backward）的时间和显存如何变化？FP32 下哪些后端可用？Flash 后端在什么条件下才能真正进入训练路径？
- 假设：FP32 下可用的非物化后端是 mem_efficient（flash 要求 BF16/FP16）；512 长度下端到端收益不大，机制价值在显存与序列边界。
- 唯一变量：Attention Kernel（是否物化完整 S² 矩阵）。BF16 只出现在明确标注的跨精度对照臂。
- 固定项：与第 19～25 期相同（模型/Tokenizer/数据、Assistant-only、token_mean、FP32、TF32 关、AdamW loop、lr 2e-5、裁剪 1、种子 20260912、batch 4、512 Token）。
- 排除项：Packing 与 Context Parallel、KV Cache、自定义 Triton kernel、用注意力微基准替代端到端训练测量。

## 文件

- `sdpa_attention.py`：本期核心。四个 Attention 实现（eager / sdpa_math / sdpa_mem_efficient / flash）、`set_attention_implementation` 注入工具、`probe_backend_support` 后端普查。
- `qwen3_model.py`：`Qwen3Attention` 增加 `attention_impl` 注入点；None 时走与第 22 期逐位相同的内联路径。
- `smoke_test.py`：注入路径逐位一致、注意力层对照、因果性反例、Padding Key 梯度阻断、CPU flash 语义、CPU 后端普查。
- `compare_correctness.py`：CUDA 后端普查、FP32 核对照、整 Epoch 轨迹、BF16 跨精度臂（无 Padding 批次）、序列容量扫描。
- `benchmark.py`：3 次重复 × 3 个 FP32 臂的整 Epoch 计时。

## 环境和数据

参考依赖环境为 Python 3.12、PyTorch `2.12.1+cu130`、CUDA 13.0。语料固定为 `sft_mini_512.jsonl @ 9b06bee2…` 的前 512 条。

```bash
cd chapter_26
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
