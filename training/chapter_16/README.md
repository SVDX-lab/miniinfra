# 第 16 期：手写 Qwen3-0.6B 全参数 SFT

## 对应篇章

对应《超轻量AI Infra教程—训练篇》第 01 章；合并后的连续编号为第 16 章。

## 显卡要求

最低与参考配置均为 1 张 GPU。

这一期只解决一个问题：不用 Transformers、Trainer 或其他训练框架，能否把一次全参数 SFT 从数据一直跑到可恢复的 Checkpoint？这里使用 Python 和 PyTorch 手写 Qwen3、Byte-level BPE Tokenizer、Causal LM Loss 与训练循环，直接加载固定版本的 `config.json`、`tokenizer.json` 和 `model.safetensors`。

本期是后续实验的原始 baseline，不追求训练效率。它使用单卡、FP32、单样本、固定长度、Eager Attention，并让除 Padding 外的完整对话 Token 参与 Loss。只监督 Assistant 的 Loss Mask 留到第 17 期，不能提前混进来。

## 实验合同

- 问题：一个最小但完整的全参数 SFT Step 包含哪些状态和计算？
- 假设：手写实现能满足 Causal LM 的数学定义，完成参数更新，并在恢复后复现同一个下一 Step。
- Baseline：`Qwen/Qwen3-0.6B-Base@da87bfb608c14b7cf20ba1ce41287e8de496c0cd` 的预训练权重。
- 唯一变化：在预训练权重上加入数据编码、Forward、Backward、梯度裁剪、AdamW、Scheduler 和 Checkpoint，建立最小完整训练系统。
- 保持不变：FP32、Micro-batch 1、单卡、固定序列长度、完整对话 Loss、Eager Attention。
- 正确性证据：手算 Shifted Cross Entropy、小模型有限差分、Padding Mask、真实权重前反向、固定样本 Loss 下降、恢复后的下一 Step 连续。
- 本期不做：Assistant-only Loss、Mini-batch、BF16、梯度累积、FlashAttention、编译和任何分布式训练。

## 文件说明

- `qwen3_tokenizer.py`：直接解析 `tokenizer.json` 的 Byte-level BPE 和本期所需 Chat Template。
- `qwen3_model.py`：手写 RMSNorm、RoPE、GQA、SwiGLU、Decoder 和 Safetensors 权重加载。
- `sft_core.py`：数据过滤、固定长度编码、完整对话 labels 和 Shifted Cross Entropy。
- `training.py`：AdamW、Scheduler、训练 Step、显存账本、生成和完整 Checkpoint。
- `prepare_data.py`：下载并校验固定 MiniMind 文件，生成 32 条确定性样本。
- `smoke_test.py`：不下载模型、不占 GPU 的最低成本检查。
- `compare_correctness.py`：有限差分与真实 Qwen3 权重的正确性检查。
- `train.py`：固定样本短训练与训练前后生成。
- `compare_checkpoint.py`：保存、恢复并复现同一个下一 Step。
- `benchmark.py`：预热后执行 3 组、每组 5 Step 的单卡 baseline。

## 从干净环境运行

参考环境使用 Python 3.12。CUDA 驱动必须能够运行 CUDA 13.0 对应的 PyTorch wheel。

```bash
cd chapter_16
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

先准备数据。脚本会下载 `sft_t2t_mini.jsonl`，核对 1,739,201,170 字节和 SHA256，再选取前 32 条本期可以无损表示的纯文本对话。固定行号和过滤统计保存在 [data-manifest.json](data-manifest.json)。

```bash
python prepare_data.py --samples 32
```

服务器无法直连 Hugging Face 主站时，可以显式设置兼容镜像；这不会改变脚本固定的仓库与 revision。

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  python prepare_data.py --samples 32
```

按成本从低到高运行：

```bash
python smoke_test.py
python compare_correctness.py --max-length 64
python train.py --steps 10 --max-length 128 --sample-cursor 0 \
  --generation-tokens 16
python compare_checkpoint.py --max-length 64
python benchmark.py --max-length 128 --warmup 2 --steps 5 --repeats 3
```

命令默认把本次运行写入 `*.local.json`。模型缓存、数据和 Checkpoint 均被 Git 忽略。`requirements-lock.txt` 是参考 GPU 环境中从四个直接依赖递归得到的精确版本闭包；不同 CUDA 主版本需要重新解析 PyTorch 与 NVIDIA wheel，不能直接照搬。


## 适用边界

- 主线模型为 `Qwen/Qwen3-0.6B-Base`，训练目标为全参数 SFT。
- `smoke_test.py` 不依赖真实模型下载或 GPU；真实权重正确性检查与 Benchmark 需要上文注明的 GPU 配置。
- 本目录只提供代码、依赖和运行说明。正确性与 Benchmark 命令会在本地生成 `*.local.json`，仓库不附带实验结果。
