# 第 17 期：Assistant-only Loss

## 对应篇章

对应《超轻量AI Infra教程—训练篇》第 02 章；合并后的连续编号为第 17 章。

## 显卡要求

最低与参考配置均为 1 张 GPU。

第 16 期让除 Padding 外的完整对话 Token 参与 Causal LM Loss，System、User 和 Assistant 都会成为预测目标。本期保持对话文本、`input_ids`、模型、FP32、单卡、固定长度、Optimizer、梯度裁剪、Scheduler 与 Eager Attention 不变，只修改 `labels` 中哪些位置为 `ignore_index=-100`。

Baseline 监督完整对话；Variant 只监督 Assistant 正文、Assistant 发出的 Tool Call 序列和该消息真实的 `<|im_end|>`。`<|im_start|>assistant\n`、System、User、Tool Response、消息间换行、Padding 以及编码器因截断额外补入的结束哨兵均不参与 Variant Loss。

## 问题、假设与边界

- 问题：相同 Token 序列上，Assistant-only Loss 如何改变有效监督 Token、Loss、梯度与短程收敛？
- 假设：前向 Logits 和计算 Shape 不变；监督集合缩小后，Loss、梯度方向和一次参数更新会改变。
- 唯一变量：`labels` 中非 `-100` 的位置。
- Baseline：除 Padding 外的完整对话 Token 参与 Shifted Cross Entropy。
- Variant：仅 Assistant 内容、Tool Call 文本及真实结束符参与 Shifted Cross Entropy。
- 固定项：`Qwen/Qwen3-0.6B-Base@da87bfb608c14b7cf20ba1ce41287e8de496c0cd`、固定数据 revision 与 source line、FP32、TF32 关闭、单卡、Micro-batch 1、固定长度、Eager Attention、非 fused/foreach AdamW、相同 Scheduler 和全局梯度裁剪阈值 1。
- 不属于本期：Mini-batch、分桶、Packing、离线 Shard、BF16、梯度累积、Activation Checkpointing、FlashAttention、`torch.compile` 和任何分布式训练。

## 模板边界

本期沿用第 16 期的 non-thinking 对话渲染，不把 `reasoning_content` 混入实验。遇到该字段或其他无法无损表示的字段会直接报错。Tool Call 与 Tool Response 的外层边界来自固定模型 revision 的 `tokenizer_config.json`：

```text
<|im_start|>assistant\n
Assistant content
<tool_call> ... </tool_call>
<|im_end|>\n

<|im_start|>user
<tool_response> ... </tool_response>
<|im_end|>\n
```

第一段中 header 不监督，正文、Tool Call 和 `<|im_end|>` 监督；第二段整体不监督。`compare_correctness.py` 使用 Jinja2 直接渲染固定上游模板中的一轮 Tool Call/Tool Response，并要求它与手写渲染逐字符一致。模板 SHA256 为 `87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5`。

## 文件说明

- `assistant_loss.py`：本期核心机制；显式渲染消息片段、追踪 Token span，并从同一份 `input_ids` 构造两种 labels。
- `boundary_fixtures.py`：多轮、System、Tool Call/Tool Response、空 Assistant 四类确定性边界样本。
- `qwen3_tokenizer.py`：直接读取 `tokenizer.json` 的 Byte-level BPE Tokenizer。
- `qwen3_model.py`：手写 RMSNorm、RoPE、GQA、SwiGLU、Decoder 和 Safetensors 权重加载。
- `sft_core.py`：固定数据版本、JSONL 读取、Shifted Cross Entropy 与统计工具。
- `training.py`：单卡 FP32 AdamW 训练 Step、CUDA Event 分段计时和显存账本。
- `prepare_data.py`：校验固定数据文件并重建与第 16 期相同的 32 条训练 shard。
- `smoke_test.py`：不下载模型、不占 GPU 的边界、Mask、Shape 和极小 Qwen3 检查。
- `compare_correctness.py`：真实 Tokenizer 边界、固定模板、真实权重 Loss/梯度/一次更新对照与 32 条样本 Mask 统计。
- `train.py`：从相同初始权重分别执行两条 10 Step 短程轨迹。
- `benchmark.py`：每组预热 2 Step，再测 3 次、每次 5 Step 的单卡对照。
- `experiment_utils.py`：环境采集、聚合和原子 JSON 写入。
- `data-manifest.json`：固定数据 revision、SHA256、筛选统计与 source line。
- `requirements.txt`、`requirements-lock.txt`：直接依赖与参考环境的完整版本闭包。

## 从干净环境运行

参考环境使用 Python 3.12。不同 CUDA 主版本需要重新解析 PyTorch 与 NVIDIA wheel。

```bash
cd chapter_17
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

准备与第 16 期相同的 32 条固定样本：

```bash
python prepare_data.py --samples 32
```

无法直连 Hugging Face 主站时，可以显式设置兼容镜像；仓库、revision、大小和 SHA256 不会改变。

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  python prepare_data.py --samples 32
```

按成本从低到高运行：

```bash
CUDA_VISIBLE_DEVICES="" python smoke_test.py
python compare_correctness.py --max-length 64 --stats-max-length 128
python train.py --steps 10 --max-length 128 --sample-cursor 0
python benchmark.py --max-length 128 --warmup 2 --steps 5 --repeats 3
```

默认输出到 `*.local.json`。模型、数据与本地结果均由 Git 忽略。


## 适用边界

- 主线模型为 `Qwen/Qwen3-0.6B-Base`，训练目标为全参数 SFT。
- `smoke_test.py` 不依赖真实模型下载或 GPU；真实权重正确性检查与 Benchmark 需要上文注明的 GPU 配置。
- 本目录只提供代码、依赖和运行说明。正确性与 Benchmark 命令会在本地生成 `*.local.json`，仓库不附带实验结果。
