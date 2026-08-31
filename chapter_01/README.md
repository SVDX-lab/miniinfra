# 第 01 期：手写 Qwen3-0.6B 推理

## 1. 对应书籍章节

对应《超轻量AI Infra教程—推理篇》第 01 章“手写 Qwen3-0.6B 推理”。

## 2. 本节目的与实现概览

本目录使用基础 Python 和 PyTorch 组件实现 Qwen3-0.6B 的 Tokenizer 与单请求自回归推理，不使用 KV Cache。Transformers 只出现在正确性对照程序中，不进入手写推理程序。

粗略步骤：读取模型配置和 Safetensors 权重；实现 BPE Tokenizer 与 non-thinking Chat Template；逐层完成 Qwen3 前向计算；使用 greedy decoding 逐 Token 生成，并与 Transformers 输出对照。

## 3. 代码使用方法

### 验证环境

- GPU：NVIDIA GeForce RTX 3080 Ti 12GB
- CPU：Intel Xeon Gold 6226R
- 内存：31 GiB
- 操作系统：Ubuntu 22.04
- Python：3.10
- GPU 驱动：595.80
- CUDA Runtime：12.6
- PyTorch：2.7.1+cu126
- Transformers：4.51.3
- 模型 revision：`c1899de289a04d12100db370d81485cdf75e47ca`

### 安装依赖

从项目根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r chapter_01/requirements-lock.txt
```

`requirements-lock.txt` 记录了测试服务器上的完整依赖版本，复现实验时应优先使用它。如果只运行手写推理，可以安装不含 Transformers 的运行依赖：

```bash
python -m pip install -r chapter_01/requirements.txt
```

如需执行 Transformers 正确性对照，则安装：

```bash
python -m pip install -r chapter_01/requirements-compare.txt
```

首次运行会从 Hugging Face 下载固定版本的 `Qwen/Qwen3-0.6B`。也可以通过 `--model` 传入已经下载好的模型目录。如果服务器连接 Hugging Face 不稳定，可以临时设置镜像端点后再运行：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 快速自检

下面的测试使用随机生成的小模型，不需要下载真实权重，也可以在 CPU 上运行：

```bash
source .venv/bin/activate
python chapter_01/smoke_test.py
```

### 运行手写推理

```bash
source .venv/bin/activate
python chapter_01/run_inference.py \
  --prompt "请用一句话介绍 KV Cache。" \
  --max-new-tokens 32
```

输出包括生成 Token IDs、生成文本、Prefill 延迟、平均 Decode 延迟和峰值显存。计时不包含 tokenizer、模型下载和模型加载。

### 与 Transformers 对照

```bash
source .venv/bin/activate
python chapter_01/compare_with_transformers.py \
  --prompt "请用一句话介绍 KV Cache。" \
  --max-new-tokens 32
```

对照程序会先检查手写 Tokenizer 与 Transformers 构造的输入 Token IDs 是否一致，再在相同输入和 greedy decoding 条件下逐步比较两种模型实现的：

- 下一个 Token ID；
- Logits 最大绝对误差；
- Logits 平均绝对误差；
- 最终生成 Token 序列和文本。

如果生成 Token 不一致，程序返回退出码 `1`；如果生成 Token 一致但 Logits 最大误差超过阈值，程序返回退出码 `2`；如果输入 Tokenizer 对照失败，程序返回退出码 `3`。

### 3080 Ti 实测记录

测试日期：2026-08-24。以下结果使用 bfloat16、greedy decoding、关闭 thinking，并强制 Transformers 使用 eager attention 和关闭 KV Cache。三组测试的手写 Tokenizer 输入也都与 Transformers 完全一致。

| Prompt | 最大生成长度 | 实际对比 Token 数 | Token 是否一致 | Logits 最大绝对误差 |
| --- | ---: | ---: | --- | ---: |
| `请用一句话介绍 KV Cache。` | 32 | 32 | 是 | 0.000000 |
| `Explain prefill in one short sentence.` | 16 | 16 | 是 | 0.000000 |
| `计算 17 乘以 23，只给出结果。` | 16 | 12（遇到 EOS） | 是 | 0.000000 |

手写 baseline 使用第一个 Prompt、19 个输入 Token、32 个输出 Token、2 次 warm-up，得到一次运行记录：

- Prefill 延迟：34.43 ms；
- 平均 Decode 延迟：33.40 ms/token；
- 端到端模型计算延迟：1069.80 ms；
- 峰值已分配显存：1173.80 MiB。

这些性能数字不包含 tokenizer、模型下载和模型加载，仅用于确认程序可运行并建立后续实验的原始记录。它们是当前服务器上的单次测量，不应视为稳定基准或外推到其他环境。

### 已知边界

- 仅支持当前固定版本的 Qwen3-0.6B dense 模型。
- 仅支持 batch size 为 1、无 padding 的输入。
- 仅实现 greedy decoding。
- 默认关闭 thinking 模式。
- 手写聊天模板只覆盖本期使用的 system/user 纯文本消息，不支持工具调用、多模态内容和 assistant 历史 reasoning。
- 每次 Decode 都重新计算完整上下文，尚未使用 KV Cache。
- 本期性能数据只用于建立后续实验的原始 baseline，不能直接外推到更大模型或其他硬件。
