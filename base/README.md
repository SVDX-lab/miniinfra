# 基础知识篇

## 快速开始

从仓库根目录运行第 01 节的浮点数演示。下面使用该节的参考环境 Python 3.12 和 CPU 版 PyTorch。

```bash
cd base
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.9.1'
python lesson_01/demo_fp32.py
```

每节的入口为 `lesson_XX/demo_*.py`。有 `requirements.txt` 的节需额外安装其中的依赖。第 21–26 节涉及 GPU，只有 CPU 时可给这些演示传入 `--cpu-smoke`。

第 07、45 节使用 Qwen3-0.6B-Base 的分词器文件，固定 revision 为 `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`。文件准备好后，从 `base/` 运行下列命令，将示例路径替换为实际位置。

```bash
python -m pip install -r lesson_07/requirements.txt -r lesson_45/requirements.txt
python lesson_07/demo_byte_bpe.py /path/to/tokenizer.json
python lesson_45/demo_chat_template.py /path/to/tokenizer.json /path/to/tokenizer_config.json
```

## 按主题找演示

| 节 | 主题 |
| --- | --- |
| [01](lesson_01/)–[04](lesson_04/) | 浮点数、张量与矩阵乘法 |
| [05](lesson_05/)–[07](lesson_07/) | 子词、BPE 与 Tokenizer 文件 |
| [08](lesson_08/)–[15](lesson_15/) | Embedding、Attention、RoPE 与 GQA |
| [16](lesson_16/)–[20](lesson_20/) | Transformer Block 与权重文件 |
| [21](lesson_21/)–[26](lesson_26/) | GPU 执行、计时、显存和算子融合 |
| [27](lesson_27/)–[32](lesson_32/) | 自回归生成、采样与性能指标 |
| [33](lesson_33/)–[44](lesson_44/) | 损失、梯度、优化器、显存与混合精度 |
| [45](lesson_45/)–[49](lesson_49/) | 对话模板、掩码与数据加载 |
| [50](lesson_50/)–[55](lesson_55/) | 分布式通信、矩阵分块与状态分片 |
