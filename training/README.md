# 训练篇

## 快速开始

从仓库根目录进入训练篇。第 16 章的冒烟测试使用合成数据，可以在已安装本章依赖的 CPU 环境中运行。

```bash
cd training/chapter_16
python smoke_test.py
```

复现单卡实验时，在 `training/chapter_16/` 创建独立环境。以下命令使用本章的参考 GPU 依赖。首次准备数据和运行真实模型时会下载固定版本的数据和 `Qwen/Qwen3-0.6B-Base` 权重。[第 16 章依赖与运行说明](chapter_16/README.md)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python prepare_data.py --samples 32
python train.py --steps 10 --max-length 128 --sample-cursor 0 --generation-tokens 16
```

其他章节的依赖、数据准备、`torchrun` 命令和基准参数见各章 README。若章节说明从 `cd chapter_XX` 开始，请先进入 `training/`。运行结果写到本地，常见后缀为 `*.local.json`。

## 章节索引与 GPU 配置

单卡实验以 RTX 3080 Ti（12 GB）为参考。最低 GPU 数按各章的核心机制计算。参考配置来自各章 README，`4+4` 表示两台服务器各有 4 张 GPU。

| 章 | 主题 | 最低 GPU | 参考 GPU |
| --- | --- | ---: | --- |
| [16](chapter_16/) | 手写 Qwen3-0.6B 全参数 SFT | 1 | 1 |
| [17](chapter_17/) | Assistant-only Loss | 1 | 1 |
| [18](chapter_18/) | 固定 Mini-batch | 1 | 1 |
| [19](chapter_19/) | 按长度分桶 | 1 | 1 |
| [20](chapter_20/) | Sequence Packing | 1 | 1 |
| [21](chapter_21/) | 离线 Token Shard | 1 | 1 |
| [22](chapter_22/) | DataLoader 预取与异步 H2D | 1 | 1 |
| [23](chapter_23/) | BF16 混合精度 | 1 | 1 |
| [24](chapter_24/) | 梯度累积 | 1 | 1 |
| [25](chapter_25/) | Activation Checkpointing | 1 | 1 |
| [26](chapter_26/) | 训练版 FlashAttention | 1 | 1 |
| [27](chapter_27/) | Fused AdamW | 1 | 1 |
| [28](chapter_28/) | `torch.compile` | 1 | 1 |
| [29](chapter_29/) | Distributed Data Parallel | 2 | 1 / 2 / 4 / 4+4 |
| [30](chapter_30/) | DDP Gradient Bucket 与通信计算重叠 | 2 | 2 / 4 / 4+4 |
| [31](chapter_31/) | 梯度累积期间延迟同步 | 2 | 2 / 4 / 4+4 |
| [32](chapter_32/) | ZeRO-1——切分优化器状态 | 2 | 2 / 4 / 4+4 |
| [33](chapter_33/) | ZeRO-2——再切分梯度 | 2 | 2 / 4 / 4+4 |
| [34](chapter_34/) | ZeRO-3 / FSDP——再切分参数 | 2 | 4 / 4+4 |
| [35](chapter_35/) | MLP Tensor Parallel | 2 | 2 / 4 |
| [36](chapter_36/) | Attention Tensor Parallel | 2 | 2 / 4 |
| [37](chapter_37/) | Vocab Parallel Embedding 与 Cross Entropy | 2 | 2 / 4 |
| [38](chapter_38/) | Sequence Parallel | 2 | 2 / 4 |
| [39](chapter_39/) | GPipe Fill-Drain Pipeline | 2 | 2 / 4 |
| [40](chapter_40/) | 1F1B Pipeline Schedule | 2 | 4 |
| [41](chapter_41/) | Interleaved 1F1B | 2 | 4 |
| [42](chapter_42/) | 混合并行与双节点 4+4 | 4 | 4+4 |
| [43](chapter_43/) | 分布式 Checkpoint 与恢复 | 2 | 2 / 4 / 4+4 |
