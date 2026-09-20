# MiniInfra

《超轻量 AI Infra 入门教程》的配套代码，包含推理篇和训练篇。推理篇以单张 NVIDIA GeForce RTX 3080 Ti（12GB）为参考环境，从手写 Qwen3-0.6B 推理逐步实现 KV Cache、请求调度和服务拆分；训练篇以 `Qwen/Qwen3-0.6B-Base` 全参数 SFT 为主线，从单卡训练扩展到 DDP、ZeRO/FSDP、Tensor Parallel、Pipeline Parallel、双节点混合并行和分布式恢复。

## 项目特点

- **一章一个变量**：每章聚焦一个机制，保留必要的 baseline、正确性检查和 Benchmark 入口。
- **章节独立运行**：每个目录都有完整源码、README、直接依赖和依赖锁文件，不从其他章节导入代码。
- **少依赖、重原理**：模型、Tokenizer、Loss、推理或训练主循环主要由 Python 与 PyTorch 手写实现。
- **训练篇不附实验结果**：新增训练章节只提交源码、依赖、运行说明和代码所需的固定数据清单；运行产生的 `*.local.json`、Checkpoint、Profiler Trace 和其他实验结果由 Git 忽略。

推理篇使用 [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B)，训练篇使用 [`Qwen/Qwen3-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-0.6B-Base)。

## 章节目录与显卡数量

“最低 GPU”是运行本章核心多卡机制所需的最少显卡数；“参考 GPU”列出 README 中覆盖的完整配置。`4+4` 表示两台服务器各 4 张 GPU。

| 代码 | 篇章 | 章节与主题 | 最低 GPU | 参考 GPU |
| --- | --- | --- | ---: | --- |
| [chapter_01](chapter_01/) | 推理篇 | 第 01 章：手写 Qwen3-0.6B 推理 | 1 | 1 |
| [chapter_02](chapter_02/) | 推理篇 | 第 02 章：KV Cache——第一个推理优化 | 1 | 1 |
| [chapter_03](chapter_03/) | 推理篇 | 第 03 章：固定批处理 | 1 | 1 |
| [chapter_04](chapter_04/) | 推理篇 | 第 04 章：Continuous Batching | 1 | 1 |
| [chapter_05](chapter_05/) | 推理篇 | 第 05 章：Paged KV Cache | 1 | 1 |
| [chapter_06](chapter_06/) | 推理篇 | 第 06 章：推理请求调度器 | 1 | 1 |
| [chapter_07](chapter_07/) | 推理篇 | 第 07 章：Chunked Prefill | 1 | 1 |
| [chapter_08](chapter_08/) | 推理篇 | 第 08 章：Prefix Cache | 1 | 1 |
| [chapter_09](chapter_09/) | 推理篇 | 第 09 章：高性能 Attention | 1 | 1 |
| [chapter_10](chapter_10/) | 推理篇 | 第 10 章：CUDA Graph 优化 Decode | 1 | 1 |
| [chapter_11](chapter_11/) | 推理篇 | 第 11 章：Greedy Speculative Decoding | 1 | 1 |
| [chapter_12](chapter_12/) | 推理篇 | 第 12 章：KV Cache Offloading | 1 | 1 |
| [chapter_13](chapter_13/) | 推理篇 | 第 13 章：KV Cache 异步传输 | 1 | 1 |
| [chapter_14](chapter_14/) | 推理篇 | 第 14 章：外部 KV Cache 与 LMCache | 1 | 1 |
| [chapter_15](chapter_15/) | 推理篇 | 第 15 章：Prefill/Decode 分离 | 1 | 1 |
| [chapter_16](chapter_16/) | 训练篇 | 第 16 章：手写 Qwen3-0.6B 全参数 SFT | 1 | 1 |
| [chapter_17](chapter_17/) | 训练篇 | 第 17 章：Assistant-only Loss | 1 | 1 |
| [chapter_18](chapter_18/) | 训练篇 | 第 18 章：固定 Mini-batch | 1 | 1 |
| [chapter_19](chapter_19/) | 训练篇 | 第 19 章：按长度分桶 | 1 | 1 |
| [chapter_20](chapter_20/) | 训练篇 | 第 20 章：Sequence Packing | 1 | 1 |
| [chapter_21](chapter_21/) | 训练篇 | 第 21 章：离线 Token Shard | 1 | 1 |
| [chapter_22](chapter_22/) | 训练篇 | 第 22 章：DataLoader 预取与异步 H2D | 1 | 1 |
| [chapter_23](chapter_23/) | 训练篇 | 第 23 章：BF16 混合精度 | 1 | 1 |
| [chapter_24](chapter_24/) | 训练篇 | 第 24 章：梯度累积 | 1 | 1 |
| [chapter_25](chapter_25/) | 训练篇 | 第 25 章：Activation Checkpointing | 1 | 1 |
| [chapter_26](chapter_26/) | 训练篇 | 第 26 章：训练版 FlashAttention | 1 | 1 |
| [chapter_27](chapter_27/) | 训练篇 | 第 27 章：Fused AdamW | 1 | 1 |
| [chapter_28](chapter_28/) | 训练篇 | 第 28 章：`torch.compile` | 1 | 1 |
| [chapter_29](chapter_29/) | 训练篇 | 第 29 章：Distributed Data Parallel | 2 | 1 / 2 / 4 / 4+4 |
| [chapter_30](chapter_30/) | 训练篇 | 第 30 章：DDP Gradient Bucket 与通信计算重叠 | 2 | 2 / 4 / 4+4 |
| [chapter_31](chapter_31/) | 训练篇 | 第 31 章：梯度累积期间延迟同步 | 2 | 2 / 4 / 4+4 |
| [chapter_32](chapter_32/) | 训练篇 | 第 32 章：ZeRO-1——切分优化器状态 | 2 | 2 / 4 / 4+4 |
| [chapter_33](chapter_33/) | 训练篇 | 第 33 章：ZeRO-2——再切分梯度 | 2 | 2 / 4 / 4+4 |
| [chapter_34](chapter_34/) | 训练篇 | 第 34 章：ZeRO-3 / FSDP——再切分参数 | 2 | 4 / 4+4 |
| [chapter_35](chapter_35/) | 训练篇 | 第 35 章：MLP Tensor Parallel | 2 | 2 / 4 |
| [chapter_36](chapter_36/) | 训练篇 | 第 36 章：Attention Tensor Parallel | 2 | 2 / 4 |
| [chapter_37](chapter_37/) | 训练篇 | 第 37 章：Vocab Parallel Embedding 与 Cross Entropy | 2 | 2 / 4 |
| [chapter_38](chapter_38/) | 训练篇 | 第 38 章：Sequence Parallel | 2 | 2 / 4 |
| [chapter_39](chapter_39/) | 训练篇 | 第 39 章：GPipe Fill-Drain Pipeline | 2 | 2 / 4 |
| [chapter_40](chapter_40/) | 训练篇 | 第 40 章：1F1B Pipeline Schedule | 2 | 4 |
| [chapter_41](chapter_41/) | 训练篇 | 第 41 章：Interleaved 1F1B | 2 | 4 |
| [chapter_42](chapter_42/) | 训练篇 | 第 42 章：混合并行与双节点 4+4 | 4 | 4+4 |
| [chapter_43](chapter_43/) | 训练篇 | 第 43 章：分布式 Checkpoint 与恢复 | 2 | 2 / 4 / 4+4 |

## 快速开始

每章都可独立安装和运行。以推理篇第 01 章为例：

```bash
git clone https://github.com/SVDX-lab/miniinfra.git
cd miniinfra

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r chapter_01/requirements-lock.txt

python chapter_01/smoke_test.py
python chapter_01/run_inference.py \
  --prompt "请用一句话介绍 KV Cache。" \
  --max-new-tokens 32
```

训练篇各章的数据准备、CPU Smoke Test、正确性检查和 Benchmark 命令见对应章节 README。首次执行真实模型路径时会下载固定 revision 的模型和数据。

## 验证边界

推理篇按单张 NVIDIA GeForce RTX 3080 Ti 12GB 设计。训练篇的单卡章节使用同一类显卡，多卡章节覆盖单节点 2/4 卡和双节点 4+4 卡；多卡聚合显存不是统一显存，实际性能还取决于 PCIe、NUMA、NCCL 和节点间网络。

不同硬件、驱动、CUDA 或 PyTorch 版本可能产生不同的性能和数值结果。仓库提供的是机制实现与复现实验入口，不附带可直接外推到其他模型、硬件或生产集群的实验结论。
