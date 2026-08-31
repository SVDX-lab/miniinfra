# MiniInfra

《超轻量 AI Infra 入门教程》的推理篇配套代码。项目面向希望从原理入门 LLM 推理系统的读者，以一张 NVIDIA GeForce RTX 3080 Ti（12GB 显存）为参考环境，从手写 Qwen3-0.6B 推理开始，逐步实现 KV Cache、批处理、请求调度、量化之外的推理优化、服务拆分等关键机制。

## 项目特点

- **单卡即可复现**：所有章节围绕单机、单张 RTX 3080 Ti 设计和验证，重点讲清完整推理链路，而不是模拟无法在本地真实验证的多机生产集群。
- **少依赖、重原理**：核心推理路径主要使用 Python 与 PyTorch 实现，不依赖 Transformers、vLLM、SGLang 等现成推理框架封装，便于直接阅读模型加载、Attention、KV Cache、调度和数据搬运等关键逻辑。模型下载、权重读取和分词等外围功能仍使用少量轻量依赖，具体版本见各章依赖文件。
- **一章一个变量**：每章聚焦一个核心知识点，尽量保留 baseline 与优化实现，通过受控实验理解机制、收益和边界。
- **章节独立运行**：每个目录都包含完整源码、README 和固定版本依赖，不需要复制前面章节的代码即可运行。

教学主模型统一使用 [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B)。除专门研究 thinking 的场景外，性能实验默认关闭 thinking。

## 章节目录

| 代码 | 《超轻量AI Infra教程—推理篇》 | 主题 |
| --- | --- | --- |
| [chapter_01](chapter_01/) | 第 01 章 | 手写 Qwen3-0.6B 推理 |
| [chapter_02](chapter_02/) | 第 02 章 | KV Cache——第一个推理优化 |
| [chapter_03](chapter_03/) | 第 03 章 | 固定批处理 |
| [chapter_04](chapter_04/) | 第 04 章 | Continuous Batching |
| [chapter_05](chapter_05/) | 第 05 章 | Paged KV Cache |
| [chapter_06](chapter_06/) | 第 06 章 | 推理请求调度器 |
| [chapter_07](chapter_07/) | 第 07 章 | Chunked Prefill |
| [chapter_08](chapter_08/) | 第 08 章 | Prefix Cache |
| [chapter_09](chapter_09/) | 第 09 章 | 高性能 Attention |
| [chapter_10](chapter_10/) | 第 10 章 | CUDA Graph 优化 Decode |
| [chapter_11](chapter_11/) | 第 11 章 | Greedy Speculative Decoding |
| [chapter_12](chapter_12/) | 第 12 章 | KV Cache Offloading |
| [chapter_13](chapter_13/) | 第 13 章 | KV Cache 异步传输 |
| [chapter_14](chapter_14/) | 第 14 章 | 外部 KV Cache 与 LMCache |
| [chapter_15](chapter_15/) | 第 15 章 | Prefill/Decode 分离 |

## 快速开始

每章都可独立安装和运行。以第 01 章为例：

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

首次执行真实模型实验时会下载固定 revision 的模型文件。其他入口、参数、正确性检查和 benchmark 命令见对应章节 README。

## 验证边界

参考环境为 Ubuntu 22.04、Python 3.10、PyTorch 2.7.1、CUDA 12.6 和 NVIDIA GeForce RTX 3080 Ti 12GB。不同硬件、驱动、CUDA 或 PyTorch 版本可能产生不同的性能和数值结果。

Qwen3-0.6B 较小，Python、CPU、分词和调度开销占比可能高于大模型。本项目中的绝对性能数据不应直接外推到 7B、32B、多 GPU 或生产集群；可复用的是机制与实验方法，而不是某个孤立的延迟或吞吐数字。
