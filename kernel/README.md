# 算子篇

## 快速开始

从仓库根目录运行第 01 期《RMSNorm 基线与性能分析》的 CPU 实验。在激活的 Python 环境中安装依赖后，脚本会依次运行冒烟测试、生成输入、检查正确性并测量性能。

```bash
cd kernel
python -m pip install -r chapter_01/requirements.txt
BACKENDS=cpu bash chapter_01/run_all.sh
```

在 GPU 宿主上先运行同一入口的基线，再用 `performance` 运行性能诊断。以下以 CUDA 为例，MUSA 宿主将 `cuda` 替换为 `musa`。

```bash
BACKENDS="cpu cuda" bash chapter_01/run_all.sh
BACKENDS=cuda bash chapter_01/run_all.sh performance
```

[第一期实验说明](chapter_01/README.md)列出共享输入、设备时间线及输出路径。

运行其他期之前，先查看本期 `run_all.sh` 开头的后端和参数说明。设备实验需要对应的 PyTorch、驱动和编译工具链。基准和分析结果通常写到本期的 `results/`。

### 模型与测试输入

- 涉及 Qwen3 的实验复用第 01 期的模型与真值。进入 `chapter_01/`，用 [`download_model.py`](chapter_01/download_model.py) 下载模型，按 [`models/MANIFEST.json`](chapter_01/models/MANIFEST.json) 核对哈希，再运行 [`model_truth.py`](chapter_01/model_truth.py) 生成探针。
- 测试输入通常由本期的 `generate_fixtures.py` 生成。第 29 期复用第 02、17、20 期的固定输入。可将 `CH29_FIXTURES_ROOT` 指向保存这些输入的算子篇目录。若同时核验原始提示词，可将 `CH29_PROMPTS_DIR` 指向书稿工程的 `chapter_29/prompts/`。
- 第 23 期的 `bash chapter_23/run_all.sh plan` 会列出引擎实验的前置检查和会话计划。

## 实验索引

第一期的[性能诊断](chapter_01/performance/)与基线共用参考实现，课程按 01–30 连续编号。第 11 期的总结材料保存在原书稿工程。

| 期 | 主题 |
| --- | --- |
| [01](chapter_01/README.md) | RMSNorm 基线与性能分析 |
| [02](chapter_02/) | 手写 RMSNorm——访存受限算子的三种组织 |
| [03](chapter_03/) | 融合 Residual Add + RMSNorm——为什么融合 |
| [04](chapter_04/) | 算子注册与接回 Qwen3——进入框架的 Dispatch |
| [05](chapter_05/) | dtype 的选择与代价（FP32 / FP16 / BF16 三路径） |
| [06](chapter_06/) | 布局即性能——contiguous 与非连续输入 |
| [07](chapter_07/) | 确定性与原子操作——浮点结合律的代价 |
| [08](chapter_08/) | 传输与重叠——时间花在搬运还是计算 |
| [09](chapter_09/) | Launch 开销与 CUDA/MUSA Graph |
| [10](chapter_10/) | Kernel 调试与工具链 |
| 11 | 三后端复盘——"兼容"的分层语义（总结期） |
| [12](chapter_12/) | 多后端 Qwen3 运行时——统一 Backend 抽象 |
| [13](chapter_13/) | Triton/DSL 跨后端算子路径 |
| [14](chapter_14/) | 自动融合的编译器路径——torch.compile/Inductor |
| [15](chapter_15/) | 量化算子与反量化融合（per-channel INT8 weight-only GEMM） |
| [16](chapter_16/) | 进阶手撕算子 I——归约、访存与逐元素 |
| [17](chapter_17/) | 朴素 GEMM——tiling 思想 |
| [18](chapter_18/) | Tensor Core / WMMA——计算单元升级 |
| [19](chapter_19/) | Kernel 内异步与流水——cp.async、双缓冲与 warp specialization 入门 |
| [20](chapter_20/) | naive Attention——prefill 与 decode 双形态 |
| [21](chapter_21/) | KV Cache 组织与 PagedAttention |
| [22](chapter_22/) | RoPE——旋转位置编码的双形态 |
| [23](chapter_23/) | 融合算子接入 vLLM/SGLang 与 vllm-musa——引擎会稀释多少收益 |
| [24](chapter_24/) | DeepSeek MLA——KV 压缩与吸收矩阵 |
| [25](chapter_25/) | DeepSeek DSA——Lightning Indexer 与稀疏注意力 |
| [26](chapter_26/) | MTP——多 Token 预测与投机解码 |
| [27](chapter_27/) | Gated DeltaNet 线性注意力——递推 vs 分块并行 |
| [28](chapter_28/) | MoE 算子——路由与分组 GEMM |
| [29](chapter_29/) | AI 辅助写 Kernel——起草与验收的分工 |
| [30](chapter_30/) | 代际边界与 2026 新结构扫描（全课程收官期） |
