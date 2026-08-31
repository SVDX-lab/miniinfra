# 第 11 期：Greedy Speculative Decoding

## 1. 对应书籍章节

对应《超轻量AI Infra教程—推理篇》第 11 章“Greedy Speculative Decoding”。

## 2. 本节目的与实现概览

本期在单张 NVIDIA GeForce RTX 3080 Ti 12GB 上独立实现 Greedy
Speculative Decoding。Qwen3-0.6B 作为 Draft Model 连续提出候选 Token，Qwen3-4B
作为 Target Model 一次验证候选块；实现最长匹配前缀、Correction Token、Bonus Token，
以及两套 KV Cache 的提交、截断和重新同步。

本期不引用其他期源码，不依赖 Transformers、vLLM 或 SGLang。Qwen3 模型、Tokenizer、
预分配 KV Cache、Target-only Baseline 和 Speculative 路径均包含在本目录中。

粗略步骤：由 Draft Model 连续提出候选 Token；Target Model 一次验证候选块；接受最长匹配前缀并补 Correction 或 Bonus Token；提交或截断两套 KV Cache，再与 Target-only baseline 对照。

## 3. 代码使用方法

### 本期边界

- 只实现 Greedy Speculative Decoding，不实现随机 Speculative Sampling。
- 不训练 MTP、EAGLE 或 Medusa 模块；MTP 是另一种候选生成器，不等于完整的
  “提出—验证—接受”执行协议。
- 主性能实验固定关闭 thinking 和 EOS，生成 32 Token，避免不同停止位置破坏对照。
- 只研究 Batch 1 的单请求 Decode 延迟，不把结果外推到高并发服务吞吐。
- 本期因为必须形成 Target/Draft 大小差异，明确使用 Qwen3-4B 作为 Target 压力模型；
  Qwen3-0.6B 仍是课程教学主模型和本期 Draft Model。
- 不使用量化或 CPU Offload 帮助双模型装入显存。

### 验证环境

| 项目 | 版本或配置 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3080 Ti 12GB |
| GPU Driver | 595.80 |
| Python | 服务器 `.venv` 中的 Python |
| PyTorch | 2.7.1+cu126 |
| CUDA Runtime | 12.6 |
| Target | `Qwen/Qwen3-4B` |
| Target revision | `1cfa9a7208912126459214e8b04321603b3df60c` |
| Draft | `Qwen/Qwen3-0.6B` |
| Draft revision | `c1899de289a04d12100db370d81485cdf75e47ca` |
| 主实验 dtype | float16 |
| Attention | PyTorch SDPA，显式 offset causal mask |

Qwen3-4B BF16 权重约 7,672.2 MiB，Qwen3-0.6B 权重约 1,136.9 MiB。两套模型加载
后 CUDA allocated 约 8,871.4 MiB；主实验观测到的最大 allocated 约 8,916.6 MiB。
这说明该组合能在参考卡上运行，但剩余空间不足以支持大 Batch 或很长的双份 KV Cache。

### 文件说明

```text
chapter_11/
├── qwen3_model.py              # Qwen3、分片权重加载与可回滚 KV Cache
├── qwen3_tokenizer.py          # 独立 BPE 与 non-thinking Chat Template
├── speculative_decode.py       # Target-only 与 Greedy Speculative 核心实现
├── experiment_utils.py         # 模型加载、环境记录、显存和计时辅助
├── smoke_test.py               # 无权重、无 GPU 的快速自检
├── compare_speculative.py      # 真实权重逐 Token 正确性实验
├── benchmark_speculative.py    # Draft Length 扫描主实验
├── benchmark_components.py     # Draft/Target/块验证组件成本
├── run_inference.py            # 自然语言功能入口
├── requirements.txt
└── requirements-lock.txt
```

### 独立安装

从项目根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r chapter_11/requirements.txt
```

首次运行会下载两个固定 revision。模型加载器同时支持 Qwen3-0.6B 的单文件权重和
Qwen3-4B 的 Safetensors 分片索引。

### CPU 快速自检

```bash
source .venv/bin/activate
python chapter_11/smoke_test.py
```

自检覆盖：最长匹配前缀、首候选拒绝、部分接受、全部接受、Correction、Bonus、EOS、
Target/Draft KV 回滚和 Target-only Token 等价性。

### 真实权重正确性实验

```bash
source .venv/bin/activate
python chapter_11/compare_speculative.py \
  --dtype float16 \
  --draft-lengths 1 2 4 \
  --max-new-tokens 16 \
  --output chapter_11/compare-speculative-results.local.json
```

参考卡上 3 个 Prompt、3 个 Draft Length 的 FP16 实验全部通过：Speculative 与
Target-only 的 Token ID 序列完全一致，Target 与 Draft 的最终 KV 逻辑长度一致。
各工作负载接受率为 0.357～0.833，说明 Draft Length 本身不能保证更多候选被接受。

#### BF16 有限精度对照

BF16 准入实验中，KV 长度和回滚状态保持一致，但一次块验证与单 Token Target 前向
选择了不同 argmax，随后输出序列分叉。两条路径的 GEMM 和 Attention Query Shape
不同，有限精度归约顺序也可能不同，且差异最终足以改变 Token。因此本期使用 FP16 作为逐 Token 等价
主实验 dtype，并把 BF16 分叉作为有效的数值边界保留，不能把数学上的 Greedy 等价写成
任意 Kernel、dtype 和硬件上的逐位保证。

### Draft Length 主实验

```bash
source .venv/bin/activate
python chapter_11/benchmark_speculative.py \
  --dtype float16 \
  --draft-lengths 1 2 4 8 \
  --max-new-tokens 32 \
  --warmup 1 \
  --repeats 3 \
  --output chapter_11/benchmark-speculative-results.local.json
```

每次正式测量使用 Wall Clock，并在边界执行 CUDA 同步。Tokenizer 和模型加载不计时；
TTFT 包含路径所需的 KV 分配和 Prefill。Speculative 路径先初始化两套 KV，因此 TTFT
高于只执行 Target Prefill 的 Baseline。TPOT 使用：

```text
(端到端延迟 - 单独测得的 TTFT) / (生成 Token 数 - 1)
```

以下是 1 次 warm-up、3 次正式重复的均值：

| 工作负载 | 路径 | 接受率 | TTFT | TPOT | TPOT 加速比 | 端到端延迟 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 中文解释，29 Token | Target-only | — | 44.89 ms | 42.98 ms | 1.000x | 1377.20 ms |
|  | γ=1 | 0.550 | 78.51 ms | 71.00 ms | 0.605x | 2279.63 ms |
|  | γ=2 | 0.469 | 96.94 ms | 80.55 ms | 0.534x | 2594.07 ms |
|  | γ=4 | 0.346 | 86.94 ms | 92.98 ms | 0.462x | 2969.17 ms |
|  | γ=8 | 0.173 | 83.95 ms | 151.16 ms | 0.284x | 4769.88 ms |
| 代码补全，28 Token | Target-only | — | 49.24 ms | 47.93 ms | 1.000x | 1534.98 ms |
|  | γ=1 | 0.938 | 87.30 ms | 62.60 ms | 0.766x | 2028.02 ms |
|  | γ=2 | 1.000 | 85.39 ms | 54.62 ms | 0.877x | 1778.65 ms |
|  | γ=4 | 0.964 | 84.49 ms | 49.64 ms | 0.965x | 1623.36 ms |
|  | γ=8 | 1.000 | 84.62 ms | 46.86 ms | 1.023x | 1537.20 ms |
| 重复格式，42 Token | Target-only | — | 45.18 ms | 42.93 ms | 1.000x | 1376.07 ms |
|  | γ=1 | 0.824 | 80.39 ms | 60.10 ms | 0.714x | 1943.51 ms |
|  | γ=2 | 0.731 | 85.30 ms | 64.05 ms | 0.670x | 2070.74 ms |
|  | γ=4 | 0.719 | 82.00 ms | 57.20 ms | 0.751x | 1855.22 ms |
|  | γ=8 | 0.554 | 95.23 ms | 89.35 ms | 0.480x | 2865.23 ms |

只有接受率为 1.0 的代码样例在 γ=8 时得到约 1.02x 的微弱 TPOT 收益；其余配置全部
是负优化。不能据此声称推测解码在 3080 Ti 上普遍有效，也不能把这组小模型结果外推
到 32B、70B 或生产级融合执行栈。

### 组件成本实验

```bash
source .venv/bin/activate
python chapter_11/benchmark_components.py \
  --draft-lengths 1 2 4 8 \
  --warmup 2 \
  --repeats 10 \
  --output chapter_11/benchmark-components-results.local.json
```

参考卡上的均值：

| 组件 | 延迟 |
| --- | ---: |
| Qwen3-4B Target，Query Length 1 | 48.16 ms |
| Qwen3-0.6B Draft，Query Length 1 | 42.51 ms |
| Target 块验证，γ=1 | 57.82 ms |
| Target 块验证，γ=2 | 57.38 ms |
| Target 块验证，γ=4 | 57.25 ms |
| Target 块验证，γ=8 | 46.34 ms |

虽然参数量相差约 6.7 倍，但当前 Eager 教学栈的 Draft 单步只比 Target 单步快约 12%。
两者都执行大量小算子并受到 Host Launch 开销影响。块验证成本没有随候选数线性增长，
但一轮仍需执行 γ+1 次 Draft 前向；接受率不足时，节省的 Target 调用无法覆盖 Proposal
成本。这也解释了为什么 MTP、EAGLE 等更轻量 Speculator 有工程价值，但它们不属于本期
自变量。

### 自然语言推理

```bash
source .venv/bin/activate
python chapter_11/run_inference.py \
  --mode speculative \
  --prompt "请用一句话解释推测解码。" \
  --max-new-tokens 32 \
  --draft-length 4
```

自然语言入口默认检查 EOS。EOS 判断需要把控制信息同步到 CPU，因此只用于功能验证；
固定长度、EOS disabled 的主实验才代表无逐 Token `.item()` 的性能路径。

### 可以推广与不能推广的结论

可以推广的机制性结论：推测解码用额外 Proposal 计算换取更少的串行 Target 调用；收益
同时取决于 Draft 单步成本、接受长度、Target 块验证效率和控制开销。KV 回滚必须与输出
提交使用同一个接受边界。

不能推广的测量性结论：本期 Eager SDPA、Qwen3-4B/0.6B、FP16、Batch 1 和 3080 Ti
上的接受率、最佳 γ、负优化幅度及 1.02x 收益，均不能外推到原生 MTP 模型、融合
Kernel、CUDA Graph、其他 Prompt 分布、更大 Target 或高并发推理框架。
