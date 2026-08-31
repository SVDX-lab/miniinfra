# 第 12 期：KV Cache Offloading（抢占式换出）

## 1. 对应书籍章节

对应《超轻量AI Infra教程—推理篇》第 12 章“KV Cache Offloading”。

## 2. 本节目的与实现概览

本目录使用基础 Python 和 PyTorch 独立实现 Qwen3-0.6B 推理引擎上的 GPU/CPU
两级 KV Cache 存储与抢占式换出。代码完整包含模型结构、Tokenizer、Paged KV
Cache（块主序布局）、Pinned CPU Pool、Chunked Prefill 调度器、Block 级准入
控制、swap/recompute 两种抢占路径、正确性检查和容量压力实验，不 import
`chapter_01`～`chapter_11` 的任何文件。

本期只改变一个核心变量：GPU Block 不足时受害请求暂停期间的 KV 去向。

- `swap`：整请求 KV 同步换出到 Pinned CPU 池，恢复时经 PCIe 换回断点续算；
- `recompute`：整请求 KV 直接丢弃，恢复时对 prompt + 已生成部分重新 Prefill。

两条路径共用相同的池容量、准入水位、受害者选择（最后接纳的 running 请求）
和恢复顺序（swapped 队列 FCFS 优先于 waiting）。另有两种对照路径：
`conservative`（悲观准入，预留 prompt+max_new，绝不抢占）和 `relaxed`
（大池，全程无抢占）。

粗略步骤：建立 GPU Block Pool 与 Pinned CPU Pool；显存不足时选择受害请求；把整请求 KV 换出或丢弃；恢复时换入断点续算或重新 Prefill，并比较容量、停顿和端到端表现。

## 3. 代码使用方法

### 实现边界

- decode 每步 Attention 读取请求全部历史 KV，因此不存在“运行中请求的部分
  换出”；换出单位是整请求，且只有被暂停请求的 KV 才“暂时不用”。
- 换出副本只用于恢复原请求，不做命中、共享或跨进程复用（第 14 期主题）。
- 单请求上下文上限仍由 GPU 池决定；换出提升的是系统级逻辑并发与上下文总量。
- 换出/换入使用默认 Stream 上的同步 blocking copy；异步传输与计算重叠属于
  第 13 期。教学引擎在拷贝期间不推进其他 GPU 工作；不把这一行为扩大为所有
  Stream 和所有硬件都无法并行。块主序布局保证一个物理块一次 contiguous 拷贝。
- 抢占与恢复发生在迭代边界；只抢占 decoding 请求，不抢占 prefilling 请求。
- 人为缩小 GPU Block Pool 是受控实验手段，所有实验都明确标注池大小。
- 不实现 SSD/远端存储、抢占策略比较、部分换出和 Copy-on-Write。

### 验证环境

| 项目 | 版本或配置 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3080 Ti 12GB |
| CPU / 内存 | Intel Xeon Gold 6248R / 31.34 GiB |
| PCIe 链路 | x16，平台最高 Gen3（理论约 15.75 GB/s） |
| GPU Driver | 595.80 |
| OS | Linux 5.15.0-113-generic x86_64 |
| Python | 3.10.12（服务器 `.venv`） |
| PyTorch | 2.7.1+cu126 |
| CUDA Runtime | 12.6 |
| 模型 | `Qwen/Qwen3-0.6B` |
| revision | `c1899de289a04d12100db370d81485cdf75e47ca` |
| 主实验 dtype | bfloat16 |
| Attention | 块主序 Paged reference，按上下文长度分组 |

Qwen3-0.6B 单 Token KV 为 28 层 × 8 KV 头 × head_dim 128 × K/V × 2 字节
≈ 112 KiB；Block Size 16 时每块 1.75 MiB。注意：RTX 3080 Ti 显卡本身支持
PCIe 4.0 x16，但参考服务器平台实际运行 Gen3 x16，本期所有带宽结论都以该
链路为准，不能外推到 Gen4 平台。

### 文件说明

```text
chapter_12/
├── qwen3_model.py            # 本期独立的 Qwen3 模型与权重加载
├── qwen3_tokenizer.py        # 独立 BPE 与 non-thinking Chat Template
├── paged_cache.py            # 块主序 GPU Block Pool + CPU Pinned Pool
├── scheduler.py              # Chunked Prefill 调度器（swapped 状态由引擎管理）
├── engine.py                 # 抢占/换出/恢复/准入主循环
├── experiment_utils.py       # 环境记录、固定负载与结果保存
├── smoke_test.py             # 无权重、无 GPU 的快速自检
├── compare_offloading.py     # 真实权重 fp32/bf16 正确性实验
├── benchmark_transfer.py     # PCIe 传输微基准
├── benchmark_recovery.py     # 256～4096 Token 恢复成本扫描
├── benchmark_copy_stall.py   # 固定 Decode batch 的同步拷贝停顿
├── benchmark_offloading.py   # 容量压力主实验（swap/recompute/对照）
├── run_inference.py          # 自然语言演示入口
├── requirements.txt
└── requirements-lock.txt
```

### 独立安装

从项目根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r chapter_12/requirements.txt
```

首次运行会下载固定 revision 的 Qwen3-0.6B。

### CPU 快速自检

```bash
source .venv/bin/activate
python chapter_12/smoke_test.py
```

自检覆盖：Chunked Prefill 调度、物理/逻辑字节与尾块碎片守恒、两级池换出/换入后
KV 逐位一致、Pinned Pool 配置不足时启动失败、swap 与 recompute 恢复后输出与
无抢占参考一致，以及保守准入不触发抢占。

### 真实权重正确性实验

```bash
source .venv/bin/activate
python chapter_12/compare_offloading.py \
  --max-new-tokens 32 \
  --output chapter_12/compare-offloading-results.local.json
```

6 个请求（短/中/长 Prompt ×2，长度 18/50/212 Token），GPU 池 = 全部 Prompt
块数 + 6，使总上下文超出池并各触发 1 次抢占。fp32 与 bfloat16 下，swap 和
recompute 的逐请求输出 Token 均与无抢占参考一致，但 Logits 并非都逐位相同：

- float32：swap 最大绝对误差 `1.62e-5`，recompute 为 `4.01e-5`；
- bfloat16：swap 最大绝对误差为 `0`，recompute 为 `0.5625`。

swap 保证的是换出前后有效 KV 字节逐位一致；调度顺序和 batch 形状仍可能影响后续
Logits。recompute 是数学等价的重新 Prefill，是否产生数值差异必须实测。当前差异
没有改变 argmax，不能据此外推任意 dtype、形状或硬件都保持 Token 一致。

### PCIe 传输微基准

```bash
source .venv/bin/activate
python chapter_12/benchmark_transfer.py
```

参考卡上的均值（CUDA event 计时，逐块同步拷贝，每块 1.75 MiB）：

| 块数 | MiB | D2H ms | H2D ms | D2H GB/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.75 | 0.187 | 0.191 | 9.82 |
| 8 | 14.00 | 1.270 | 1.343 | 11.56 |
| 64 | 112.00 | 9.850 | 10.528 | 11.92 |
| 256 | 448.00 | 39.403 | 42.168 | 11.92 |

- 逐块拷贝时间随块数近似线性，渐近约 11.9 GB/s；1 块时固定发起开销使 D2H
  有效带宽下降到 9.82 GB/s。
- 一次 contiguous 大拷贝（448 MiB）D2H 13.13 GB/s、H2D 12.32 GB/s，是逐块
  路径的上界参考；差距来自逐块拷贝的固定开销而非带宽本身。
- pageable 对照（64 块）仅 4.16/4.60 GB/s，显著慢于 pinned；本期主路径因此
  固定使用 Pinned Memory，但差距幅度只适用于本机。
- 本轮分配 448 MiB pinned 的 3 次均值为 101.9 ms，且该指标容易受主机内存状态
  影响；它仍明显高于单次换入/换出，Pinned Pool 必须预分配而不是抢占时现场申请。

### 受害请求上下文扫描

```bash
source .venv/bin/activate
python chapter_12/benchmark_recovery.py
```

两个同长请求同时到达，GPU Pool 仅保留一块增长余量，使最后接纳的请求恰好发生
一次抢占和恢复。每个长度运行 3 次：

| 上下文 Token | swap-out ms | swap-in ms | recompute Prefill ms |
| ---: | ---: | ---: | ---: |
| 256 | 2.53 | 2.70 | 95.40 |
| 512 | 4.98 | 5.33 | 173.37 |
| 1024 | 9.91 | 10.56 | 265.03 |
| 2048 | 19.67 | 21.01 | 509.12 |
| 4096 | 39.27 | 42.15 | 953.77 |

这些点表明，本环境和长度区间内 swap 按整块搬运近似线性，recompute 曲线由真实
Prefill 决定，不能预先假定为严格线性。交叉点和差距幅度不能外推到其他模型。

### 固定 Decode batch 的同步拷贝停顿

```bash
source .venv/bin/activate
python chapter_12/benchmark_copy_stall.py
```

该微基准固定两个 running 请求和 slot，从 512 Token 初始上下文继续 Decode，并在
相邻 Decode 间对第二个请求当时的全部 KV（33～35 块）做同步 D2H+H2D round-trip。
10 次正式重复的均值为：相邻无拷贝 Decode 对照 40.82 ms，round-trip 10.71 ms，
event-aligned 间隔增量 10.70 ms。它只隔离同步拷贝停顿，不代表真实抢占的暂停等待
或容量收益。

### 容量压力主实验

```bash
source .venv/bin/activate
python chapter_12/benchmark_offloading.py \
  --pool-blocks 230,320 \
  --max-new-tokens 192 \
  --repeats 3 \
  --warmup 1 \
  --output chapter_12/benchmark-offloading-results.local.json
```

固定负载：8 个合成请求（Prompt 1536/768/1280/1024/1792/896/1152/640 Token，
seed=12），输出预算 192 Token，到达间隔 40 ms 逻辑时间，Block Size 16，
Token Budget 256，最大运行 6，bfloat16，greedy，EOS disabled，计时含调度、
前向、Cache 管理与换出/换入，不含模型加载。全部 Prompt 合计 568 块，
单请求最坏 124 块；GPU 池人为缩至 230 块（约 402 MiB KV）和 320 块
（约 560 MiB）。`relaxed` 负对照忽略扫描值，自动使用可容纳全部最坏上下文的
664 块大池；分别在两个扫描组中独立重复。3 次重复取均值：

| 池 | 模式 | Makespan | tok/s | ITL p50/p95/max (ms) | 抢占 | 暂停 p95 (ms) |
| ---: | --- | ---: | ---: | --- | ---: | ---: |
| 230 | swap | 41.5 s | 37.0 | 55.3 / 67.9 / 9781 | 2 | 9703 |
| 230 | recompute | 40.8 s | 37.6 | 52.2 / 64.9 / 10101 | 2 | 9434 |
| 230 | conservative | 43.1 s | 35.7 | 52.0 / 56.9 / 111 | 0 | 0 |
| 664（230 组负对照） | relaxed | 33.7 s | 45.5 | 94.3 / 107.5 / 222 | 0 | 0 |
| 320 | swap | 34.2 s | 44.9 | 72.7 / 84.3 / 4078 | 3 | 3992 |
| 320 | recompute | 35.0 s | 43.9 | 72.3 / 104.5 / 4368 | 3 | 3892 |
| 320 | conservative | 37.7 s | 40.7 | 62.0 / 66.8 / 155 | 0 | 0 |
| 664（320 组负对照） | relaxed | 33.7 s | 45.6 | 94.1 / 110.4 / 233 | 0 | 0 |

机制级明细（3 次均值）：

| 指标 | swap | recompute |
| --- | ---: | ---: |
| 池 230：换出/换入总量 | 266.0 MiB / 266.0 MiB | 0 / 0（丢弃 266.0 MiB KV） |
| 池 230：换出/换入总耗时 | 23.6 ms（11.82 GB/s）/ 25.0 ms（11.14 GB/s） | — |
| 池 230：重算 Token / Prefill 暴露时间 | 0 | 2426 / 672.7 ms |
| 池 320：换出/换入总量 | 341.2 MiB / 341.2 MiB | 0 / 0（丢弃 341.2 MiB KV） |
| 池 320：换出/换入总耗时 | 30.1 ms（11.87 GB/s）/ 32.4 ms（11.05 GB/s） | — |
| 池 320：重算 Token / Prefill 暴露时间 | 0 | 3107 / 763.6 ms |
| 池 320：事件窗口内共存请求 ITL | 均值 84.2 ms / 最大 128.9 ms | — |

换出总量均为实际 PCIe 物理字节。池 230 的双向尾块碎片合计 1.75 MiB，池 320
为 4.16 MiB；逻辑有效字节和实际传输字节已在原始 JSON 中分列保存。

主结论：

1. **容量收益成立**：同一池下 swap 优于保守准入——吞吐 37.0 vs 35.7
   tok/s（池 230）和 44.9 vs 40.7 tok/s（池 320），逻辑并发峰值从 2～3
   升到 4～5。悲观预留把池“用不满”，
   乐观准入 + 抢占让池始终打满。
2. **抢占本身有代价**：两种抢占模式都低于 relaxed 大池（45.5/45.6 tok/s），
   差距来自暂停等待；池越接近真实需求，抢占收益越接近上界，容量充足时
   收益为零（负对照通过）。
3. **swap 与 recompute 的系统结果受负载支配**：池 230 中 recompute 的均值略好，
   但 swap 的 Makespan 标准差达 1.45 s，不能把 0.7 s 均值差写成稳定优势；池 320
   中 swap 则快约 0.8 s。机制计时更明确：swap 的双向传输约 49～63 ms，recompute
   的额外 Prefill 暴露约 673～764 ms。暂停等待仍占 4～10 s，恢复方式只是其中
   一部分，不能仅靠一次端到端均值判断优劣。
4. **ITL max 约等于受害者暂停时长**（秒级）：抢占的延迟代价主要落在被
   抢占请求自身；主实验中的事件窗口 ITL 仍包含正常 Decode。固定 batch 微基准
   进一步测得同步 round-trip 带来的 event-aligned 增量约 10.70 ms，不能把秒级
   暂停归因于 PCIe 拷贝。如何异步隐藏这部分停顿是第 13 期主题。
5. **吞吐与单请求延迟的权衡**：relaxed 模式吞吐最高但 ITL p50 最差
   （约 94 ms，6 个请求同批 decode），压力模式下驻留更少、单步更快
   （约 52～73 ms）。抢占改变了驻留规模，间接改变 decode batch 大小。

### 自然语言演示

```bash
source .venv/bin/activate
python chapter_12/run_inference.py
```

默认 4 个请求（Prompt 22/25/42/48 Token）、输出预算 32、GPU 池 14 块
（约 24.5 MiB KV 空间）。参考卡上请求 3 在生成中途被换出（4 块 7.0 MiB，
0.69 ms，10.56 GB/s），其余请求完成后换回（0.72 ms，10.12 GB/s），全部输出
正常完成。该入口默认关闭 EOS，用固定输出预算保证抢占压力可预测。

### 可以推广与不能推广的结论

可以推广的机制性结论：

- 换出把显存容量问题转化为带宽与 CPU 容量问题：实际传输量按物理 Block 计算，
  随逻辑长度阶梯式增长，宏观近似线性。
- 乐观准入 + 抢占使系统级逻辑并发超过 GPU 池容量；容量不足从失败语义变成
  延迟语义。
- swap 保留搬运的 KV 字节；后续 Logits 仍可能因执行形状变化。recompute 只有
  数学等价，数值等价必须实测。
- 本期默认 Stream 教学路径中的同步拷贝会增加 event-aligned 间隔，这是第 13 期
  异步化的对象。
- 换出不提升单请求上下文上限；每步 Attention 需要全量历史 KV。

不能推广的测量性结论：

- 具体毫秒数、GB/s、最优池比例和抢占次数只适用本期环境、模型与负载。
- 本服务器 PCIe 为 Gen3 x16；Gen4 平台的传输收益需重新测量。
- Qwen3-0.6B 上 swap 对 recompute 的差距幅度不能外推到 7B/32B；不同模型的
  参数量、层数、KV 头数、Attention Kernel 和 PCIe 平台都会改变两条成本曲线。
- eager 教学栈的拷贝与调度开销不能代表生产引擎。
