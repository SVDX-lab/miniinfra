# 第 13 期：KV Cache 异步传输

## 1. 对应书籍章节

对应《超轻量AI Infra教程—推理篇》第 13 章“KV Cache 异步传输”。

## 2. 本节目的与实现概览

本目录从模型加载开始，独立实现 Qwen3-0.6B 推理、Paged KV Cache、Pinned CPU
Pool、Continuous Batching、Chunked Prefill、抢占式 Offloading，以及同步/异步
两条 KV 传输路径。运行时不 import `chapter_01`～`chapter_12` 的任何代码。

本期只改变一个核心变量：GPU 与 Pinned CPU Memory 之间的传输执行方式。

- `sync`：默认计算 Stream 上逐块传输，完成后调度器继续；
- `async`：独立 Copy Stream 上逐块提交，通过 CUDA Event 延迟转移 Block 所有权，
  传输期间调度器继续推进无数据依赖的 Prefill 或 Decode。

GPU/CPU 存储层级、块布局、Pinned Pool 容量、抢占触发、受害者选择、恢复顺序、
请求 Trace 和模型条件在两组中保持不变。

粗略步骤：保留默认 Stream 同步拷贝 baseline；在独立 Copy Stream 提交 D2H/H2D；用 CUDA Event 管理完成通知与 Block 所有权；调度无依赖计算尝试覆盖传输，并测量实际重叠程度。

## 3. 代码使用方法

### 核心语义

异步提交后，请求会经过两个显式中间状态：

```text
running -> swapping_out -> swapped -> swapping_in -> running
```

- D2H 完成前，GPU 源 Block 不能释放或改写；
- H2D 完成前，GPU 目标 Block 不能读取，CPU 源 Block不能释放；
- `Event.query()` 只轮询，不等待；只有没有可执行工作或下一次 Decode 必须申请
  Block 时，调度器才等待对应 Event；
- 计算计时只同步默认计算 Stream，不调用会等待全部 Stream 的设备级同步。

“异步提交”不等于“传输已完成”，也不保证端到端更快。能隐藏多少取决于是否有
足够的独立计算、抢占触发是否过晚，以及并发计算和传输是否争用资源。

### 验证环境

| 项目 | 版本或配置 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3080 Ti 12GB |
| CPU / 内存 | Intel Xeon Gold 6248R / 31.34 GiB |
| PCIe | 平台最高 Gen3 x16（空闲查询可能降到 Gen1） |
| Python | 3.10.12 |
| PyTorch | 2.7.1+cu126 |
| CUDA Runtime | 12.6 |
| 模型 | `Qwen/Qwen3-0.6B` |
| revision | `c1899de289a04d12100db370d81485cdf75e47ca` |
| 主实验 dtype | bfloat16 |
| Block Size | 16 |

具体驱动、内核和链路状态会写入每个结果 JSON，不能把本机测量值外推到其他平台。

### 文件说明

```text
chapter_13/
├── qwen3_model.py          # 独立 Qwen3 模型与权重加载
├── qwen3_tokenizer.py      # 独立 BPE 与 non-thinking Chat Template
├── scheduler.py            # 独立 Chunked Prefill 调度器
├── paged_cache.py          # GPU Block Pool 与 Pinned CPU Pool
├── transfer.py             # Copy Stream、Event、传输任务与完成语义
├── engine_base.py          # 独立 Prefill/Decode 公共执行底座
├── engine.py               # sync/async 抢占、恢复与主循环
├── experiment_utils.py     # 环境、固定负载与 JSON 输出
├── smoke_test.py           # 无权重 CPU 快速自检
├── compare_async.py        # 真实权重正确性实验
├── benchmark_overlap.py    # 受控传输—计算重叠微基准
├── benchmark_async.py      # 容量压力主实验
├── run_inference.py        # 自然语言演示入口
├── requirements.txt
└── requirements-lock.txt
```

### 独立安装

从项目根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r chapter_13/requirements.txt
```

首次真实权重实验会下载固定 revision 的 Qwen3-0.6B。

### 自然语言演示

```bash
source .venv/bin/activate
python chapter_13/run_inference.py --transfer-mode async
```

### CPU 快速自检

```bash
source .venv/bin/activate
python chapter_13/smoke_test.py
```

自检覆盖延迟 Block 所有权、D2H/H2D 往返逐位一致、sync/async 抢占次数一致、
输出与无抢占参考一致，以及逻辑/物理传输字节守恒。

### 真实权重正确性

```bash
source .venv/bin/activate
python chapter_13/compare_async.py \
  --output chapter_13/compare-async-results.local.json
```

实验分别在 float32 与 bfloat16 下运行无抢占参考、同步小池和异步小池，报告逐请求
Token 一致性、首个分叉位置和最大 Logits 误差。有限精度下数学等价不代表逐位等价，
Token 一致性也不能替代 Block 生命周期检查。

### 受控重叠微基准

```bash
source .venv/bin/activate
python chapter_13/benchmark_overlap.py \
  --output chapter_13/benchmark-overlap-results.local.json
```

固定真实 Qwen3 KV Block 形状，分别测 D2H/H2D 与独立 BF16 GEMM 的串行、并发
时间。`overlap_efficiency` 只描述这个受控窗口，不等于推理引擎端到端隐藏比例。

参考服务器正式结果（10 次重复均值）：

| 方向 | Block | 串行窗口 | 并发窗口 | 加速 |
| --- | ---: | ---: | ---: | ---: |
| D2H | 16 | 6.69 ms | 4.64 ms | 1.44× |
| H2D | 16 | 6.50 ms | 4.32 ms | 1.50× |
| D2H | 64 | 13.09 ms | 9.08 ms | 1.44× |
| H2D | 64 | 13.75 ms | 9.73 ms | 1.41× |
| D2H | 128 | 22.08 ms | 18.09 ms | 1.22× |
| H2D | 128 | 23.41 ms | 19.41 ms | 1.21× |

这证明本机具备传输—计算重叠能力，不证明引擎主循环能获得同等加速。

### 容量压力主实验

```bash
source .venv/bin/activate
python chapter_13/benchmark_async.py \
  --pool-blocks 230,320 \
  --max-new-tokens 192 \
  --warmup 1 \
  --repeats 3 \
  --output chapter_13/benchmark-async-results.local.json
```

固定 8 请求 Trace、输出预算、Token Budget、GPU Pool 和调度策略，只比较 sync 与
async；`relaxed` 大池路径是无抢占负对照。主要指标包括 TTFT、ITL、Makespan、
吞吐、传输设备时间、提交时间、显式等待时间和方向拆分。

参考服务器 3 次正式重复均值：

| GPU Pool | 模式 | Makespan | tok/s | ITL p95 | 抢占 | 设备传输 / 暴露等待 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 230 | sync | 38.79 s | 39.61 | 62.2 ms | 2 | 48.3 / 48.3 ms |
| 230 | async | 39.07 s | 39.31 | 63.3 ms | 2 | 44.8 / 20.0 ms |
| 320 | sync | 33.78 s | 45.55 | 81.0 ms | 3 | 62.2 / 62.2 ms |
| 320 | async | 32.27 s | 47.62 | 77.6 ms | 3 | 57.4 / 19.2 ms |

异步路径确实减少了显式传输等待，但端到端结果不是单向正优化：230 Block 组略慢，
320 Block 组较快。单次模型执行波动明显，且 Makespan 差值远大于几十毫秒传输窗口，
因此不能把全部端到端差异归因于隐藏 PCIe 传输。机制结论以分阶段 Event 指标为准。

真实权重正确性实验中，float32 与 bfloat16 的 reference/sync/async 抢占次数均为
0/1/1；两个小池路径的所有请求 Token 和所比较 Logits 均与无抢占参考一致。本结果
只适用于当前 batch 形状、权重、dtype 和硬件，不能外推为所有有限精度执行都逐位一致。

### 已知边界

- 逐块异步提交不额外引入 pack/unpack 临时缓冲区；连续大拷贝只应作为独立上界，
  不能与主实验混为同一个优化变量。
- D2H 在紧急 Block 水位触发，完成前不能释放 GPU Block，因此可能大部分暴露；
  本期不通过提前抢占或新水位策略人为扩大重叠窗口。
- H2D 请求在完成前占用 GPU Block 和运行名额；其他请求存在时才可能隐藏传输。
- 单请求上下文上限仍由 GPU Pool 决定；CPU 副本不做共享、命中或跨进程复用。
- 不实现 SSD、远端 KV、RDMA、多 GPU、传输优先级或预测性换出。
- Qwen3-0.6B 的绝对时间不能直接外推到更大模型或生产推理框架。
