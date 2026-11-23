# 第 14 期：外部 KV Cache

## 1. 对应书籍章节

对应《超轻量AI Infra教程—推理篇》第 14 章“外部 KV Cache 与 LMCache”。

## 2. 本节目的与实现概览

本目录从模型加载开始，独立实现 Qwen3-0.6B 推理、本地 Paged KV Cache、进程无关的
Token Chunk 标识、独立 CPU Cache Service、Lookup/Load/Store、固定容量 leaf-LRU、
跨进程复用和 Load 与重新 Prefill 对照。运行时不 import `chapter_01`～`chapter_13`
的任何文件。

本期只改变一个核心变量：新的引擎进程能否使用外部 Cache 中已有的 Prompt KV。

- `recompute`：新请求完整重新 Prefill；
- `external`：查找最长连续 Token Chunk，加载到新申请的本地 GPU Block，只 Prefill
  未命中的后缀。

模型、revision、dtype、Prompt、输出预算、Block Size、External Chunk Size、Token
Budget、greedy decoding、EOS 条件和计时边界保持不变。

粗略步骤：用模型命名空间和 Token Chunk 构造进程无关 CacheKey；由独立服务完成 Lookup/Load/Store；新引擎把命中的 CPU KV 重建到本地 GPU Block；只 Prefill 未命中后缀，并与完整重算对照。

## 3. 代码使用方法

### 与第 08、12、13 期的边界

- 第 08 期 Prefix Cache 共享同一进程中的 GPU Physical Block ID；本期外部对象不含
  Block ID，加载进新引擎时必须重新分配 Block 并重建 Block Table。
- 第 12 期 CPU 副本只恢复原请求；本期 KV 由 Token Chunk 标识，可以被另一请求和
  另一进程命中。
- 第 13 期研究 GPU/CPU 异步传输；本期固定使用同步 TCP + CPU Bytes + H2D 路径，
  不把异步协议或零拷贝同时加入主实验。

### 核心设计

#### 进程无关的 CacheKey

```text
digest = SHA256(
    model namespace
    || parent chunk digest
    || current chunk token ids
)
```

Namespace 包含模型 ID、固定 revision、dtype、层数、KV Head、Head Dim、RoPE Theta、
引擎 Block Size、外部布局、External Chunk Size 和格式版本。哈希命中后，服务端还会
比较父摘要、Token 数和完整 Token IDs。

外部 Cache 保存逻辑连续布局：

```text
[num_layers, K/V, num_kv_heads, chunk_tokens, head_dim]
```

它不保存请求 ID、CUDA 地址或本地 Physical Block ID。默认 GPU Block Size 为 16
Token，External Chunk Size 为 256 Token；一个外部 Chunk 加载后会重新映射到 16 个
本地 GPU Block。

本期只保存 `prompt_length - 1` 范围内的完整 External Chunk。KV 本身不包含最后
Prompt 位置的 LM Head Logits，因此至少重新执行最后一个 Prompt Token，才能产生首
Token。

#### Cache Service

`cache_server.py` 是独立进程，只管理不可变 CPU Bytes、Token Index、容量和父子关系。
协议使用本机 TCP 上的长度前缀 JSON Header 与原始 Payload，Payload 使用 SHA-256
校验。Store 完整接收并校验后才发布；父 Chunk 不存在时拒绝发布子 Chunk。

容量按实际 Payload 字节限制，固定使用 leaf-LRU：只淘汰没有后继的叶子，避免留下
无法从 Prompt 开头到达的孤儿 Chunk。外部服务不可用、Lookup/Load 失败或 Payload
校验失败时，引擎记录错误并安全回退完整 Prefill；Store 失败不影响已经产生的推理
结果。

### 文件结构

```text
chapter_14/
├── qwen3_model.py                 # 独立手写 Qwen3 模型与权重加载
├── qwen3_tokenizer.py             # 独立 BPE 与 non-thinking Chat Template
├── paged_cache.py                 # 独立本地 Paged KV Cache 与外部格式转换
├── cache_protocol.py              # CacheKey、TCP 协议和客户端 Connector
├── cache_server.py                # 独立 CPU Cache Service 与 leaf-LRU
├── engine.py                      # recompute/external 推理路径
├── experiment_utils.py            # 固定环境、服务进程与 JSON 工具
├── smoke_test.py                  # 无权重 CPU 自检
├── compare_external_cache.py      # 真实权重 cold/warm 正确性
├── cross_process_worker.py        # Producer/Consumer 独立工作进程
├── cross_process_validate.py      # Producer 退出后的跨进程验证
├── benchmark_external_cache.py    # Load vs Prefill 长度扫描
├── benchmark_eviction.py          # 固定容量 LRU Trace
└── run_inference.py               # 自然语言演示
```

### 验证环境

| 项目 | 版本或配置 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3080 Ti，12,491,292,672 Bytes |
| 操作系统 | Ubuntu 22.04.5 LTS，Linux 5.15.0-113-generic |
| Python | 3.10.12 |
| PyTorch | 2.7.1+cu126 |
| CUDA Runtime | 12.6 |
| 模型 | `Qwen/Qwen3-0.6B` |
| revision | `c1899de289a04d12100db370d81485cdf75e47ca` |
| 主实验 dtype | bfloat16 |
| GPU Block / External Chunk | 16 / 256 Token |

每份结果 JSON 还记录实际 PID、设备、dtype 和关键参数。Qwen3 thinking 默认通过本期
独立 Chat Template 关闭。真实性能实验使用固定合成 Token，排除 Tokenizer 和网络
请求入口时间。

### 独立安装

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r chapter_14/requirements-lock.txt
```

首次运行会下载固定 revision；也可以用 `--model` 指向完整本地模型目录。

### CPU 快速自检

```bash
source .venv/bin/activate
python chapter_14/smoke_test.py
```

自检覆盖：链式 CacheKey、TCP Service、Paged KV 导出/导入逐位一致、cold/warm 输出、
namespace 隔离、本地 Block 全释放和固定容量 LRU 淘汰。

### 自然语言演示

```bash
source .venv/bin/activate
python chapter_14/run_inference.py \
  --mode both \
  --external-chunk-size 16 \
  --max-new-tokens 16
```

`external` 会先运行 cold Producer，再在同一 Cache Service 上运行 warm Consumer。
默认自然语言 Prompt 较短，演示命令使用 16 Token Chunk；正式实验固定为 256。

### 真实权重正确性

```bash
source .venv/bin/activate
python chapter_14/compare_external_cache.py \
  --dtype bfloat16 \
  --prompt-length 513 \
  --max-new-tokens 4 \
  --output chapter_14/compare-external-cache-bfloat16-results.local.json

python chapter_14/compare_external_cache.py \
  --dtype float32 \
  --prompt-length 513 \
  --max-new-tokens 4 \
  --output chapter_14/compare-external-cache-float32-results.local.json
```

3080 Ti 实测：

| dtype | cold/warm 输出与 recompute 一致 | 首 Token Logits 最大误差 | warm 命中 | warm 实际 Prefill | 本地 Block 全释放 |
| --- | --- | ---: | ---: | ---: | --- |
| bfloat16 | 是 / 是 | 0 / 0 | 512 Token | 1 Token | 是 |
| float32 | 是 / 是 | 0 / 0 | 512 Token | 1 Token | 是 |

这里的逐位结果只适用于当前模型、输入、Chunk 切分和普通 PyTorch 路径，不能扩大为
所有有限精度 Kernel 都逐位一致。

### 真正的跨进程验证

```bash
source .venv/bin/activate
python chapter_14/cross_process_validate.py \
  --dtype bfloat16 \
  --prompt-length 513 \
  --max-new-tokens 4 \
  --output chapter_14/cross-process-results.local.json
```

脚本保持 Cache Service 存活，依次启动两个独立 Python 进程。Producer 完成 Store 后
完全退出，Consumer 再加载模型并执行 Lookup/Load。3080 Ti 实测两个 Worker PID
不同；Producer 为 0 Token cold miss，Consumer 命中 512 Token，双方输出 Token
完全一致。这验证的是同机、顺序执行的跨进程复用，不是多 GPU 并发或跨机收益。

### Load 与重新 Prefill

```bash
source .venv/bin/activate
python chapter_14/benchmark_external_cache.py \
  --dtype bfloat16 \
  --prefix-lengths 256,512,1024,2048 \
  --warmup 1 \
  --repeats 3 \
  --output chapter_14/benchmark-external-cache-results.local.json
```

每个长度使用 `prefix_length + 1` 个 Prompt Token，确保命中完整前缀后仍重新计算最后
一个 Token。每组先清空 Cache Service，再单独 Populate；正式统计不包含首次 Store，
只比较 warm Load TTFT 与完整重新 Prefill TTFT。TTFT 包含 Lookup、TCP Payload、导入
本地 GPU Block、后缀 Prefill 和首 Token 选择，不含模型加载、Tokenizer 和 Store。

正式结果会原样保留负优化；外部缓存的机制正确不代表当前传输实现一定更快。

3080 Ti 上 1 次 warm-up、3 次正式重复的中位数：

| 命中前缀 | 重新 Prefill TTFT | warm External TTFT | Lookup | TCP Load | 导入/H2D | 末 Token Prefill | 相对速度 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 Token | 88.88 ms | 273.46 ms | 0.95 ms | 178.96 ms | 54.81 ms | 33.47 ms | 0.33× |
| 512 Token | 145.12 ms | 500.92 ms | 0.93 ms | 387.69 ms | 77.36 ms | 30.98 ms | 0.29× |
| 1024 Token | 242.90 ms | 1028.19 ms | 0.94 ms | 847.73 ms | 143.21 ms | 31.22 ms | 0.24× |
| 2048 Token | 480.62 ms | 2131.68 ms | 1.44 ms | 1831.88 ms | 260.16 ms | 33.27 ms | 0.23× |

本实现中所有长度都是负优化。Lookup 始终约 1 ms，主要成本来自 TCP 传输和按 Chunk
重建 GPU Block；2048 Token 的 bfloat16 KV Payload 为 234,881,024 Bytes，而完整
重新 Prefill 中位数仅 480.62 ms。这个结果说明“已经命中”不等于“值得加载”，也不能
把本教学协议的负优化扩大为 LMCache、共享内存、RDMA 或更大模型上的结论。

### 与 LMCache 的工程映射

本期不把安装现有框架当作核心实现，也不把手写 PyTorch 引擎和 vLLM + LMCache 的
性能放进同一受控对照。概念上的对应关系如下：

| 本期教学实现 | LMCache 中的对应职责 |
| --- | --- |
| `ExternalCacheClient` | 推理引擎 Connector |
| `TokenChunker` | Token Index / Chunk 标识 |
| 不可变原始 KV Payload | Memory Object |
| `CacheStore` | CPU Backend、容量与 LRU |
| `lookup/load/store` | 命中查询、加载和写回路径 |

LMCache 当前还提供异步 Offloading、分层存储和多种远端 Connector，这些能力会同时
改变传输实现和存储层级，不纳入本期自变量。对应架构与配置应以复现时固定版本的官方
文档为准：<https://github.com/LMCache/LMCache/blob/dev/docs/source/developer_guide/architecture.rst>。

### 固定容量 LRU Trace

```bash
source .venv/bin/activate
python chapter_14/benchmark_eviction.py \
  --trace A,B,A,C,A,B \
  --capacity-entries 2 \
  --output chapter_14/benchmark-eviction-results.local.json
```

该 Trace 实测 2 次命中、4 次 miss 和 2 次淘汰。它验证容量与 LRU 语义，不代表真实
在线命中率；命中率由请求 Trace、容量和 Chunk 粒度共同决定。

### 已知边界

- Cache Service 是本机教学实现，不提供认证、TLS、多租户配额或不可信网络防护；
- Payload 保存在服务进程 CPU 内存，Cache Service 自身退出后不会持久化；
- TCP 路径存在用户态复制，导入按 Chunk 和逐层 Tensor 执行，不是共享内存、CUDA
  IPC、GPUDirect、RDMA 或 LMCache 的高性能 Connector；
- Store/Load 固定同步执行，不研究异步流水、压缩、量化和零拷贝；
- 不比较 External Chunk Size、LRU/FIFO/LFU、CPU/SSD/Redis 等后端；
- 不实现多个冷请求的 in-flight Prefill 去重；
- 单卡只验证顺序跨进程语义，不声称多实例并发吞吐、跨机网络或 RDMA 收益；
- KV-aware 调度与 Prefill/Decode 分离分别留给后续期次；
- Qwen3-0.6B 很小，Python、TCP 和逐 Tensor 搬运开销占比很高，绝对结果不能外推到
  7B、32B 或生产框架。
