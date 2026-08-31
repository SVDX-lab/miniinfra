# 第 15 期：Prefill/Decode 分离

## 1. 对应书籍章节

对应《超轻量AI Infra教程—推理篇》第 15 章“Prefill/Decode 分离”。

## 2. 本节目的与实现概览

本目录从模型加载开始，独立实现 Qwen3-0.6B 推理、Paged KV Cache、monolithic
Baseline、跨进程 Prefill/Decode Worker、请求级 KV Handoff、ACK 所有权转移和失败后
重新 Prefill。运行时不 import `chapter_01`～`chapter_14` 的任何文件。

本期只改变一个核心变量：一次请求的 Prefill 和 Decode 是否由同一个 Worker 完成。

- `monolithic`：同一进程完成 Prompt Prefill、首 Token 和后续 Decode；
- `disaggregated`：Prefill Worker 产生 Prompt KV 和首 Token，经固定同步 CPU
  Staging/TCP 路径交给 Decode Worker。

模型、revision、dtype、Prompt、输出预算、Block Size、Token Budget、greedy decoding、
EOS 条件和计时方式保持不变。本期不引入异步传输、压缩、共享内存、CUDA IPC、RDMA、
负载均衡或 KV-aware 调度。

粗略步骤：由 Prefill Worker 计算 Prompt KV 与首 Token；通过服务把逻辑 KV Payload 交给 Decode Worker；Decode 侧重建本地 Paged KV 并返回 ACK；确认所有权后继续生成，失败时回退完整 Prefill。

## 3. 代码使用方法

### 与第 14 期的边界

第 14 期外部 KV Cache 研究跨请求、跨进程复用，具有 Token Chunk Key、Lookup、命中、
Store 和 LRU。第 15 期 Handoff 是同一 `Request ID + Attempt ID` 的一次所有权交接：

- Producer 主动发布，Consumer 不做前缀 Lookup；
- Payload 只服务当前请求，不保留给后续请求命中；
- Decode ACK 后 Producer 才释放本地 KV，Service 随即释放 Payload；
- 传入协议的是逻辑连续 KV，不是进程内 Physical Block ID。

### 首 Token 与 ACK 语义

Prefill Worker 计算完整 Prompt KV 和最后位置 Logits，并选择首 Token。Decode Worker
导入 Prompt KV 后，以首 Token 作为下一次输入，产生第二个 Token。本期固定采用
ACK-gated 输出：Decode 完成校验、本地 Block 分配和 H2D 导入并返回 ACK 后，首 Token
才被视为可以发送给客户端。因此：

```text
split TTFT = Prefill + Export + Publish + Receive + Import + ACK
```

Payload 损坏或 Namespace 不匹配时，Decode Worker 发送 fallback ACK，再用原始 Prompt
完整重新 Prefill。失败路径优先保证输出正确和资源可回收。

### 文件结构

```text
chapter_15/
├── qwen3_model.py                  # 独立手写 Qwen3 与固定 revision 加载
├── qwen3_tokenizer.py              # 独立 BPE 与 non-thinking Chat Template
├── paged_cache.py                  # 独立 Paged KV、逻辑导出与本地重建
├── handoff_protocol.py             # 长度前缀 TCP 协议、摘要与客户端
├── handoff_server.py               # 请求级一次性交接 Service 与超时中止
├── engine.py                       # monolithic、Prefill、Export、Import、Decode
├── experiment_utils.py             # 模型、服务进程、环境与 JSON 工具
├── worker.py                       # 三种独立 Worker 入口
├── smoke_test.py                   # 无权重 CPU 自检
├── validate_disaggregated.py       # 正确性与故障回退验证
├── benchmark_handoff.py            # Prompt 长度与成本拆解
├── run_inference.py                # 自然语言跨进程演示
├── requirements.txt
└── requirements-lock.txt
```

### 验证环境

| 项目 | 版本或配置 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3080 Ti，12,491,292,672 Bytes |
| 操作系统 | Ubuntu 22.04.5 LTS，Linux 5.15.0-113-generic |
| Python | 3.10.12 |
| PyTorch | 2.7.1+cu126 |
| CUDA Runtime | 12.6 |
| NVIDIA 驱动 | 595.80 |
| 模型 | `Qwen/Qwen3-0.6B` |
| revision | `c1899de289a04d12100db370d81485cdf75e47ca` |
| 主实验 dtype | bfloat16 |
| GPU Block Size | 16 Token |
| Handoff 路径 | 同步 GPU→CPU Bytes→本机 TCP→CPU Bytes→GPU |
| 大模型下载组件 | hf_xet 1.6.0 |

每份结果 JSON 记录实际 PID、设备、dtype 和关键参数。正式性能实验使用固定合成 Token，
排除 Tokenizer、模型加载、Worker 启动和网络请求入口时间。每个 Worker 在模型加载后、
正式计时前在同一 CUDA 进程内执行 1 次完整 warm-up。

`Qwen/Qwen3-1.7B` 只作为明确标注的模型规模压力对照。课程教学主模型仍为
`Qwen/Qwen3-0.6B`；压力实验保持 KV 结构和交接协议不变，只提高模型计算量。

### 独立安装

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r chapter_15/requirements-lock.txt
```

首次运行会下载固定 revision；也可以用 `--model` 指向完整本地模型目录。

### CPU 快速自检

```bash
source .venv/bin/activate
python chapter_15/smoke_test.py
```

自检覆盖 Toy Model 的 monolithic/split 输出、Paged KV 导出与导入、
Publish/Receive/ACK/Abort/Release、Checksum 拒绝、Namespace 拒绝和资源释放。

### 真实权重正确性

```bash
source .venv/bin/activate
python chapter_15/validate_disaggregated.py \
  --dtype bfloat16 \
  --prompt-length 513 \
  --max-new-tokens 4 \
  --worker-warmup 1 \
  --output chapter_15/validate-bfloat16-results.local.json

python chapter_15/validate_disaggregated.py \
  --dtype float32 \
  --prompt-length 513 \
  --max-new-tokens 4 \
  --worker-warmup 1 \
  --output chapter_15/validate-float32-results.local.json
```

3080 Ti 实测 BF16 与 FP32 均满足：

- monolithic、Prefill、Decode 来自三个不同 PID；
- monolithic 与 Prefill 的首 Token Logits SHA-256 一致；
- monolithic 与 disaggregated 的完整输出 Token 一致；
- Producer 在 ACK 后释放本地 Block；
- Decode 完成后释放本地 Block；
- Handoff Service 最终不保留 Payload。

这里的摘要一致只适用于当前普通 PyTorch 路径、模型、dtype、Prompt 和 Query Shape，
不能扩大为所有有限精度 Kernel 的逐位保证。

### 故障回退

```bash
source .venv/bin/activate
python chapter_15/validate_disaggregated.py \
  --dtype bfloat16 --prompt-length 257 --max-new-tokens 4 \
  --failure checksum \
  --output chapter_15/validate-checksum-fallback-results.local.json

python chapter_15/validate_disaggregated.py \
  --dtype bfloat16 --prompt-length 257 --max-new-tokens 4 \
  --failure namespace \
  --output chapter_15/validate-namespace-fallback-results.local.json
```

两种故障均在 Decode 侧导入前被拒绝，Producer 收到 fallback ACK 后释放 KV，Decode
重新 Prefill 后的输出与 monolithic baseline 一致，服务端最终零残留。

### KV Handoff 长度扫描

```bash
source .venv/bin/activate
python chapter_15/benchmark_handoff.py \
  --dtype bfloat16 \
  --prompt-lengths 256,512,1024,2048 \
  --max-new-tokens 8 \
  --worker-warmup 1 \
  --warmup 0 \
  --repeats 3 \
  --output chapter_15/benchmark-handoff-results.local.json
```

`--warmup 0` 表示不额外丢弃整个进程样本；每个 Worker 内部仍固定执行 1 次完整
warm-up。正式计时不包含模型加载、进程启动与 warm-up。

<!-- BENCHMARK_RESULTS_START -->
3080 Ti 上每个 Worker 内 warm-up 1 次、3 次正式重复的中位数：

| Prompt | KV Payload | mono TTFT | split TTFT | split/mono | mono E2E | split E2E |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 Token | 28 MiB | 56.08 ms | 321.36 ms | 5.73× | 303.17 ms | 582.14 ms |
| 512 Token | 56 MiB | 108.18 ms | 647.47 ms | 5.99× | 361.51 ms | 916.97 ms |
| 1024 Token | 112 MiB | 215.92 ms | 1271.60 ms | 5.89× | 470.86 ms | 1534.67 ms |
| 2048 Token | 224 MiB | 433.39 ms | 2378.55 ms | 5.49× | 685.59 ms | 2648.27 ms |

ACK-gated split TTFT 的阶段中位数如下。各列分别取中位数，所以横向相加不要求严格
等于“split TTFT”列的样本总和中位数。

| Prompt | Prefill | Export D2H | Publish | Receive | Import H2D | ACK |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 55.07 ms | 58.17 ms | 114.12 ms | 43.63 ms | 50.66 ms | 0.58 ms |
| 512 | 109.96 ms | 119.87 ms | 223.86 ms | 90.09 ms | 100.71 ms | 0.62 ms |
| 1024 | 222.86 ms | 218.17 ms | 442.13 ms | 189.93 ms | 205.95 ms | 0.67 ms |
| 2048 | 442.73 ms | 433.22 ms | 895.72 ms | 357.73 ms | 252.93 ms | 0.64 ms |

所有长度都是明确负优化。2048 Token 的 Prompt KV 为 224 MiB，除了 Prefill，还要
执行约 433 ms 的逻辑导出、896 ms 的 Producer→Service Publish、358 ms 的
Service→Decoder Receive 和 253 ms 的本地导入。当前同步教学路径没有任何独立 GPU
计算可以覆盖这些成本，因此角色分离只增加了数据移动和协议开销。
<!-- BENCHMARK_RESULTS_END -->

Handoff TTFT 使用 ACK-gated 定义，包含 Prefill、D2H Export、Producer→Service
Publish、Service→Decoder Receive、H2D Import 和 ACK。同步教学协议中的 Payload 实际
经过两次 Socket 传递，因此 `publish_ms + receive_ms` 不能解释成单次网络传输。

### 较大模型压力对照

压力模型固定为：

```text
Qwen/Qwen3-1.7B
revision: 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
```

该官方模型标识为 1.7B，BF16 Safetensors 总大小约 4.06 GB。它与主模型同为 28 层、
8 个 KV Head、Head Dim 128，所以相同 Prompt 长度的 KV Payload 完全相同；隐藏维度
由 1024 增至 2048、MLP 中间维度由 3072 增至 6144，传输字节数不变。这使它适合作为
“计算量相对交接成本”的单变量压力对照。

```bash
source .venv/bin/activate
python chapter_15/validate_disaggregated.py \
  --model Qwen/Qwen3-1.7B \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --dtype bfloat16 --prompt-length 257 --max-new-tokens 4 \
  --worker-warmup 1 \
  --output chapter_15/validate-qwen3-1.7b-results.local.json

python chapter_15/benchmark_handoff.py \
  --model Qwen/Qwen3-1.7B \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --dtype bfloat16 --prompt-lengths 256,512,1024,2048 \
  --max-new-tokens 8 --worker-warmup 1 --warmup 0 --repeats 3 \
  --output chapter_15/benchmark-handoff-qwen3-1.7b-results.local.json
```

<!-- MODEL_SCALE_RESULTS_START -->
BF16 正确性实验通过：三个 Worker PID 不同、首 Token Logits 摘要一致、完整输出 Token
一致，Producer、Decoder 和 Service 最终均无 KV/Payload 残留。单 Worker 正式请求峰值
allocated 约 3.65 GB、reserved 约 3.70 GB；两个模型进程可以在 12GB GPU 上完成交接。

相同实验方法下的 TTFT 中位数：

| Prompt | KV Payload | 0.6B mono | 0.6B split | 0.6B 倍数 | 1.7B mono | 1.7B split | 1.7B 倍数 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 28 MiB | 56.08 ms | 321.36 ms | 5.73× | 59.99 ms | 320.41 ms | 5.34× |
| 512 | 56 MiB | 108.18 ms | 647.47 ms | 5.99× | 111.63 ms | 594.22 ms | 5.32× |
| 1024 | 112 MiB | 215.92 ms | 1271.60 ms | 5.89× | 220.58 ms | 1209.42 ms | 5.48× |
| 2048 | 224 MiB | 433.39 ms | 2378.55 ms | 5.49× | 458.62 ms | 2397.39 ms | 5.23× |

端到端延迟的相对倍数略有缩小，但仍未转正：

| Prompt | 0.6B mono/split E2E | 0.6B 倍数 | 1.7B mono/split E2E | 1.7B 倍数 |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 303.17 / 582.14 ms | 1.92× | 324.50 / 597.03 ms | 1.84× |
| 512 | 361.51 / 916.97 ms | 2.54× | 371.22 / 863.06 ms | 2.32× |
| 1024 | 470.86 / 1534.67 ms | 3.26× | 478.29 / 1469.25 ms | 3.07× |
| 2048 | 685.59 / 2648.27 ms | 3.86× | 722.57 / 2679.69 ms | 3.71× |

结果没有支持“换成 1.7B 后负优化消失”。1.7B 的 warmed Prefill 只比 0.6B 略慢，
而同长度 KV 交接仍需要近似相同的时间。一个合理解释是 0.6B 在 Batch 1、256 Token
Chunk 下没有充分利用 GPU，1.7B 更大的矩阵提高了硬件利用率，使参数量增加没有按比例
转化为延迟；本期没有采集逐 Kernel 利用率，因此这一原因只作为数据支持的推断。
<!-- MODEL_SCALE_RESULTS_END -->

这个实验说明相对惩罚有小幅缩小，但“负优化完全由模型太小造成”并不成立。在单卡串行
路径中，split 始终比 monolithic 多出 KV 交接，因此不能用它证明较大模型的单请求
TTFT 会由负转正。多 GPU P/D 并行吞吐收益不在本期实测范围内。

### 自然语言演示

```bash
source .venv/bin/activate
python chapter_15/run_inference.py \
  --prompt "请用一句话解释 Prefill 和 Decode 的区别。" \
  --max-new-tokens 32
```

脚本用本期独立 Tokenizer 生成 non-thinking Chat Prompt，再启动三个独立 Worker，打印
monolithic 与 disaggregated 文本及 Token 一致性。

### 已知边界

- 两个模型 Worker 共享同一张 GPU，不具备真正的 GPU 资源隔离；
- 为验证 ACK 前所有权，Producer 持有模型和 KV 时 Decode Worker 会加载第二份模型；
- 不能用本期单卡串行结果声称多 GPU P/D 分离具有吞吐收益；
- Handoff Service 是本机教学实现，不提供认证、TLS、多租户或不可信网络防护；
- Payload 使用普通 CPU Bytes 和同步 TCP，存在用户态复制，不是共享内存、CUDA IPC、
  GPUDirect、NIXL、NVLink 或 RDMA；
- 本期不实现多个 Prefill/Decode 副本、路由、负载均衡、超时重试队列或请求迁移；
- 本期不缓存交接 Payload，也不支持后续请求 Lookup；
- 只实现 greedy decoding，没有随机采样 RNG 状态交接；
- Qwen3-0.6B 很小，Python、TCP 和逐层 Tensor 搬运占比较高，绝对结果不能外推到
  7B、32B 或生产框架；
- KV-aware 调度与 Mooncake 不纳入本推理篇。它们需要多个独立执行位置才能真实测量
  缓存位置、执行负载和跨设备传输之间的路由权衡；单卡模拟结果不冒充实测收益。
