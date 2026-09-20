# 第 22 期：DataLoader 预取与异步 H2D

## 对应篇章

对应《超轻量AI Infra教程—训练篇》第 07 章；合并后的连续编号为第 22 章。

## 显卡要求

最低与参考配置均为 1 张 GPU。

第 21 期把 Tokenize 移到了离线的二进制 Shard 里，但读取 Shard、校验、组批和 Host-to-Device 拷贝仍然发生在每一步的关键路径上。本期只改变这些数据工作的调度方式：Baseline 同步执行——每步在主线程内读取 Shard、组批、阻塞式 H2D 到默认流，然后计算；Variant 用一个 Worker 线程把未来 Batch 物化进 Pinned Memory（有界队列深度 2），并以双缓冲方式在第 k 步计算期间，于独立 CUDA Stream 上异步发起第 k+1 步的 H2D。两条路径使用同一个第 21 期 Shard、同一 Batch 计划和同一个训练步体。

## 问题、假设与边界

- 问题：预取与异步 H2D 能隐藏多少数据等待？可隐藏上限与同步路径的数据成本是什么关系？
- 假设：可隐藏量不超过同步路径的数据侧耗时；当数据成本远小于训练 Step 时，端到端收益也会受限。
- 唯一变量：数据准备与 H2D 的调度方式——步内同步串行与跨步重叠双缓冲。
- 固定项：与第 19～21 期相同（模型/Tokenizer/数据 revision、512 条样本、Assistant-only Labels、`token_mean`、FP32、TF32 关闭、Eager Attention、AdamW、学习率 2e-5、梯度裁剪 1、种子 20260912、Batch Size 4）；Worker 数固定为 1，两条路径消费同一 Shard 与同一 Batch 计划。
- 排除项：多进程 DataLoader 与多 Worker、Tokenize 阶段与分桶策略（沿用第 21 期）、CUDA Graph、Pinned Memory 池化复用、BF16、梯度累积、分布式。

## 文件

- `prefetch_pipeline.py`：本期核心实现。`SyncBatchLoader`（步内物化 + 阻塞 H2D）；`PrefetchBatchLoader`（`PrefetchWorker` 线程 + Pinned Memory 有界队列 + 侧流 `non_blocking` 双缓冲拷贝 + 事件等待）；`materialize_host_batch`/`pin_host_batch`/`to_device_batch`。
- `data_paths.py`：沿用第 21 期的 `ShardCorpus` 与共享组批 `pad_pairs`。
- `training.py`：`train_device_batch_step`——与第 18～21 期同一训练步体，直接接收设备上的 Batch。
- `token_shard.py`：与第 21 期相同的 Shard 读写实现。
- `smoke_test.py`：CPU 上的 Loader 一致性与顺序、队列上界、Worker 异常传播、组批账本检查。
- `compare_correctness.py`：GPU 上的设备 Batch 逐位对照与整 Epoch Loss 轨迹逐位对照。
- `benchmark.py`：3 次重复 × 双路径的逐步数据等待、计算分段与吞吐测量。

## 环境和数据

参考依赖环境为 Python 3.12、PyTorch `2.12.1+cu130`、CUDA 13.0。语料固定为 `sft_mini_512.jsonl @ 9b06bee2…` 的前 512 条。

```bash
cd chapter_22
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 python prepare_data.py --samples 512
```

## 运行顺序

```bash
CUDA_VISIBLE_DEVICES="" python smoke_test.py
python compare_correctness.py --output correctness-results.local.json
python benchmark.py --output benchmark-results.local.json
```

## 双缓冲的调度约定

预取路径的重叠来自错拍：第 k 步计算前，主线程先在拷贝流上发起第 k+1 步 Batch 的 H2D（`non_blocking=True`，源缓冲为 Pinned Memory）并记录事件，再让计算流 `wait_event` 第 k 步的拷贝事件。这样第 k+1 步的 H2D 与第 k 步的计算并行；计算流使用某 Batch 前必须等到它的拷贝事件完成，Pinned 源缓冲在事件完成前不会被复用。Worker 异常通过哨兵与 `check_error` 在主线程显式抛出；队列深度 2 限制宿主内存占用。


## 适用边界

- 主线模型为 `Qwen/Qwen3-0.6B-Base`，训练目标为全参数 SFT。
- `smoke_test.py` 不依赖真实模型下载或 GPU；真实权重正确性检查与 Benchmark 需要上文注明的 GPU 配置。
- 本目录只提供代码、依赖和运行说明。正确性与 Benchmark 命令会在本地生成 `*.local.json`，仓库不附带实验结果。
