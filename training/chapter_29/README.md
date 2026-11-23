# 第 29 期：Distributed Data Parallel

## 对应篇章

对应《超轻量AI Infra教程—训练篇》第 14 章；合并后的连续编号为第 29 章。

## 显卡要求

最低 2 张 GPU；参考配置覆盖 1、2、4 张 GPU，以及双节点 4+4 张 GPU。

从第 17 期起，训练一直发生在单张 GPU 上。第 23～28 期把单卡步体压到了各自的机制边界，但数据并行的第一维还没有打开：本期让两张卡各持一份完整模型、完整梯度和完整优化器状态，各自处理全局批次的一个分片，在 Backward 之后用一次 AllReduce SUM 得到相同的全局梯度。本期只改变一件事：数据并行副本数量（1 → 2）。梯度 Bucket 重叠、累积期延迟同步、状态分片分别属于第 30～33 期。

## 问题、假设与边界

- 问题：副本数增加时，训练语义能否严格等价（相同全局有效 Token 的 token_mean 目标）？吞吐如何变化？后置全量梯度 AllReduce 在关键路径上暴露多少？
- 假设：本地损失按监督 Token 占比缩放 + AllReduce SUM 与单卡大 batch 数学等价；两卡经 PCIe PHB（无 NVLink）互联，每步约 2.4 GB FP32 梯度的 AllReduce 完全暴露，Strong Scaling 效率显著低于 2。
- 唯一变量：数据并行副本数量。
- 固定项：与第 19～28 期相同（模型/Tokenizer/数据、Assistant-only、全局 token_mean、FP32、TF32 关、Eager Attention、AdamW loop、lr 2e-5、裁剪 1.0、种子 20260912、Strong 组全局 batch 4）。
- 排除项：Bucket 重叠（第 30 期）、no_sync（第 31 期）、ZeRO（第 32/33 期）和 BF16。

## 文件

- `ddp_core.py`：本期核心。分片批次（`shard_members`：成员 `rank::world_size` 切片）、损失缩放（`supervised_local / supervised_global`）、`all_reduce_gradients`（Backward 后全量 SUM）、`ddp_train_step`（分段计时：h2d / forward / backward / grad_allreduce / scalar_sync / optimizer）、`spawn_workers`（env:// 启动与异常传播）、`parameter_count_bytes`。
- `ring_allreduce.py`：教学实现。Gloo 点对点上的 Ring AllReduce（Reduce-Scatter + AllGather 各 W-1 步）与数据量公式 `2(W-1)/W·D`；用于 smoke 的手算对照，不替代 NCCL。
- `benchmark_allreduce.py`：NCCL AllReduce 微基准（1～2360 MiB，algbw/busbw），用于解释主实验的通信数字。
- `compare_correctness.py`：真实权重两阶段对照——单卡大 batch 参考（Phase A）对照 2 卡 DDP（Phase B）：step-0 梯度、8 步轨迹、跨 Rank 逐位一致性。step-0 梯度两侧同点捕获——AllReduce 之后、`clip_grad_norm_` 之前（`after_allreduce` 钩子），对应参考路径"裁剪前 stash"；DDP 侧若在裁剪后读取，误差会恒等于裁剪系数 `1 - 裁剪范数/全局范数`，是比较时点错误而非梯度错误。
- `smoke_test.py`：CPU/Gloo——环形 AllReduce 逐位对照、分片不变量、缩放反例、一步一致性。
- `benchmark.py`：Strong/Weak Scaling 三臂（strong_dp1 / strong_dp2 / weak_dp2）× 3 次重复。

## 环境和数据

参考依赖环境为 Python 3.12、PyTorch `2.12.1+cu130` 和 NCCL 2.29.7。语料固定为 `sft_mini_512.jsonl @ 9b06bee2…` 的前 512 条。

```bash
cd chapter_29
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 python prepare_data.py --samples 512
```

## 运行顺序

```bash
CUDA_VISIBLE_DEVICES="" python smoke_test.py
# 12GB 显卡使用 max-length 128，并启用 expandable_segments 降低碎片化风险。
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python compare_correctness.py --world-size 2 --max-length 128 --output correctness-results.local.json
python benchmark_allreduce.py --output allreduce-results.local.json
python benchmark.py --output benchmark-results.local.json
```


## 适用边界

- 主线模型为 `Qwen/Qwen3-0.6B-Base`，训练目标为全参数 SFT。
- `smoke_test.py` 不依赖真实模型下载或 GPU；真实权重正确性检查与 Benchmark 需要上文注明的 GPU 配置。
- 本目录只提供代码、依赖和运行说明。正确性与 Benchmark 命令会在本地生成 `*.local.json`，仓库不附带实验结果。
