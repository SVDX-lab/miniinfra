# 第 30 期：DDP Gradient Bucket 与通信计算重叠

## 对应篇章

对应《超轻量AI Infra教程—训练篇》第 15 章；合并后的连续编号为第 30 章。

## 显卡要求

最低 2 张 GPU；参考配置覆盖 2、4 张 GPU，以及双节点 4+4 张 GPU。

第 29 期的梯度 AllReduce 严格排在 Backward 结束之后，整个通信时间都暴露在关键路径上。真实的 DDP 不是这样工作的：参数被分成若干桶，某个桶的梯度一齐就绪时就立刻发起 AllReduce，与后续 Backward 计算重叠。本期只改变一件事：梯度 AllReduce 的启动时机（Backward 后一次性 对照 Backward 中按 Bucket 分批异步）。模型、World Size、全局 Batch、通信数据总量与第 29 期完全相同。

## 问题、假设与边界

- 问题：Bucket 化能在多大程度上把通信藏进计算？Bucket 大小如何影响隐藏效果、尾部等待与临时显存？
- 假设：梯度按反向就绪顺序（参数逆序）就绪，小桶先发先收，可与计算重叠；但 PCIe PHB 无 NVLink、通信与计算共享带宽，收益可能有限；桶过小时固定延迟占比上升反而变慢，桶过大退化为第 29 期的全暴露形态。
- 唯一变量：AllReduce 启动时机。
- 固定项：与第 29 期相同（2 卡、全局 batch 4、FP32、每步约 2.4 GB 梯度、种子 20260912）。
- 排除项：累积期 no_sync（第 31 期）、状态/梯度分片（第 32/33 期）、参数粒度以下的桶内拆分。

## 文件

- `bucketed_ddp.py`：本期核心。`build_buckets`（参数逆序装桶、参数粒度、不跨桶拆分）、`BucketCoordinator`（`register_post_accumulate_grad_hook` 钩子：拷入桶缓冲→计数→桶满即 `async_op=True` AllReduce；`wait_and_attach` 尾部等待并把梯度以视图形式挂回参数）、`bucketed_train_step`（与第 29 期同口径的分段计时）。
- `ddp_core.py`：第 29 期复制件，`flat` 臂的基线步体。
- `compare_correctness.py` / `compare_reference.py`：真实权重三重对照——桶化梯度对大 batch 参考（容差闸门）、桶化对 flat 同权重同批次逐位一致、8 步轨迹。
- `smoke_test.py`：CPU/Gloo——桶布局、钩子覆盖、双桶档位逐位 parity、跨 Rank 一致。
- `benchmark.py`：flat + 1/16/128/1024 MiB 四档 Bucket，3 次重复。

## 环境和数据

依赖、模型和数据与第 29 期相同；分布式正确性检查至少需要 2 张 GPU。

```bash
cd chapter_30
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 python prepare_data.py --samples 512
```

## 运行顺序

```bash
CUDA_VISIBLE_DEVICES="" python smoke_test.py
python compare_correctness.py --world-size 2 --output correctness-results.local.json
python benchmark.py --output benchmark-results.local.json
```


## 适用边界

- 主线模型为 `Qwen/Qwen3-0.6B-Base`，训练目标为全参数 SFT。
- `smoke_test.py` 不依赖真实模型下载或 GPU；真实权重正确性检查与 Benchmark 需要上文注明的 GPU 配置。
- 本目录只提供代码、依赖和运行说明。正确性与 Benchmark 命令会在本地生成 `*.local.json`，仓库不附带实验结果。
