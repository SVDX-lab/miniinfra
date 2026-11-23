# 第 33 期：ZeRO-2——再切分梯度

## 对应篇章

对应《超轻量AI Infra教程—训练篇》第 18 章；合并后的连续编号为第 33 章。

## 显卡要求

最低 2 张 GPU；参考配置覆盖 2、4 张 GPU，以及双节点 4+4 张 GPU。

第 32 期切分了优化器状态，但每卡仍保留完整梯度——backward 结束那一刻它必须存在，之后却只有本 Rank 的那份会被本 Rank 的优化器用到。本期只改变一件事：梯度是否切分（完整 AllReduce 对照 Reduce-Scatter 只保留本 Rank 分片）。优化器状态沿用 ZeRO-1 的切分，参数副本策略不变（仍是 ZeRO-2 而非 ZeRO-3）。

## 问题、假设与边界

- 问题：Reduce-Scatter 的通信时间与 AllReduce 差多少？每卡梯度显存省多少？0.6B 在 12GB 上的 Batch/Length 准入边界如何变化？
- 假设：Reduce-Scatter 与 AllReduce 的每链接数据量同为 2(W-1)/W·D，W=2 时通信时间应接近；分片梯度与完整 AllReduce 结果的对应切片逐位一致（W=2 逐元素求和交换律）；增加 Micro-batch 或序列长度时，ZeRO-2 应比 ZeRO-1 更晚 OOM。梯度切分后 `clip_grad_norm_` 不再可用，全局范数要由"各 Rank 分片局部范数平方和 → 一次标量 AllReduce"合成，再缩放本地分片。
- 唯一变量：梯度是否切分。
- 固定项：优化器状态分片（1/W）、参数副本、梯度路径输入、通信量口径均与第 32 期 zero1 臂一致。
- 排除项：参数切分与按需 AllGather（第 34 期）、Bucket 化 Reduce-Scatter（真实 ZeRO 的扁平桶实现；本期逐参数发起，代价如实测量）。

## 文件

- `zero2_core.py`：本期核心。`ShardedGradAdamW`（梯度分片缓冲随状态一起持久分配）、`reduce_scatter_gradients`（逐参数 Reduce-Scatter，整除时直接以梯度视图为输入、完成后立即释放完整梯度；非整除按零填充，填充位置求和仍为零）、`clip_grad_norm_sharded`（分片空间的全局范数裁剪：局部平方和 → 标量 AllReduce → 缩放本地分片）、`all_gather_parameters`（与第 32 期相同的分块实现）、`zero2_train_step`（分段计时）、`ring_reduce_scatter`（Gloo/CPU 的环状回退，教学用）。
- `ddp_core.py` / `zero1_core.py`：第 29/32 期复制件，`zero1` 臂基线。
- `compare_correctness.py` / `compare_reference.py`：真实权重对照（含逐参数分片逐位断言）。
- `smoke_test.py`：CPU——Reduce-Scatter 数学（含奇数 numel 填充路径）、线性层 3 步全流程逐位对照、tiny Qwen3、账本比例。
- `benchmark.py`：zero1 / zero2 两臂 × 3 次重复。
- `capacity_probe.py`：0.6B × ddp/zero1/zero2 × Micro-batch/Length 网格容量探针。

## 环境和数据

与第 29 期完全相同，模型固定为 `Qwen/Qwen3-0.6B-Base@da87bfb608c14b7cf20ba1ce41287e8de496c0cd`。

```bash
cd chapter_33
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 python prepare_data.py --samples 512
```

## 运行顺序

```bash
CUDA_VISIBLE_DEVICES="" python smoke_test.py
python compare_correctness.py --world-size 2 --output correctness-results.local.json
python benchmark.py --output benchmark-results.local.json
python capacity_probe.py --max-lengths 128 256 512 --micro-batch-sizes 1 2 4 --output capacity-results.local.json
```


## 适用边界

- 主线模型为 `Qwen/Qwen3-0.6B-Base`，训练目标为全参数 SFT。
- `smoke_test.py` 不依赖真实模型下载或 GPU；真实权重正确性检查与 Benchmark 需要上文注明的 GPU 配置。
- 本目录只提供代码、依赖和运行说明。正确性与 Benchmark 命令会在本地生成 `*.local.json`，仓库不附带实验结果。
