# 第 32 期：ZeRO-1——切分优化器状态

## 对应篇章

对应《超轻量AI Infra教程—训练篇》第 17 章；合并后的连续编号为第 32 章。

## 显卡要求

最低 2 张 GPU；参考配置覆盖 2、4 张 GPU，以及双节点 4+4 张 GPU。

DDP 每卡持有完整模型、完整梯度和完整优化器状态。0.6B 在 12GB 上已经足以形成容量压力。本期只改变 FP32 优化器状态是否在 Data Parallel Rank 间切分（ZeRO-1）。每卡仍保存完整参数和完整梯度；AdamW 动量分片存放在各 Rank，更新只发生在参数分片的所有者上，更新完成后用一次分块 AllGather 把新参数同步给其他 Rank。

## 问题、假设与边界

- 问题：切分后的训练语义与复制式 AdamW 是否一致？每卡节省多少状态显存、付出多少额外通信？0.6B 在 12GB 上的 Batch/Length 准入边界在哪里？
- 假设：AdamW 是逐元素更新，"本 Rank 状态分片 × 本 Rank 参数分片 × 本 Rank 梯度分片"与全量 torch AdamW 逐位一致（公式逐操作镜像）；代价是每步新增约 1×参数字节的 AllGather（DDP 没有），收益是状态显存减半。增加 Micro-batch 或序列长度时，ZeRO-1 应比复制式 DDP 更晚 OOM。
- 唯一变量：优化器状态是否切分。
- 固定项：与第 29 期相同（梯度路径仍是完整梯度 + 一次 AllReduce SUM）。
- 排除项：梯度切分（第 33 期）、参数切分（第 34 期）、框架配置对照（可引用不实测）。

## 文件

- `zero1_core.py`：本期核心。`shard_boundaries`（连续分片、余数分给前几个 Rank）、`ShardedStateAdamW`（只为本地分片分配 exp_avg/exp_avg_sq，`step()` 逐操作镜像 torch AdamW 公式）、`all_gather_parameters`（分块 AllGather，块缓冲 128MB，非整除参数按 padded 长度打包）、`memory_ledger`（状态所有权账本）。
- `ddp_core.py`：第 29 期复制件，梯度路径与 `ddp` 臂步体。
- `compare_correctness.py` / `compare_reference.py`：真实权重 8 步对照 + 状态账本。
- `smoke_test.py`：CPU——分片数学、线性层三步逐位对照、tiny Qwen3 一步、跨卡一致、账本比例。
- `benchmark.py`：ddp / zero1 两臂 × 3 次重复（AllGather 时间单独计时、按加法计入步时间）。
- `capacity_probe.py`：0.6B × ddp/zero1 × Micro-batch/Length 网格的 2 步真实容量探针，记录成功或 OOM 位置。

## 环境和数据

与第 29 期完全相同，模型固定为 `Qwen/Qwen3-0.6B-Base@da87bfb608c14b7cf20ba1ce41287e8de496c0cd`。

```bash
cd chapter_32
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
