# 第 31 期：梯度累积期间延迟同步

## 对应篇章

对应《超轻量AI Infra教程—训练篇》第 16 章；合并后的连续编号为第 31 章。

## 显卡要求

最低 2 张 GPU；参考配置覆盖 2、4 张 GPU，以及双节点 4+4 张 GPU。

第 30 期把单次 Backward 内的通信藏进了计算；本期处理累积维度：当一个 Optimizer Step 由 K 个 Micro-step 组成时，每个 Micro-step 都同步一次梯度是极大的浪费——前 K-1 次纯本地累积（no_sync）、最后一次统一同步，梯度数学不变、通信量降为 1/K。本期只改变一件事：微步间梯度同步的时机（每微步同步 对照 仅最后一次同步）。Bucket 机制不再使用（第 30 期变量），所有 AllReduce 仍是 Backward 结束后的平铺 SUM。

## 问题、假设与边界

- 问题：两种同步模式的最终梯度和参数更新是否数学等价？每步通信量与步时间差多少？
- 假设：每微步损失按监督 Token 占比缩放后，两种模式都等于全局 token_mean 梯度；"每微步同步"的正确实现必须用独立累加器（对已归约的累积缓冲再次 AllReduce 会把先前微步贡献放大 World Size 倍）；K=2 时 no_sync 省一次全量 AllReduce。
- 唯一变量：微步间同步时机。
- 固定项：与第 29 期相同；K 主对照为 2（全局 batch 4），容量对照 K=4（全局 batch 8，每卡每微步 1 样本，切片方式同第 24 期）。
- 排除项：Bucket 重叠（第 30 期）、状态/梯度分片（第 32/33 期）、BF16。

## 文件

- `no_sync_core.py`：本期核心。`build_micro_batches`（Rank 分片再切成 K 个连续微批，记账监督 Token）、`accumulate_step`（`every`：微步 backward→AllReduce SUM→独立累加器；`last`：全部微步本地累积后一次 AllReduce SUM；分段计时 + AllReduce 计数与字节数）。
- `ddp_core.py`：第 29 期复制件，`k1` 臂（单微步锚点）与损失缩放工具。
- `compare_correctness.py` / `compare_reference.py`：真实权重两种模式对照大 batch 参考（step-0 梯度、8 步轨迹、跨卡一致、AllReduce 计数）。
- `smoke_test.py`：CPU/Gloo——两种模式梯度一致（FP32 结合律容差）、计数断言、一步跨卡一致。
- `benchmark.py`：5 臂（k1 / k2_every / k2_last / k4_every / k4_last）× 3 次重复。

## 环境和数据

依赖、模型和数据与第 29 期相同；分布式正确性检查至少需要 2 张 GPU。

```bash
cd chapter_31
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
