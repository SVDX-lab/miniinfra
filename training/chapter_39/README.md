# 第 39 期：GPipe Fill-Drain Pipeline

## 对应篇章

对应《超轻量AI Infra教程—训练篇》第 24 章；合并后的连续编号为第 39 章。

## 显卡要求

最低 2 张 GPU；参考配置覆盖 2 张和 4 张 GPU。

第 38 期已掌握单层内部的张量与激活切分，本期把模型按层跨卡切分：28 层连续均分，PP=2 每 Stage 14 层，PP=4 每 Stage 7 层（不能整除时前段 Stage 多一层并记录，0.6B 恰好整除）。但只切层还不够——一个 micro-batch 从头走到尾时，任一时刻只有一张卡在算。本期只改变一件事：在完全相同的 Stage 划分下，多个 Micro-batch 是逐个走完全流水（baseline），还是先连续 Forward 再连续 Backward 组成 GPipe Fill-Drain 流水（实验组）。1F1B 是第 40 期，虚拟 Stage 是第 41 期，均不在本期。

## 问题、假设与边界

- 问题：Fill-Drain 能把 Stage 间的等待压缩多少（Bubble）？激活驻留显存要为此多付多少（每 Stage 同时持有 M 份激活图 vs 1 份）？P2P 传输本身占多少时间？小模型在 PCIe P2P 上，调度收益是否会被通信吃掉？
- 假设：两种调度计算同一个数学目标——相同 micro-batch、相同 loss scale、相同更新——所以正确性只差 FP32 归约顺序；Bubble 随 micro-batch 数增加而下降（填充-排空模型给 (S-1)/(M+S-1) 的上界），峰值激活显存随 M 线性上升。0.6B 太小、P2P 走 PCIe，端到端加速比可能远小于理论值甚至为负；负优化如实保留。
- 唯一变量：流水调度。`serial_ppN` 臂（一个 micro-batch 完整走完所有 Stage 再下一个）对照 `gpipe_ppN` 臂（先所有 Forward 再所有 Backward）。两臂权重、Stage 划分、micro-batch 序列、loss 口径、优化器完全一致。
- 固定项：模型/数据/标签/dtype（FP32、TF32 关闭）/eager attention/优化器公式/种子；每 micro 样本数 = 2 × Stage 数，每步全局 batch = micro × 2 × Stage 数。
- 排除项：1F1B（第 40 期）、Interleaved/虚拟 Stage（第 41 期）、自动负载均衡（只用连续均分）、TP/SP 与流水组合（第 42 期）、激活重算/编译/FlashAttention（第 25/26/28 期机制）。

## 机制与易错点

1. **Stage = 连续层 + 边界件**。Stage 0 持有 Embedding，最后一个 Stage 持有最终 RMSNorm 与 LM Head；中间 Stage 只有 `layers[j]`。`load_pipeline_stage` 用 safetensors 只读本 Stage 的层窗口（其他 Stage 的层权重从不物化），参数不复制——Stage 划分只是层切分。
2. **Backward 靠 P2P 梯度种子驱动**。最后一个 Stage 从 loss 反传；其余每个 Stage 收到"自己的输出激活的输入梯度"后用 `torch.autograd.backward(out, grad_tensors=g)` 启动本 Stage 反传，再把本 Stage 输入激活的 `.grad` 传回上一个 Stage。这个收到的梯度就是"缺失的 loss 项"的梯度链，语义与单卡完全一致。非最后 Stage 的 loss 分片是全零占位，只用于各 Rank 记账统一。
3. **Loss 聚合**。每个 micro 的 loss 按 `supervised_i / supervised_global` 加权，梯度总和精确等于全局 token_mean。每个 Stage 都能本地算出 scale（数据流水确定性，各 Stage 编码同一份语料），P2P 载荷因此只有激活与梯度。
4. **tied Embedding/LM Head 的代价**。Qwen3-0.6B 共享 Embedding 与 LM Head，但 PP 把它们放在首末两个 Stage。两份同值副本各自累积不同的部分梯度（lookup vs logits）；`exchange_tied_gradients` 在反向结束后做一对 P2P（末→首传 LM Head 梯度，首→末传 Embedding 梯度），两侧都拿到 `g_embed + g_lm_head` 再更新，副本逐步保持逐位一致，语义与 tied 参考相同。这是真实的 PP+tied-embedding 工程问题，代价是每步 2 × vocab×hidden 字节的 P2P。
5. **参数更新边界**。所有 Stage 的所有 micro 反传完成 + tied 交换完成后，各 Stage 用精确全局范数（参数互斥，1 次标量 AllReduce）裁剪并各自 AdamW step。更新公式逐参数与单卡一致。
6. **调度与死锁**。阻塞式 P2P 下，fill 阶段消息全部向右、drain 阶段全部向左，各自无环，gloo（CPU smoke）不会死锁；NCCL 异步入队，Stage 0 能提前跑下一个 micro——这才是流水本身。
7. **显存对照**。serial 每 Stage 同一时刻只驻留 1 个 micro 的激活图；gpipe 驻留 M 份。两臂峰值显存差是本期机制性结论之一。

## 文件

- `pipeline_parallel.py`：本期核心。`split_layers`（连续均分 + 余数规则）、`PipelineStage`（层块 + 边界件 + `forward_layers`/`head_loss`）、`load_pipeline_stage`（safetensors 只读本 Stage 窗口；tied LM Head 从 Embedding 键读入）、`gpipe_step` / `serial_step`（两种调度，P2P 契约相同）、`exchange_tied_gradients`（tied 梯度 P2P 汇合）、`clip_grad_norm_pp`（精确全局范数）、`pp_train_step`（step 边界 + 分段指标）、`StageTimeline` + `compute_bubble`（每 micro 每 Stage 的 Forward/Backward/P2P 事件时间线，CUDA Event + Wall Clock，可复算 Bubble）、`build_micro_batches` 在 `pp_runtime`（全局 batch 行切分）。
- `qwen3_model.py`：第 33/35 期纯复制版（无 TP 工厂与分片加载器；PP 只切层，权重整读）。
- `pp_runtime.py`：数据流水与进程组通用件（deterministic 编码、第 22 期 shuffle、env:// 启动、micro-batch 组装与 loss scale）。
- `smoke_test.py`：CPU gloo，tiny 模型（hidden 32 / intermediate 64 / vocab 64 / 4 层 / 8 Q Head / 4 KV Head / head_dim 8 / seq 12），PP=2 与 uneven PP=3 [2,1,1] 两轮：切分规则、gpipe 对照参考（loss/全部梯度/边界输入梯度/tied 交换）、serial 与 gpipe 梯度一致、一次 AdamW step 两种模式对照、时间线事件与 fill-drain 排序。
- `compare_correctness.py`：真实 Qwen3-0.6B 权重，PP=2 与单卡 FP32 参考对照（参考跑同一 micro 序列）：loss 轨迹、各 Stage 梯度对照切片、Stage 边界输入梯度对照 hook 捕获、serial/gpipe 梯度一致性、一次更新后参数、时间线 JSON。
- `benchmark.py`：每 world size 两臂 `serial_ppN` / `gpipe_ppN`（W=2 主实验、W=4 对照），micro 扫描 {2,4,8}：Step Time、Bubble 比例、每 Stage Forward/Backward/P2P 时间、P2P 字节（含 tied 交换）、每 Rank 峰值显存、最后一步完整时间线 JSON、有效监督 Token/s。
- `prepare_data.py` / `sft_core.py` / `assistant_loss.py` / `mini_batch.py` / `qwen3_tokenizer.py` / `experiment_utils.py`：与第 36 期一致的稳定工具（schema `chapter39.v1`，`environment_record` 增加 `pp_degree`）；数据文件名为 `data/chapter39_train.jsonl` 与 `data/chapter39_manifest.json`。

## 环境和数据

与第 36 期完全相同，模型固定为 `Qwen/Qwen3-0.6B-Base@da87bfb608c14b7cf20ba1ce41287e8de496c0cd`，数据固定为 `sft_mini_512.jsonl@9b06bee29c0d5685647ac11b0dcea9f572c5064a`（SHA256 `475039fa…7852`，1,232,540,940 字节，下载后校验）。

```bash
cd chapter_39
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 python prepare_data.py --samples 512
```

模型权重在首次运行时由 `resolve_model_directory` 按固定 revision 下载。

## 运行顺序

```bash
# 1. CPU smoke（不需要 GPU 与模型下载；PP=2 与 uneven PP=3 两轮）
CUDA_VISIBLE_DEVICES="" python smoke_test.py

# 2. 真实权重正确性：PP=2（单机双卡，mp.spawn）
python compare_correctness.py --world-size 2 --output correctness-results.local.json
#    torchrun 等价形式（--world-size 必须等于 WORLD_SIZE）：
#    torchrun --nproc-per-node 2 compare_correctness.py --world-size 2
#    PP=4（单机四卡；参考与 stash 约 10 GB，显存紧张时下调 --max-length）

# 3. Benchmark 主实验：W=2（同轮内先跑 serial_pp2 再跑 gpipe_pp2，micro {2,4,8}）
python benchmark.py --world-sizes 2 --output benchmark-results.local.json
#    W=4（单机四卡）
python benchmark.py --world-sizes 4 --output benchmark-results.local.json
#    只跑部分 micro：--micro-batches 2,4
```

所有输出默认写入 `*.local.json`，并由 Git 忽略。

## 指标口径与计时边界

- `step_time_ms` 为端到端 Wall Clock（覆盖所有 micro 的 forward/backward、tied 梯度 P2P、裁剪与 AdamW），两端正确同步；每个 micro 每 Stage 的 Forward/Backward/P2P 用 CUDA Event 记录（CPU 退化为 perf_counter），时间线以 step 起点为 0 的秒为单位写入 JSON，Bubble 可从原始事件复算。
- Bubble 比例定义为 `1 - 计算busy/窗口`（每 Rank 自己的时间线，P2P 时间单列不折算进 busy）；两种口径（串行 vs 流水、不同 micro 数）都在同一全局工作量下对比（Strong Scaling）。
- P2P 记录时间与字节数（激活 `[b, s, h]`、输入梯度、tied 交换各一类）；tied 交换每步固定 2 次 P2P。
- 显存报告每 Rank 峰值 allocated/reserved（`per_rank_memory`），并预期 serial < gpipe（1 份 vs M 份激活图）；OOM 记录失败配置、失败阶段与当时 allocated/reserved。
- 主吞吐为有效监督 Token/s（全局 batch = micro × 2 × Stage 数，只计一次），同时报告物理 Token/s。


## 适用边界

- 主线模型为 `Qwen/Qwen3-0.6B-Base`，训练目标为全参数 SFT。
- `smoke_test.py` 不依赖真实模型下载或 GPU；真实权重正确性检查与 Benchmark 需要上文注明的 GPU 配置。
- 本目录只提供代码、依赖和运行说明。正确性与 Benchmark 命令会在本地生成 `*.local.json`，仓库不附带实验结果。
