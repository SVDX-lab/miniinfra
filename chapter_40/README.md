# 第 40 期：1F1B Pipeline Schedule

## 对应篇章

对应《超轻量AI Infra教程—训练篇》第 25 章；合并后的连续编号为第 40 章。

## 显卡要求

最低 2 张 GPU；参考配置为 4 张 GPU。

第 39 期已经让按层切分的模型跑起了 GPipe Fill-Drain 流水，但 Fill-Drain 的代价是每个 Stage 从 Fill 开始就要驻留所有 micro-batch 的激活图，直到 Drain 结束才释放。本期只改变一件事：在完全相同的 Stage 划分、Stage 数、Micro-batch 数和 P2P 协议下，把调度从 Fill-Drain 换成非交错 1F1B——Stage s 先做 min(S-1-s, M) 个 Warm-up Forward，Steady State 交替执行一次 Forward 与一次 Backward，最后 Cool-down 排空。虚拟 Stage 是第 41 期，不在本期。

## 问题、假设与边界

- 问题：激活生命周期缩短多少（每 Stage 同时驻留的激活图从 M 份降到 min(S-s, M) 份）？Bubble 和 Stage 空闲时间相对 Fill-Drain 如何变化？在小模型 + PCIe P2P 上，调度收益是否会被更多次的收发等待吃掉？
- 假设：两种调度计算同一个数学目标——相同 micro-batch、相同 loss scale、相同更新——所以正确性只差 FP32 归约顺序；1F1B 的主要价值首先体现为峰值激活显存下降，而不是绝对吞吐大幅上升；Bubble 上界仍服从 (S-1)/(M+S-1) 的填充-排空模型，micro 增大时两臂差距收窄。负优化如实保留。
- 唯一变量：流水调度。`gpipe_ppN` 臂（第 39 期 Fill-Drain，先所有 Forward 再所有 Backward）对照 `onefoone_ppN` 臂（Warm-up / Steady 1F1B / Cool-down）。两臂权重、Stage 划分、micro-batch 序列、loss 口径、优化器完全一致。
- 固定项：模型/数据/标签/dtype（FP32、TF32 关闭）/eager attention/优化器公式/种子；每 micro 样本数 = 2 × Stage 数，每步全局 batch = micro × 2 × Stage 数。
- 排除项：Interleaved/虚拟 Stage（第 41 期）、自动负载均衡（只用连续均分）、TP/SP 与流水组合（第 42 期）、激活重算/编译/FlashAttention（第 25/26/28 期机制）。

## 机制与易错点

1. **调度公式**。Stage s 的 Warm-up 数是 `min(S-1-s, M)`：Stage 越靠前，第一个 Backward 到达得越晚，能提前做的 Forward 就越多；`M` 封顶（micro 不够时全部算 Warm-up，退化为 Fill-Drain 形状）。Steady State 执行 `M - warmup` 轮"一次 Forward + 一次 Backward"，Cool-down 再排空 `warmup` 个 Backward。每个 Stage 的执行序完全由 `(stage_index, num_stages, num_micro)` 决定，无需任何额外同步。
2. **in-flight 激活数**。Warm-up 结束时驻留 `warmup` 份激活图，Steady 每轮先多拿一份（刚 Forward 的）再放掉一份（刚 Backward 的），峰值是 `min(S-s, M)`。这是 1F1B 的主要价值假设，benchmark 用每 Rank 峰值 allocated 显存对照 `gpipe` 的 M 份。
3. **迭代内操作顺序与死锁**。Steady 一轮按"收激活 → Forward → 收梯度 → Backward → 发梯度 → 发激活"执行：所有阻塞 recv 先于本轮的 send 挂出。gloo 这类严格阻塞的传输下，若先把激活 send 出去再等上一级的梯度 recv，相邻 Stage 可能互相等对方的 recv 形成环；先收后发打断了这个环（推理见 `pipeline_1f1b.py` 模块注释）。NCCL 异步入队，两种顺序都成立，因此 CPU smoke 与 GPU 用同一条代码路径。每对 Stage 之间每个方向的消息仍严格按 micro 序排列，P2P 协议与第 39 期逐字节相同。
4. **梯度语义不变**。最后一个 Stage 从 loss 反传；其余 Stage 收到输出激活的输入梯度后用 `torch.autograd.backward(out, grad_tensors=g)` 启动反传，再把输入激活的 `.grad` 传回。哪个 micro 先算完不影响任何一份梯度——各参数的梯度都按 micro 序累加，与 `gpipe` 逐位一致（容差内）。
5. **tied Embedding/LM Head 与更新边界**。与第 39 期完全相同：所有 micro 反传完成后做一对 P2P 交换（末→首传 LM Head 梯度，首→末传 Embedding 梯度），两侧都拿 `g_embed + g_lm_head` 更新；然后用精确全局范数（1 次标量 AllReduce）裁剪并各自 AdamW step。调度只重排时间，不改变更新边界。
6. **Bubble 与时间线**。复用第 39 期的 `StageTimeline`/`compute_bubble`：每 micro 每 Stage 的 Forward/Backward/P2P 事件写入 JSON，Bubble 可从原始事件复算。1F1B 的时间线签名是 Warm-up 的连续 F 之后 F/B 交替，与 Fill-Drain 的 F…F B…B 块状结构形成对照。

## 文件

- `pipeline_1f1b.py`：本期核心。`split_layers` / `PipelineStage` / `load_pipeline_stage` / `exchange_tied_gradients` / `clip_grad_norm_pp` / `StageTimeline` / `compute_bubble` 与第 39 期逐行一致（唯一变量是调度，这些必须不动）；`schedule_1f1b`（Warm-up/Steady/Cool-down 与 in-flight 公式）、`onef1b_step`（本期变体调度）、`gpipe_step`（第 39 期基线，原样保留）、`pp_train_step`（`'gpipe'|'onef1b'` 两种模式，输出调度字段与分段时间）。
- `qwen3_model.py`：第 33/35 期纯复制版（PP 只切层，权重整读）。
- `pp_runtime.py`：数据流水与进程组通用件（deterministic 编码、第 22 期 shuffle、env:// 启动、micro-batch 组装与 loss scale），与第 39 期一致。
- `smoke_test.py`：CPU gloo，tiny 模型（hidden 32 / intermediate 64 / vocab 64 / 4 层 / 8 Q Head / 4 KV Head / head_dim 8 / seq 12），PP=2 与 uneven PP=3 [2,1,1] 两轮、各 4 micro × 2 样本：切分与调度公式、gpipe 对照参考、onef1b 与 gpipe 梯度一致、onef1b 时间线逐 Stage 检查（warmup F + F/B 交替 + cooldown B，Forward/Backward 各按 micro 序）、一次 AdamW step 两种模式对照。
- `compare_correctness.py`：真实 Qwen3-0.6B 权重，PP=2 与单卡 FP32 参考对照（参考跑同一 micro 序列）：onef1b 的 loss 轨迹、各 Stage 梯度对照切片、Stage 边界输入梯度对照 hook 捕获、onef1b/gpipe 梯度一致性、一次更新后参数、时间线 JSON。
- `benchmark.py`：每 world size 两臂 `gpipe_ppN` / `onefoone_ppN`（W=2 主实验、W=4 对照），micro 扫描 {2,4,8}：Step Time、Bubble 比例、每 Stage Forward/Backward/P2P 时间、P2P 字节（含 tied 交换）、调度理论 in-flight 数、每 Rank 峰值显存、最后一步完整时间线 JSON、有效监督 Token/s。
- `prepare_data.py` / `sft_core.py` / `assistant_loss.py` / `mini_batch.py` / `qwen3_tokenizer.py` / `experiment_utils.py`：与第 39 期一致的稳定工具（schema `chapter40.v1`）；数据文件名为 `data/chapter40_train.jsonl` 与 `data/chapter40_manifest.json`。

## 环境和数据

与第 39 期完全相同，模型固定为 `Qwen/Qwen3-0.6B-Base@da87bfb608c14b7cf20ba1ce41287e8de496c0cd`，数据固定为 `sft_mini_512.jsonl@9b06bee29c0d5685647ac11b0dcea9f572c5064a`（SHA256 `475039fa…7852`，1,232,540,940 字节，下载后校验）。

```bash
cd chapter_40
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 python prepare_data.py --samples 512
```

模型权重在首次运行时由 `resolve_model_directory` 按固定 revision 下载。

## 运行顺序

```bash
# 1. CPU smoke（不需要 GPU 与模型下载；PP=2 与 uneven PP=3 两轮，4 micro）
CUDA_VISIBLE_DEVICES="" python smoke_test.py

# 2. 真实权重正确性：PP=2（单机双卡，mp.spawn）
python compare_correctness.py --world-size 2 --output correctness-results.local.json
#    torchrun 等价形式（--world-size 必须等于 WORLD_SIZE）：
#    torchrun --nproc-per-node 2 compare_correctness.py --world-size 2
#    PP=4（单机四卡；参考与 stash 约 10 GB，显存紧张时下调 --max-length）

# 3. Benchmark 主实验：W=2（同轮内先跑 gpipe_pp2 再跑 onefoone_pp2，micro {2,4,8}）
python benchmark.py --world-sizes 2 --output benchmark-results.local.json
#    W=4（单机四卡）
python benchmark.py --world-sizes 4 --output benchmark-results.local.json
#    只跑部分 micro：--micro-batches 2,4
```

所有输出默认写入 `*.local.json`，并由 Git 忽略。

## 指标口径与计时边界

- `step_time_ms` 为端到端 Wall Clock（覆盖所有 micro 的 forward/backward、tied 梯度 P2P、裁剪与 AdamW），两端正确同步；每个 micro 每 Stage 的 Forward/Backward/P2P 用 CUDA Event 记录（CPU 退化为 perf_counter），时间线以 step 起点为 0 的秒为单位写入 JSON，Bubble 可从原始事件复算。
- Bubble 比例定义为 `1 - 计算busy/窗口`（每 Rank 自己的时间线，P2P 时间单列不折算进 busy）；两臂在同一全局工作量下对比（Strong Scaling）。
- P2P 记录时间与字节数（激活 `[b, s, h]`、输入梯度、tied 交换各一类）；两臂的 P2P 消息数完全相同（每 micro 每边一条），差异只应体现在等待时间上。
- 调度理论 in-flight 数（`gpipe` = M，`onef1b` = min(S-s, M)）随每步指标输出，与每 Rank 峰值 allocated/reserved 显存（`per_rank_memory`）对照；OOM 记录失败配置、失败阶段与当时 allocated/reserved。
- 主吞吐为有效监督 Token/s（全局 batch = micro × 2 × Stage 数，只计一次），同时报告物理 Token/s。


## 适用边界

- 主线模型为 `Qwen/Qwen3-0.6B-Base`，训练目标为全参数 SFT。
- `smoke_test.py` 不依赖真实模型下载或 GPU；真实权重正确性检查与 Benchmark 需要上文注明的 GPU 配置。
- 本目录只提供代码、依赖和运行说明。正确性与 Benchmark 命令会在本地生成 `*.local.json`，仓库不附带实验结果。
