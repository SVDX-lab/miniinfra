# 第 41 期：Interleaved 1F1B

## 对应篇章

对应《超轻量AI Infra教程—训练篇》第 26 章；合并后的连续编号为第 41 章。

## 显卡要求

最低 2 张 GPU；参考配置为 4 张 GPU。

第 40 期的 1F1B 已把激活生命周期压到 min(S-s, M)，但每个物理 Rank 仍只持有一个连续 Stage，流水级数 S 被物理卡数锁死。本期只改变一件事：每个物理 Rank 保存多个非连续的虚拟 Stage（模型块，total stage = P × vpp），调度换成 Pipeline-parallel 论文的 Interleaved 1F1B——micro-batch 按 P 分组，组内按块扫 Forward、按块逆序扫 Backward。混合并行是第 42 期，不在本期。

## 问题、假设与边界

- 问题：虚拟 Stage 把流水级数从 P 增到 2P，Bubble 能缩短多少？同一 `[b, s, h]` 载荷要飞过 2P-1 条 Stage 边而不是 P-1 条，P2P 次数约翻倍——通信代价与 Bubble 收益孰大孰小？每 Rank 的 in-flight 激活（Warm-up 数增大）涨多少？
- 假设：两种 Placement 计算同一个数学目标——相同 micro-batch、相同 loss scale、相同更新——所以正确性只差 FP32 归约顺序；Bubble 因虚拟 Stage 更短而下降、P2P 次数与 in-flight 显存上升；0.6B 太小、P2P 走 PCIe 时端到端吞吐可能为负优化，如实保留。
- 唯一变量：每个物理 Rank 保存一个 Stage 还是多个非连续虚拟 Stage（Placement 表与其上的调度）。`onefoone_ppN` 臂（第 40 期非交错 1F1B，每 Rank 一个连续 Stage）对照 `interleaved_ppN` 臂（每 Rank 2 个虚拟 Stage）。两臂每 Rank 层数相同（W=4 均 7 层）、全局 batch 相同、权重切分互补不重叠。
- 固定项：模型/数据/标签/dtype（FP32、TF32 关闭）/eager attention/优化器公式/种子；每 micro 样本数 = 2 × 物理 Rank 数；micro 扫描 {4,8}（interleaved 要求 micro 能被 P 整除，作为调度前置条件显式报错）。
- 排除项：vpp>2 与自动负载均衡、ZeRO/DP×PP、TP/SP 与流水组合（第 42 期）、激活重算/编译/FlashAttention（第 25/26/28 期机制）。

## 机制与易错点

1. **Placement 表是唯一的真相来源**。虚拟 Stage s 位于物理 Rank `s % P`，是 Rank 的第 `s // P` 个模型块；Rank r 拥有 `{r, r+P, ...}`——层区间非连续（28 层 / P=4 / vpp=2：8 个虚拟 Stage 依次 [4,4,4,4,3,3,3,3] 层，每 Rank 合计仍 7 层，除法余数规则记录在 split 元数据里）。所有收发都经 `stage_owner` 查表，不改代码即可换 Placement。
2. **本地与远程两种传递**。虚拟 Stage 的前后驱属于 `owner(s) ± 1 (mod P)`：块内边在相邻 Rank，块边界是 wrap 对 P-1↔0（不再是相邻 Rank）。`pass_activation`/`receive_activation` 同时实现两种路径——不同 Rank 走 `dist.isend` 投递 + `dist.recv` 接收（每对每方向仍严格按固定顺序，载荷不变），同 Rank 走本地 mailbox 直传（strided placement 下相邻虚拟 Stage 永不同属一个 Rank，该路径不会触发，但保留它使 Placement 表可替换，smoke 有单元检查）。
3. **调度映射**。Forward 第 i 项 = (块 `(i mod P·vpp) // P`，micro `(i // (P·vpp))·P + i mod P`)；Backward 第 i 项的块号按 `vpp-1-…` 逆序。micro 按 P 分组是调度的结构前提，`num_micro % P ≠ 0` 直接报错而不是静默重排。Rank r 的 Warm-up 数是 `min(2(P-1-r) + (vpp-1)P, vpp·M)`，每 Rank 的 F/B 序列仍呈 `F×warmup + (F,B)×steady + B×cooldown`。
4. **异步发送是死锁与正确性的双重前提**。与第 39/40 期的线性流水不同，Interleaved 的依赖在块边界闭合成环：P=2、vpp=2 下，Rank A 下一块的 Forward 依赖邻居 B 当前块的 Backward，而 B 的这次 Backward 又依赖 A 上一块的产物——阻塞式 `send` 在严格阻塞的传输（CPU gloo）上互相等待成死锁（开发期 CPU smoke 实测复现）；Megatron 在 NCCL 上靠点对点异步入队回避。本实现统一用 `dist.isend` 立即投递、步末统一 `wait`：所有阻塞等待都是 recv，其链条必终于 Embedding 前向（无 recv）或 loss 反向（无 recv），可证明无环。同时每条消息必须按每对每方向的固定 FIFO 顺序流动——激活在自己的前向算完后立即投递；任何"延迟到反向之后再发"都会静默打乱接收端的匹配顺序，把梯度当成激活（同样在开发期实测复现为 loss 漂移）。时间线把每条 send 记为"投递→步末 drain"，即真实暴露通信。
5. **梯度语义与更新边界不变**。最后一个虚拟 Stage（Rank P-1 的最后一块）从 loss 反传；其余虚拟 Stage 收到输出激活的输入梯度后 `torch.autograd.backward(out, grad_tensors=g)` 启动反传，再把输入激活的 `.grad` 传给前一个虚拟 Stage 的属主。所有反传完成后，物理首末 Rank 做一对 tied 梯度 P2P 交换，再按精确全局范数（1 次标量 AllReduce，参数在所有虚拟 Stage 上互斥）裁剪并各自 AdamW step。
6. **对照的可比性**。两臂每 Rank 层数相同、每步全局工作量相同（Strong Scaling）；P2P 计数单独成列——interleaved 的消息数由 Stage 边数决定（`(2P-1)·M` 条激活 + 同数梯度 vs `(P-1)·M`），消息更小更密，等待时间与次数分开读。

## 文件

- `interleaved.py`：本期核心。`split_virtual_stages`（divmod 余数规则 + 元数据）、`stage_owner`/`stage_chunk`（strided placement 表）、`VirtualStage`（块 + 边界件，`virtual_index`/`owner_rank`/`local_chunk`）、`load_pipeline_model`（safetensors 只读本 Rank 各块的层窗口，tied LM Head 从 Embedding 键读入最后一块）、`pass_activation`/`receive_activation`（本地 mailbox 与远程 P2P 双路径）、`interleaved_schedule`（Warm-up/Steady/Cool-down 与 in-flight 公式，micro 分组校验）、`interleaved_1f1b_step`（本期变体）、`onef1b_step` + `onef1b_schedule`（第 40 期基线，共享同一传输层）、`exchange_tied_gradients`（物理首末 Rank）、`clip_grad_norm_chunks`、`pp_train_step`（`'onef1b'|'interleaved'`）、`StageTimeline` + `compute_bubble`（事件带 chunk/virtual_stage 信息）。
- `qwen3_model.py`：第 33/35 期纯复制版（PP 只切层，权重整读）。
- `pp_runtime.py`：数据流水与进程组通用件，与第 39/40 期一致（world size = 物理流水大小）。
- `smoke_test.py`：CPU gloo 五轮同参考：A 轮 interleaved 2×2（total stage 4）、B 轮 onef1b 4 卡（等效 total stage）、C 轮 onef1b 2 卡（同物理卡）、D 轮 interleaved 4×2（total stage 8，每块 1 层，覆盖 wrap 对与最深 Warm-up）、E 轮 interleaved 2×2 × 8 micro（两个 micro-batch group）；切分/placement/调度公式、对照参考的 loss/逐块梯度/逐 micro 边界梯度/tied 交换/一次 AdamW step、时间线模式、A→B/C 的梯度 stash 交叉对照、P2P 计数对照（B 轮总数与 A 相同——Stage 图相同；C 轮严格小于 A——虚拟化的通信代价）、本地 mailbox 单元检查。
- `compare_correctness.py`：真实 Qwen3-0.6B 权重，W=2、vpp=2 与单卡 FP32 参考对照（参考按 interleaved 划分挂 hook）：interleaved 的 loss 轨迹、各虚拟 Stage 梯度按全局层名对照、逐块逐 micro 边界输入梯度、onef1b（vpp=1）对参考与对 interleaved 的梯度一致性、一次更新后参数、时间线 JSON。
- `benchmark.py`：每 world size 两臂 `onefoone_ppN` / `interleaved_ppN`，micro 扫描 {4,8}：Step Time、Bubble、每 (块, micro) 的 Forward/Backward/P2P 时间、P2P 次数/字节（含 tied 交换）、调度理论 in-flight、每 Rank 峰值显存、最后一步完整时间线 JSON、有效监督 Token/s。W=4 为主实验，W=2 附带。
- `prepare_data.py` / `sft_core.py` / `assistant_loss.py` / `mini_batch.py` / `qwen3_tokenizer.py` / `experiment_utils.py`：与第 39/40 期一致的稳定工具（schema `chapter41.v1`）；数据文件名为 `data/chapter41_train.jsonl` 与 `data/chapter41_manifest.json`。

## 环境和数据

与第 39/40 期完全相同，模型固定为 `Qwen/Qwen3-0.6B-Base@da87bfb608c14b7cf20ba1ce41287e8de496c0cd`，数据固定为 `sft_mini_512.jsonl@9b06bee29c0d5685647ac11b0dcea9f572c5064a`（SHA256 `475039fa…7852`，1,232,540,940 字节，下载后校验）。

```bash
cd chapter_41
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 python prepare_data.py --samples 512
```

模型权重在首次运行时由 `resolve_model_directory` 按固定 revision 下载。

## 运行顺序

```bash
# 1. CPU smoke（不需要 GPU 与模型下载；五轮：interleaved 2×2 / onef1b 4 卡 / onef1b 2 卡 / interleaved 4×2 / interleaved 2×2×8micro）
CUDA_VISIBLE_DEVICES="" python smoke_test.py

# 2. 真实权重正确性：W=2、vpp=2（单机双卡，mp.spawn；micro 必须能被 2 整除）
python compare_correctness.py --world-size 2 --vpp 2 --output correctness-results.local.json
#    torchrun 等价形式（--world-size 必须等于 WORLD_SIZE）：
#    torchrun --nproc-per-node 2 compare_correctness.py --world-size 2
#    W=4（单机四卡；参考与 stash 约 10 GB，显存紧张时下调 --max-length）

# 3. Benchmark 主实验：W=4（同轮内先跑 onefoone_pp4 再跑 interleaved_pp4，micro {4,8}）
python benchmark.py --world-sizes 4 --output benchmark-results.local.json
#    W=2 附带对照
python benchmark.py --world-sizes 2 --output benchmark-results.local.json
```

所有输出默认写入 `*.local.json`，并由 Git 忽略。

## 指标口径与计时边界

- `step_time_ms` 为端到端 Wall Clock（覆盖所有 (块, micro) 的 forward/backward、tied 梯度 P2P、裁剪与 AdamW），两端正确同步；每 (虚拟 Stage, micro) 的 Forward/Backward/P2P 用 CUDA Event 记录（CPU 退化为 perf_counter），事件带 `chunk`/`virtual_stage` 字段，时间线以 step 起点为 0 的秒为单位写入 JSON，Bubble 可从原始事件复算。
- Bubble 比例定义为 `1 - 计算busy/窗口`（每 Rank 自己的时间线，P2P 时间单列不折算进 busy）；两臂在同一全局工作量、同一每 Rank 层数下对比（Strong Scaling）。
- P2P 记录时间、字节数与次数（激活 `[b, s, h]`、输入梯度、tied 交换各一类；本地直传记 `local_handoff`，不计入 P2P）。interleaved 的消息次数由 `(2P-1)·M` 条激活边决定，次数与等待时间分开读。
- 调度理论 in-flight 数随每步指标输出（`onefoone` = min(P-r, M)，`interleaved` = warmup+1），与每 Rank 峰值 allocated/reserved 显存（`per_rank_memory`）对照；OOM 记录失败配置、失败阶段与当时 allocated/reserved。
- 主吞吐为有效监督 Token/s（全局 batch = micro × 2 × 物理 Rank 数，只计一次），同时报告物理 Token/s。


## 适用边界

- 主线模型为 `Qwen/Qwen3-0.6B-Base`，训练目标为全参数 SFT。
- `smoke_test.py` 不依赖真实模型下载或 GPU；真实权重正确性检查与 Benchmark 需要上文注明的 GPU 配置。
- 本目录只提供代码、依赖和运行说明。正确性与 Benchmark 命令会在本地生成 `*.local.json`，仓库不附带实验结果。
