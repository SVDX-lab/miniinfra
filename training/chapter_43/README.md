# 第 43 期：分布式 Checkpoint 与恢复

## 对应篇章

对应《超轻量AI Infra教程—训练篇》第 28 章；合并后的连续编号为第 43 章。

## 显卡要求

最低 2 张 GPU；参考配置覆盖 2、4 张 GPU，以及双节点 4+4 张 GPU。

第 42 期把 DP、TP、PP 组合成三维并行后，训练状态仍只存在于各 Rank 内存：崩溃或停机即从头再来。本期只改变一件事——完整训练状态是集中保存为一个文件（rank 0 聚合写出），还是各 Rank 并行保存 Shard（分片 + manifest 清单）。训练配置固定为第 29 期 DDP（4 卡，全参数 FP32，带 Warm-up 调度器），`torch.distributed.checkpoint`（DCP）作为官方对照路径参与吞吐与恢复语义对照。

## 问题、假设与边界

- 问题：两种模式的保存/恢复吞吐与文件体积差多少？中断后能否逐位接上不中断的参考轨迹？4 卡保存的 Checkpoint 能否被 2 卡加载（重分片）？一个 Rank 被 Kill 后，训练能否从最近 Checkpoint 无缝续跑？
- 假设：DDP 各 Rank 在梯度 AllReduce 后持有逐位相同的参数与优化器状态，所以 sharded 的"分片"是可互换的完整副本——两种模式恢复出的训练状态必须完全一致，差别只在 I/O（串行单文件 vs 并行多文件）；DDP 下的重分片退化为"副本选择 + 校验"，真正的字节级重分片属于第 34 期 ZeRO/FSDP 的状态形态。
- 唯一变量：`centralized`（rank 0 聚合写单文件，其它 Rank 在 barrier 等待）对照 `sharded`（每 Rank 写自足分片 + rank 0 写 manifest 清单）。`dcp` 模式为官方对照，不参与唯一变量论证。
- 固定项：模型/数据/标签、DDP 语义、优化器与调度器公式、确定性数据计划（第 22 期 shuffle）、种子；保存清单两种模式完全一致（见下）。
- 排除项：异步/影子 Checkpoint、增量保存、远端对象存储、ZeRO/FSDP 分片状态的真重分片、TP/PP 切分状态的实切保存（拓扑元数据入清单即可）。

## 保存清单（两种模式一致）

- 模型参数（完整 FP32 state dict）；Optimizer 状态（AdamW 一二阶矩）；LR Scheduler 状态；
- 全局 Step、已消费有效监督 Token、已消费样本数；
- Sampler 状态：计划种子、全局 batch、批次总数、游标；
- RNG：Python `random`、NumPy、torch CPU、torch CUDA——**每 Rank 一份**；
- DP/TP/PP 拓扑元数据（本期 dp=W、tp=pp=1）与分片元数据（文件名、Rank、校验和、字节）；
- 软件版本（Python/torch/CUDA/NCCL/NumPy/Git revision）与训练配置。

schema 统一为 `chapter43.checkpoint.v1`；全部写盘走 `*.tmp` + `os.replace` 原子替换——被 Kill 的进程不可能留下会被恢复信任的半写文件。

## 机制与易错点

1. **副本一致性是契约不是假设**。保存前聚合各 Rank 的模型校验和，必须逐位相等，否则直接报错——AllReduce 契约一旦破坏，Checkpoint 会把分叉静默固化。恢复时再次用清单校验和核对加载结果。
2. **LR Scheduler 的恢复陷阱**。`LRScheduler.load_state_dict` 只更新调度器簿记，**不**回写 optimizer 的 param_groups：新建调度器在构造时已把 optimizer 设为初始 Warm-up LR，直接续训的第一步会用错学习率。恢复后必须用 state dict 里的 `_last_lr` 回填（`sync_scheduler_lr`）。第 16 期的 checkpoint 测试恰好被平坦 Warm-up 掩盖了这个错误——两处 LR 数值相等使断言失效。
3. **RNG 是每 Rank 状态**。Sharded 模式每个分片带自己的 RNG；centralized 模式聚合全部 Rank 的 RNG 成列表。同 World Size 恢复各取自己的槽位；重分片时新 Rank `r` 采用旧 Rank `r % W_old` 的状态——确定性且对本章无损（训练循环不消耗随机数：数据计划确定、无 dropout），任何引入随机性的模块都需要从种子重新派生每 Rank 流，这一边界写进恢复记录而不是隐藏。
4. **重分片 = 副本选择 + 校验**。4 卡保存 → 2 卡加载：rank 0 先核对 manifest 里各分片校验和一致，新 Rank 读 `shard-{r % 4}`。之所以合法，正因为 DDP 分片是可互换完整副本；跨 World Size 后每 Rank 数据分片改变，接续段 loss 只有全局值可比（FP32 归约顺序变化，容差覆盖），RNG 指纹不做跨布局对比。
5. **DCP 是官方对照不是替代理解**。`dcp.save` 对重复 Tensor 自动去重、`dcp.load` 在新 World Size 下重新分配读取——即本章手写模式在 DDP 上退化出的语义的完整版。Loop 状态（进度/Sampler/调度器/每 Rank RNG）DCP 不负责，走 `worker-meta.pt` sidecar。依赖缺失时显式报错。
6. **故障注入的真实性**。被 Kill 的 Rank 在 Checkpoint 落盘（`os.replace` 完成）之后 `os._exit(70)`，不做任何分布式清理；torchrun 判定失败并收回其余 Rank。恢复脚本断言"这次启动确实失败了"——Kill 没发生的注入是假注入。
7. **恢复即验证**。每次恢复都核对：schema 版本、模型校验和、进度字段。连续性脚本对照五样东西：接续段逐步全局 loss、LR 序列、下一 batch 预览、RNG 指纹、最终参数校验和——同构恢复要求逐位相等（确定性路径），不是"容差内一致"。

## 文件

- `checkpointing.py`：本期核心。`capture_rng`/`restore_rng`/`rng_fingerprint`（RNG 状态机）、`collect_local_state`（全清单）、`save_centralized`/`load_centralized`（rank 0 单文件）、`save_sharded`/`load_sharded`/`read_manifest`（并行分片 + 清单，重分片=副本选择）、`detect_mode`/`load_for_resume`、`sync_scheduler_lr`（LR 回填）、`dcp_available`/`dcp_save_checkpoint`/`dcp_load_checkpoint`（官方对照）、`export_inference_weights`（单卡推理权重导出）、模型/优化器 float64 校验和。
- `train_worker.py`：torchrun 入口。第 29 期 DDP 语义 + LambdaLR 的训练循环，`--save-dir/--save-every/--save-modes centralized,sharded,dcp`、`--resume-dir/--resume-mode`、`--fault-at-step/--fault-rank`（Checkpoint 落盘后硬退出）；每步记录全局 loss、LR、RNG 指纹、下一 batch 预览。
- `compare_checkpoint.py`：连续性验证。Phase 1 参考轨迹 + 各模式保存；Phase 2 全新进程集恢复续训；逐位对照（loss/LR/预览/RNG/校验和）；`--resume-from --reference` 支持跨 World Size 重分片恢复（4 卡保存 → 2 卡加载）。
- `benchmark_checkpoint.py`：4 卡保存/恢复吞吐（GB/s、字节、barrier 界定墙钟）、保存时间/训练时间占比、每 Rank 峰值显存；`--resume-from` 在任意 World Size 下量重分片加载。
- `fault_inject.py`：端到端故障注入。参考运行 → 带 Kill 运行（断言失败）→ 恢复运行 → 接续段逐位对照；默认 CPU tiny，`--real-weights` 上 GPU。
- `export_weights.py`：从新训练或 `--resume-from` 的状态导出单卡可推理 safetensors 权重（plain Qwen3 布局）。
- `smoke_test.py`：CPU gloo 全流程——RNG 状态机、safetensors 导出往返、2 卡三种模式恢复逐位连续、4 卡保存 → 2 卡加载重分片。
- `ddp_core.py`：第 29 期复制（DDP 语义与工具）；`qwen3_model.py` / `qwen3_tokenizer.py` / `assistant_loss.py` / `mini_batch.py` / `sft_core.py` / `experiment_utils.py`：稳定工具（schema `chapter43.v1`）；数据文件为 `data/chapter43_train.jsonl` 与 `data/chapter43_manifest.json`。

## 环境和数据

模型固定为 `Qwen/Qwen3-0.6B-Base@da87bfb608c14b7cf20ba1ce41287e8de496c0cd`，数据固定为 `sft_mini_512.jsonl@9b06bee29c0d5685647ac11b0dcea9f572c5064a`（SHA256 `475039fa…7852`，下载后校验）。tiny 冒烟路径不依赖模型与数据下载。

```bash
cd chapter_43
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 python prepare_data.py --samples 512
```

## 运行顺序

```bash
# 1. CPU smoke（不需要 GPU 与下载：RNG 状态机、三种模式恢复、4→2 重分片、导出往返）
CUDA_VISIBLE_DEVICES="" python smoke_test.py

# 2. 真实权重连续性：4 卡，6 步、第 3 步保存，三种模式各恢复一次（单机）
torchrun --nproc-per-node 4 compare_checkpoint.py --output checkpoint-continuity.local.json

# 3. 重分片恢复：4 卡保存的 sharded Checkpoint 由 2 卡加载（先保留第 2 步输出目录）
torchrun --nproc-per-node 2 compare_checkpoint.py \
  --resume-from checkpoint-continuity-run/checkpoints/step-00003/sharded \
  --reference checkpoint-continuity-run/reference-records.json \
  --output reshard-results.local.json

# 4. 故障注入端到端（默认 CPU tiny；GPU 复跑加 --real-weights）
python fault_inject.py --output fault-results.local.json

# 5. 保存/恢复吞吐 benchmark（4 卡；2 卡重分片加载用 --resume-from）
torchrun --nproc-per-node 4 benchmark_checkpoint.py --output checkpoint-benchmark.local.json
torchrun --nproc-per-node 2 benchmark_checkpoint.py --tiny --resume-from <dir> \
  --output checkpoint-reshard-benchmark.local.json

# 6. 导出单卡推理权重
torchrun --nproc-per-node 4 export_weights.py --output exported/qwen3-exported.safetensors

# （可选）双节点 4+4 同拓扑恢复：两台机器各 --nproc-per-node 4 --nnodes 2 --node-rank R，
# Checkpoint 目录必须放在共享文件系统上；只作案例分析（见已知限制）。
```

所有输出默认写入 `*.local.json`，并由 Git 忽略。

## 指标口径与计时边界

- 保存/恢复时间以 barrier 界定墙钟窗口：centralized 的写盘只在 rank 0，其它 Rank 在 barrier 内等待——报告的是"训练整体停摆时间"，不是 rank 0 单点写时间；sharded 的写盘重叠，窗口覆盖最慢 Rank。
- 字节数取模式目录落盘后的实际文件总和；GB/s = 字节 / 墙钟。DDP 分片不省字节：sharded 的总体积约等于 centralized，收益只在不并行就得不到的 I/O 时间。
- 保存时间 / 训练时间占比 = 保存墙钟 / 同配置单步训练均值（3 步均值），两者同一脚本同一进程内测得。
- 连续性口径：同构恢复的 loss 绝对误差要求为 0（确定性路径逐位一致），重分片恢复用相对容差（只差 FP32 归约顺序）；LR、下一 batch 预览、进度字段要求精确相等；RNG 指纹只在同 World Size 下对比。
- Checkpoint 体积的理论构成（0.6B FP32）：参数 2.4GB + AdamW 状态 4.8GB ≈ 7.2GB/副本；实测值以目录字节数为准。


## 适用边界

- 主线模型为 `Qwen/Qwen3-0.6B-Base`，训练目标为全参数 SFT。
- `smoke_test.py` 不依赖真实模型下载或 GPU；真实权重正确性检查与 Benchmark 需要上文注明的 GPU 配置。
- 本目录只提供代码、依赖和运行说明。正确性与 Benchmark 命令会在本地生成 `*.local.json`，仓库不附带实验结果。
