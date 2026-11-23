# 第 42 期：混合并行与双节点 4+4

## 对应篇章

对应《超轻量AI Infra教程—训练篇》第 27 章；合并后的连续编号为第 42 章。

## 显卡要求

最低 4 张 GPU；完整参考配置为双节点 4+4 张 GPU。

第 41 期之前，DP（第 29 期）、TP（第 35/36 期）与 PP/GPipe（第 39 期）已分别验证。本期是综合章：不发明任何新的并行原语，把三者组合成三维并行，并在双节点 4+4 卡上验证 DP、TP、PP 同时启用的拓扑。受控性由分阶段对照保证——每组实验只在已验证的单维配置上增加一个维度。

## 问题、假设与边界

- 问题：三维组合能否在不写新机制的前提下得到与单卡一致的训练语义？每增加一个维度付出多少通信与 Bubble 代价？双节点上把 TP/PP 留在节点内、只让 DP 跨节点时，跨节点通信量是什么口径？
- 假设：组内通信语义不变（TP 组内 f/g AllReduce、PP 邻接 P2P、DP 组梯度 AllReduce）时，三维组合与单卡参考计算同一个全局 token_mean 目标，差异只在 FP32 归约顺序；0.6B 上组合可能是负优化，如实保留。
- 唯一变量与受控结构：本期没有单一"两臂对照"，而是三组单变量差分加上一个综合配置——A1 `a1_tp2`（TP2/PP1/DP1）→ A2 `a2_tp2pp2`（只加 PP）；B1 与 A1 同构（同配置只跑一次，记录复用）→ B2 `b2_tp2dp2`（只加 DP）；C1 `c1_pp2`（TP1/PP2/DP1）→ C2 `c2_pp2dp2`（只加 DP）；实验 D `d_tp2pp2dp2`（TP2/PP2/DP2，8 rank，只在 A2 上加 DP）。任意两维布局（DP2TP2PP1、DP1TP2PP2、DP2TP1PP2）之间同时改变了多个维度，横向性能差异不写成单变量因果结论，只报告容量、吞吐、通信量、Bubble 与拓扑路径。
- 固定项：模型/数据/标签/dtype（FP32、TF32 关闭）/eager attention/优化器公式/种子；Stage 用连续层 divmod 均分（28 层 / PP=2 → [14,14]，余数记录在 split 元数据）；每 Stage 的层在 TP 组内按第 36 期窗口切片。
- 排除项：1F1B 与 Interleaved 调度（第 40/41 期）、ZeRO/FSDP 组合、Sequence/Vocab Parallel、激活检查点、编译优化、FlashAttention、RDMA 调优。

## 逻辑 Rank 映射表（placement 表）

rank 分解公式（唯一的 placement 真相来源，代码内 `ParallelTopology`）：

```text
rank = dp_rank * (pp_size * tp_size) + pp_rank * tp_size + tp_rank
```

实验 D（TP=2, PP=2, DP=2，8 rank，双节点 4+4）的映射表：

| rank | dp | pp | tp | 节点 | 本地 GPU (LOCAL_RANK) |
|---:|---:|---:|---:|---|---:|
| 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 1 | 0 | 1 |
| 2 | 0 | 1 | 0 | 0 | 2 |
| 3 | 0 | 1 | 1 | 0 | 3 |
| 4 | 1 | 0 | 0 | 1 | 0 |
| 5 | 1 | 0 | 1 | 1 | 1 |
| 6 | 1 | 1 | 0 | 1 | 2 |
| 7 | 1 | 1 | 1 | 1 | 3 |

组表（每 Rank 只属于每族的一个组，代码逐 Rank 断言并写入结果 JSON）：

- TP 组（节点内）：{0,1} {2,3} {4,5} {6,7}
- PP 组（节点内，邻接 P2P 按 tp_rank 配对）：{0,2} {1,3} {4,6} {5,7}
- DP 组（跨节点）：{0,4} {1,5} {2,6} {3,7}

即每节点一份完整 TP2×PP2 模型副本，训练主路径只有 DP 同步跨节点。`node_index` 由主机名的匿名哈希（sha256 前 8 位）聚合得到，结果不落主机名；若实际节点布局与 dp_rank 不一致，脚本默认报错（`CHAPTER27_ALLOW_CROSS_NODE_TP_PP=1` 才放行并在结果中标记"拓扑解释失效"）。逻辑 rank 到物理 GPU 的最终映射以各节点 `nvidia-smi topo -m` 复核，不默认连续编号最快。

## 机制与易错点

1. **三维组是一次性建表**。TP/PP/DP 三族的成员表由 rank 公式导出并断言"恰好划分 world"；`dist.new_group` 是全局集合操作——每个 rank 必须按相同顺序创建族内所有组、只保留自己所在的那一个，只建"自己的组"会让其他 rank 永久等待（开发期自查发现）。PP 不建组：邻接 P2P 直接用全局 rank 寻址（第 39 期契约）。
2. **数据只按 DP 切**。全局 batch = `num_micro × samples_per_micro × DP`（流水 micro 循环即梯度累积，即 micro × accumulation × DP）。DP rank 取 `members[dp_rank::dp_size]`（第 29 期 stride 规则）后切成 `num_micro` 个 micro-batch；TP 组内每 rank 同一 micro-batch（TP 切权重不切数据），PP 按 micro 顺序流过，Stage k>0 只需要 mask/shape/scale，P2P 载荷保持纯激活。
3. **梯度所有权三类不变，DP 只做一次**。TP shard 权重梯度 rank-local（组内不通信）；残差流输入梯度由 f 算子在 backward 内已按 TP 组求和；replicated 权重（norms、Embedding、LM Head、q/k norm 钩子归约后）在 TP 组内值相同——TP 组内再 AllReduce 会把它们乘以 tp_size，是错误不是浪费。DP 维度上所有参数（两类都是数据偏部分量）只需要一次梯度 AllReduce。
4. **精确全局范数要防重复计数**。全局平方范数 = Σ_{dp,pp}( replicated_sq + Σ_tp shard_sq )：先在 TP 组归约 shard 部分，再把二者之和在全 world 归约后除以 tp_size——每个 (dp,pp) 单元恰好被计 tp_size 次。跳过 TP 归约直接 world 归约原始平方和，replicated 部分会被乘 tp_size，与 shard 部分需要的 TP 求和不抵消。
5. **tied Embedding/LM Head 的 pp=1 特例**。PP=1 时首末两端在同一 Stage，本期把 LM Head 就地绑定到 Embedding（单参数，两份梯度自动累加，即第 16 期语义）；PP≥2 时沿用第 39 期首末配对交换，按相同 tp_rank 配对（载荷 TP 内相同，t↔t 对保持一致）。漏掉 pp=1 特例会让 a1/b2 臂在第一步就偏离 tied 参考。
6. **边界载荷在 TP 组内是冗余的**。Stage 输出激活经过 Row-parallel 的 g 归约、输入梯度经过下一个 Stage 首块的 f 归约，二者在 TP 组内已完整且相同；每条边界消息按 tp_rank 配对飞行 tp_size 次（字节冗余 = tp_size 倍），这是本布局的真实代价，计入 p2p_bytes 指标而不隐藏。
7. **对照的可比性**。每臂与单卡 FP32 参考对照：loss 轨迹（DP 侧按监督 Token 全局加权、TP 侧复制一致、PP 侧 Stage 切分）、一次更新后每 DP 副本的全模型参数校验和（TP 分片按 rank 序聚合；tie 计两次与参考 state_dict 对齐）、组成员断言。

## 文件

- `hybrid_parallel.py`：本期核心。`ParallelTopology`（rank 公式、分解、三族组表、describe JSON）、`build_parallel_groups`（建组 + 每 Rank 成员断言）、`HybridParallelStage`（第 39 期 Stage + 第 36 期层工厂，pp=1 tied 绑定特例）、`load_hybrid_stage`（层窗口 × TP 切片双窗口 safetensors 读取）、`gpipe_step`（第 39 期 fill-drain，边界按 tp_rank 配对）、`exchange_tied_gradients`、`allreduce_gradients_dp`、`clip_grad_norm_hybrid`（TP 标量 + world 标量/(tp_size×dp_size)，pp≥2 时 tied 对按逻辑参数计一次）、`hybrid_train_step`（分段计时 + 时间线 + Bubble）、`state_ledger`（状态所有权台账）、`ARMS`/`arm_topology`、`StageTimeline`/`compute_bubble`/`split_layers`（第 39 期复制）。
- `hybrid_runtime.py`：进程组/数据流水通用件。`node_mapping_record`（匿名节点哈希、node_index==dp_rank 断言）、`shard_micro_batches`（DP stride 分片 + micro 切分 + 监督 Token 账本）、`micro_dicts`、`global_batch_members`。
- `mlp_tp.py` / `attention_tp.py` / `qwen3_model.py`：第 36 期复制（TP 组件与层工厂逐字节兼容；group 参数换成本 Rank 的 TP 组）。
- `smoke_test.py`：CPU gloo 三轮 + 单元检查——A 轮 8 进程模拟 2×4 节点拓扑（TP2/PP2/DP2：组成员断言、DP 数据分片、对照参考 loss/范数/参数校验和、通信计数公式）、B 轮 TP2/PP1/DP1（tied 绑定、零 P2P/DP）、C 轮 TP1/PP2/DP2（零 TP 计数）；placement 表/tied Stage/数据划分单元检查。
- `compare_correctness.py`：真实 Qwen3-0.6B 权重，任一臂 vs 单卡 FP32 参考（loss 轨迹、梯度范数、参数校验和、tied 一致、组成员、节点映射断言）。
- `benchmark.py`：每臂一次调用，warm-up + 测量步：Step Time、TP/PP/DP 分段通信、Bubble、每 Rank 峰值显存、状态所有权台账、最后一步完整时间线 JSON、全局有效监督 Token/s。
- `prepare_data.py` / `sft_core.py` / `assistant_loss.py` / `mini_batch.py` / `qwen3_tokenizer.py` / `experiment_utils.py`：与第 39-41 期一致的稳定工具（schema `chapter42.v1`）；数据文件为 `data/chapter42_train.jsonl` 与 `data/chapter42_manifest.json`。

## 环境和数据

与第 39-41 期完全相同，模型固定为 `Qwen/Qwen3-0.6B-Base@da87bfb608c14b7cf20ba1ce41287e8de496c0cd`，数据固定为 `sft_mini_512.jsonl@9b06bee29c0d5685647ac11b0dcea9f572c5064a`（SHA256 `475039fa…7852`，下载后校验）。

```bash
cd chapter_42
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 python prepare_data.py --samples 512
```

## 运行顺序

```bash
# 1. CPU smoke（不需要 GPU 与模型下载；8 进程模拟 2×4 拓扑 + 两个小拓扑轮）
CUDA_VISIBLE_DEVICES="" python smoke_test.py

# 2. 单机正确性（每臂一次；--world-size 必须等于臂的世界大小）
python compare_correctness.py --arm a1_tp2     --output correctness-a1.local.json          # 2 卡
python compare_correctness.py --arm a2_tp2pp2  --output correctness-a2.local.json          # 4 卡
python compare_correctness.py --arm b2_tp2dp2  --output correctness-b2.local.json          # 4 卡
python compare_correctness.py --arm c1_pp2     --output correctness-c1.local.json          # 2 卡
python compare_correctness.py --arm c2_pp2dp2  --output correctness-c2.local.json          # 4 卡
#    torchrun 等价形式：torchrun --nproc-per-node 4 compare_correctness.py --arm a2_tp2pp2

# 3. 单机 Benchmark（同一全局 batch 时按 DP 比例放大 --samples-per-micro）
torchrun --nproc-per-node 2 benchmark.py --arm a1_tp2 --samples-per-micro 2 --output benchmark-a1.local.json
torchrun --nproc-per-node 4 benchmark.py --arm a2_tp2pp2 --samples-per-micro 2 --output benchmark-a2.local.json
torchrun --nproc-per-node 4 benchmark.py --arm b2_tp2dp2 --samples-per-micro 1 --output benchmark-b2.local.json
torchrun --nproc-per-node 2 benchmark.py --arm c1_pp2 --samples-per-micro 2 --output benchmark-c1.local.json
torchrun --nproc-per-node 4 benchmark.py --arm c2_pp2dp2 --samples-per-micro 1 --output benchmark-c2.local.json

# 4. 双节点 4+4 实验 D（在两台服务器上执行；<master> 为节点 0 内网地址，见 .env）
# 节点 0：
torchrun --nnodes 2 --node-rank 0 --nproc-per-node 4 \
  --master-addr <master> --master-port 29500 \
  benchmark.py --arm d_tp2pp2dp2 --output benchmark-d.local.json
# 节点 1：
torchrun --nnodes 2 --node-rank 1 --nproc-per-node 4 \
  --master-addr <master> --master-port 29500 \
  benchmark.py --arm d_tp2pp2dp2 --output benchmark-d.local.json
# 实验 D 的正确性同理（compare_correctness.py --arm d_tp2pp2dp2）。
```

所有输出默认写入 `*.local.json`，并由 Git 忽略。

## 指标口径与计时边界

- `step_time_ms` 为端到端 Wall Clock（覆盖 fill-drain 流水、tied 交换、DP AllReduce、裁剪与 AdamW），两端正确同步；每 micro 的 Forward/Backward/P2P 用 CUDA Event 记录（CPU 退化为 perf_counter），时间线以 step 起点为 0 的秒写入 JSON，Bubble 可从原始事件复算。
- 分段通信三个口径分开读：TP AllReduce（前向/反向分列，COMM_RECORDER 事件对；含裁剪 TP 标量单独字段）；PP P2P（时间线事件，次数/字节/秒，含 tied 交换）；DP AllReduce（每参数梯度一次 + 裁剪 world 标量，字节为梯度字节数）。
- DP 跨节点通信：本实现的 DP AllReduce 是"每参数一次"的朴素归约，NCCL 会把它合并成环/树跨越节点边界，无法从 API 层面按"节点内段/跨节点段"直接分段；跨节点口径以 world size、节点映射表、`nvidia-smi topo -m` 与 `NCCL_DEBUG=INFO` 记录的 Transport（P2P/SHM/Socket）解释，不另行给出分段数字。
- 主吞吐为全局有效监督 Token/s（DP 求和、TP 去重：world loss 分片和 ÷ tp_size），同时报告全局物理 Token/s；两臂比较时声明 Strong Scaling（同全局 batch，按 DP 比例放大 `--samples-per-micro`）或固定每卡工作量的口径。
- 每 Rank 峰值 allocated/reserved 显存与状态所有权台账（replicated/shard 字节、通信路径清单）同轮输出；OOM 记录失败配置、失败阶段与当时 allocated/reserved。


## 适用边界

- 主线模型为 `Qwen/Qwen3-0.6B-Base`，训练目标为全参数 SFT。
- `smoke_test.py` 不依赖真实模型下载或 GPU；真实权重正确性检查与 Benchmark 需要上文注明的 GPU 配置。
- 本目录只提供代码、依赖和运行说明。正确性与 Benchmark 命令会在本地生成 `*.local.json`，仓库不附带实验结果。
