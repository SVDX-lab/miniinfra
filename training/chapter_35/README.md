# 第 35 期：MLP Tensor Parallel

## 对应篇章

对应《超轻量AI Infra教程—训练篇》第 20 章；合并后的连续编号为第 35 章。

## 显卡要求

最低 2 张 GPU；参考配置覆盖 2 张和 4 张 GPU。

第 34 期已从数据并行维度切分模型状态（ZeRO-3/FSDP），但单层内部的矩阵计算与权重仍然每卡完整。本期只改变一件事：SwiGLU MLP 权重是完整复制，还是沿中间维度切分。Attention、Embedding、LM Head、LayerNorm、残差连接全部保持每 Rank 完整复制，从而单独测清 MLP TP 这一个机制。Attention Head 切分是第 36 期，词表切分是第 37 期，均不在本期。

## 问题、假设与边界

- 问题：MLP 权重沿中间维度切分后，训练语义是否与完整复制数学等价？每卡省多少参数与状态显存？每层新增的 2 次 TP AllReduce（forward 的 down_proj 输出求和、backward 的 MLP 输入梯度求和）暴露在关键路径上，吞掉多少收益？0.6B 在 12GB 上的容量边界如何移动？
- 假设：Column Parallel（Gate/Up 沿 dim0 切）+ Row Parallel（Down 沿 dim1 切）在数学上等价于完整 SwiGLU MLP，FP32 下只有归约顺序带来的极小数值分叉。0.6B 单卡本可完整常驻，因此 TP 更可能是负优化——这是课程预期内的合法结果，失败与负优化如实保留。
- 唯一变量：MLP 权重完整复制（tp_size=1，即 `replicated_w1` 臂）对照沿中间维度切分（`tp2_w2`/`tp4_w4`）。TP 度即 world size；TP 组内数据复制，每个 Rank 处理相同的全局 batch（每步 2 × tp_size 个样本）。
- 固定项：模型/数据/标签/dtype（FP32、TF32 关闭）/eager attention/优化器公式/种子与既有章节一致；loss 为全局 token_mean（TP 组内 loss_scale 恒为 1.0）。
- 排除项：Attention TP（第 36 期）、Vocab Parallel 与分布式 Cross Entropy（第 37 期）、Sequence Parallel（第 38 期）、FlashAttention / 激活重算 / 编译优化（第 25/26/28 期机制，本期不混入）、DP×TP 混合与 Process Group 细分（第 42 期）。

## 梯度所有权分析（本期易错点）

设 TP 组内有 N 个 Rank，每个 Rank 持有相同的输入激活 x 和自己的权重分片。每个 Rank 局部计算：

```text
G_r = x @ W_gate_r^T ,  U_r = x @ W_up_r^T ,  H_r = silu(G_r) * U_r
P_r = H_r @ W_down_r^T ,  y = AllReduce_sum(P_r)      （forward 通信）
```

参数梯度分三类，处理方式完全不同：

1. **MLP 分片权重（gate/up/down）：梯度只在本地，不需要任何跨 Rank 通信。** `dW_down_r = dY^T @ H_r`、`dW_gate_r`、`dW_up_r` 只依赖本 Rank 的分片前向和复制的 x；把各 Rank 的分片梯度按顺序拼接就是完整梯度。不需要、也不应该对它们做 AllReduce。
2. **MLP 输入激活的残差流梯度：必须做一次 backward AllReduce（每层一次）。** Rank r 的 autograd 只算得出自己的贡献 `dX_r`，而真实梯度是 `sum_r dX_r`。down_proj 的 forward AllReduce 并不能自动补上这一步——它的 backward 是恒等变换，若不做输入梯度的 AllReduce，一个"部分梯度"会悄悄流进下面的 Attention 路径，训练语义被静默改坏。实现上这是 Megatron 意义上的 f 算子（forward 恒等、backward AllReduce），在 `ParallelMLP` 入口处发起一次，而不是 gate/up 两个分支各发起一次。
3. **复制参数（Attention、LayerNorm、Embedding/LM Head）：不需要梯度 AllReduce。** 纯 TP 的每个 Rank 处理同一份数据，forward AllReduce 之后（backward AllReduce 之前）所有激活在各 Rank 上逐位相同，因此这些参数的梯度在每个 Rank 上数学相同。再做一次 AllReduce SUM 会把它们放大 tp_size 倍——这是错误，不是浪费。只有当 Rank 之间持有不同数据（引入 DP 维度，第 42 期）时才需要梯度同步，且那发生在 DP 组之间，TP 组内永远不需要。

`compare_correctness.py` 对第 3 类做了实证检查：各 Rank 的复制参数梯度在完全没有参数梯度通信的情况下必须一致，且等于单卡参考的完整梯度（若被错误地 AllReduce SUM，会恰好放大 tp_size 倍，检查会失败）。优化器侧唯一的跨 Rank 通信是 `clip_grad_norm_tp` 的 1 次标量 AllReduce：全局范数平方 = 复制部分平方和（各 Rank 相同，只计一次）+ 分片部分平方和（跨 Rank 求和），这保证裁剪行为与单卡 `clip_grad_norm_` 一致。

## 文件

- `mlp_tp.py`：本期核心。`ColumnParallelLinear`（权重沿 output dim 切 N 份，输入复制、输出分片；含独立使用时的输入梯度 AllReduce）、`RowParallelLinear`（权重沿 input dim 切 N 份，输入分片、输出 AllReduce 求和）、`ParallelMLP`（组合三者，f/g 两个通信算子带注释与计时）、`split_tensor_parallel_weights` / `assemble_tensor_parallel_weights`（完整 state dict 与每 Rank 分片互转，含整除校验）、`tp_train_step`（分段计时）、`clip_grad_norm_tp`（精确全局范数）、`gather_tensor_parallel_parameters`（分片重组，仅用于正确性对照）、`tp_memory_ledger`（每卡所有权账本）、`TPCommRecorder`（TP AllReduce 事件计时）。
- `qwen3_model.py`：第 33 期副本，新增 `mlp_factory` 装配点与 `load_handwritten_model_tp`——非 MLP 权重每个 Rank 完整读取，MLP 权重用 safetensors 的 `PySafeSlice` 只读取本 Rank 的分片窗口，不在内存中物化完整 MLP 权重再切分。
- `tp_runtime.py`：数据流水与进程组通用件（确定性编码、第 22 期洗牌、env:// 启动、`build_replicated_batch` 的 TP 数据复制语义）。
- `smoke_test.py`：CPU gloo，tiny 模型（hidden 32 / intermediate 64 / vocab 64 / 2 层 / tp=2）：forward、input gradient、weight gradient 分片、一次 AdamW step 重组对照、梯度所有权、整除错误提示。
- `compare_correctness.py`：真实 Qwen3-0.6B 权重，TP=2/4 与单卡 FP32 参考对照（loss < 1e-5 红线 + 梯度/中间输出/一步更新/分片重组）。
- `benchmark.py`：`replicated_w1` / `tp2_w2` / `tp4_w4`（另含 `tp8_w8` 跨节点边界档），3 次重复，分段计时与 TP 通信暴露占比。
- `capacity_probe.py`：tp {1,2,4} × Micro-batch {1,2,4} × 序列 {128,256,512} 容量网格。
- `prepare_data.py` / `sft_core.py` / `assistant_loss.py` / `mini_batch.py` / `qwen3_tokenizer.py` / `experiment_utils.py`：与第 33 期一致的稳定工具；数据文件名为 `data/chapter35_train.jsonl` 与 `data/chapter35_manifest.json`；`experiment_utils` 的 schema 为 `chapter35.v1` 并增加 `tp_degree` 字段。

## 环境和数据

与第 29/33 期完全相同，模型固定为 `Qwen/Qwen3-0.6B-Base@da87bfb608c14b7cf20ba1ce41287e8de496c0cd`，数据固定为 `sft_mini_512.jsonl@9b06bee29c0d5685647ac11b0dcea9f572c5064a`（SHA256 `475039fa…7852`，1,232,540,940 字节，下载后校验）。

```bash
cd chapter_35
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 python prepare_data.py --samples 512
```

模型权重在首次运行时由 `resolve_model_directory` 按固定 revision 下载；`prepare_data.py` 输出 512 条纯 system/user/assistant 样本并写 manifest。

## 运行顺序

```bash
# 1. CPU smoke（不需要 GPU 与模型下载）
CUDA_VISIBLE_DEVICES="" python smoke_test.py

# 2. 真实权重正确性：TP=2（单机双卡，mp.spawn）
python compare_correctness.py --world-size 2 --output correctness-results.local.json
#    TP=4（单机四卡）
python compare_correctness.py --world-size 4 --output correctness-results.local.json
#    torchrun 等价形式（--world-size 必须等于 WORLD_SIZE）：
#    torchrun --nproc-per-node 2 compare_correctness.py --world-size 2

# 3. Benchmark：TP=1/2（单机两卡内完成基线与实验组）
python benchmark.py --world-sizes 1,2 --output benchmark-results.local.json
#    TP=4（单机四卡）
python benchmark.py --world-sizes 4 --output benchmark-results.local.json
#    W=8（双节点 4+4，跨节点 TP 仅作边界对照，先在两个节点分别同步同一 revision 与数据）
#    节点 0：torchrun --nnodes 2 --nproc-per-node 4 --node-rank 0 benchmark.py --world-sizes 8 --output benchmark-results.local.json
#    节点 1：torchrun --nnodes 2 --nproc-per-node 4 --node-rank 1 benchmark.py --world-sizes 8 --output benchmark-results.local.json

# 4. 容量探针：tp {1,2,4} × batch {1,2,4} × length {128,256,512}（单机四卡一轮完成）
python capacity_probe.py --output capacity-results.local.json
```

所有输出默认写入 `*.local.json`，并由 Git 忽略。

## 指标口径与计时边界

- `step_time_ms` 为端到端 Wall Clock，两端正确同步；forward/backward 分段用 CUDA Event，且各自包含层内 TP AllReduce，再单独减去通信时间给出纯计算段。
- TP AllReduce 时间由每次 Collective 的 Event 对累计（`TPCommRecorder`），每步只在结尾同步一次；它们全部暴露在关键路径上（forward AllReduce 阻塞后续 forward，backward AllReduce 阻塞后续 backward），暴露占比 = TP 通信总时间 / step_time。
- 主吞吐为有效监督 Token/s；TP 组内数据复制，全局 batch = 每 Rank batch = 2 × tp_size，Token 只计一次，不按 Rank 数重复计。同时报告物理 Token/s 与非 Padding Token/s。
- 显存同时报告每卡峰值 allocated/reserved 与理论所有权账本（复制参数/分片参数/优化器状态各自字节数）。优化器状态在首个 step 才物化，账本在 warm-up 后采集。
- OOM 记录失败配置、失败阶段与当时 allocated/reserved，不做静默降级。


## 适用边界

- 主线模型为 `Qwen/Qwen3-0.6B-Base`，训练目标为全参数 SFT。
- `smoke_test.py` 不依赖真实模型下载或 GPU；真实权重正确性检查与 Benchmark 需要上文注明的 GPU 配置。
- 本目录只提供代码、依赖和运行说明。正确性与 Benchmark 命令会在本地生成 `*.local.json`，仓库不附带实验结果。
