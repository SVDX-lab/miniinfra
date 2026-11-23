# 第 36 期：Attention Tensor Parallel

## 对应篇章

对应《超轻量AI Infra教程—训练篇》第 21 章；合并后的连续编号为第 36 章。

## 显卡要求

最低 2 张 GPU；参考配置覆盖 2 张和 4 张 GPU。

第 35 期已经把 SwiGLU MLP 沿中间维度切分，但每层的 Attention 投影与 Head 仍然完整复制。本期只改变一件事：在已有 MLP TP 的基础上，Attention Head 是否也按 TP Rank 切分。词表切分是第 37 期，Sequence Parallel 是第 38 期，均不在本期。

Qwen3-0.6B 使用 GQA：32 个 Q Head、8 个 KV Head（4:1 分组）、head_dim 128。切分必须满足整除约束，每 Rank 拿到整数个完整 Head：

| TP 度 | 每 Rank Q Head | 每 Rank KV Head | 局部 GQA 组数 | 说明 |
| --- | --- | --- | --- | --- |
| 1 | 32 | 8 | 4 | 完整复制（MLP TP 臂） |
| 2 | 16 | 4 | 4 | q/k/v 各切一半行，o 切一半列 |
| 4 | 8 | 2 | 4 | 每 Rank 8 个 Q Head 共享 2 个 KV Head |

`head_dim` 永不切分；Qwen3 的 Q/K per-head RMSNorm 作用在 head_dim 内，随 Head 留在本 Rank。

## 问题、假设与边界

- 问题：Attention Head 切分后，训练语义是否与完整复制数学等价？RoPE 与 GQA 的 `repeat_kv` 在局部 Head 集合内重放是否精确？相比仅切 MLP，每层多出的通信（o_proj 输出 AllReduce、attention 输入梯度 AllReduce、q/k norm 梯度同步）吞掉多少收益？每卡参数/激活账本如何变化？
- 假设：Q/KV Head 沿 head 维连续切块时，每块恰好是整数个完整 Head；RoPE 是 Head 内的固定旋转，局部重放与全局逐位等价；当 Q 数与 KV 数同时被 tp_size 整除时，局部 `repeat_kv` 的组数等于全局组数（4），局部 GQA 与全局 GQA 在本 Rank 的 Head 子集上完全一致。FP32 下只有归约顺序带来的极小数值分叉。0.6B 的 Attention 占比小，预期收益有限甚至为负——负优化如实保留。
- 唯一变量：在 MLP TP 基础上，Attention Head 完整复制（`mlp_tp_wN` 臂）对照按 TP Rank 切分（`attn_tp_wN` 臂）。两个臂在同一 world size 下只差 Attention 分片这一个机制。
- 固定项：MLP 分片方式（第 35 期）、模型/数据/标签/dtype（FP32、TF32 关闭）/eager attention/优化器公式/种子；loss 为全局 token_mean（TP 组内 loss_scale 恒为 1.0）；每步全局 batch = 2 × tp_size。
- 排除项：Vocab Parallel 与分布式 Cross Entropy（第 37 期）、Sequence Parallel（第 38 期）、FlashAttention / 激活重算 / 编译优化（第 25/26/28 期机制，本期不混入）、DP×TP 混合与 Process Group 细分（第 42 期）。

## 梯度所有权分析（本期易错点）

每层现在有两个并行块，各自带一对 Megatron 意义上的 f/g 算子（沿用第 35 期的实现）：

1. **分片权重（q/k/v/o 与 gate/up/down）：梯度只在本地，不需要任何跨 Rank 通信。** Rank r 的 q/k/v 分片梯度等于完整梯度中本 Rank Head 行块/列块的切片——这正是 GQA Head 映射的数值证明：拿错 Head 的 Rank 会与错误的切片比较，立即在 O(1) 量级失败。
2. **两个并行块的输入残差流梯度：各做一次 backward AllReduce（每层两次）。** Attention 的 f 算子在块入口发起一次，覆盖 q/k/v 三个分支；漏掉任何一次都会让"部分梯度"沿残差流静默传播。
3. **复制参数（外部 LayerNorm、Embedding/LM Head）：不需要梯度 AllReduce。** g 与下一个 f 之间的激活在各 Rank 逐位相同，这些参数的梯度已经一致；再 AllReduce SUM 会放大 tp_size 倍，是错误不是浪费。
4. **本期特有的陷阱：q_norm/k_norm。** 它们的权重值在每个 Rank 完整复制，但每个 Rank 只累积自己 Head 的梯度项——原始梯度是部分和，各 Rank 不同。若不做处理，各 Rank 会对同一份参数施加不同更新，副本从第一次 optimizer step 起静默分叉。`ParallelAttention` 用 post-accumulate-grad hook 对每个 norm 权重做一次 backward AllReduce SUM，恢复与参考一致的完整梯度；这次通信计入 backward 计时。

优化器侧与第 35 期相同：`clip_grad_norm_tp` 用 1 次标量 AllReduce 重建精确全局范数（复制部分计一次、分片部分跨 Rank 求和）。

## 文件

- `attention_tp.py`：本期核心。`gqa_head_layout`（每 Rank 的 Q/KV Head 索引清单与全部整除检查）、`validate_gqa_layout`（重推 `repeat_interleave` 映射）、`ParallelAttention`（局部 QKV 投影、局部 RoPE、局部 GQA、Row Parallel o_proj；f 算子覆盖三分支；q/k norm 梯度同步 hook）、`split/assemble_transformer_parallel_weights`（完整 state dict 与每 Rank 分片互转，含 Attention 与 MLP 全部整除校验）、`full_attention_reference`（并排摆放的复制版 GQA 定义）。
- `mlp_tp.py`：第 35 期实现的原样副本（ColumnParallelLinear / RowParallelLinear、f/g 算子、计时、精确范数裁剪、账本）。本期所有通信算子直接复用这里。
- `qwen3_model.py`：第 33/35 期副本，新增 `attn_factory` 装配点与 `load_handwritten_model_atp`——Attention 与 MLP 权重用 safetensors 的 `PySafeSlice` 只读取本 Rank 分片窗口，不在内存物化完整权重再切分；norm、Embedding、LM Head 每 Rank 完整读取。
- `tp_runtime.py`：数据流水与进程组通用件；`load_tokenizer_and_model_tp(attention_parallel=...)` 在两个臂之间选择权重布局。
- `smoke_test.py`：CPU gloo，tiny 模型（hidden 32 / intermediate 64 / vocab 64 / 2 层 / 8 Q Head / 4 KV Head / head_dim 8），tp=2 与 tp=4 两轮：forward、input gradient、q/k/v/o 分片梯度逐切片对照、GQA 映射与全局并集、一次 AdamW step 重组对照、梯度所有权（含 q/k norm 同步后复制梯度一致）、整除错误提示。
- `compare_correctness.py`：真实 Qwen3-0.6B 权重，TP=2/4 与单卡 FP32 参考对照：forward loss、Attention 输出（g 算子后逐 Rank 一致）、input gradient、全部层的 Attention/MLP 分片梯度对照、layer-0 分片拼接、一次更新后重组参数、每 Rank 的 Q/KV Head 索引清单与映射校验写入结果 JSON。
- `benchmark.py`：每 world size 两臂 `mlp_tp_wN` / `attn_tp_wN`（N=2/4，另含 N=8 跨节点边界档），3 次重复，分段计时与 TP 通信暴露占比，两臂在同一 TP 度下对比。
- `capacity_probe.py`：tp {2,4} × Micro-batch {1,2} × 序列 {128,256,512} 容量网格（attention 分片布局）。
- `prepare_data.py` / `sft_core.py` / `assistant_loss.py` / `mini_batch.py` / `qwen3_tokenizer.py` / `experiment_utils.py`：与第 35 期一致的稳定工具；数据文件名为 `data/chapter36_train.jsonl` 与 `data/chapter36_manifest.json`；schema 为 `chapter36.v1`。

## 环境和数据

与第 35 期完全相同，模型固定为 `Qwen/Qwen3-0.6B-Base@da87bfb608c14b7cf20ba1ce41287e8de496c0cd`，数据固定为 `sft_mini_512.jsonl@9b06bee29c0d5685647ac11b0dcea9f572c5064a`（SHA256 `475039fa…7852`，1,232,540,940 字节，下载后校验）。

```bash
cd chapter_36
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 python prepare_data.py --samples 512
```

模型权重在首次运行时由 `resolve_model_directory` 按固定 revision 下载。

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

# 3. Benchmark：W=2（同轮内先跑 mlp_tp_w2 再跑 attn_tp_w2）
python benchmark.py --world-sizes 2 --output benchmark-results.local.json
#    W=4（单机四卡）
python benchmark.py --world-sizes 4 --output benchmark-results.local.json
#    W=8（双节点 4+4，跨节点 TP 仅作边界对照，先在两个节点分别同步同一 revision 与数据）
#    节点 0：torchrun --nnodes 2 --nproc-per-node 4 --node-rank 0 benchmark.py --world-sizes 8 --output benchmark-results.local.json
#    节点 1：torchrun --nnodes 2 --nproc-per-node 4 --node-rank 1 benchmark.py --world-sizes 8 --output benchmark-results.local.json

# 4. 容量探针：tp {2,4} × batch {1,2} × length {128,256,512}
python capacity_probe.py --output capacity-results.local.json
```

所有输出默认写入 `*.local.json`，并由 Git 忽略。

## 指标口径与计时边界

- `step_time_ms` 为端到端 Wall Clock，两端正确同步；forward/backward 分段用 CUDA Event，且各自包含层内 TP AllReduce，再单独减去通信时间给出纯计算段。
- TP AllReduce 时间由每次 Collective 的 Event 对累计（`TPCommRecorder`），每步只在结尾同步一次，全部暴露在关键路径上。`attn_tp` 臂每层 2 次前向（o_proj 输出、down_proj 输出）、4 次反向（两个块输入梯度 + q/k norm 同步）；`mlp_tp` 臂每层 1 次前向、1 次反向。两臂的 AllReduce 计数随结果一并记录。
- 主吞吐为有效监督 Token/s；TP 组内数据复制，全局 batch = 每 Rank batch = 2 × tp_size，Token 只计一次。同时报告物理 Token/s 与非 Padding Token/s。
- 显存同时报告每卡峰值 allocated/reserved 与理论所有权账本（复制参数/分片参数/优化器状态各自字节数；`attn_tp` 臂的分片部分包含 Attention 权重）。
- OOM 记录失败配置、失败阶段与当时 allocated/reserved，不做静默降级。


## 适用边界

- 主线模型为 `Qwen/Qwen3-0.6B-Base`，训练目标为全参数 SFT。
- `smoke_test.py` 不依赖真实模型下载或 GPU；真实权重正确性检查与 Benchmark 需要上文注明的 GPU 配置。
- 本目录只提供代码、依赖和运行说明。正确性与 Benchmark 命令会在本地生成 `*.local.json`，仓库不附带实验结果。
