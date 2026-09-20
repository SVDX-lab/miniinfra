# 第 37 期：Vocab Parallel Embedding 与 Cross Entropy

## 对应篇章

对应《超轻量AI Infra教程—训练篇》第 22 章；合并后的连续编号为第 37 章。

## 显卡要求

最低 2 张 GPU；参考配置覆盖 2 张和 4 张 GPU。

第 36 期已把 MLP 与 Attention Head 全部切分，但 Embedding、LM Head 与完整词表 Logits 仍每卡完整。本期只改变一件事：Embedding/LM Head 的词表维度是否也按 TP Rank 切分，并用不物化完整 Logits 的分布式 Cross Entropy 闭合训练。Sequence Parallel 是第 38 期，不在本期。

Qwen3-0.6B 词表 151936：TP=2 每 Rank 75968 行，TP=4 每 Rank 37984 行。Qwen3 的 Embedding 与 LM Head 共享权重（tie），词表分片后两者共享同一个分片 Parameter。batch 2 × seq 512 的完整 Logits 是 151936 × 1024 × 4B ≈ 622 MiB——这是 TP 模型里最大的单项激活，也是本期的主变量。

## 问题、假设与边界

- 问题：词表切分后，训练语义是否与完整复制数学等价？不物化完整 Logits 的 Cross Entropy（全局 max 用 AllReduce MAX、全局 exp 和用 AllReduce SUM、目标 token logit 按所属 Rank 用 AllReduce SUM 取回）与完整词表 CE 是否在 FP32 容差内一致？分片 Logits 省下的显存与新增通信如何权衡？
- 假设：CE 的 log-sum-exp 可以按词表维度任意分块归约（max 与 sum 都可分块）；目标 token 的 logit 只由其所属 Rank 贡献，用一次 SUM 即可取回；CE 的 backward（softmax − onehot）只需要本 Rank 分片和三个全局标量，因此 backward 不需要任何通信。隐藏状态 h 是复制的，LM Head 输出不归约，但其输入梯度 `dlogits_r @ W_r` 是部分和，需要一次 backward AllReduce（Megatron 的 f 算子）——这是词表切分在 backward 侧的固定代价。0.6B 在 tp=2 下词表切分的收益可能仍为负，负优化如实保留。
- 唯一变量：在 Attention+MLP TP（第 36 期布局）基础上，Embedding/LM Head 完整复制（`attn_tp_wN` 臂）对照沿词表切分 + 分布式 CE（`vocab_tp_wN` 臂）。两臂在同一 world size 下只差词表分片这一个机制。
- 固定项：MLP/Attention 分片方式（第 35/36 期）、模型/数据/标签/dtype（FP32、TF32 关闭）/eager attention/优化器公式/种子；loss 为全局 token_mean（两臂使用相同的 `reduce_losses` 归约，TP 组内 loss_scale 恒为 1.0）；每步全局 batch = 2 × tp_size。
- 排除项：Sequence Parallel（第 38 期）、FlashAttention / 激活重算 / 编译优化（第 25/26/28 期机制，本期不混入）、DP×TP 混合与 Process Group 细分（第 42 期）、Bucket 化/重叠通信（第 30 期机制）。

## 梯度所有权分析（本期易错点）

1. **Embedding 分片（lookup 路径）：梯度只在本地。** 越界 token 先被 mask 成 0 贡献，一次 forward AllReduce 求和后每个 Rank 拿到完整隐状态；backward 的 g 算子是恒等，`dW_r` 只累加本 Rank 行的散射项——完整梯度的行切片，无需任何通信。
2. **LM Head 分片（logits 路径）：权重梯度本地，输入梯度要归约。** `dW_r = dlogits_r^T @ h` 只用本 Rank 的 logits 分片，是完整梯度的行切片；但 `d(h) = dlogits_r @ W_r` 只是部分和——第 36 期 LM Head 复制时每卡就有完整梯度，本期必须补一次 hidden 梯度的 backward AllReduce（f 算子，每步一次）。漏掉它，`d(h)` 变成部分梯度流入最终 LayerNorm，训练语义被静默改坏：本期的 smoke 与正确性对照里表现为 LayerNorm 梯度跨 Rank 不一致、tied Embedding 梯度对照失败。
3. **Tie 的合并：** Embedding 与 LM Head 是同一个 Parameter，lookup 与 logits 两条路径的梯度自动累加进同一个分片，与参考实现的 tied 梯度（两条路径之和）逐切片一致。
4. **复制参数只剩 LayerNorm：** 梯度跨 Rank 天然一致（f 与 g 之间的激活逐 Rank 相同），不需要也不能做 AllReduce；q/k norm 的部分梯度由第 36 期的 backward 同步机制处理。
5. **优化器侧：** 与第 35/36 期相同，精确全局范数裁剪用 1 次标量 AllReduce（复制部分计一次、分片部分跨 Rank 求和，分片类别扩展到词表）。

分布式 Cross Entropy 的通信账本（每步、vocab_tp 臂）：forward 新增 embedding 的 1 次 g + 3 次标量级 Collective（row-wise MAX、SUM、目标 logit SUM，尺寸是 token 数而非词表数）；backward 新增 LM Head 的 1 次输入梯度 AllReduce，**CE 本身 0 次**。对照 attn_tp 臂：完整 logits 物化 + 完整词表 CE，无这些 Collective，但要付 622 MiB 量级的 Logits 显存与其读写带宽。

## 文件

- `vocab_tp.py`：本期核心。`VocabParallelEmbedding`（范围 mask + 本地 lookup + 输出 AllReduce）、`VocabParallelLMHead`（分片 logits，输入 f 算子）、`_VocabParallelCrossEntropy`（单 autograd Function：3 次 forward Collective、backward 0 通信，IGNORE_INDEX 行零 loss 零梯度）、`batch_loss_vocab_parallel`（与 `mini_batch.reduce_losses` 相同的 token_mean 归约）、`split_all_parameters` / `clip_grad_norm_all` / `gather_all_tensor_parallel_parameters` / `tp_memory_ledger_all`（第 35 期工具的词表扩展版）、`tp_train_step`（按臂分派 CE，记录 local/full logits 字节数）。
- `attention_tp.py` / `mlp_tp.py`：第 36/35 期核心的原样副本，两臂共同基线。
- `qwen3_model.py`：新增 `embed_factory` / `lm_head_factory` 装配点与 `load_handwritten_model_vocab_tp`——Embedding 行、Attention Head、MLP 中间维全部用 safetensors `PySafeSlice` 只读本 Rank 分片；加载后校验 tie 身份（`lm_head.weight is embed_tokens.weight`）。
- `tp_runtime.py`：`load_tokenizer_and_model_tp(vocab_parallel=...)` 在两臂之间选择权重布局；数据流水与第 36 期一致。
- `smoke_test.py`：CPU gloo，tiny 模型（vocab 64、8 Q/4 KV Head、2 层），tp=2：4 词极小词表（每 Rank 2 词）的手算 CE 推导与分布式实现对照（含目标属于两个 Rank、IGNORE_INDEX 行零 loss 零梯度三种情况）、Embedding 查表与行切片梯度、shard logits 形状与梯度切片、tied Embedding 梯度（lookup+head 两路合并）、一次 AdamW step 重组、所有权、通信计数（含 CE backward 0 通信）、logits 显存记账、整除错误。
- `compare_correctness.py`：真实 Qwen3-0.6B 权重，TP=2/4 与单卡 FP32 参考对照：vocab-parallel CE loss、shard logits 梯度对照（stash 完整 logits 梯度再切分片）、tied Embedding 分片梯度、Attention/MLP 分片梯度、layer-0 分片拼接（含 Embedding 行拼接）、一次更新后重组参数、LayerNorm 梯度跨 Rank 一致性。
- `benchmark.py`：每 world size 两臂 `attn_tp_wN` / `vocab_tp_wN`（N=2/4，另含 N=8 跨节点边界档），3 次重复；每步记录 `local_logits_bytes` 与 `full_logits_bytes`（batch 2 × seq 512 时完整 logits ≈ 622 MiB，vocab_tp 臂除以 TP 度）。
- `capacity_probe.py`：tp {2,4} × Micro-batch {1,2} × 序列 {128,256,512} 容量网格（vocab 分片布局）。
- `prepare_data.py` / 稳定工具：与第 36 期一致；数据文件名 `data/chapter37_train.jsonl` 与 `data/chapter37_manifest.json`；schema 为 `chapter37.v1`。

## 环境和数据

与第 35/36 期完全相同，模型固定为 `Qwen/Qwen3-0.6B-Base@da87bfb608c14b7cf20ba1ce41287e8de496c0cd`，数据固定为 `sft_mini_512.jsonl@9b06bee29c0d5685647ac11b0dcea9f572c5064a`（SHA256 `475039fa…7852`，1,232,540,940 字节，下载后校验）。

```bash
cd chapter_37
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 python prepare_data.py --samples 512
```

## 运行顺序

```bash
# 1. CPU smoke（不需要 GPU 与模型下载）
CUDA_VISIBLE_DEVICES="" python smoke_test.py

# 2. 真实权重正确性：TP=2（单机双卡）
python compare_correctness.py --world-size 2 --output correctness-results.local.json
#    TP=4（单机四卡）
python compare_correctness.py --world-size 4 --output correctness-results.local.json
#    torchrun 等价形式（--world-size 必须等于 WORLD_SIZE）：
#    torchrun --nproc-per-node 2 compare_correctness.py --world-size 2

# 3. Benchmark：W=2（同轮内先跑 attn_tp_w2 再跑 vocab_tp_w2）
python benchmark.py --world-sizes 2 --output benchmark-results.local.json
#    W=4（单机四卡）
python benchmark.py --world-sizes 4 --output benchmark-results.local.json
#    W=8（双节点 4+4，跨节点 TP 仅作边界对照）
#    节点 0：torchrun --nnodes 2 --nproc-per-node 4 --node-rank 0 benchmark.py --world-sizes 8 --output benchmark-results.local.json
#    节点 1：torchrun --nnodes 2 --nproc-per-node 4 --node-rank 1 benchmark.py --world-sizes 8 --output benchmark-results.local.json

# 4. 容量探针：tp {2,4} × batch {1,2} × length {128,256,512}
python capacity_probe.py --output capacity-results.local.json
```

所有输出默认写入 `*.local.json`，并由 Git 忽略。

## 指标口径与计时边界

- `step_time_ms` 为端到端 Wall Clock；forward/backward 分段用 CUDA Event，包含层内 AllReduce 后再单独减出纯计算段。TP AllReduce 由 `TPCommRecorder` 的 Event 对累计，每步结尾同步一次，全部暴露在关键路径上。
- 两臂的 AllReduce 计数不同且随结果记录：vocab_tp 臂 forward = 每层 2 次块输出 + embedding 1 次 + CE 3 次；backward = 每层 2 次块输入 + 2 次 q/k norm 同步 + LM Head 输入梯度 1 次。attn_tp 臂不含 embedding/CE/LM Head 项。
- Logits 显存：每步记录 `local_logits_bytes`（实际物化的 logits）与 `full_logits_bytes`（等价完整 logits 理论值）；attn_tp 臂两者相等，vocab_tp 臂比值 = 1/tp_size。
- 主吞吐为有效监督 Token/s（TP 组内数据复制，Token 只计一次），同时报告物理 Token/s 与非 Padding Token/s。
- 显存同时报告每卡峰值 allocated/reserved 与所有权账本（复制参数仅剩 LayerNorm，分片参数含词表行）。
- OOM 记录失败配置、失败阶段与当时 allocated/reserved，不做静默降级。


## 适用边界

- 主线模型为 `Qwen/Qwen3-0.6B-Base`，训练目标为全参数 SFT。
- `smoke_test.py` 不依赖真实模型下载或 GPU；真实权重正确性检查与 Benchmark 需要上文注明的 GPU 配置。
- 本目录只提供代码、依赖和运行说明。正确性与 Benchmark 命令会在本地生成 `*.local.json`，仓库不附带实验结果。
