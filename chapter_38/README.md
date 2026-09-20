# 第 38 期：Sequence Parallel

## 对应篇章

对应《超轻量AI Infra教程—训练篇》第 23 章；合并后的连续编号为第 38 章。

## 显卡要求

最低 2 张 GPU；参考配置覆盖 2 张和 4 张 GPU。

第 37 期已切分 MLP、Attention 和词表，但 TP 区域之间的残余激活仍在各 Rank 重复保存：残差流与 LayerNorm 的激活是完整的 `[batch, seq, hidden]`，与权重是否分片无关。本期只改变一件事：在 Attention+MLP 权重分片（第 36 期布局）保持不变的前提下，这些非 TP 区域的激活是否沿 Sequence 维切分。Context Parallel / Ring Attention 明确不在本期——Attention 内部仍然每 Rank 跑全序列，序列维只在 TP 区域之外切分。

## 问题、假设与边界

- 问题：LayerNorm、残差这些区域按序列切分后，训练语义是否与复制版数学等价？每卡激活显存下降多少？用 AllGather/Reduce-Scatter 换来的容量收益，在 0.6B、L=512 的主实验里能不能覆盖边界通信的代价？
- 假设：Qwen3 的 RMSNorm 逐 Token 独立（均值只在 hidden 维），残差加法逐元素，所以在 `[b, s/N, h]` 分片上计算等于在 `[b, s, h]` 上计算后切片；Attention 是唯一的序列耦合算子，它在 TP 区域内部拿 AllGather 后的全序列计算。FP32 下只有归约顺序带来的极小数值分叉。代价是每个并行块多一次 AllGather + 一次 Reduce-Scatter，边界通信上升；0.6B 激活占比小，端到端吞吐可能为负优化，收益预期首先出现在长序列容量。负优化如实保留。
- 唯一变量：TP 区域之间原本重复保存的激活，是否沿 Sequence 维分片。`tp_wN` 臂（第 36 期布局，激活复制）对照 `sp_wN` 臂（同权重分片 + 序列分片）。
- 固定项：权重布局（与第 36 期完全一致）、模型/数据/标签/dtype（FP32、TF32 关闭）/eager attention/优化器公式/种子；loss 为全局 token_mean；每步全局 batch = 2 × tp_size；主实验 W=2、L=512。
- 排除项：Context Parallel / Ring Attention（第 38 期明确排除）、Megatron 把 f-AllReduce 与 g-backward 融合为一次 Reduce-Scatter 的字节优化（本期保留第 36 期 f 算子 + 本地切片）、Vocab Parallel（第 37 期）、FlashAttention / 激活重算 / 编译优化、DP×TP 混合（第 42 期）。

## 机制与梯度所有权（本期易错点）

每层区域布局（`b` batch、`s` sequence、`h` hidden、`N` = tp_size）：

```text
x_r [b, s/N, h]                        序列分片，Rank 本地
normed_r = RMSNorm(x_r)                逐 Token，分片上精确
normed   = AllGather_seq(normed_r)     g 算子，进入 TP 区域
attn_r   = o_proj 部分和（不做 AllReduce）
out_r    = ReduceScatter_seq(attn_r)   ḡ 算子，离开 TP 区域
x_r     += out_r                       残差加法，仍在分片上
```

1. **非 TP 权重的梯度从"天然一致"变成"部分和"**：第 35/36 期里 LayerNorm 权重梯度在各 Rank 一致、再 AllReduce 反而是错的；序列切分后每个 Rank 只看到自己的 Token，LayerNorm、最终 norm、Embedding/LM Head 的原始梯度都是部分和。`ParallelLayerNorm` 和 `register_replicated_gradient_sync` 用 post-accumulate-grad hook 对每个权重做一次 backward AllReduce SUM，恢复与参考一致的完整梯度。漏掉任何一个，副本从第一次 optimizer step 起静默分叉。
2. **g 算子的 backward 是本地切片，不是 Reduce-Scatter**：f 算子（第 36 期的块输入梯度 AllReduce）先把各 Rank 的部分梯度求和成一致的完整梯度，切片即可；对一致的副本再做 Reduce-Scatter 会放大 tp_size 倍。Megatron 把这两步融合成一次 Reduce-Scatter 省一半反向字节，该融合记录为本期不做。
3. **ḡ 算子 forward 是 Reduce-Scatter（输入是各 Rank 的部分和）、backward 是 AllGather（各 Rank 持有真实梯度的互斥片段）**：两种"分散"的语义不同，不能互换。
4. **Loss**：每个 Rank 只对自己序列窗口算 CE，`(sum, count)` 打包成一个标量张量做一次 AllReduce，backward loss = `sum_local / count_global`，梯度总和精确等于全局 token_mean 梯度。完整词表 logits 永远只物化 `[b, s/N, vocab]`。
5. 分片权重（q/k/v/o、gate/up/down）与 Attention 内部 q/k norm 的语义与第 36 期完全一致：本地梯度、q/k norm hook 同步。

优化器侧沿用 `clip_grad_norm_tp`（复制部分计一次、分片部分跨 Rank 求和的精确全局范数）；hook 同步之后复制梯度恰好是完整参考值，账本分类不变。

## 文件

- `sequence_parallel.py`：本期核心。`sequence_shard_bounds`（整除检查与窗口）、`_GatherFromSequenceParallelRegion` / `_ScatterToSequenceParallelRegion`（g/ḡ autograd 算子）、`ParallelLayerNorm`（分片 RMSNorm + 权重梯度同步 hook）、`SPDecoderLayer`、`SequenceParallelModel` / `SequenceParallelForCausalLM`（Embedding/最终 norm/LM Head 只算本地窗口，梯度同步注册）、`sp_loss_terms` + `sp_global_loss_scale`（sum/count 打包 AllReduce）、`sp_train_step`（分段计时 + 通信账本）、`ActivationLedger`（saved_tensors_hooks 测量 saved-for-backward 字节，排除参数存储）、`arm_byte_ledger`（两臂 ring 模型字节账本）、`SPCommRecorder`（AllReduce 计时上叠加 AllGather/Reduce-Scatter 计时与字节）、`load_handwritten_model_sp`（权重读取与第 36 期相同的分片窗口）。
- `mlp_tp.py` / `attention_tp.py`：第 35/36 期实现副本；唯一改动是 `ParallelMLP` / `ParallelAttention` 把 `reduce_output` 透传给 Row-parallel 层——SP 需要未归约的部分和去做 Reduce-Scatter，其余逐字节兼容。
- `qwen3_model.py`：第 33/35/36 期副本（SP 模型组装在 `sequence_parallel.py`，状态字典命名与复制版一致）。
- `tp_runtime.py`：数据流水与进程组通用件；新增 `load_tokenizer_and_model_sp`。
- `smoke_test.py`：CPU gloo，tiny 模型（hidden 32 / intermediate 64 / vocab 64 / 2 层 / 8 Q Head / 4 KV Head / head_dim 8 / seq 12），tp=2 与 tp=4 两轮：g/ḡ 算子对照、分片 LayerNorm 对照、SP DecoderLayer 逐切片对照、完整模型 loss/梯度所有权/一次 AdamW step 重组对照、saved activation bytes（SP < 纯 TP）、整除错误提示。
- `compare_correctness.py`：真实 Qwen3-0.6B 权重，同一 Rank 先后跑两臂（纯 TP → SP），逐名对照 forward loss、全部参数梯度、loss 轨迹、一次更新后重组参数；SP 复制参数梯度跨 Rank 一致；每 Rank saved activation bytes 与比值写入 JSON。
- `benchmark.py`：每 world size 两臂 `tp_wN` / `sp_wN`（默认 W=2 主实验，另含 W=4），3 次重复；峰值显存、状态账本、saved-for-backward 字节、AllGather/Reduce-Scatter/AllReduce 时间与 ring 字节账本、Step Time、有效监督 Token/s。
- `capacity_probe.py`：arm {tp,sp} × max_length {512,1024,2048} × micro {1,2}；语料本身不超过 512 Token，长序列档把 batch 固定 pad 到物理长度 L（只解读显存边界，不解读算力）。
- `prepare_data.py` / `sft_core.py` / `assistant_loss.py` / `mini_batch.py` / `qwen3_tokenizer.py` / `experiment_utils.py`：与第 36 期一致的稳定工具；数据文件名为 `data/chapter38_train.jsonl` 与 `data/chapter38_manifest.json`；schema 为 `chapter38.v1`。

## 环境和数据

与第 36 期完全相同，模型固定为 `Qwen/Qwen3-0.6B-Base@da87bfb608c14b7cf20ba1ce41287e8de496c0cd`，数据固定为 `sft_mini_512.jsonl@9b06bee29c0d5685647ac11b0dcea9f572c5064a`（SHA256 `475039fa…7852`，1,232,540,940 字节，下载后校验）。

```bash
cd chapter_38
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 python prepare_data.py --samples 512
```

模型权重在首次运行时由 `resolve_model_directory` 按固定 revision 下载。

## 运行顺序

```bash
# 1. CPU smoke（不需要 GPU 与模型下载）
CUDA_VISIBLE_DEVICES="" python smoke_test.py

# 2. 真实权重正确性：TP=2（单机双卡，mp.spawn；同一 Rank 先后跑两臂）
python compare_correctness.py --world-size 2 --output correctness-results.local.json
#    TP=4（单机四卡）
python compare_correctness.py --world-size 4 --output correctness-results.local.json
#    torchrun 等价形式（--world-size 必须等于 WORLD_SIZE）：
#    torchrun --nproc-per-node 2 compare_correctness.py --world-size 2

# 3. Benchmark 主实验：W=2（同轮内先跑 tp_w2 再跑 sp_w2）
python benchmark.py --world-sizes 2 --output benchmark-results.local.json
#    W=4（单机四卡）
python benchmark.py --world-sizes 4 --output benchmark-results.local.json

# 4. 容量探针：arm {tp,sp} × L {512,1024,2048} × micro {1,2}
python capacity_probe.py --tp-size 2 --output capacity-results.local.json
```

所有输出默认写入 `*.local.json`，并由 Git 忽略。

## 指标口径与计时边界

- `step_time_ms` 为端到端 Wall Clock，两端正确同步；forward/backward 分段用 CUDA Event，各自包含通信时间，再减去记录的 Collective 时间给出纯计算段。
- 通信分三类记录：TP AllReduce（f 算子 + q/k norm 同步 + hook 同步）、SP 边界算子（g/ḡ 的 AllGather/Reduce-Scatter）、loss 聚合（一次打包标量 AllReduce）。`SPCommRecorder` 用 Event 对累计，每步只在结尾同步一次；两臂的算子计数随结果一并记录。字节为 ring 模型推算（`arm_byte_ledger`），不是线速测量。
- 激活显存两条线：每卡峰值 allocated/reserved；每 Rank 测量的 saved-for-backward 字节（`ActivationLedger`，排除参数存储，每臂在计时外测一次）。两者都不替代理论账本，结论以同 batch 两臂差值为准。
- 主吞吐为有效监督 Token/s；TP/SP 组内数据复制，全局 batch = 每 Rank batch = 2 × tp_size，Token 只计一次。同时报告物理 Token/s 与非 Padding Token/s。
- OOM 记录失败配置、失败阶段与当时 allocated/reserved，不做静默降级。


## 适用边界

- 主线模型为 `Qwen/Qwen3-0.6B-Base`，训练目标为全参数 SFT。
- `smoke_test.py` 不依赖真实模型下载或 GPU；真实权重正确性检查与 Benchmark 需要上文注明的 GPU 配置。
- 本目录只提供代码、依赖和运行说明。正确性与 Benchmark 命令会在本地生成 `*.local.json`，仓库不附带实验结果。
