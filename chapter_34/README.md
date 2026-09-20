# 第 34 期：ZeRO-3 / FSDP——再切分参数

## 对应篇章

对应《超轻量AI Infra教程—训练篇》第 19 章；合并后的连续编号为第 34 章。

## 显卡要求

最低 2 张 GPU；参考配置为 4 张 GPU，并覆盖双节点 4+4 张 GPU。

第 32 期切分了优化器状态，第 33 期切分了梯度，但每卡仍常驻完整参数——它只在 forward/backward 里被读取，却在两个 Step 之间整份占着显存。本期只改变一件事：训练参数是否也只以 Shard 形式常驻每个 Rank（ZeRO-2 对照 ZeRO-3/FSDP）。参数从此有了完整生命周期：分片常驻 → Forward 前 AllGather → 计算 → Reshard（释放全量存储）→ Backward 前按需 AllGather → Reduce-Scatter 梯度 → 分片 Optimizer Step；下一个 Forward 的 AllGather 本身就是参数刷新，ZeRO-1/2 那次 post-step AllGather 被取消。

## 问题、假设与边界

- 问题：参数分片把多少常驻显存换成多少通信？通信从 ZeRO-2 的一次 post-step AllGather 搬进 Forward/Backward 热路径后，Step Time 如何变化？0.6B 在 12GB 上的 Batch/Length 准入边界如何变化？
- 假设：每卡常驻模型状态从 ZeRO-2 的 `2N+14N/W` 降到 ZeRO-3 的 `16N/W` 字节（FP32 账本）；0.6B 上 AllGather 进热路径很可能让 Step Time 变长——负优化如实保留；FSDP 的梯度 Reduce-Scatter 取各 Rank 平均，本地损失乘 W 校正后与 ZeRO-2 的全局 token_mean 梯度在 FP32 容差内一致（归约顺序不同允许极小分叉，不等于实现错误）。
- 唯一变量：低精度训练参数是否也只以 Shard 形式常驻每个 Rank。
- 固定项：模型、Tokenizer、数据、FP32 dtype、AdamW 公式与超参、梯度分片（ZeRO-2 机制）、状态分片（ZeRO-1 机制）、loss 口径（全局 token_mean）均与第 32/33 期一致。
- 排除项：激活重算与 CPU Offload、Bucket 化扁平参数与通信重叠调优（FSDP 内部策略按默认使用，不扫描）、BF16 混合精度、FSDP2/DTensor、TP/SP/PP。

## 文件

- `zero3_core.py`：本期核心（手写机制层）。三个 autograd 单元 `sharded_linear` / `sharded_embedding` / `sharded_scale` 实现参数的完整生命周期——forward 里 AllGather 出全量权重并在autograd 不可见的 no-grad 区内完成计算，因此全量存储在单元返回时真正释放（Reshard 是实打实的）；backward 按需重新 AllGather，算出的权重梯度当场 Reduce-Scatter，只有本 Rank 分片落进 `.grad`。`sharded_swiglu_mlp` 用普通 autograd 把三个单元拼成 Qwen3 MLP 形状。`ShardedLeafAdamW` 在分片叶子上执行与第 32 期逐操作相同的 AdamW 公式；`unshard_parameter`/`memory_ledger_zero3` 用于检查与账本。`GATHER_CALLS`/`LAST_FULL_GATHER` 计数器让 smoke test 能机械地证明"forward 返回时全量已释放、backward 恰好一次按需 AllGather"。
- `fsdp_train.py`：官方实现层。`wrap_model_fsdp` 用 `torch.distributed.fsdp` FULL_SHARD 包装手写 Qwen3-0.6B：`transformer_auto_wrap_policy` 按 `Qwen3DecoderLayer` 切单元、`BackwardPrefetch.BACKWARD_PRE`、`CPUOffload=None`、FP32、`use_orig_params=True`（`tie_word_embeddings=True` 时 FSDP 要求开启，共享的 Embedding/LM Head 权重保持同一对象、同一分片）。`fsdp_train_step` 提供 ZeRO-2 同款分段指标，并在本地损失上乘 W 以抵消 FSDP Reduce-Scatter 的平均语义；`get_full_state_dicts`/`load_full_state_dicts` 是 FULL_STATE_DICT（模型+优化器）保存/恢复路径；`fsdp_theoretical_ledger` 给出 16N/W 账本。
- `zero2_core.py` / `ddp_core.py`：第 33/29 期复制件，`zero2` 臂 baseline 引擎。
- `compare_reference.py`：第 30 期起共用的大 batch 参考轨迹（含 step-0 梯度暂存与尺度相对误差报告）。
- `compare_correctness.py`：真实权重三相对照——大 batch 参考 / ZeRO-2 baseline / FSDP variant，含 FULL_STATE_DICT 保存恢复回放。
- `smoke_test.py`：CPU——分片数学、Reshard 与按需 AllGather 证明、3 步逐位对照、SwiGLU 块、tie 单元、账本比例、FSDP tiny 模型一步。
- `benchmark.py`：zero2 / fsdp 两臂 × 3 次重复，W ∈ {2,4,8}。
- `capacity_probe.py`：0.6B × zero2/fsdp × Micro-batch {1,2} × Length {128,256,512} 容量探针。

## 环境和数据

与第 32/33 期完全相同，模型固定为 `Qwen/Qwen3-0.6B-Base@da87bfb608c14b7cf20ba1ce41287e8de496c0cd`，数据固定为 `jingyaogong/minimind_dataset@sft_mini_512.jsonl@9b06bee29c0d5685647ac11b0dcea9f572c5064a`（SHA256 `475039fa9b80ad365bc23d837b1a7695c7c680d950713a91359621fbb7607852`，1,232,540,940 字节），Tokenizer 与 Chat Template 复用第 21 期的手写实现，固定 512 条纯文本对话。

```bash
cd chapter_34
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 python prepare_data.py --samples 512
```

## 运行顺序

```bash
CUDA_VISIBLE_DEVICES="" python smoke_test.py
python compare_correctness.py --world-size 2 --output correctness-results.local.json
python benchmark.py --world-sizes 2 --output benchmark-results-w2.local.json
python capacity_probe.py --max-lengths 128 256 512 --micro-batch-sizes 1 2 --output capacity-results.local.json
```

单机 4 卡（`--world-sizes` 传子集，`all` 表示 {2,4,8} 全跑，需要单机 8 卡）：

```bash
python benchmark.py --world-sizes 2,4 --output benchmark-results-w2w4.local.json
python compare_correctness.py --world-size 4 --output correctness-results-w4.local.json
python capacity_probe.py --world-size 4 --output capacity-results-w4.local.json
```

双节点 4+4（W=8），torchrun 两种等价形式——静态组网（输出 JSON 只由节点 0 的 rank 0 落盘，两个节点使用同一条命令与同一个 `--output` 路径）：

```bash
# 节点 0
MASTER_ADDR=<节点0地址> MASTER_PORT=29500 torchrun --nnodes 2 --nproc-per-node 4 --node-rank 0 \
  benchmark.py --world-sizes 8 --output benchmark-results-w8.local.json
# 节点 1
MASTER_ADDR=<节点0地址> MASTER_PORT=29500 torchrun --nnodes 2 --nproc-per-node 4 --node-rank 1 \
  benchmark.py --world-sizes 8 --output benchmark-results-w8.local.json
```

或 c10d rendezvous（两节点命令完全相同，只需共享同一 rendezvous 端点）：

```bash
torchrun --nnodes 2 --nproc-per-node 4 --rdzv-backend c10d --rdzv-endpoint <节点0地址>:29500 --rdzv-id chapter34 \
  benchmark.py --world-sizes 8 --output benchmark-results-w8.local.json
```

`compare_correctness.py` / `capacity_probe.py` 同理（分别把脚本名与 `--world-size 8` 换入；torchrun 下 `--world-size` 必须等于 `WORLD_SIZE`）。注意 12GB 上 FP32 0.6B 的容量尚未校准：正确性对照若 OOM，按合同候选档位降档重试（如 `--max-length 128`）。

## 指标口径

- 主吞吐为全局有效监督 Token/s，同时记录物理 Token/s 与非 Padding Token/s；全局 batch = 每 rank 2 × W（弱扩放口径，与第 32/33 期相同）。
- Step Time 为端到端 Wall Clock，边界同步；CUDA 区间用 Event。
- zero2 臂的通信是暴露的：backward 后的逐参数 Reduce-Scatter 链（`grad_comm`）与 post-step 参数 AllGather（`param_allgather`）单独计时并计入 Step Time。
- fsdp 臂的通信嵌入 Forward/Backward 分段内（FULL_SHARD 的单元 AllGather 与梯度 Reduce-Scatter），`grad_allreduce_ms`/`param_allgather_ms` 恒为 0 且结果中标注 `communication_embedded: true`；不单列独立通信段，比较时把 zero2 的 forward/backward 与两段通信加在一起看。
- 显存报告每卡峰值 allocated/reserved，并与理论账本（ZeRO-2 `2N+14N/W` 对照 ZeRO-3 `16N/W`）同列；OOM 记录失败阶段与已分配/保留显存。


## 适用边界

- 主线模型为 `Qwen/Qwen3-0.6B-Base`，训练目标为全参数 SFT。
- `smoke_test.py` 不依赖真实模型下载或 GPU；真实权重正确性检查与 Benchmark 需要上文注明的 GPU 配置。
- 本目录只提供代码、依赖和运行说明。正确性与 Benchmark 命令会在本地生成 `*.local.json`，仓库不附带实验结果。
