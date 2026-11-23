# 第 19 期：按长度分桶

## 对应篇章

对应《超轻量AI Infra教程—训练篇》第 04 章；合并后的连续编号为第 19 章。

## 显卡要求

最低与参考配置均为 1 张 GPU。

第 18 期已经能把变长样本右侧 Padding 后组成固定 Mini-batch。同一 Batch 内的长度差异会让短样本占用大量没有有效 Token 的槽位。本期只改变 Batch Sampler：Baseline 用固定种子全局 Shuffle 后顺序切块；Variant 用同一随机排列切成有限 Shuffle Window，窗口内按编码长度排序后切块，最后打乱 Batch 顺序。两条路径使用相同样本集合、Batch Size、Assistant-only Labels、Loss reduction 和每 Epoch 相同数量的 AdamW Step。

## 问题、假设与边界

- 问题：在固定随机顺序组 Batch 时，同一 Batch 内的长度差异带来多少 Padding？在有限 Shuffle Window 内按长度分桶能否减少这些浪费，代价是什么？
- 假设：窗口越大，Batch 内长度越接近，物理 Padding Token 越少，有效监督 Token/s 越高；同时 Batch 成员越来越由长度决定，跨 Epoch 的成员共现重复率上升，全排序还可能带来可测的训练顺序损失。
- 唯一变量：Batch 成员的组织方式——纯随机切块与有限窗口内按长度分桶。
- 固定项：模型与 Tokenizer revision、数据 revision、固定 512 条样本、Assistant-only Labels、`token_mean` 归约、FP32、TF32 关闭、Eager Attention、AdamW（betas=(0.9, 0.95)，eps=1e-8，foreach=false，fused=false）、学习率 2e-5（无调度器）、全局梯度裁剪阈值 1、最大长度 512、Batch Size 4、种子 20260912。每个 Epoch 内每条样本恰好出现在一个 Batch 中，两条路径的 Step 数相同。
- 排除项：Sequence Packing、离线 Tokenize、DataLoader 预取与异步 H2D、BF16、梯度累积、动态 Batch Size、Token Budget Batch、DDP 及其他分布式训练。

## 文件

- `length_bucket.py`：本期核心实现。`BatchPlan`（一个 Epoch 的批次划分）、`random_plan`（随机切块）、`bucketed_plan`（窗口内按长度分桶）、`project_plan_tokens`（仅由长度投影 Padding 与每 Batch 工作量）、`batch_length_spreads` 与 `companion_repeat_ratio`（数据随机性度量）。
- `mini_batch.py`：沿用第 18 期的右侧 Padding 构造与 `token_mean` 归约；逐样本累加路径已在第 18 期证明等价，本期不再重复。
- `qwen3_model.py`、`qwen3_tokenizer.py`、`assistant_loss.py`：本期目录内的手写 Qwen3 模型、Byte-level BPE Tokenizer 与 Assistant-only Labels。
- `training.py`：plan 驱动的 FP32 训练 Step、CUDA Event 分段计时、显存与遥测。
- `smoke_test.py`：CPU 极小样本集上的划分、退化对照、确定性、窗口边界与投影检查。
- `compare_correctness.py`：真实 Qwen3 FP32 的划分检查、成员不变性、同成员孪生更新和串行锚定对照。
- `benchmark.py`：固定 512 样本、Batch Size 4、窗口 {4(随机), 8, 16, 32, 64, 128, 256, 512} 的单 Epoch 扫描。
- `train.py`：随机、中等窗口与全排序三个配置的双 Epoch Loss 轨迹。
- `survey_lengths.py`：测量固定 512 样本在 cap 1024 下的自然长度向量，供投影分析使用。
- `prepare_data.py`：数据下载、大小与 SHA256 校验、512 条固定样本筛选。
- `data-manifest.json`：数据 revision、文件校验和与 source line 清单。

## 环境和数据

参考依赖环境为 Python 3.12、PyTorch `2.12.1+cu130`、CUDA 13.0。

本期使用 `sft_mini_512.jsonl @ jingyaogong/minimind_dataset@9b06bee29c0d5685647ac11b0dcea9f572c5064a`（大小 1,232,540,940 字节，SHA256 `475039fa…`）的前 512 条固定样本。第 16～18 期使用 `sft_t2t_mini.jsonl`，两组数据互不混用。

```bash
cd chapter_19
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python prepare_data.py --samples 512
```

无法直连 Hugging Face 主站时，可显式设置兼容镜像并禁用 Xet（镜像下必需）：

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  python prepare_data.py --samples 512
```

## 运行顺序

```bash
CUDA_VISIBLE_DEVICES="" python smoke_test.py
python compare_correctness.py --output correctness-results.local.json
python benchmark.py --output benchmark-results.local.json
python train.py --output train-results.local.json
python survey_lengths.py --output corpus-lengths-cap1024.local.json
```

默认结果写入 `*.local.json`。Smoke Test 不下载模型，不占用 GPU。真实正确性和 Benchmark 使用一张 GPU。样本编码在每条命令开始时在线执行（本期不预生成 Token Shard，那是第 21 期的变量）。

## Sampler 设计

`random_plan` 用 `random.Random(f"{seed}|{epoch}|shuffle")` 打乱全部样本下标，再按 Batch Size 顺序切块。`bucketed_plan` 用同一种子得到同一排列，切成大小为 `window` 的窗口，窗口内按编码长度稳定排序（等长样本保持随机顺序），再切块，最后用同一随机流打乱 Batch 顺序。窗口必须不小于 Batch Size 且为其整数倍，否则窗口无法切成满 Batch。`window == batch_size` 时每个窗口就是一个随机块，排序不改变成员，因此分桶成员与 Baseline 完全一致——这一退化性质是正确性对照的一部分。

plan 是 `(lengths, seed, epoch, batch_size, window)` 的纯函数，结果 JSON 保存完整 plan，任何一次运行都能从记录重放。

## 指标与计时

`physical_tokens` 是 Batch 右 Padding 后的槽位数 `Σ_b |b| × max_len(b)`；`non_padding_tokens` 是 `Σ len_i`；`supervised_tokens` 按 shift 后 `labels[:, 1:] != -100` 计数。主吞吐是有效监督 Token/s，同时报告非 Padding Token/s、物理 Token/s。每条命令内的 Epoch 监督 Token 总量必须等于编码语料的总量，脚本会断言这一点；记录的物理 Token 总量还必须等于长度投影值。

数据随机性用两个度量刻画：`batch_length_spread_mean`（Batch 内最大与最小长度之差的均值，直接决定 Padding）和 `companion_repeat_ratio`（两个 Epoch 的 Batch 内共现样本对中，重复共现的比例；随机组队接近 0，成员越被长度固定越接近 1）。另报告每 Batch 物理 Token 的变异系数与最大均值比，作为 DDP 同步步长下负载不均衡的前瞻代理；本章未实测多卡。

Step Time 包含内存组 Batch、同步 H2D、Forward、Backward、全局梯度裁剪和 AdamW；不包含 JSONL 读取、Chat Template、Tokenize、plan 构造、Checkpoint 和 `nvidia-smi` 查询。每个配置先跑 4 个 Warm-up Step（消费该 Epoch plan 的前 4 个 Batch），重置峰值显存后测量其余 124 个 Step。CUDA 区间使用 Event，端到端使用同步后的 Wall Clock。


## 适用边界

- 主线模型为 `Qwen/Qwen3-0.6B-Base`，训练目标为全参数 SFT。
- `smoke_test.py` 不依赖真实模型下载或 GPU；真实权重正确性检查与 Benchmark 需要上文注明的 GPU 配置。
- 本目录只提供代码、依赖和运行说明。正确性与 Benchmark 命令会在本地生成 `*.local.json`，仓库不附带实验结果。
