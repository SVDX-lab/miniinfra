# 第 21 期：离线 Token Shard

## 对应篇章

对应《超轻量AI Infra教程—训练篇》第 06 章；合并后的连续编号为第 21 章。

## 显卡要求

最低与参考配置均为 1 张 GPU。

第 20 期把多个样本装进同一物理序列，提高了槽位利用率。但 Chat Template 渲染与 BPE 编码仍然发生在训练的数据路径上。本期只改变 Tokenize 的执行阶段：Baseline 在线读取 JSONL、逐条渲染模板并 Tokenize；Variant 训练前一次性执行同一编码，写入带版本头、模型/数据 revision、样本 ID、input IDs、labels、长度与逐记录校验和的二进制 Shard，训练时按索引随机读取并校验。两条路径共享同一个训练 Step，Batch 计划完全相同。

## 问题、假设与边界

- 问题：把 Tokenize 移出训练热路径，能省多少数据时间与 GPU 等待？Shard 的校验与版本机制能否保证与在线编码逐 Token 一致？
- 假设：离线 Shard 可以移除训练热路径中的 Chat Template 与 BPE 编码成本；收益大小由在线编码成本相对训练 Step 的占比决定，需要通过 Benchmark 测量。
- 唯一变量：Tokenize 与 Chat Template 的执行阶段——训练在线执行与离线预处理执行。
- 固定项：与第 34/35 期相同（模型/Tokenizer/数据 revision、512 条样本、Assistant-only Labels、`token_mean`、FP32、TF32 关闭、Eager Attention、AdamW、学习率 2e-5、梯度裁剪 1、种子 20260912）；两条路径使用同一 Batch 计划（B=4，128 Batch/Epoch）。
- 排除项：多进程 DataLoader 与 GPU 预取流水（第 22 期变量）、Pinned Memory 与异步 H2D、labels 位掩码压缩、Sequence Packing 组合、BF16、梯度累积、分布式。

## 文件

- `token_shard.py`：本期核心实现。Shard 二进制格式（8 字节 magic、格式版本、40 字节模型/数据 revision、最大长度、记录数；每条记录 source_line + 长度 + input_ids(u32) + labels(i32) + FNV-1a 64 校验和；尾部自描述索引表）。`TokenShardWriter`/`TokenShardReader` 提供顺序写与 O(1) 随机读，读取端强制校验头部与校验和。
- `data_paths.py`：`OnlineCorpus`（JSONL 读取 + 模板渲染 + BPE）与 `ShardCorpus`（寻址读取 + 校验和）暴露同一 `materialize(index)` 接口；`pad_pairs` 是两条路径共享的组批函数。
- `qwen3_model.py`、`qwen3_tokenizer.py`、`assistant_loss.py`、`mini_batch.py`、`training.py`：与第 18～20 期相同的模型、Tokenizer、Loss、组批与训练步。
- `smoke_test.py`：CPU 上的读写往返、头部校验、载荷翻转检测、索引截断拒绝、随机访问与组批等价检查。
- `compare_correctness.py`：真实语料的逐 Token 一致性（512 条）、相同成员 Batch 的孪生更新、损坏检测。
- `benchmark.py`：一次性离线构建成本、冷启动首批延迟、每 Epoch 数据时间、随机访问成本、GPU Data Wait 两条路径对照。

## 环境和数据

参考依赖环境为 Python 3.12、PyTorch `2.12.1+cu130`、CUDA 13.0。语料固定为 `sft_mini_512.jsonl @ 9b06bee2…` 的前 512 条。

```bash
cd chapter_21
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 python prepare_data.py --samples 512
```

## 运行顺序

```bash
CUDA_VISIBLE_DEVICES="" python smoke_test.py
python compare_correctness.py --output correctness-results.local.json
python benchmark.py --output benchmark-results.local.json
```

## Shard 格式与校验机制

Shard 要替代的是内存中的编码结果，因此必须自带证据：头部固定 100 字节（magic `MISFTSKD`、格式版本、40 字节模型 revision、40 字节数据 revision、最大长度、记录数）；每条记录以 source_line 和长度开头，接 input_ids（u32）与 labels（i32，含 -100），以 FNV-1a 64 校验和结尾；文件尾部是「索引区起始位置」自描述指针加记录偏移表。读取端强制校验：magic 与版本不符、revision 不符、索引指针错位、载荷校验和不匹配、文件截断，一律抛错而非静默回退。FNV-1a 不是密码学哈希，它的职责是检测意外损坏，不是抵御篡改。


## 适用边界

- 主线模型为 `Qwen/Qwen3-0.6B-Base`，训练目标为全参数 SFT。
- `smoke_test.py` 不依赖真实模型下载或 GPU；真实权重正确性检查与 Benchmark 需要上文注明的 GPU 配置。
- 本目录只提供代码、依赖和运行说明。正确性与 Benchmark 命令会在本地生成 `*.local.json`，仓库不附带实验结果。
