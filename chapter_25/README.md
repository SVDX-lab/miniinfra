# 第 25 期：Activation Checkpointing

## 对应篇章

对应《超轻量AI Infra教程—训练篇》第 10 章；合并后的连续编号为第 25 章。

## 显卡要求

最低与参考配置均为 1 张 GPU。

第 24 期解决了大 Batch 的显存准入问题，但 autograd 仍会为 Backward 保留中间激活。本期只改变一件事：不再保存全部激活，只保存每层（段）的输入并在 Backward 时重算。本期手写 `RecomputeCheckpoint`（含 RNG 状态的保存与恢复），并与 `torch.utils.checkpoint`（`use_reentrant=False`）对照。

## 问题、假设与边界

- 问题：把中间激活改为"只存边界、Backward 重算"，能省多少显存、付出多少时间？粒度扫描呈现怎样的权衡曲线？容量边界（最大 batch/序列）右移多少？
- 假设：逐层 checkpoint 把 27/28 的层内激活换成段输入，峰值显存大幅下降；代价是 Backward 内多出约一次前向的重算。
- 唯一变量：层间激活「保存到 Backward」还是「只存边界、Backward 时重算」。
- 固定项：与第 19～24 期相同（模型/Tokenizer/数据、Assistant-only、token_mean、FP32、TF32 关、Eager Attention、AdamW loop、lr 2e-5、裁剪 1、种子 20260912、batch 4、512 Token）。
- 排除项：选择性 checkpoint 单个算子、BF16/SDPA/编译叠加、RNG 语义的形式化证明（实现保存并恢复 RNG，dropout 反例验证其必要性）。

## 文件

- `activation_checkpoint.py`：本期核心。`RecomputeCheckpoint`（手写 autograd Function：forward 存输入与 RNG，backward 恢复 RNG → `enable_grad` 重算 → 传播并返回输入梯度）；`CheckpointedQwen3Model/ForCausalLM`（按 `layers_per_segment` 分段，权重加载规则与原模型一致）。
- `training.py`：与第 37/38 期相同的共享步体——两臂只差 forward 所用的模块。
- `smoke_test.py`：dropout RNG 正反例、tiny Qwen3 逐位对照（segments=1/2）、state_dict 兼容、完整一步逐位对照。
- `compare_correctness.py`：双 Epoch 轨迹逐位对照、官方 checkpoint 对照、粒度扫描 {1,2,4,7,14,28}、batch/序列容量扫描。
- `benchmark.py`：3 次重复 × 双臂整 Epoch 计时。

## 环境和数据

参考依赖环境为 Python 3.12、PyTorch `2.12.1+cu130`、CUDA 13.0。语料固定为 `sft_mini_512.jsonl @ 9b06bee2…` 的前 512 条。

```bash
cd chapter_25
python3.12 -m venv --system-site-packages .venv
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


## 适用边界

- 主线模型为 `Qwen/Qwen3-0.6B-Base`，训练目标为全参数 SFT。
- `smoke_test.py` 不依赖真实模型下载或 GPU；真实权重正确性检查与 Benchmark 需要上文注明的 GPU 配置。
- 本目录只提供代码、依赖和运行说明。正确性与 Benchmark 命令会在本地生成 `*.local.json`，仓库不附带实验结果。
