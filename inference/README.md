# 推理篇

## 快速开始

从仓库根目录运行以下命令。第 01 章的冒烟测试在 CPU 上使用小模型检查实现，可以先于真实模型推理运行。

```bash
cd inference
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r chapter_01/requirements-lock.txt

python chapter_01/smoke_test.py
python chapter_01/run_inference.py \
  --prompt "请用一句话介绍 KV Cache。" \
  --max-new-tokens 32
```

首次运行真实模型时会下载固定 revision 的 `Qwen/Qwen3-0.6B`。已有权重时可用 `--model /path/to/model` 指定本地目录。单卡实验以 RTX 3080 Ti（12 GB）为参考。

其他章节的安装、正确性对照和基准测试以各章 README 为准。章节说明里的「从项目根目录执行」指本篇目录 `inference/`。

## 章节索引

| 章 | 主题 |
| --- | --- |
| [01](chapter_01/) | 手写 Qwen3-0.6B 推理 |
| [02](chapter_02/) | KV Cache——第一个推理优化 |
| [03](chapter_03/) | 固定批处理 |
| [04](chapter_04/) | Continuous Batching |
| [05](chapter_05/) | Paged KV Cache |
| [06](chapter_06/) | 推理请求调度器 |
| [07](chapter_07/) | Chunked Prefill |
| [08](chapter_08/) | Prefix Cache |
| [09](chapter_09/) | 高性能 Attention |
| [10](chapter_10/) | CUDA Graph 优化 Decode |
| [11](chapter_11/) | Greedy Speculative Decoding |
| [12](chapter_12/) | KV Cache Offloading |
| [13](chapter_13/) | KV Cache 异步传输 |
| [14](chapter_14/) | 外部 KV Cache 与 LMCache |
| [15](chapter_15/) | Prefill/Decode 分离 |
