# 第 01 期 · RMSNorm 基线与性能分析

在 CPU、CUDA 与 MUSA 上运行同一份 PyTorch RMSNorm 表达式。先与 NumPy FP32 参考结果比对，再测延迟、有效带宽和核函数启动开销。有本地 Qwen3-0.6B 权重时，还可对照逐层探针、首步 logits 与生成序列。

在仓库根目录进入 `kernel/`，按设备安装 PyTorch 后端，再安装 `chapter_01/requirements.txt` 所列依赖。用对应的后端参数执行。

```bash
BACKENDS=cpu bash chapter_01/run_all.sh
BACKENDS=cuda bash chapter_01/run_all.sh performance
```

第一个命令运行边界用例、结果比对和延迟基线，也可将 `cpu` 改为 `cuda` 或 `musa`。第二个命令运行 Roofline 基线、设备探针和逐元素加法尺寸扫描，MUSA 宿主改用 `BACKENDS=musa`。

- [评测配置](experiment-contract.yaml)和 [Qwen3 权重清单](models/MANIFEST.json)位于本目录。用 `download_model.py` 准备权重，推理参考由 `model_truth.py` 生成。默认模式将本地结果写入 `chapter_01/results/`。
- [性能诊断代码](performance/)及其[评测配置](performance/experiment-contract.yaml)已归入第一期的 `performance/` 目录，输出写入 `chapter_01/performance/results/`。CUDA 的设备时间线在该目录用 [run_nsys.sh](performance/implementations/cuda/run_nsys.sh) 采集，附加参数 `32768` 运行大工作集对照。
