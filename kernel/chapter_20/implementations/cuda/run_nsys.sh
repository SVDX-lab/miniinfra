#!/usr/bin/env bash
# ============================================================================
# 第 20 期 Nsight Systems 证据采集（CUDA 4090；ncu 计数器权限受限
# ERR_NVGPUCTRPERM —— 第 01 期已实测，证据链 = nsys kernsum + 有效带宽对账）。
# 在 chapter_20/ 目录执行：bash implementations/cuda/run_nsys.sh
# 产出（results/ 下）：
#   nsys-ch21-prefill-n2048-kernsum.txt   prefill 核 + 物化三 kernel 满阵流量证据
#   nsys-ch21-decode-n32768-kernsum.txt   decode 核（单 kernel 吞吐证据）
#   nsys-ch21-*-apisum.txt                API 汇总（launch 开销）
#   （.nsys-rep 时间线文件不回传仓库，见大纲 §11 大文件约定）
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."    # chapter_20/
mkdir -p results

BENCH="python3 benchmark.py"

echo "=== [1/3] prefill S=2048 B=1 fp16（核 + 物化变体同进程对照）==="
nsys profile -o results/nsys-ch21-prefill-n2048 --force-overwrite true --trace=cuda \
    $BENCH --backend cuda --prefill-lens 2048 --decode-lens 2048 --batches 1 \
    --dtypes fp16 --skip-sdpa --out results/nsys-prefill-n2048.json \
    > results/nsys-ch21-prefill-n2048-run.log 2>&1
echo "exit=$?（bench 自身输出见 results/nsys-ch21-prefill-n2048-run.log）"

echo "=== [2/3] decode N=32768 B=1 fp16（单核长上下文）==="
nsys profile -o results/nsys-ch21-decode-n32768 --force-overwrite true --trace=cuda \
    $BENCH --backend cuda --prefill-lens 64 --decode-lens 32768 --batches 1 \
    --dtypes fp16 --skip-sdpa --out results/nsys-decode-n32768.json \
    > results/nsys-ch21-decode-n32768-run.log 2>&1
echo "exit=$?"

echo "=== [3/3] nsys stats 提取 ==="
for tag in ch21-prefill-n2048 ch21-decode-n32768; do
  if [ -f "results/nsys-${tag}.nsys-rep" ]; then
    nsys stats --report cuda_gpu_kern_sum --report cuda_api_sum \
        "results/nsys-${tag}.nsys-rep" > "results/nsys-${tag}-stats.txt" 2>&1
    awk '/^cuda_gpu_kern_sum/{f=1} /^cuda_api_sum/{f=0} f' "results/nsys-${tag}-stats.txt" \
        > "results/nsys-${tag}-kernsum.txt"
    awk '/^cuda_api_sum/{f=1} f' "results/nsys-${tag}-stats.txt" \
        > "results/nsys-${tag}-apisum.txt"
    rm -f "results/nsys-${tag}.nsys-rep"   # 大文件不回传
  else
    echo "警告：results/nsys-${tag}.nsys-rep 不存在（profile 可能失败，见 run.log）"
  fi
done
echo "=== 完成 ==="
head -15 results/nsys-ch21-prefill-n2048-kernsum.txt 2>/dev/null
