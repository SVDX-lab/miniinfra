#!/usr/bin/env bash
# ============================================================================
# 第 02 期 Nsight Systems 证据采集（CUDA 4090；ncu 计数器权限受限
# ERR_NVGPUCTRPERM —— 第 01 期已实测，证据链降级为 nsys kernsum +
# 有效带宽对账口径，参照 chapter_01/performance/implementations/cuda/run_nsys.sh）。
# 在 chapter_02/ 目录执行：bash implementations/cuda/run_nsys.sh
# 产出（results/ 下）：
#   nsys-ch03-kernsum.txt      每 kernel 次数/总时长（独立 bench 进程，纯 kernel）
#   nsys-ch03-apisum.txt       CUDA API 汇总（launch 开销证据）
#   nsys-ch03-gpusum.txt       GPU 总体忙碌/间隙
#   （.nsys-rep 时间线文件不回传仓库，见大纲 §11 大文件约定）
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."    # chapter_02/
mkdir -p results

BENCH=implementations/cuda/build/rmsnorm_bench
PEAK=918.8   # ch02 实测 D2D 拷贝峰值 GB/s（达成率分母单源）

echo "=== [1/3] nsys profile 独立 bench（H=1024 rows=2048 fp16，20 次）==="
nsys profile -o results/nsys-ch03 --force-overwrite true --trace=cuda \
    "$BENCH" --h 1024 --rows 2048 --dtype fp16 --tiles 256 --warmup 3 --repeats 20 \
             --peak-gbs "$PEAK" --check true \
    > results/nsys-ch03-profile.log 2>&1
echo "profile exit=$?（bench 自身 JSON 与自检见 results/nsys-ch03-profile.log）"

echo "=== [2/3] nsys profile 标量路径（H=129）==="
nsys profile -o results/nsys-ch03-scalar --force-overwrite true --trace=cuda \
    "$BENCH" --h 129 --rows 2048 --dtype fp16 --tiles 256 --warmup 3 --repeats 20 \
             --peak-gbs "$PEAK" --check true \
    > results/nsys-ch03-profile-scalar.log 2>&1
echo "profile exit=$?"

echo "=== [3/3] nsys stats 提取 ==="
for tag in ch03 ch03-scalar; do
  nsys stats --report cuda_gpu_kern_sum --report cuda_api_sum --report gpu_sum \
      results/nsys-${tag}.nsys-rep > results/nsys-${tag}-stats.txt 2>&1
  awk '/^cuda_gpu_kern_sum/{f=1} /^cuda_api_sum/{f=0} f' results/nsys-${tag}-stats.txt \
      > results/nsys-${tag}-kernsum.txt
  awk '/^cuda_api_sum/{f=1} /^gpu_sum/{f=0} f' results/nsys-${tag}-stats.txt \
      > results/nsys-${tag}-apisum.txt
  awk '/^gpu_sum/{f=1} f' results/nsys-${tag}-stats.txt > results/nsys-${tag}-gpusum.txt
done
rm -f results/*.nsys-rep   # 大文件不回传
echo "=== 完成：results/nsys-ch03-kernsum.txt 等 ==="
head -20 results/nsys-ch03-kernsum.txt
