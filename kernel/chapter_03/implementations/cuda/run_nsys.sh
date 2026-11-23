#!/usr/bin/env bash
# ============================================================================
# 第 03 期 Nsight Systems 证据采集（CUDA 4090；ncu 计数器权限受限
# ERR_NVGPUCTRPERM —— 第 01/02 期已实测，证据链降级为 nsys kernsum +
# apisum（Kernel Launch 数直接证据）+ 有效带宽对账口径）。
# 在 chapter_03/ 目录执行：bash implementations/cuda/run_nsys.sh
# 产出（results/ 下）：
#   nsys-ch04-sep-kernsum.txt   分离路径每 kernel 次数/时长（2 kernel/迭代）
#   nsys-ch04-sep-apisum.txt    分离路径 CUDA API 汇总（cudaLaunchKernel 计数）
#   nsys-ch04-fused-kernsum.txt 融合路径每 kernel 次数/时长（1 kernel/迭代）
#   nsys-ch04-fused-apisum.txt  融合路径 CUDA API 汇总（launch 计数对照）
#   （.nsys-rep 时间线文件不回传仓库，见大纲 §11 大文件约定）
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."    # chapter_03/
mkdir -p results

BENCH=implementations/cuda/build/fused_bench
PEAK=918.8   # ch02 实测 D2D 拷贝峰值 GB/s（达成率分母单源）

echo "=== [1/3] nsys profile 分离路径（H=1024 rows=2048 fp16，20 次）==="
nsys profile -o results/nsys-ch04-sep --force-overwrite true --trace=cuda \
    "$BENCH" --h 1024 --rows 2048 --dtype fp16 --mode separated --tiles 256 \
             --warmup 3 --repeats 20 --peak-gbs "$PEAK" --check true \
    > results/nsys-ch04-profile-sep.log 2>&1
echo "profile exit=$?（bench 自身 JSON 与自检见 results/nsys-ch04-profile-sep.log）"

echo "=== [2/3] nsys profile 融合路径（同形状同口径）==="
nsys profile -o results/nsys-ch04-fused --force-overwrite true --trace=cuda \
    "$BENCH" --h 1024 --rows 2048 --dtype fp16 --mode fused --tiles 256 \
             --warmup 3 --repeats 20 --peak-gbs "$PEAK" --check true \
    > results/nsys-ch04-profile-fused.log 2>&1
echo "profile exit=$?"

echo "=== [3/3] nsys stats 提取 ==="
for tag in ch04-sep ch04-fused; do
  nsys stats --report cuda_gpu_kern_sum --report cuda_api_sum \
      results/nsys-${tag}.nsys-rep > results/nsys-${tag}-stats.txt 2>&1
  # 2025.3.2 的分节标题为 " ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):"
  # （行首有空格与 **，第 02 期的 ^cuda_api_sum 锚点匹配不到——本期修正）
  awk '/CUDA GPU Kernel Summary/{f=1} /CUDA API Summary/{f=0} f' results/nsys-${tag}-stats.txt \
      > results/nsys-${tag}-kernsum.txt
  awk '/CUDA API Summary/{f=1} f' results/nsys-${tag}-stats.txt \
      > results/nsys-${tag}-apisum.txt
done
rm -f results/*.nsys-rep   # 大文件不回传
echo "=== 完成：results/nsys-ch04-{sep,fused}-{kernsum,apisum}.txt ==="
grep -E "add_kernel|rmsnorm_kernel|fused_add" results/nsys-ch04-sep-kernsum.txt results/nsys-ch04-fused-kernsum.txt | head -10
