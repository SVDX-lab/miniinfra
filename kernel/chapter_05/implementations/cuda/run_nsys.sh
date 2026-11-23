#!/usr/bin/env bash
# ============================================================================
# 第 05 期 Nsight Systems 证据采集（CUDA 4090；ncu 计数器权限受限
# ERR_NVGPUCTRPERM —— 第 01 期已实测，证据链沿用 nsys kernsum +
# 有效带宽对账口径）。
# 在 chapter_05/ 目录执行：bash implementations/cuda/run_nsys.sh
# 产出（results/ 下）：
#   nsys-ch06-kernsum.txt      三 dtype kernel 次数/时长（独立 bench 进程）
#   nsys-ch06-apisum.txt       CUDA API 汇总
#   nsys-ch06-gpusum.txt       GPU 总体忙碌/间隙
#   （.nsys-rep 时间线文件不回传仓库，见大纲 §11 大文件约定）
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."    # chapter_05/
mkdir -p results

BENCH=implementations/cuda/build/rmsnorm_bench
PEAK=918.8   # ch02 实测 D2D 拷贝峰值 GB/s（达成率分母单源）

echo "=== [1/4] nsys profile：H=1024 rows=2048 三 dtype（L2 区，各 20 次）==="
nsys profile -o results/nsys-ch06-contract --force-overwrite true --trace=cuda \
    bash -c "\"$BENCH\" --h 1024 --rows 2048 --dtype fp32 --acc fp32 --tiles 256 --warmup 3 --repeats 20 --peak-gbs $PEAK --check true && \
             \"$BENCH\" --h 1024 --rows 2048 --dtype fp16 --acc fp32 --tiles 256 --warmup 3 --repeats 20 --peak-gbs $PEAK --check true && \
             \"$BENCH\" --h 1024 --rows 2048 --dtype bf16 --acc fp32 --tiles 256 --warmup 3 --repeats 20 --peak-gbs $PEAK --check true" \
    > results/nsys-ch06-profile-contract.log 2>&1
echo "profile exit=$?"

echo "=== [2/4] nsys profile：DRAM 区同字节对照（fp32@32768 vs fp16@65536 vs bf16@65536，H=2048）==="
nsys profile -o results/nsys-ch06-dram --force-overwrite true --trace=cuda \
    bash -c "\"$BENCH\" --h 2048 --rows 32768 --dtype fp32 --acc fp32 --tiles 256 --warmup 3 --repeats 15 --peak-gbs $PEAK --check false && \
             \"$BENCH\" --h 2048 --rows 65536 --dtype fp16 --acc fp32 --tiles 256 --warmup 3 --repeats 15 --peak-gbs $PEAK --check false && \
             \"$BENCH\" --h 2048 --rows 65536 --dtype bf16 --acc fp32 --tiles 256 --warmup 3 --repeats 15 --peak-gbs $PEAK --check false" \
    > results/nsys-ch06-profile-dram.log 2>&1
echo "profile exit=$?"

echo "=== [3/4] nsys profile：acc=fp16 对照（溢出场景 rows=4 H=1024 fp32 存储）==="
nsys profile -o results/nsys-ch06-acc --force-overwrite true --trace=cuda \
    bash -c "\"$BENCH\" --h 1024 --rows 4 --dtype fp32 --acc fp16 --tiles 256 --warmup 3 --repeats 20 --peak-gbs $PEAK --check false" \
    > results/nsys-ch06-profile-acc.log 2>&1
echo "profile exit=$?"

echo "=== [4/4] nsys stats 提取 ==="
for tag in ch06-contract ch06-dram ch06-acc; do
  nsys stats --report cuda_gpu_kern_sum --report cuda_api_sum --report gpu_sum \
      results/nsys-${tag}.nsys-rep > results/nsys-${tag}-stats.txt 2>&1
  awk '/^cuda_gpu_kern_sum/{f=1} /^cuda_api_sum/{f=0} f' results/nsys-${tag}-stats.txt \
      > results/nsys-${tag}-kernsum.txt
  awk '/^cuda_api_sum/{f=1} /^gpu_sum/{f=0} f' results/nsys-${tag}-stats.txt \
      > results/nsys-${tag}-apisum.txt
done
rm -f results/*.nsys-rep   # 大文件不回传
echo "=== 完成：results/nsys-ch06-*-kernsum.txt ==="
head -24 results/nsys-ch06-contract-kernsum.txt
