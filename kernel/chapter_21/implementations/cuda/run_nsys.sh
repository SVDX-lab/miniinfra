#!/usr/bin/env bash
# ============================================================================
# 第 21 期 Nsight Systems 证据采集（CUDA 4090；ncu 计数器权限受限 ERR_NVGPUCTRPERM
# ——第 01 期实测，证据链 = nsys kernsum + 有效带宽对账，任务书"双口径"）。
# 在 chapter_21/ 目录执行：bash implementations/cuda/run_nsys.sh
# 产出（results/ 下）：
#   nsys-ch22-decode-n8192-b16-kernsum.txt  contig/paged-seq/paged-rand 同进程
#     对照（B=16 = 256 blocks 占用率层退出瓶颈，隔离间接寻址与局部性效应）
#   nsys-ch22-gather-n8192-kernsum.txt      gather-only 核（纯取数代价层）
#   nsys-ch22-*-apisum.txt
#   （.nsys-rep 时间线不回传仓库）
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."    # chapter_21/
mkdir -p results

echo "=== [1/3] decode N=8192 B=16 fp16（contig + paged-seq + paged-rand 同进程）==="
nsys profile -o results/nsys-ch22-decode-n8192-b16 --force-overwrite true --trace=cuda \
    python3 benchmark.py --backend cuda --decode-lens 8192 --batches 16 \
    --dtypes fp16 --variants contig,paged_seq,paged_rand \
    --out results/nsys-decode-n8192-b16.json \
    > results/nsys-ch22-decode-n8192-b16-run.log 2>&1
echo "exit=$?（bench 输出见 results/nsys-ch22-decode-n8192-b16-run.log）"

echo "=== [2/3] decode N=8192 B=1 fp16（占用率瓶颈层对照）==="
nsys profile -o results/nsys-ch22-decode-n8192-b1 --force-overwrite true --trace=cuda \
    python3 benchmark.py --backend cuda --decode-lens 8192 --batches 1 \
    --dtypes fp16 --variants contig,paged_seq,paged_rand \
    --out results/nsys-decode-n8192-b1.json \
    > results/nsys-ch22-decode-n8192-b1-run.log 2>&1
echo "exit=$?"

echo "=== [3/3] gather-only + prefill paged 对照（归因层）==="
nsys profile -o results/nsys-ch22-gather-prefill --force-overwrite true --trace=cuda \
    python3 benchmark.py --backend cuda --decode-lens 8192 --batches 1 \
    --dtypes fp16 --variants gather,prefill \
    --out results/nsys-gather-prefill.json \
    > results/nsys-ch22-gather-prefill-run.log 2>&1
echo "exit=$?"

echo "=== [4/4] nsys stats 提取 ==="
for tag in ch22-decode-n8192-b16 ch22-decode-n8192-b1 ch22-gather-prefill; do
  if [ -f "results/nsys-${tag}.nsys-rep" ]; then
    nsys stats --report cuda_gpu_kern_sum --report cuda_api_sum \
        "results/nsys-${tag}.nsys-rep" > "results/nsys-${tag}-stats.txt" 2>&1
    awk '/^cuda_gpu_kern_sum/{f=1} /^cuda_api_sum/{f=0} f' "results/nsys-${tag}-stats.txt" \
        > "results/nsys-${tag}-kernsum.txt"
    awk '/^cuda_api_sum/{f=1} f' "results/nsys-${tag}-stats.txt" \
        > "results/nsys-${tag}-apisum.txt"
    rm -f "results/nsys-${tag}.nsys-rep"
  else
    echo "警告：results/nsys-${tag}.nsys-rep 不存在（profile 可能失败，见 run.log）"
  fi
done
echo "=== 完成 ==="
head -15 results/nsys-ch22-decode-n8192-b16-kernsum.txt 2>/dev/null
