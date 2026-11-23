#!/usr/bin/env bash
# 第一期 RMSNorm 基线的 CUDA 时间线，可从任意目录运行。
#   bash run_nsys.sh          # 2048 行
#   bash run_nsys.sh 32768    # 大工作集对照
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD"
mkdir -p results

rows="${1:-2048}"
case "$rows" in
  2048)
    stem=results/nsys-rmsnorm
    iters=20
    ;;
  32768)
    stem=results/nsys-rmsnorm-large
    iters=7
    ;;
  *)
    echo "Usage: bash run_nsys.sh [2048|32768]" >&2
    exit 2
    ;;
esac

nsys profile -o "$stem" --force-overwrite true --trace=cuda \
    python3 implementations/cuda/ncu_profile_rmsnorm.py \
    --rows "$rows" --hidden 1024 --warmup 3 --iters "$iters" \
    > "$stem-profile.log" 2>&1

nsys stats --force-export=true --report cuda_gpu_kern_sum \
    "$stem.nsys-rep" > "$stem-kernsum.txt"
nsys stats --report cuda_api_sum "$stem.nsys-rep" > "$stem-apisum.txt"
if ! nsys stats --report gpu_sum "$stem.nsys-rep" > "$stem-gpusum.txt" 2>&1; then
  printf 'GPU 汇总不可用，详见 %s-gpusum.txt\n' "$stem" >&2
fi
printf '时间线与汇总已写入 %s-*\n' "$stem"
