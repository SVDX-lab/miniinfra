#!/usr/bin/env bash
# ============================================================================
# 第 25 期 Nsight Systems 证据采集（CUDA 4090；ncu 计数器权限受限
# ERR_NVGPUCTRPERM —— 第 01 期已实测，证据链降级为 nsys kernsum +
# 有效带宽对账口径，参照 chapter_01/performance/03 的 run_nsys.sh）。
# 在 chapter_24/ 目录执行：bash implementations/cuda/run_nsys.sh
# 产出（results/ 下）：
#   nsys-ch26-core-n32768-kernsum.txt  N=32768 核级：dsa_indexer/count/compact/gather_split/merge
#                                      （自定义）vs cuBLAS bmm 系（未吸收链）
#   nsys-ch26-core-n32768-apisum.txt   API 汇总（launch 开销证据）
#   nsys-ch26-full-n2048-kernsum.txt   full step N=2048（两形式每步 kernel 数证据）
#   nsys-ch26-full-n2048-apisum.txt
#   （.nsys-rep 时间线文件不回传仓库，见大纲 §11 大文件约定）
# 注：--profile-mode core 的同一进程里先测未吸收核链再测吸收核（含 splits
# 扫描），kernsum 按 kernel 名区分。
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."    # chapter_25/
mkdir -p results

BENCH="python3 benchmark.py"

echo "=== [1/3] 核级对比 N=32768 fp16（独立进程，30 次重复）==="
nsys profile -o results/nsys-ch26-core-n32768 --force-overwrite true --trace=cuda \
    $BENCH --backend cuda --profile-mode core --kv-len 32768 --warmup 3 --repeats 30 \
    > results/nsys-ch26-core-n32768-run.log 2>&1
echo "exit=$?（bench 自身输出见 results/nsys-ch26-core-n32768-run.log）"

echo "=== [2/3] full decode step N=2048（两形式每步 kernel 数证据）==="
nsys profile -o results/nsys-ch26-full-n2048 --force-overwrite true --trace=cuda \
    $BENCH --backend cuda --profile-mode full --kv-len 2048 --warmup 3 --repeats 30 \
    > results/nsys-ch26-full-n2048-run.log 2>&1
echo "exit=$?"

echo "=== [3/3] nsys stats 提取 ==="
for tag in ch26-core-n32768 ch26-full-n2048; do
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
head -15 results/nsys-ch26-core-n32768-kernsum.txt 2>/dev/null
