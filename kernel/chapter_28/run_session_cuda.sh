#!/usr/bin/env bash
# 第 28 期 CUDA 列回填会话（4090 上执行；先取 cuda4090.lock）
# 用法：bash run_session_cuda.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "=== [1] torch 版本门禁（要求 2.12.1+cu130）==="
python3 - <<'EOF'
import torch
v = torch.__version__
print("torch:", v)
assert "2.12.1" in v and "cu130" in v, f"torch 版本门禁失败: {v}"
print("cuda_available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
EOF

echo "=== [2] nvcc ==="
nvcc --version | tail -3

echo "=== [3] build（nvcc 实编，sm_89）==="
bash implementations/cuda/build.sh 2>&1 | tee implementations/cuda/build-log-cuda.txt

echo "=== [4] selftest（K1-K4+G 全链 device vs host naive 对拍）==="
./implementations/cuda/moe_route --selftest 2>&1 | tee implementations/cuda/selftest.log

echo "=== [5] bench 30 格分段计时矩阵 ==="
./implementations/cuda/moe_route --bench 2>&1 | tee implementations/cuda/bench.log

echo "=== [6] imbalance：均匀 vs 80/20-Zipf ==="
./implementations/cuda/moe_route --imbalance 2>&1 | tee implementations/cuda/imbalance.log

echo "=== 完成 ==="
ls -la results/ | grep cuda || true
