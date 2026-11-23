#!/usr/bin/env bash
# build.sh for ai_gemm (AI draft, episode 30)
set -euo pipefail
cd "$(dirname "$0")"
ARCH="${CUDA_ARCH:-89}"
nvcc -O3 -std=c++17 -arch="sm_${ARCH}" ai_gemm.cu -o ai_gemm
echo "built ai_gemm (sm_${ARCH})"
