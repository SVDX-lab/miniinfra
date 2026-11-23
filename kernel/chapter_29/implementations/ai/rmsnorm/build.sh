#!/usr/bin/env bash
# build.sh for ai_rmsnorm (AI draft, episode 30)
set -euo pipefail
cd "$(dirname "$0")"
ARCH="${CUDA_ARCH:-89}"
nvcc -O3 -std=c++17 -arch="sm_${ARCH}" ai_rmsnorm.cu -o ai_rmsnorm
echo "built ai_rmsnorm (sm_${ARCH})"
