#!/usr/bin/env bash
# build.sh for ai_decode_attn (AI draft, episode 30)
set -euo pipefail
cd "$(dirname "$0")"
ARCH="${CUDA_ARCH:-89}"
nvcc -O3 -std=c++17 -arch="sm_${ARCH}" ai_decode_attn.cu -o ai_decode_attn
echo "built ai_decode_attn (sm_${ARCH})"
