#!/usr/bin/env bash
# ============================================================================
# 第 05 期 MUSA 构建脚本（CUDA 版 build.sh 的迁移对应物；第 02 期模式沿用）。
#
# 构建选择记录：
#   - 编译器：mcc（clang 14 底座），SDK 在 /usr/local/musa；
#   - 独立 bench：先试 .cu 后缀，失败则复制为 build/*.mu 再编
#     （musa_extension._is_musa_file 只认 .mu/.muh —— 第 02 期实机结论）；
#   - torch 扩展路径：torch_musa.utils.musa_extension.load（build_ext.py，
#     源自动复制为 .mu）；失败则退到 rmsnorm_bench 文件 IO 对比路径并记录；
#   - Kernel 模板实例化 TILE {64,128,256,512,1024} × VEC {0,1} ×
#     dtype {fp32,fp16,bf16} × acc {fp32,fp16} = 60 个一次编入；
#   - BF16 设备类型 __mt_bfloat16（musa_bf16.h），__float2bfloat16_rn
#     显式 RNE —— 见 rmsnorm_kernel.cu 文件头差异条目。
# 用法：./build.sh            （构建 bench + torch 扩展）
#       BUILD_EXT=0 ./build.sh（只构建 bench）
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

BUILD_DIR="build"
mkdir -p "$BUILD_DIR"

echo "[build] mcc 版本："
mcc --version || true

echo "[build] mcc 构建 rmsnorm_bench（试 .cu 后缀）"
if mcc -O3 -std=c++17 \
       -o "$BUILD_DIR/rmsnorm_bench" rmsnorm_bench.cu rmsnorm_kernel.cu 2> "$BUILD_DIR/mcc_cu_err.log"; then
  echo "[build] .cu 后缀可用（mcc 接受 .cu）"
else
  echo "[build] .cu 失败（错误摘要见 $BUILD_DIR/mcc_cu_err.log），复制为 .mu 重试"
  cp rmsnorm_bench.cu "$BUILD_DIR/rmsnorm_bench.mu"
  cp rmsnorm_kernel.cu "$BUILD_DIR/rmsnorm_kernel.mu"
  mcc -O3 -std=c++17 \
      -o "$BUILD_DIR/rmsnorm_bench" "$BUILD_DIR/rmsnorm_bench.mu" "$BUILD_DIR/rmsnorm_kernel.mu"
  echo "[build] .mu 后缀可用（.cu→.mu 重命名是必需步骤）"
fi
echo "[build] $BUILD_DIR/rmsnorm_bench OK"

if [ "${BUILD_EXT:-1}" = "1" ]; then
  echo "[build] 构建 torch_musa 扩展（musa_extension 路径，源复制为 .mu）"
  if python3 build_ext.py --verbose; then
    echo "[build] torch_musa 扩展 OK"
  else
    echo "[build] torch_musa 扩展路径失败（平台边界，已记录）——实验退到 rmsnorm_bench 文件 IO 对比路径"
  fi
fi
echo "[build] 完成"
