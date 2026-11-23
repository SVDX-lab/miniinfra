// ============================================================================
// 第 15 期 MUSA 迁移版（MTT S4000 / mcc 5.2.0）——ch18 同款入口包装
//
// 迁移策略：与 CUDA 版同一份源码（implementations/cuda/int8_gemm.cu），
// 全部平台差异集中在文件头 USE_MUSA 兼容 shim；本文件只是入口包装：
//     diff int8_gemm.mu ../cuda/int8_gemm.cu  → 仅此两行 define/include
// M-43 教训：入口 .mu 必须自带 #define USE_MUSA 1（不能依赖外部 -D，
// 否则走 #else 分支报 fatal error: 'cuda_runtime.h' file not found）。
// ============================================================================
#define USE_MUSA 1
#include "../cuda/int8_gemm.cu"
