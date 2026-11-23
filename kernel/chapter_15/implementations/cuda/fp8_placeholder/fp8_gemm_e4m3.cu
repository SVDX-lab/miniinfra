// ============================================================================
// 第 15 期 FP8（E4M3）weight-only 融合反量化 kernel——【框架代码，未编译未实测】
// ============================================================================
//
// 状态：4090 待修复后的回填位（implementations/cuda/fp8_placeholder/README.md）。
// 结构与 gemm_int8_fused_kernel（../int8_gemm.cu）完全同构，唯一差异是
// B-tile 加载端的数据类型与转换 intrinsic：
//   INT8:  (float)int8_val * scale[gn]          （整数→float 转换 + FP32 乘）
//   FP8:   __half2float(__nv_cvt_fp8_to_halfraw(val, __NV_E4M3)) * scale[gn]
//          （fp8→half 硬件转换指令 + FP32 乘；或 half 乘后 FMA——回填时二选一
//           并把选择与依据记录进 EXPERIMENT_LOG）
//
// 依赖：cuda_fp8.h（CUDA 12.0+ 自带；4090 CUDA 13.0 就绪）；
// 编译：-arch=sm_89（FP8 转换指令在 sm_89 Ada 可用，无 Tensor Core 依赖时
// 走 CUDA Core 转换——与本期 INT8 kernel 同"朴素 CUDA Core"口径，对照公平）。
//
// 【回填 TODO 清单】
//  1. 确认 4090 新实例 cuda_fp8.h + __nv_fp8_e4m3 可编译；
//  2. 与 int8_gemm.cu 合编（EP16_ENABLE_FP8 宏）或独立 binary；
//  3. selftest（host 参考：numpy ml_dtypes float8_e4m3fn 或手工 E4M3 解码）；
//  4. bench decode 档矩阵 + decodeblock；
//  5. FP8 量化误差层（La1_fp8）与 INT8 La1 对数。
// ============================================================================

#if defined(EP16_ENABLE_FP8)  // 整个文件仅在显式开启时参与编译

#include <cuda_fp8.h>

#include "int8_gemm.cu"  // 复用 shim/转换/计时/JSON 基建

template <int TILE_M, int TILE_N, int TILE_K, int REG_M, int REG_N>
__global__ void gemm_fp8_fused_kernel(const KERNEL_HALF_TYPE* __restrict__ A,
                                      const __nv_fp8_e4m3* __restrict__ W,
                                      const float* __restrict__ scale,
                                      KERNEL_HALF_TYPE* __restrict__ C, int M, int N, int K) {
    __shared__ float As[TILE_M][TILE_K + 1];
    __shared__ float Bs[TILE_K][TILE_N + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int nthreads = blockDim.x * blockDim.y;
    const int tid = ty * blockDim.x + tx;

    const int m0 = blockIdx.y * TILE_M;
    const int n0 = blockIdx.x * TILE_N;

    float acc[REG_M][REG_N];
#pragma unroll
    for (int m = 0; m < REG_M; ++m)
#pragma unroll
        for (int n = 0; n < REG_N; ++n) acc[m][n] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        for (int idx = tid; idx < TILE_M * TILE_K; idx += nthreads) {
            int r = idx / TILE_K;
            int c = idx % TILE_K;
            int gm = m0 + r;
            int gk = k0 + c;
            As[r][c] = (gm < M && gk < K) ? to_float_val(A[(size_t)gm * K + gk]) : 0.0f;
        }
        for (int idx = tid; idx < TILE_K * TILE_N; idx += nthreads) {
            int r = idx / TILE_N;
            int c = idx % TILE_N;
            int gk = k0 + r;
            int gn = n0 + c;
            Bs[r][c] = (gk < K && gn < N)
                           ? (__half2float(__nv_cvt_fp8_to_halfraw(
                                  W[(size_t)gk * N + gn].__x, __NV_E4M3)) *
                              scale[gn])
                           : 0.0f;
        }
        __syncthreads();
        // …… FMA 主循环与 gemm_int8_fused_kernel 逐指令相同（回填时完整展开或
        //    改成双数据类型模板——见下）……

        __syncthreads();
    }
    // 写回同 int8 版（略——回填时合并为模板）
}

// 备选设计（回填时评估）：把 gemm_int8_fused_kernel 泛化为 DequantLoad 策略
// 模板（int8→float / fp8→half→float 两个特化），避免复制 FMA 主循环——
// 但会改动已实测的 int8 kernel 源码结构，回填时需重跑 int8 selftest 保证等价。

#endif  // EP16_ENABLE_FP8
