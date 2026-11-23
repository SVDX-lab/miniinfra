// ============================================================================
// stub_syntax_check.cpp —— 本地（无 nvcc/mcc）对 paged_kernels.cuh 的语法与
// 模板实例化检查（ep21 "stub 语法检查" 模式）。
//
// 方法：把 __global__/__device__/__shared__ 等限定符 stub 成空、把 shuffle/
// syncthreads 等 device builtin stub 成平凡实现后，用普通 g++ 编译核文件并
// 显式实例化全部模板组合（dtype × warp_size × 5 核）——语法错误、类型错误、
// smem 声明错误在上服务器前暴露。kernel launch 语法（<<<>>>）只出现在
// .cu/.mu 入口文件里（本文件不包含），那部分由服务器构建验证。
// 运行：g++ -std=c++17 -I../cuda -o /tmp/stub_check stub_syntax_check.cpp && /tmp/stub_check
// ============================================================================

#define ATTN_NO_FP16_INCLUDE 1

#include <cmath>
#include <cstdio>
#include <cstring>

// ---- CUDA/MUSA 限定符与 builtin stub（仅语法检查，不执行）----
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __shared__
#define __restrict__
#define __syncthreads() ((void)0)
struct Dim3Stub { int x, y, z; };
inline Dim3Stub threadIdx{0, 0, 0};
inline Dim3Stub blockIdx{0, 0, 0};
inline Dim3Stub blockDim{128, 1, 1};
inline float __shfl_xor_sync(unsigned, float v, int) { return v; }
inline double __shfl_xor_sync(unsigned, double v, int) { return v; }
constexpr int warpSize = 32;

// ---- __half stand-in（接口对齐 cuda_fp16.h / musa_fp16.h 的两函数）----
struct HalfStub { unsigned short bits; };
inline float __half2float(HalfStub) { return 0.0f; }
inline HalfStub __float2half(float) { return HalfStub{0}; }
#define ATTN_HALF_TYPE HalfStub

#include "paged_kernels.cuh"

template struct pkg::IO<float>;
template struct pkg::IO<HalfStub>;

// 显式实例化（与 .cu/.mu 发射面一致的全组合）
template __global__ void pkg::paged_decode_kernel<float, 32>(
    const float*, const float*, const float*, const int*, const int*, float*,
    int, int, int, int, float);
template __global__ void pkg::paged_decode_kernel<float, 128>(
    const float*, const float*, const float*, const int*, const int*, float*,
    int, int, int, int, float);
template __global__ void pkg::paged_decode_kernel<HalfStub, 32>(
    const HalfStub*, const HalfStub*, const HalfStub*, const int*, const int*,
    HalfStub*, int, int, int, int, float);
template __global__ void pkg::paged_decode_kernel<HalfStub, 128>(
    const HalfStub*, const HalfStub*, const HalfStub*, const int*, const int*,
    HalfStub*, int, int, int, int, float);
template __global__ void pkg::contig_decode_kernel<float, 32>(
    const float*, const float*, const float*, const int*, float*, int, int, int, float);
template __global__ void pkg::contig_decode_kernel<float, 128>(
    const float*, const float*, const float*, const int*, float*, int, int, int, float);
template __global__ void pkg::contig_decode_kernel<HalfStub, 32>(
    const HalfStub*, const HalfStub*, const HalfStub*, const int*, HalfStub*,
    int, int, int, float);
template __global__ void pkg::contig_decode_kernel<HalfStub, 128>(
    const HalfStub*, const HalfStub*, const HalfStub*, const int*, HalfStub*,
    int, int, int, float);
template __global__ void pkg::paged_prefill_kernel<float>(
    const float*, const float*, const float*, const int*, float*,
    int, int, int, int, int, float);
template __global__ void pkg::paged_prefill_kernel<HalfStub>(
    const HalfStub*, const HalfStub*, const HalfStub*, const int*, HalfStub*,
    int, int, int, int, int, float);
template __global__ void pkg::contig_prefill_kernel<float>(
    const float*, const float*, const float*, float*, int, int, int, float);
template __global__ void pkg::contig_prefill_kernel<HalfStub>(
    const HalfStub*, const HalfStub*, const HalfStub*, HalfStub*,
    int, int, int, float);
template __global__ void pkg::gather_kv_kernel<float>(
    const float*, const float*, const int*, const int*, float*, float*,
    int, int, int, int, int, long long);
template __global__ void pkg::gather_kv_kernel<HalfStub>(
    const HalfStub*, const HalfStub*, const int*, const int*, HalfStub*, HalfStub*,
    int, int, int, int, int, long long);

int main() {
    // 静态断言：smem 预算口径（M-43：MUSA 静态上限 28672B）
    // prefill: q_s BR*D*sizeof(T) + s_s BR*BC*4 + acc BR*D*4 + 3*BR*4
    constexpr int br = pkg::BR, bc = pkg::BC, d = pkg::D;
    constexpr size_t prefill_smem_fp16 =
        static_cast<size_t>(br) * d * 2 + static_cast<size_t>(br) * bc * 4
        + static_cast<size_t>(br) * d * 4 + 3 * br * 4;
    constexpr size_t prefill_smem_fp32 = prefill_smem_fp16 + static_cast<size_t>(br) * d * 2;
    constexpr size_t musa_smem_limit = 28672;
    static_assert(prefill_smem_fp16 <= musa_smem_limit, "fp16 prefill smem 超 MUSA 限制");
    static_assert(prefill_smem_fp32 <= musa_smem_limit, "fp32 prefill smem 超 MUSA 限制");
    std::printf("[stub] paged_kernels.cuh 语法+模板实例化检查通过\n");
    std::printf("[stub] prefill smem: fp16=%zu B / fp32=%zu B（MUSA 上限 %zu B）\n",
                prefill_smem_fp16, prefill_smem_fp32, musa_smem_limit);
    constexpr size_t decode_smem_fp32 =
        static_cast<size_t>(d) * 4 + d * 4 + 2 * bc * 4 + 2 * pkg::THREADS * 4;
    std::printf("[stub] decode smem: %zu B\n", decode_smem_fp32);
    return 0;
}
