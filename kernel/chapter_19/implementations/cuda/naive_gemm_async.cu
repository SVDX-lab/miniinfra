// ============================================================================
// 第 19 期：Kernel 内异步与流水——cp.async、双缓冲与 warp specialization 入门
// ============================================================================
//
// 载体（对应大纲 §13 第 19 期设计）：第 17 期 tiled GEMM（单层 tiling +
// shared memory 片上缓冲 + 寄存器分块）。本期只改"数据供给组织"这一层——
// 算法、tile 配置、block/grid 映射、计时口径、输出语义全部与第 17 期一致，
// 三级递进在同 tile 同形状下逐级对照：
//
//   level a  gemm_sync_kernel      —— 第 17 期 gemm_tiled_kernel 原样复用
//                                     （同步 load：LDG→寄存器→STS→__syncthreads）；
//   level b  gemm_cpasync1_kernel  —— cp.async 单缓冲：global→shared 异步拷贝
//                                     （绕过寄存器文件），计算前 wait_group 0；
//   level c  gemm_cpasync2_kernel  —— cp.async 双缓冲：第 i+1 块的搬运与
//                                     第 i 块的计算 overlap（wait_group 1）；
//   level c+ cpasync2v16           —— 16B 向量化 cp.async.cg 变体（+4 padding
//                                     保持 16B 行对齐；量化"4B vs 16B issue
//                                     效率"，与 bank 无关的对照见 EXPERIMENT_LOG）；
//   level d  gemm_warpspec_kernel  —— 简单 warp specialization：warp 0 专职
//                                     生产者（只发 cp.async），其余 warp 消费
//                                     （只算 FMA）；barrier 仍用 __syncthreads；
//   组合    gemm_wmma_pipe_kernel  —— 第 18 期 WMMA（计算升级）+ 本期 cp.async
//                                     双缓冲（供给升级）：FP16 输入 + FP32 累加，
//                                     TILE_M×TILE_N×KSTEP=64×64×16（16×16×16
//                                     fragment 天然对齐）；两个正交维度的实测点；
//   race 演示 cpasync2_broken / sync2_broken —— 去掉双缓冲第二个 __syncthreads
//                                     的故意竞态版（racecheck 报告形态的阳性对照，
//                                     只可通过显式 --kernel 调用，不进任何表）。
//
// cp.async 说明（sm_80+，Ada sm_89 支持）：
//   - cp.async.ca.shared.global [dst],[src],4 —— 4B 拷贝（L1 缓存路径）；
//     本期 smem 布局沿用第 17 期的 +1 padding（行距 TK+1/TN+1 floats），
//     16B 行对齐不成立，故主对照用 4B .ca——与 baseline 布局/bank 行为
//     完全一致，"只换 load 通道"归因干净；
//   - cp.async.cg.shared.global [dst],[src],16 —— 16B 拷贝（bypass L1），
//     仅用于 cpasync2v16（padding 换成 +4，行距 16B 对齐）；
//   - cp.async.commit_group / cp.async.wait_group N —— 组提交/等待；
//   - 越界：整 tile 越界检查（M/N/K 全为 TILE 倍数）才走 cp.async 快路，
//     否则回退第 17 期同步逐元素 guard 路径（正确性/边界用，基准形状
//     {256,512,1024}³ 全部整除，永远走快路）。
//
// MUSA 迁移：同一份源码，平台差异集中在文件头 shim（第 17 期方案延续）。
// 异步拷贝机制是 MUSA 侧"待实机确认"项（大纲 §14 准入第 10 条）：
//   - 编译级探测：mcc 是否接受 cp.async 内联 PTX / __cvta_generic_to_shared /
//     pipeline intrinsics（implementations/musa/async_probe/ 留全过程证据）；
//   - 探测通过 → build.sh 加 -DHAVE_MUSA_CP_ASYNC 编出 cpasync1/cpasync2；
//   - 探测失败 → 仅编译 sync 级，cp.async 记平台边界（报错全文入台账）。
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

#if defined(USE_MUSA)
  #include <musa_runtime.h>
  #if defined(__has_include)
    #if __has_include(<musa_fp16.h>)
      #include <musa_fp16.h>
      #define HAVE_MUSA_FP16_H 1
    #endif
  #endif
  // ---- GPU API 兼容 shim：CUDA 命名 → MUSA 命名（机械替换单点化，第 17 期同款）----
  #define gpuError_t          musaError_t
  #define gpuSuccess          musaSuccess
  #define gpuGetErrorString   musaGetErrorString
  #define gpuGetLastError     musaGetLastError
  #define gpuDeviceProp_t     musaDeviceProp
  #define gpuGetDeviceProperties musaGetDeviceProperties
  #define gpuMalloc           musaMalloc
  #define gpuFree             musaFree
  #define gpuMemset           musaMemset
  #define gpuMemcpy           musaMemcpy
  #define gpuMemcpyHostToHost musaMemcpyHostToHost
  #define gpuMemcpyHostToDevice musaMemcpyHostToDevice
  #define gpuMemcpyDeviceToHost musaMemcpyDeviceToHost
  #define gpuEvent_t          musaEvent_t
  #define gpuEventCreate      musaEventCreate
  #define gpuEventDestroy     musaEventDestroy
  #define gpuEventRecord      musaEventRecord
  #define gpuEventSynchronize musaEventSynchronize
  #define gpuEventElapsedTime musaEventElapsedTime
  #define gpuDeviceSynchronize musaDeviceSynchronize
  #define gpuFuncAttributes   musaFuncAttributes
  #define gpuFuncGetAttributes musaFuncGetAttributes
#else
  #include <cuda_runtime.h>
  #include <cuda_fp16.h>
  // ---- CUDA 分支：同样的 gpu* 统一命名（与 MUSA shim 对称）----
  #define gpuError_t          cudaError_t
  #define gpuSuccess          cudaSuccess
  #define gpuGetErrorString   cudaGetErrorString
  #define gpuGetLastError     cudaGetLastError
  #define gpuDeviceProp_t     cudaDeviceProp
  #define gpuGetDeviceProperties cudaGetDeviceProperties
  #define gpuMalloc           cudaMalloc
  #define gpuFree             cudaFree
  #define gpuMemset           cudaMemset
  #define gpuMemcpy           cudaMemcpy
  #define gpuMemcpyHostToHost cudaMemcpyHostToHost
  #define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
  #define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
  #define gpuEvent_t          cudaEvent_t
  #define gpuEventCreate      cudaEventCreate
  #define gpuEventDestroy     cudaEventDestroy
  #define gpuEventRecord      cudaEventRecord
  #define gpuEventSynchronize cudaEventSynchronize
  #define gpuEventElapsedTime cudaEventElapsedTime
  #define gpuDeviceSynchronize cudaDeviceSynchronize
  #define gpuFuncAttributes   cudaFuncAttributes
  #define gpuFuncGetAttributes cudaFuncGetAttributes
#endif

// ---------------------------------------------------------------------------
// cp.async 可用性门控（本期核心变量）：
//   - CUDA（sm_80+，本期 sm_89）：默认可用（-DDISABLE_CPASYNC 显式关闭）；
//   - MUSA：仅当 build.sh 的探测通过并传 -DHAVE_MUSA_CP_ASYNC 才启用
//     （探测脚本与证据在 implementations/musa/async_probe/）。
//     MUSA 实测（2026-11-21，probe-result.json）：CUDA 风格内联 PTX
//     `cp.async.*` 被 mcc 后端拒绝（"This new pipeline of instruction hasn't
//     been supported for inline asm!"）——但 <musa_pipeline.h> 提供
//     __pipeline_memcpy_async/__pipeline_commit/__pipeline_wait_prior 且
//     编译+运行全链验证通过（P4）→ MUSA 的 cp.async 对应物以 intrinsics
//     形态存在，cp_async_* helpers 走 MUSA_PIPELINE_INTRINSICS 实现。
// ---------------------------------------------------------------------------
#if !defined(USE_MUSA) && !defined(DISABLE_CPASYNC)
  #define HAVE_CP_ASYNC 1
#endif
#if defined(USE_MUSA) && defined(HAVE_MUSA_CP_ASYNC)
  #define HAVE_CP_ASYNC 1
  #if defined(__has_include)
    #if __has_include(<musa_pipeline.h>)
      #include <musa_pipeline.h>
      #define MUSA_PIPELINE_INTRINSICS 1
    #endif
  #endif
#endif

// ---------------------------------------------------------------------------
// FP16 类型与转换（第 17/18 期同款：CUDA __half；MUSA musa_fp16.h 或位级
// fallback——本期 FP16 只用于 wmma_pipe 组合核，scalar 三级保持 FP32 主线）
// ---------------------------------------------------------------------------
#if !defined(USE_MUSA) || defined(HAVE_MUSA_FP16_H)
  #define KERNEL_HALF_TYPE __half
#endif
#if defined(USE_MUSA) && !defined(HAVE_MUSA_FP16_H)
  #define KERNEL_HALF_TYPE gpu_half_bits
#endif

#if defined(USE_MUSA) && !defined(HAVE_MUSA_FP16_H)
struct gpu_half_bits { unsigned short bits; };
__host__ __device__ inline unsigned short float_to_half_rne(float value) {
    unsigned int bits;
    memcpy(&bits, &value, sizeof(bits));
    unsigned int sign = (bits >> 16) & 0x8000u;
    unsigned int exp32 = (bits >> 23) & 0xFFu;
    unsigned int mant = bits & 0x7FFFFFu;
    unsigned short out;
    if (exp32 == 0xFF) {
        out = (unsigned short)(mant ? 0xFE00u : 0x7C00u);
    } else if (exp32 == 0 && mant == 0) {
        out = 0;
    } else {
        int e = (int)exp32 - 127;
        if (e > 15) { out = 0x7C00u; }
        else if (e >= -14) {
            unsigned int hm = mant >> 13;
            unsigned int rem = mant & 0x1FFFu;
            unsigned int h = ((unsigned int)(e + 15) << 10) | hm;
            if (rem > 0x1000u || (rem == 0x1000u && (hm & 1u))) h += 1;
            out = (unsigned short)h;
        } else if (e >= -25) {
            unsigned int full = 0x800000u | mant;
            int shift = -e - 1;
            if (shift > 24) { out = 0; }
            else {
                unsigned int s = full >> shift;
                unsigned int rem = full & ((1u << shift) - 1u);
                unsigned int half_point = 1u << (shift - 1);
                if (rem > half_point || (rem == half_point && (s & 1u))) s += 1;
                out = (unsigned short)s;
            }
        } else { out = 0; }
    }
    return (unsigned short)(sign | out);
}
#endif

template <typename T>
__device__ __host__ inline T to_type(float v);
template <typename T>
__device__ __host__ inline float to_float_val(T v);

template <>
__device__ __host__ inline float to_type<float>(float v) { return v; }
template <>
__device__ __host__ inline float to_float_val<float>(float v) { return v; }

#if !defined(DISABLE_FP16)
#if defined(USE_MUSA)
#if defined(HAVE_MUSA_FP16_H)
template <> __device__ __host__ inline __half to_type<__half>(float v) { return __float2half(v); }
template <> __device__ __host__ inline float to_float_val<__half>(__half v) { return __half2float(v); }
#else
template <> __device__ __host__ inline gpu_half_bits to_type<gpu_half_bits>(float v) {
    gpu_half_bits h; h.bits = float_to_half_rne(v); return h;
}
template <> __device__ __host__ inline float to_float_val<gpu_half_bits>(gpu_half_bits v) {
    unsigned short h = v.bits;
    unsigned int sign = (unsigned int)(h & 0x8000u) << 16;
    unsigned int exp = (h >> 10) & 0x1Fu;
    unsigned int mant = h & 0x3FFu;
    unsigned int bits;
    if (exp == 0) {
        if (mant == 0) { bits = sign; }
        else {
            int e = -14;
            unsigned int m = mant;
            while ((m & 0x400u) == 0) { m <<= 1; e -= 1; }
            m &= 0x3FFu;
            bits = sign | ((unsigned int)(127 + e) << 23) | (m << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}
#endif
#else  // CUDA
template <> __device__ __host__ inline __half to_type<__half>(float v) { return __float2half(v); }
template <> __device__ __host__ inline float to_float_val<__half>(__half v) { return __half2float(v); }
#endif
#endif  // !DISABLE_FP16

// ---------------------------------------------------------------------------
// 错误检查（第 17 期同款）
// ---------------------------------------------------------------------------
#define GPU_CHECK(call)                                                        \
    do {                                                                       \
        gpuError_t err_ = (call);                                              \
        if (err_ != gpuSuccess) {                                              \
            fprintf(stderr, "[GPU-ERROR] %s 失败 (%s:%d): %s\n", #call,        \
                    __FILE__, __LINE__, gpuGetErrorString(err_));              \
            exit(2);                                                           \
        }                                                                      \
    } while (0)

static void check_launch(const char* where) {
    gpuError_t err = gpuGetLastError();
    if (err != gpuSuccess) {
        fprintf(stderr, "[GPU-ERROR] launch %s: %s\n", where, gpuGetErrorString(err));
        exit(2);
    }
}

// ---------------------------------------------------------------------------
// cp.async helpers（本期教学主体）：
//   - CUDA：内联 PTX，指令逐条可见（cp.async.ca/cg + commit_group/wait_group）；
//   - MUSA：同一 helper 接口换 <musa_pipeline.h> intrinsics 实现
//     （__pipeline_memcpy_async 尺寸 4/8/16；__pipeline_commit；
//     __pipeline_wait_prior(N) 语义 = 允许 N 组未完成，与 wait_group N 对应）。
// ---------------------------------------------------------------------------
#if defined(HAVE_CP_ASYNC)
#if defined(MUSA_PIPELINE_INTRINSICS)
// 4B 异步拷贝（MUSA intrinsics 版）
__device__ __forceinline__ void cp_async_ca4(float* smem_dst, const float* gmem_src) {
    __pipeline_memcpy_async(smem_dst, gmem_src, 4);
}
// 2×half（=4B）拷贝，wmma_pipe 组合核用
__device__ __forceinline__ void cp_async_ca4_h(KERNEL_HALF_TYPE* smem_dst,
                                               const KERNEL_HALF_TYPE* gmem_src) {
    __pipeline_memcpy_async(smem_dst, gmem_src, 4);
}
// 16B 异步拷贝（要求 dst/src 均 16B 对齐）——cpasync2v16 用
__device__ __forceinline__ void cp_async_cg16(float* smem_dst, const float* gmem_src) {
    __pipeline_memcpy_async(smem_dst, gmem_src, 16);
}
__device__ __forceinline__ void cp_async_commit_group() {
    __pipeline_commit();
}
template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    __pipeline_wait_prior(N);
}
#else  // CUDA 内联 PTX（教学主体：指令逐条可见）
// 4B 异步拷贝（.ca：L1+L2 缓存路径；4B 对齐即可——+1 padding 布局的约束下
// 与第 17 期 baseline 的 smem 布局完全一致）
__device__ __forceinline__ void cp_async_ca4(float* smem_dst, const float* gmem_src) {
    unsigned int dst = (unsigned int)__cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(dst), "l"(gmem_src));
}
// 2×half（=4B）拷贝，wmma_pipe 组合核用（half tile 的 4B 对齐块）
__device__ __forceinline__ void cp_async_ca4_h(KERNEL_HALF_TYPE* smem_dst,
                                               const KERNEL_HALF_TYPE* gmem_src) {
    unsigned int dst = (unsigned int)__cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(dst), "l"(gmem_src));
}
// 16B 异步拷贝（.cg：bypass L1；要求 dst/src 均 16B 对齐）——cpasync2v16 用
__device__ __forceinline__ void cp_async_cg16(float* smem_dst, const float* gmem_src) {
    unsigned int dst = (unsigned int)__cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(gmem_src));
}
__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;");
}
template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N));
}
#endif  // MUSA_PIPELINE_INTRINSICS
#endif  // HAVE_CP_ASYNC

// ---------------------------------------------------------------------------
// WMMA 头文件探测（第 18 期同款；wmma_pipe 组合核 = WMMA + cp.async 双缓冲）
// ---------------------------------------------------------------------------
#if !defined(DISABLE_WMMA)
  #if !defined(USE_MUSA)
    #include <mma.h>
    #define HAVE_WMMA 1
    #define WMMA_NS nvcuda
  #else
    // MUSA：第 18 期已实测 <mma.h> 存在（mtmusa::wmma）但多 warp 有 vendor
    // 缺陷（M-23）；组合核默认在 MUSA 关闭（-DHAVE_MUSA_WMMA 显式打开=复测）。
    #if defined(HAVE_MUSA_WMMA) && defined(__has_include)
      #if __has_include(<mma.h>)
        #include <mma.h>
        #define HAVE_WMMA 1
        #if !defined(WMMA_NS)
          #define WMMA_NS mtmusa
        #endif
      #endif
    #endif
  #endif
#endif
#if defined(HAVE_WMMA)
  namespace wmma_api = WMMA_NS::wmma;
#endif

// ---------------------------------------------------------------------------
// Kernel a：gemm_sync_kernel —— 第 17 期 gemm_tiled_kernel 原样复用
//（diff 审计口径：除改名与去 EPI 实例外逐行一致；EPI 模板参数保留以证明
// 逐字复制，只实例化 EPI=0。同步 load = LDG→寄存器→STS，两个 __syncthreads。）
// ---------------------------------------------------------------------------
template <typename TIn, typename TOut, int TILE_M, int TILE_N, int TILE_K, int REG_M, int REG_N,
          int EPI>
__global__ void gemm_sync_kernel(const TIn* __restrict__ A, const TIn* __restrict__ B,
                                 const float* __restrict__ bias, TOut* __restrict__ C, int M,
                                 int N, int K) {
    __shared__ float As[TILE_M][TILE_K + 1];
    __shared__ float Bs[TILE_K][TILE_N + 1];

    const int tx = threadIdx.x;  // 列方向（快维）
    const int ty = threadIdx.y;  // 行方向
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
            Bs[r][c] = (gk < K && gn < N) ? to_float_val(B[(size_t)gk * N + gn]) : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            float a_reg[REG_M];
            float b_reg[REG_N];
#pragma unroll
            for (int m = 0; m < REG_M; ++m) a_reg[m] = As[ty * REG_M + m][kk];
#pragma unroll
            for (int n = 0; n < REG_N; ++n) b_reg[n] = Bs[kk][tx * REG_N + n];
#pragma unroll
            for (int m = 0; m < REG_M; ++m)
#pragma unroll
                for (int n = 0; n < REG_N; ++n) acc[m][n] += a_reg[m] * b_reg[n];
        }
        __syncthreads();
    }

#pragma unroll
    for (int m = 0; m < REG_M; ++m) {
        int gm = m0 + ty * REG_M + m;
#pragma unroll
        for (int n = 0; n < REG_N; ++n) {
            int gn = n0 + tx * REG_N + n;
            if (gm < M && gn < N) {
                float v = acc[m][n];
                if (EPI >= 1) v += bias[gn];
                C[(size_t)gm * N + gn] = to_type<TOut>(v);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 同步逐元素加载（第 17 期 guard 路径抽出的公共形态；cp.async 核的
// 非"整 tile"回退路径用它，保证 ragged 形状正确性与 ep18 完全一致。
// PAD = smem 行 padding（+1 为第 17 期原布局；v16 变体 +4））
// ---------------------------------------------------------------------------
template <int TILE_M, int TILE_N, int TILE_K, int PAD>
__device__ __forceinline__ void sync_load_tile(float as[TILE_M][TILE_K + PAD],
                                               float bs[TILE_K][TILE_N + PAD], const float* A,
                                               const float* B, int M, int N, int K, int m0,
                                               int n0, int k0, int tid, int nthreads) {
    for (int idx = tid; idx < TILE_M * TILE_K; idx += nthreads) {
        int r = idx / TILE_K;
        int c = idx % TILE_K;
        int gm = m0 + r;
        int gk = k0 + c;
        as[r][c] = (gm < M && gk < K) ? A[(size_t)gm * K + gk] : 0.0f;
    }
    for (int idx = tid; idx < TILE_K * TILE_N; idx += nthreads) {
        int r = idx / TILE_N;
        int c = idx % TILE_N;
        int gk = k0 + r;
        int gn = n0 + c;
        bs[r][c] = (gk < K && gn < N) ? B[(size_t)gk * N + gn] : 0.0f;
    }
}

#if defined(HAVE_CP_ASYNC)
// cp.async 4B 版整 tile 加载（快路：仅当 M/N/K 全为 TILE 倍数时调用；
// 末尾 commit_group——一次"搬运提交"把 A/B 两个 tile 打进同一组）
template <int TILE_M, int TILE_N, int TILE_K>
__device__ __forceinline__ void cpasync_load_tile4(float as[TILE_M][TILE_K + 1],
                                                   float bs[TILE_K][TILE_N + 1], const float* A,
                                                   const float* B, int M, int N, int K, int m0,
                                                   int n0, int k0, int tid, int nthreads) {
    (void)M; (void)N; (void)K;  // 快路无 guard（调用方已保证整 tile 在界内）
    for (int idx = tid; idx < TILE_M * TILE_K; idx += nthreads) {
        int r = idx / TILE_K;
        int c = idx % TILE_K;
        cp_async_ca4(&as[r][c], &A[(size_t)(m0 + r) * K + k0 + c]);
    }
    for (int idx = tid; idx < TILE_K * TILE_N; idx += nthreads) {
        int r = idx / TILE_N;
        int c = idx % TILE_N;
        cp_async_ca4(&bs[r][c], &B[(size_t)(k0 + r) * N + n0 + c]);
    }
    cp_async_commit_group();
}

// cp.async 16B 版整 tile 加载（cpasync2v16：+4 padding 布局，行距 16B 对齐）
template <int TILE_M, int TILE_N, int TILE_K>
__device__ __forceinline__ void cpasync_load_tile16(float as[TILE_M][TILE_K + 4],
                                                    float bs[TILE_K][TILE_N + 4], const float* A,
                                                    const float* B, int K, int N, int m0, int n0,
                                                    int k0, int tid, int nthreads) {
    // A：每行 TILE_K floats（TK=8→32B/TK=16→64B，均为 16B 整数倍）→ 按行内 16B 块展开
    int a_row_chunks = TILE_K / 4;
    int a_chunks = TILE_M * a_row_chunks;
    int b_row_chunks = TILE_N / 4;
    int total = a_chunks + TILE_K * b_row_chunks;
    for (int ch = tid; ch < total; ch += nthreads) {
        if (ch < a_chunks) {
            int r = ch / a_row_chunks;
            int c16 = (ch % a_row_chunks) * 4;
            cp_async_cg16(&as[r][c16], &A[(size_t)(m0 + r) * K + k0 + c16]);
        } else {
            int ch2 = ch - a_chunks;
            int r = ch2 / b_row_chunks;
            int c16 = (ch2 % b_row_chunks) * 4;
            cp_async_cg16(&bs[r][c16], &B[(size_t)(k0 + r) * N + n0 + c16]);
        }
    }
    cp_async_commit_group();
}
#endif  // HAVE_CP_ASYNC

// ---------------------------------------------------------------------------
// Kernel b：gemm_cpasync1_kernel —— cp.async 单缓冲
// 同一 k-tile 内：cp.async 发起 → wait_group 0 → __syncthreads → 计算。
// 与 level a 的差异只在 load 通道（绕过寄存器文件 + 单指令等待）；
// 与 level c 的差异：搬运与计算仍串行（同缓冲，无 overlap）。
// ---------------------------------------------------------------------------
#if defined(HAVE_CP_ASYNC)
template <typename TOut, int TILE_M, int TILE_N, int TILE_K, int REG_M, int REG_N>
__global__ void gemm_cpasync1_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                     const float* __restrict__ bias, TOut* __restrict__ C, int M,
                                     int N, int K) {
    (void)bias;  // EPI=0 口径，bias 不参与
    __shared__ float As[TILE_M][TILE_K + 1];
    __shared__ float Bs[TILE_K][TILE_N + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int nthreads = blockDim.x * blockDim.y;
    const int tid = ty * blockDim.x + tx;

    const int m0 = blockIdx.y * TILE_M;
    const int n0 = blockIdx.x * TILE_N;
    const bool full = (M % TILE_M == 0) && (N % TILE_N == 0) && (K % TILE_K == 0);

    float acc[REG_M][REG_N];
#pragma unroll
    for (int m = 0; m < REG_M; ++m)
#pragma unroll
        for (int n = 0; n < REG_N; ++n) acc[m][n] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        if (full) {
            cpasync_load_tile4<TILE_M, TILE_N, TILE_K>(As, Bs, A, B, M, N, K, m0, n0, k0, tid,
                                                       nthreads);
            cp_async_wait_group<0>();  // 单缓冲：计算前等全部搬运落地
        } else {
            sync_load_tile<TILE_M, TILE_N, TILE_K, 1>(As, Bs, A, B, M, N, K, m0, n0, k0, tid,
                                                      nthreads);
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            float a_reg[REG_M];
            float b_reg[REG_N];
#pragma unroll
            for (int m = 0; m < REG_M; ++m) a_reg[m] = As[ty * REG_M + m][kk];
#pragma unroll
            for (int n = 0; n < REG_N; ++n) b_reg[n] = Bs[kk][tx * REG_N + n];
#pragma unroll
            for (int m = 0; m < REG_M; ++m)
#pragma unroll
                for (int n = 0; n < REG_N; ++n) acc[m][n] += a_reg[m] * b_reg[n];
        }
        __syncthreads();
    }

#pragma unroll
    for (int m = 0; m < REG_M; ++m) {
        int gm = m0 + ty * REG_M + m;
#pragma unroll
        for (int n = 0; n < REG_N; ++n) {
            int gn = n0 + tx * REG_N + n;
            if (gm < M && gn < N) C[(size_t)gm * N + gn] = to_type<TOut>(acc[m][n]);
        }
    }
}
#endif  // HAVE_CP_ASYNC

// ---------------------------------------------------------------------------
// Kernel c：gemm_cpasync2_kernel —— cp.async 双缓冲（本期主实现）
//   smem = 2×(As+Bs)；主循环第 i 步：发起 i+1 块搬运（commit）→ wait_group 1
//   （等第 i 块落地、允许 i+1 块仍在飞）→ __syncthreads → 从 buf[i&1] 计算
//   → __syncthreads（消费完才允许 i+2 块覆盖同一缓冲，WAR 保护）。
//   省的等待 = 第 i+1 块的 global→smem 搬运时间（原第 i+1 步开头必须干等）。
// ---------------------------------------------------------------------------
#if defined(HAVE_CP_ASYNC)
template <typename TOut, int TILE_M, int TILE_N, int TILE_K, int REG_M, int REG_N>
__global__ void gemm_cpasync2_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                     const float* __restrict__ bias, TOut* __restrict__ C, int M,
                                     int N, int K) {
    (void)bias;
    __shared__ float As[2][TILE_M][TILE_K + 1];
    __shared__ float Bs[2][TILE_K][TILE_N + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int nthreads = blockDim.x * blockDim.y;
    const int tid = ty * blockDim.x + tx;

    const int m0 = blockIdx.y * TILE_M;
    const int n0 = blockIdx.x * TILE_N;
    const bool full = (M % TILE_M == 0) && (N % TILE_N == 0) && (K % TILE_K == 0);
    const int steps = (K + TILE_K - 1) / TILE_K;

    float acc[REG_M][REG_N];
#pragma unroll
    for (int m = 0; m < REG_M; ++m)
#pragma unroll
        for (int n = 0; n < REG_N; ++n) acc[m][n] = 0.0f;

    // 预取第 0 块
    if (full) {
        cpasync_load_tile4<TILE_M, TILE_N, TILE_K>(As[0], Bs[0], A, B, M, N, K, m0, n0, 0, tid,
                                                   nthreads);
    }

    for (int i = 0; i < steps; ++i) {
        const int k0 = i * TILE_K;
        const int cur = i & 1;
        if (full) {
            const bool has_next = (i + 1 < steps);
            if (has_next) {
                cpasync_load_tile4<TILE_M, TILE_N, TILE_K>(As[cur ^ 1], Bs[cur ^ 1], A, B, M, N,
                                                           K, m0, n0, k0 + TILE_K, tid, nthreads);
                cp_async_wait_group<1>();  // 双缓冲：只等当前块，允许下一块在飞
            } else {
                cp_async_wait_group<0>();
            }
        } else {
            sync_load_tile<TILE_M, TILE_N, TILE_K, 1>(As[cur], Bs[cur], A, B, M, N, K, m0, n0,
                                                      k0, tid, nthreads);
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            float a_reg[REG_M];
            float b_reg[REG_N];
#pragma unroll
            for (int m = 0; m < REG_M; ++m) a_reg[m] = As[cur][ty * REG_M + m][kk];
#pragma unroll
            for (int n = 0; n < REG_N; ++n) b_reg[n] = Bs[cur][kk][tx * REG_N + n];
#pragma unroll
            for (int m = 0; m < REG_M; ++m)
#pragma unroll
                for (int n = 0; n < REG_N; ++n) acc[m][n] += a_reg[m] * b_reg[n];
        }
        __syncthreads();  // WAR 保护：消费完 buf[cur] 才允许 i+2 块覆盖它
    }

#pragma unroll
    for (int m = 0; m < REG_M; ++m) {
        int gm = m0 + ty * REG_M + m;
#pragma unroll
        for (int n = 0; n < REG_N; ++n) {
            int gn = n0 + tx * REG_N + n;
            if (gm < M && gn < N) C[(size_t)gm * N + gn] = to_type<TOut>(acc[m][n]);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel c+：gemm_cpasync2v16 —— 16B 向量化 cp.async.cg 双缓冲变体
//   布局差异：padding +1 → +4（行距 (TK+4)/(TN+4) floats = 16B 对齐，16B 拷贝
//   的硬要求）；bank 角度：+4 使 As 跨行读（ty*RM 步长）的 bank 步长为
//   RM*12 mod 32——reg2x2/reg4x4 配置的 warp 读模式无冲突（EXPERIMENT_LOG
//   记录验证推理），对照价值 = "每次 issue 搬 4 倍数据"的 issue 开销差。
//   仅实例化 32x32x8_reg4x4 与 64x64x8_reg4x4。
// ---------------------------------------------------------------------------
template <typename TOut, int TILE_M, int TILE_N, int TILE_K, int REG_M, int REG_N>
__global__ void gemm_cpasync2v16_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                        const float* __restrict__ bias, TOut* __restrict__ C,
                                        int M, int N, int K) {
    (void)bias;
    __shared__ __align__(16) float As[2][TILE_M][TILE_K + 4];
    __shared__ __align__(16) float Bs[2][TILE_K][TILE_N + 4];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int nthreads = blockDim.x * blockDim.y;
    const int tid = ty * blockDim.x + tx;

    const int m0 = blockIdx.y * TILE_M;
    const int n0 = blockIdx.x * TILE_N;
    const bool full = (M % TILE_M == 0) && (N % TILE_N == 0) && (K % TILE_K == 0);
    const int steps = (K + TILE_K - 1) / TILE_K;

    float acc[REG_M][REG_N];
#pragma unroll
    for (int m = 0; m < REG_M; ++m)
#pragma unroll
        for (int n = 0; n < REG_N; ++n) acc[m][n] = 0.0f;

    if (full) {
        cpasync_load_tile16<TILE_M, TILE_N, TILE_K>(As[0], Bs[0], A, B, K, N, m0, n0, 0, tid,
                                                    nthreads);
    }

    for (int i = 0; i < steps; ++i) {
        const int k0 = i * TILE_K;
        const int cur = i & 1;
        if (full) {
            const bool has_next = (i + 1 < steps);
            if (has_next) {
                cpasync_load_tile16<TILE_M, TILE_N, TILE_K>(As[cur ^ 1], Bs[cur ^ 1], A, B, K, N,
                                                            m0, n0, k0 + TILE_K, tid, nthreads);
                cp_async_wait_group<1>();
            } else {
                cp_async_wait_group<0>();
            }
        } else {
            sync_load_tile<TILE_M, TILE_N, TILE_K, 4>(As[cur], Bs[cur], A, B, M, N, K, m0, n0,
                                                      k0, tid, nthreads);
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            float a_reg[REG_M];
            float b_reg[REG_N];
#pragma unroll
            for (int m = 0; m < REG_M; ++m) a_reg[m] = As[cur][ty * REG_M + m][kk];
#pragma unroll
            for (int n = 0; n < REG_N; ++n) b_reg[n] = Bs[cur][kk][tx * REG_N + n];
#pragma unroll
            for (int m = 0; m < REG_M; ++m)
#pragma unroll
                for (int n = 0; n < REG_N; ++n) acc[m][n] += a_reg[m] * b_reg[n];
        }
        __syncthreads();
    }

#pragma unroll
    for (int m = 0; m < REG_M; ++m) {
        int gm = m0 + ty * REG_M + m;
#pragma unroll
        for (int n = 0; n < REG_N; ++n) {
            int gn = n0 + tx * REG_N + n;
            if (gm < M && gn < N) C[(size_t)gm * N + gn] = to_type<TOut>(acc[m][n]);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel d：gemm_warpspec_kernel —— 简单 warp specialization（cp.async 双缓冲）
//   1D block：warp 0 = 生产者（只发 cp.async + wait，不做 FMA）；
//   其余 warp = 消费者（只算 FMA，不碰 global）；协同仍用 __syncthreads
//   （无 mbarrier 的最简分工——mbarrier/TMA 版归第 30 期）。
//   compute 槽位 = 原二级 block 的 (TN/RN)×(TM/RM)，block = n_compute + 32 线程。
// ---------------------------------------------------------------------------
template <int TILE_M, int TILE_N, int TILE_K, int REG_M, int REG_N>
__global__ void gemm_warpspec_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                     float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[2][TILE_M][TILE_K + 1];
    __shared__ float Bs[2][TILE_K][TILE_N + 1];

    const int tid = threadIdx.x;                       // 1D
    const int n_compute = blockDim.x - 32;             // 消费者线程数
    const bool is_producer = (tid < 32);
    int tx = 0, ty = 0;
    if (!is_producer) {
        int v = tid - 32;
        tx = v % (TILE_N / REG_N);
        ty = v / (TILE_N / REG_N);
    }

    const int m0 = blockIdx.y * TILE_M;
    const int n0 = blockIdx.x * TILE_N;
    const bool full = (M % TILE_M == 0) && (N % TILE_N == 0) && (K % TILE_K == 0);
    const int steps = (K + TILE_K - 1) / TILE_K;

    float acc[REG_M][REG_N];
#pragma unroll
    for (int m = 0; m < REG_M; ++m)
#pragma unroll
        for (int n = 0; n < REG_N; ++n) acc[m][n] = 0.0f;

    if (full) {
        if (is_producer) {
            // 预取第 0 块（生产者专属；tid 直接当 load 展开序号）
            cpasync_load_tile4<TILE_M, TILE_N, TILE_K>(As[0], Bs[0], A, B, M, N, K, m0, n0, 0,
                                                       tid, 32);
        }
    } else if (!is_producer) {
        // ragged 回退：消费者自搬自算（生产者纯旁观），分配序号按消费者 tid
        sync_load_tile<TILE_M, TILE_N, TILE_K, 1>(As[0], Bs[0], A, B, M, N, K, m0, n0, 0,
                                                  tid - 32, n_compute);
    }

    for (int i = 0; i < steps; ++i) {
        const int k0 = i * TILE_K;
        const int cur = i & 1;
        if (full) {
            if (is_producer) {
                const bool has_next = (i + 1 < steps);
                if (has_next) {
                    cpasync_load_tile4<TILE_M, TILE_N, TILE_K>(As[cur ^ 1], Bs[cur ^ 1], A, B, M,
                                                               N, K, m0, n0, k0 + TILE_K, tid, 32);
                    cp_async_wait_group<1>();
                } else {
                    cp_async_wait_group<0>();
                }
            }
        } else if (!is_producer) {
            sync_load_tile<TILE_M, TILE_N, TILE_K, 1>(As[cur], Bs[cur], A, B, M, N, K, m0, n0,
                                                      k0, tid - 32, n_compute);
        }
        __syncthreads();

        if (!is_producer) {
#pragma unroll
            for (int kk = 0; kk < TILE_K; ++kk) {
                float a_reg[REG_M];
                float b_reg[REG_N];
#pragma unroll
                for (int m = 0; m < REG_M; ++m) a_reg[m] = As[cur][ty * REG_M + m][kk];
#pragma unroll
                for (int n = 0; n < REG_N; ++n) b_reg[n] = Bs[cur][kk][tx * REG_N + n];
#pragma unroll
                for (int m = 0; m < REG_M; ++m)
#pragma unroll
                    for (int n = 0; n < REG_N; ++n) acc[m][n] += a_reg[m] * b_reg[n];
            }
        }
        __syncthreads();
    }

    if (!is_producer) {
#pragma unroll
        for (int m = 0; m < REG_M; ++m) {
            int gm = m0 + ty * REG_M + m;
#pragma unroll
            for (int n = 0; n < REG_N; ++n) {
                int gn = n0 + tx * REG_N + n;
                if (gm < M && gn < N) C[(size_t)gm * N + gn] = acc[m][n];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 故意竞态演示核（racecheck 阳性对照；不进任何配置表，仅显式 --kernel 调用）
//   cpasync2_broken：双缓冲结构但【去掉第二个 __syncthreads】——迭代 i 的
//   计算未同步确认完成，迭代 i+1 的 cp.async 已向同一缓冲写入 → WAR 竞态。
//   sync2_broken：同步 load 双缓冲的同一缺陷（racecheck 必报的保守对照，
//   防"racecheck 不跟踪 cp.async 写"的报告盲区干扰归因）。
// ---------------------------------------------------------------------------
template <int TILE_M, int TILE_N, int TILE_K, int REG_M, int REG_N, bool USE_ASYNC>
__global__ void gemm_pipe2_broken_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                         float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[2][TILE_M][TILE_K + 1];
    __shared__ float Bs[2][TILE_K][TILE_N + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int nthreads = blockDim.x * blockDim.y;
    const int tid = ty * blockDim.x + tx;

    const int m0 = blockIdx.y * TILE_M;
    const int n0 = blockIdx.x * TILE_N;
    const int steps = (K + TILE_K - 1) / TILE_K;

    float acc[REG_M][REG_N];
#pragma unroll
    for (int m = 0; m < REG_M; ++m)
#pragma unroll
        for (int n = 0; n < REG_N; ++n) acc[m][n] = 0.0f;

    if (USE_ASYNC) {
        cpasync_load_tile4<TILE_M, TILE_N, TILE_K>(As[0], Bs[0], A, B, M, N, K, m0, n0, 0, tid,
                                                   nthreads);
    } else {
        sync_load_tile<TILE_M, TILE_N, TILE_K, 1>(As[0], Bs[0], A, B, M, N, K, m0, n0, 0, tid,
                                                  nthreads);
    }

    for (int i = 0; i < steps; ++i) {
        const int k0 = i * TILE_K;
        const int cur = i & 1;
        if (USE_ASYNC) {
            const bool has_next = (i + 1 < steps);
            if (has_next) {
                cpasync_load_tile4<TILE_M, TILE_N, TILE_K>(As[cur ^ 1], Bs[cur ^ 1], A, B, M, N,
                                                           K, m0, n0, k0 + TILE_K, tid, nthreads);
                cp_async_wait_group<1>();
            } else {
                cp_async_wait_group<0>();
            }
        } else {
            sync_load_tile<TILE_M, TILE_N, TILE_K, 1>(As[cur ^ 1], Bs[cur ^ 1], A, B, M, N, K,
                                                      m0, n0, k0 + TILE_K, tid, nthreads);
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            float a_reg[REG_M];
            float b_reg[REG_N];
#pragma unroll
            for (int m = 0; m < REG_M; ++m) a_reg[m] = As[cur][ty * REG_M + m][kk];
#pragma unroll
            for (int n = 0; n < REG_N; ++n) b_reg[n] = Bs[cur][kk][tx * REG_N + n];
#pragma unroll
            for (int m = 0; m < REG_M; ++m)
#pragma unroll
                for (int n = 0; n < REG_N; ++n) acc[m][n] += a_reg[m] * b_reg[n];
        }
        // 故意缺失：此处应有一个 __syncthreads() 保护 buf[cur] 的 WAR 边界
    }

#pragma unroll
    for (int m = 0; m < REG_M; ++m) {
        int gm = m0 + ty * REG_M + m;
#pragma unroll
        for (int n = 0; n < REG_N; ++n) {
            int gn = n0 + tx * REG_N + n;
            if (gm < M && gn < N) C[(size_t)gm * N + gn] = acc[m][n];
        }
    }
}
#endif  // HAVE_CP_ASYNC

// ---------------------------------------------------------------------------
// Kernel 组合：gemm_wmma_pipe_kernel —— WMMA（第 18 期计算升级）×
// cp.async 双缓冲（本期供给升级）。FP16 输入 + FP32 累加，TILE_M×TILE_N×KSTEP
// = 64×64×16（16×16×16 fragment；KSTEP=16 → 每 warp 每 k 步 1 个 k-fragment）。
// warp 布局 (TILE_M/16)×(TILE_N/16) = 4×4 = 16 warps = 512 线程。
// smem：As/Bs half 双缓冲（+8 half padding 保持 WMMA ldm=8 倍数与 16B 对齐）
// + Cs_f fp32 暂存。对照锚点 = 第 18 期 gemm_wmma_kernel（同步 load 同 tile）。
// ---------------------------------------------------------------------------
#if defined(HAVE_WMMA) && defined(HAVE_CP_ASYNC) && !defined(DISABLE_FP16)
template <int TILE_M, int TILE_N, int KSTEP>
__global__ void __launch_bounds__(512) gemm_wmma_pipe_kernel(
    const KERNEL_HALF_TYPE* __restrict__ A, const KERNEL_HALF_TYPE* __restrict__ B,
    KERNEL_HALF_TYPE* __restrict__ C, int M, int N, int K) {
    __shared__ __align__(32) KERNEL_HALF_TYPE As[2][TILE_M][KSTEP + 8];
    __shared__ __align__(32) KERNEL_HALF_TYPE Bs[2][KSTEP][TILE_N + 8];
    __shared__ __align__(32) float Cs_f[TILE_M][TILE_N];

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int warps_n = TILE_N / 16;
    const int warp_row = (tid / 32) / warps_n;
    const int warp_col = (tid / 32) % warps_n;

    const int m0 = blockIdx.y * TILE_M;
    const int n0 = blockIdx.x * TILE_N;
    const bool full = (M % TILE_M == 0) && (N % TILE_N == 0) && (K % KSTEP == 0);
    const int steps = (K + KSTEP - 1) / KSTEP;

    using acc_t = float;
    wmma_api::fragment<wmma_api::accumulator, 16, 16, 16, acc_t> acc;
    wmma_api::fill_fragment(acc, 0.0f);

    // 半 tile 加载：4B = 2 half 块粒度（A 行 KSTEP*2B=32B、B 行 TILE_N*2B 均
    // 为 4B 对齐；+8 padding 行距 16B 对齐——4B 拷贝无硬要求，取整块粒度）
    const int a_elems = TILE_M * KSTEP;
    const int b_elems = KSTEP * TILE_N;
    auto load_async = [&](int buf, int k0) {
        const int total_chunks = (a_elems + b_elems) / 2;
        for (int ch = tid; ch < total_chunks; ch += nthreads) {
            if (ch < a_elems / 2) {
                int r = (ch * 2) / KSTEP;
                int c = (ch * 2) % KSTEP;
                cp_async_ca4_h(&As[buf][r][c], &A[(size_t)(m0 + r) * K + k0 + c]);
            } else {
                int ch2 = (ch * 2 - a_elems);
                int r = ch2 / TILE_N;
                int c = ch2 % TILE_N;
                cp_async_ca4_h(&Bs[buf][r][c], &B[(size_t)(k0 + r) * N + n0 + c]);
            }
        }
        cp_async_commit_group();
    };
    auto load_sync = [&](int buf, int k0) {
        for (int idx = tid; idx < a_elems; idx += nthreads) {
            int r = idx / KSTEP;
            int c = idx % KSTEP;
            int gm = m0 + r;
            int gk = k0 + c;
            As[buf][r][c] =
                (gm < M && gk < K) ? A[(size_t)gm * K + gk] : to_type<KERNEL_HALF_TYPE>(0.0f);
        }
        for (int idx = tid; idx < b_elems; idx += nthreads) {
            int r = idx / TILE_N;
            int c = idx % TILE_N;
            int gk = k0 + r;
            int gn = n0 + c;
            Bs[buf][r][c] =
                (gk < K && gn < N) ? B[(size_t)gk * N + gn] : to_type<KERNEL_HALF_TYPE>(0.0f);
        }
    };

    if (full) load_async(0, 0);

    for (int i = 0; i < steps; ++i) {
        const int k0 = i * KSTEP;
        const int cur = i & 1;
        if (full) {
            const bool has_next = (i + 1 < steps);
            if (has_next) {
                load_async(cur ^ 1, k0 + KSTEP);
                cp_async_wait_group<1>();
            } else {
                cp_async_wait_group<0>();
            }
        } else {
            load_sync(cur, k0);
        }
        __syncthreads();

        wmma_api::fragment<wmma_api::matrix_a, 16, 16, 16, KERNEL_HALF_TYPE, wmma_api::row_major>
            a_frag;
        wmma_api::fragment<wmma_api::matrix_b, 16, 16, 16, KERNEL_HALF_TYPE, wmma_api::row_major>
            b_frag;
        wmma_api::load_matrix_sync(a_frag, &As[cur][warp_row * 16][0], KSTEP + 8);
        wmma_api::load_matrix_sync(b_frag, &Bs[cur][0][warp_col * 16], TILE_N + 8);
        wmma_api::mma_sync(acc, a_frag, b_frag, acc);
        __syncthreads();
    }

    wmma_api::store_matrix_sync(&Cs_f[warp_row * 16][warp_col * 16], acc, TILE_N,
                                wmma_api::mem_row_major);
    __syncthreads();
    for (int idx = tid; idx < TILE_M * TILE_N; idx += nthreads) {
        int r = idx / TILE_N;
        int c = idx % TILE_N;
        int gm = m0 + r;
        int gn = n0 + c;
        if (gm < M && gn < N) C[(size_t)gm * N + gn] = to_type<KERNEL_HALF_TYPE>(Cs_f[r][c]);
    }
}
#endif  // HAVE_WMMA && HAVE_CP_ASYNC && !DISABLE_FP16

// ---------------------------------------------------------------------------
// 随机填充 kernel（第 17 期同款 LCG hash，bench/selftest 数据，确定性）
// ---------------------------------------------------------------------------
__host__ __device__ inline unsigned int hash_u32(size_t i, unsigned long long seed) {
    unsigned int x = (unsigned int)(i * 2654435761ULL) ^ (unsigned int)(seed & 0xFFFFFFFFu) ^
                     (unsigned int)((seed >> 32) & 0xFFFFu);
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    x *= 0x846CA68Bu;
    x ^= x >> 16;
    return x;
}

__global__ void fill_random_float(float* out, size_t count, unsigned long long seed) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        out[idx] = ((float)hash_u32(idx, seed) / 2147483648.0f) - 1.0f;
    }
}

#if !defined(DISABLE_FP16)
__global__ void fill_random_half(KERNEL_HALF_TYPE* out, size_t count, unsigned long long seed) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        out[idx] = to_type<KERNEL_HALF_TYPE>(((float)hash_u32(idx, seed) / 2147483648.0f) - 1.0f);
    }
}
#endif

// ---------------------------------------------------------------------------
// 配置注册表（第 17 期 tile 配置族的 ep20 相关子集；Python 侧单源
// async_gemm_configs.py 与之严格一致）
// ---------------------------------------------------------------------------
struct Config {
    const char* name;  // 例如 "32x32x8_reg4x4"
    int tm, tn, tk;
    int rm, rn;
};

static const std::vector<Config>& config_table() {
    static const std::vector<Config> table = {
        {"16x16x16_reg2x2", 16, 16, 16, 2, 2},
        {"16x16x16_reg4x4", 16, 16, 16, 4, 4},
        {"32x32x8_reg2x2", 32, 32, 8, 2, 2},
        {"32x32x8_reg4x4", 32, 32, 8, 4, 4},
        {"32x64x8_reg4x4", 32, 64, 8, 4, 4},
        {"64x32x8_reg4x4", 64, 32, 8, 4, 4},
        {"64x64x8_reg2x2", 64, 64, 8, 2, 2},
        {"64x64x8_reg4x4", 64, 64, 8, 4, 4},
    };
    return table;
}

static const Config* find_config(const std::string& tile, const std::string& reg) {
    std::string want = tile + "_reg" + reg;
    for (const auto& c : config_table())
        if (want == c.name) return &c;
    return nullptr;
}

// smem（单缓冲，+1 padding 布局）：与 ep18 smem_bytes 公式一致
static size_t smem_bytes_of(const Config& c) {
    return (size_t)(c.tm * (c.tk + 1) + c.tk * (c.tn + 1)) * sizeof(float);
}
// v16 变体（+4 padding）
static size_t smem_bytes_v16(const Config& c) {
    return (size_t)(c.tm * (c.tk + 4) + c.tk * (c.tn + 4)) * sizeof(float);
}

// kernel 类型（CLI --kernel 取值）
//  sync / cpasync1 / cpasync2 / cpasync2v16 / warpspec / wmma_pipe
//  sync2_broken / cpasync2_broken（race 演示）
static int kernel_kind_from(const std::string& k) {
    if (k == "sync") return 0;
    if (k == "cpasync1") return 1;
    if (k == "cpasync2") return 2;
    if (k == "cpasync2v16") return 3;
    if (k == "warpspec") return 4;
    if (k == "wmma_pipe") return 5;
    if (k == "sync2_broken") return 6;
    if (k == "cpasync2_broken") return 7;
    return -1;
}
static const char* kernel_name_of(int kind) {
    static const char* names[] = {"sync",       "cpasync1",        "cpasync2",     "cpasync2v16",
                                  "warpspec",   "wmma_pipe",       "sync2_broken", "cpasync2_broken"};
    return names[kind];
}

// ---------------------------------------------------------------------------
// 分发（模板实例化表；MUSA 无 cp.async 时 1/2/3/4/7 类不存在 → 显式报错）
// ---------------------------------------------------------------------------
#define CONFIG_MATCH(cfg, TM, TN, TK, RM, RN) \
    ((cfg).tm == TM && (cfg).tn == TN && (cfg).tk == TK && (cfg).rm == RM && (cfg).rn == RN)

static void unsupported_kernel_exit(const char* kind, const char* why) {
    fprintf(stderr, "[ERROR] kernel=%s 在本二进制不可用：%s\n", kind, why);
    exit(2);
}

// 该 kernel 类是否已在本二进制编译（MUSA 无 cp.async 时 1/2/3/4/7 缺席）
static bool kernel_compiled(int kind) {
    switch (kind) {
        case 0: return true;  // sync 永远可用
        case 1:
        case 2:
        case 3:
        case 4:
        case 6:
        case 7:
#if defined(HAVE_CP_ASYNC)
            return true;
#else
            return false;
#endif
        case 5:
#if defined(HAVE_WMMA) && defined(HAVE_CP_ASYNC) && !defined(DISABLE_FP16)
            return true;
#else
            return false;
#endif
        default: return false;
    }
}

template <typename TOut>
static bool launch_scalar(int kind, const Config& cfg, dim3 grid, dim3 block, const float* dA,
                          const float* dB, float* dC, int M, int N, int K) {
#define TRY_SYNC(TM, TN, TK, RM, RN)                                                   \
    if (kind == 0 && CONFIG_MATCH(cfg, TM, TN, TK, RM, RN)) {                          \
        gemm_sync_kernel<float, TOut, TM, TN, TK, RM, RN, 0>                           \
            <<<grid, block>>>(dA, dB, nullptr, dC, M, N, K);                           \
        return true;                                                                   \
    }
    TRY_SYNC(16, 16, 16, 2, 2);
    TRY_SYNC(16, 16, 16, 4, 4);
    TRY_SYNC(32, 32, 8, 2, 2);
    TRY_SYNC(32, 32, 8, 4, 4);
    TRY_SYNC(32, 64, 8, 4, 4);
    TRY_SYNC(64, 32, 8, 4, 4);
    TRY_SYNC(64, 64, 8, 2, 2);
    TRY_SYNC(64, 64, 8, 4, 4);
#undef TRY_SYNC
#if defined(HAVE_CP_ASYNC)
#define TRY_CP1(TM, TN, TK, RM, RN)                                                    \
    if (kind == 1 && CONFIG_MATCH(cfg, TM, TN, TK, RM, RN)) {                          \
        gemm_cpasync1_kernel<TOut, TM, TN, TK, RM, RN>                                 \
            <<<grid, block>>>(dA, dB, nullptr, dC, M, N, K);                           \
        return true;                                                                   \
    }
#define TRY_CP2(TM, TN, TK, RM, RN)                                                    \
    if (kind == 2 && CONFIG_MATCH(cfg, TM, TN, TK, RM, RN)) {                          \
        gemm_cpasync2_kernel<TOut, TM, TN, TK, RM, RN>                                 \
            <<<grid, block>>>(dA, dB, nullptr, dC, M, N, K);                           \
        return true;                                                                   \
    }
#define TRY_CP2V16(TM, TN, TK, RM, RN)                                                 \
    if (kind == 3 && CONFIG_MATCH(cfg, TM, TN, TK, RM, RN)) {                          \
        gemm_cpasync2v16_kernel<TOut, TM, TN, TK, RM, RN>                              \
            <<<grid, block>>>(dA, dB, nullptr, dC, M, N, K);                           \
        return true;                                                                   \
    }
    TRY_CP1(16, 16, 16, 2, 2);
    TRY_CP1(16, 16, 16, 4, 4);
    TRY_CP1(32, 32, 8, 2, 2);
    TRY_CP1(32, 32, 8, 4, 4);
    TRY_CP1(32, 64, 8, 4, 4);
    TRY_CP1(64, 32, 8, 4, 4);
    TRY_CP1(64, 64, 8, 2, 2);
    TRY_CP1(64, 64, 8, 4, 4);
    TRY_CP2(16, 16, 16, 2, 2);
    TRY_CP2(16, 16, 16, 4, 4);
    TRY_CP2(32, 32, 8, 2, 2);
    TRY_CP2(32, 32, 8, 4, 4);
    TRY_CP2(32, 64, 8, 4, 4);
    TRY_CP2(64, 32, 8, 4, 4);
    TRY_CP2(64, 64, 8, 2, 2);
    TRY_CP2(64, 64, 8, 4, 4);
    TRY_CP2V16(32, 32, 8, 4, 4);
    TRY_CP2V16(64, 64, 8, 4, 4);
#undef TRY_CP1
#undef TRY_CP2
#undef TRY_CP2V16
#else
    (void)dA; (void)dB; (void)dC; (void)M; (void)N; (void)K;
#endif
    return false;
}

static bool launch_warpspec(const Config& cfg, dim3 grid, unsigned nthreads_all, const float* dA,
                            const float* dB, float* dC, int M, int N, int K) {
#if !defined(HAVE_CP_ASYNC)
    (void)cfg; (void)grid; (void)nthreads_all; (void)dA; (void)dB; (void)dC; (void)M; (void)N; (void)K;
    unsupported_kernel_exit("warpspec", "本构建未启用 cp.async（依赖生产者 warp 的异步搬运）");
    return false;
#else
#define TRY_WS(TM, TN, TK, RM, RN)                                                     \
    if (CONFIG_MATCH(cfg, TM, TN, TK, RM, RN)) {                                       \
        gemm_warpspec_kernel<TM, TN, TK, RM, RN>                                       \
            <<<grid, nthreads_all>>>(dA, dB, dC, M, N, K);                             \
        return true;                                                                   \
    }
    TRY_WS(16, 16, 16, 2, 2);
    TRY_WS(16, 16, 16, 4, 4);
    TRY_WS(32, 32, 8, 2, 2);
    TRY_WS(32, 32, 8, 4, 4);
    TRY_WS(32, 64, 8, 4, 4);
    TRY_WS(64, 32, 8, 4, 4);
    // 注：64x64x8_reg2x2（1024 计算槽 + 32 生产者 = 1056 > 1024 上限）不实例化
    TRY_WS(64, 64, 8, 4, 4);
#undef TRY_WS
#endif
    return false;
}

static bool launch_broken(int kind, const Config& cfg, dim3 grid, dim3 block, const float* dA,
                          const float* dB, float* dC, int M, int N, int K) {
#if !defined(HAVE_CP_ASYNC)
    (void)kind; (void)cfg; (void)grid; (void)block; (void)dA; (void)dB; (void)dC; (void)M; (void)N; (void)K;
    unsupported_kernel_exit(kernel_name_of(kind), "本构建未启用 cp.async");
    return false;
#else
#define TRY_BROKEN(TM, TN, TK, RM, RN)                                                 \
    if (CONFIG_MATCH(cfg, TM, TN, TK, RM, RN)) {                                       \
        if (kind == 6)                                                                 \
            gemm_pipe2_broken_kernel<TM, TN, TK, RM, RN, false>                        \
                <<<grid, block>>>(dA, dB, dC, M, N, K);                                \
        else                                                                           \
            gemm_pipe2_broken_kernel<TM, TN, TK, RM, RN, true>                         \
                <<<grid, block>>>(dA, dB, dC, M, N, K);                                \
        return true;                                                                   \
    }
    TRY_BROKEN(32, 32, 8, 4, 4);
    TRY_BROKEN(64, 64, 8, 4, 4);
#undef TRY_BROKEN
#endif
    return false;
}

#if defined(HAVE_WMMA) && defined(HAVE_CP_ASYNC) && !defined(DISABLE_FP16)
static bool launch_wmma_pipe(dim3 grid, unsigned nthreads, const KERNEL_HALF_TYPE* dA,
                             const KERNEL_HALF_TYPE* dB, KERNEL_HALF_TYPE* dC, int M, int N,
                             int K) {
    if (M % 64 || N % 64 || K % 16) return false;  // 组合核只走整 tile（快路演示）
    gemm_wmma_pipe_kernel<64, 64, 16><<<grid, nthreads>>>(dA, dB, dC, M, N, K);
    return true;
}
#endif

// ---------------------------------------------------------------------------
// 计时（第 17 期同款：事件对、P50/P95/mean/min、全样本）
// ---------------------------------------------------------------------------
struct TimingStats {
    int warmup = 0;
    int repeats = 0;
    float p50_ms = 0, p95_ms = 0, mean_ms = 0, min_ms = 0;
    std::vector<float> samples_ms;
};

template <typename LaunchFn>
static TimingStats time_launches(int warmup, int repeats, LaunchFn launch_once) {
    for (int i = 0; i < warmup; ++i) launch_once();
    GPU_CHECK(gpuDeviceSynchronize());

    std::vector<gpuEvent_t> starts(repeats), ends(repeats);
    for (int r = 0; r < repeats; ++r) {
        GPU_CHECK(gpuEventCreate(&starts[r]));
        GPU_CHECK(gpuEventCreate(&ends[r]));
    }
    for (int r = 0; r < repeats; ++r) {
        GPU_CHECK(gpuEventRecord(starts[r], 0));
        launch_once();
        GPU_CHECK(gpuEventRecord(ends[r], 0));
    }
    GPU_CHECK(gpuDeviceSynchronize());

    TimingStats st;
    st.warmup = warmup;
    st.repeats = repeats;
    st.samples_ms.resize(repeats);
    for (int r = 0; r < repeats; ++r) {
        float ms = 0.f;
        GPU_CHECK(gpuEventElapsedTime(&ms, starts[r], ends[r]));
        st.samples_ms[r] = ms;
        GPU_CHECK(gpuEventDestroy(starts[r]));
        GPU_CHECK(gpuEventDestroy(ends[r]));
    }
    std::vector<float> sorted = st.samples_ms;
    std::sort(sorted.begin(), sorted.end());
    st.min_ms = sorted.front();
    st.p50_ms = sorted[repeats / 2];
    int p95_idx = std::min(repeats - 1, (int)std::lround(repeats * 0.95) - 1);
    p95_idx = std::max(p95_idx, 0);
    st.p95_ms = sorted[p95_idx];
    double sum = 0;
    for (float v : st.samples_ms) sum += v;
    st.mean_ms = (float)(sum / repeats);
    return st;
}

static std::string fnum(float v) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%.6f", v);
    return buf;
}

// ---------------------------------------------------------------------------
// grid/block 推导（与 ep18 相同的映射：block=(TN/RN, TM/RM)，warp-spec 例外）
// ---------------------------------------------------------------------------
static void launch_geometry(int kind, const Config& cfg, int m, int n, int k, dim3& grid,
                            dim3& block, unsigned& nthreads_1d) {
    const int bx = cfg.tn / cfg.rn;
    const int by = cfg.tm / cfg.rm;
    if (kind == 4) {  // warpspec：1D = bx*by + 32
        nthreads_1d = (unsigned)(bx * by + 32);
        block = dim3(nthreads_1d, 1, 1);
    } else if (kind == 5) {  // wmma_pipe：4×4 warp = 512
        nthreads_1d = 512;
        block = dim3(nthreads_1d, 1, 1);
    } else {
        block = dim3(bx, by);
        nthreads_1d = 0;
    }
    grid = dim3((unsigned)((n + cfg.tn - 1) / cfg.tn), (unsigned)((m + cfg.tm - 1) / cfg.tm));
    (void)k;
}

static size_t smem_total_of(int kind, const Config& cfg) {
    if (kind == 3) return smem_bytes_v16(cfg) * 2;
    if (kind == 1 || kind == 0) return smem_bytes_of(cfg);
    if (kind == 5) {
        // As/Bs half 双缓冲 + Cs_f fp32
        return (size_t)(2 * (cfg.tm * (16 + 8) + 16 * (cfg.tn + 8)) * 2) +
               (size_t)cfg.tm * cfg.tn * 4;
    }
    return smem_bytes_of(cfg) * 2;  // cpasync2 / warpspec / broken 双缓冲
}

// ---------------------------------------------------------------------------
// bench 模式：单（shape × kernel × tile）测延迟，JSON 单行输出
// ---------------------------------------------------------------------------
static void mode_bench(const std::map<std::string, std::string>& args) {
    int m = std::stoi(args.at("--m"));
    int n = std::stoi(args.at("--n"));
    int k = std::stoi(args.at("--k"));
    std::string tile = args.count("--tile") ? args.at("--tile") : "32x32x8";
    std::string reg = args.count("--reg") ? args.at("--reg") : "4x4";
    std::string kernel_s = args.count("--kernel") ? args.at("--kernel") : "sync";
    std::string dtype = args.count("--dtype") ? args.at("--dtype") : "fp32";
    int warmup = args.count("--warmup") ? std::stoi(args.at("--warmup")) : 10;
    int repeats = args.count("--repeats") ? std::stoi(args.at("--repeats")) : 50;
    unsigned long long seed = args.count("--seed") ? std::stoull(args.at("--seed")) : 20261118ULL;

    const int kind = kernel_kind_from(kernel_s);
    if (kind < 0) {
        fprintf(stderr, "[ERROR] 未知 kernel=%s\n", kernel_s.c_str());
        exit(2);
    }
    const Config* cfg = find_config(tile, reg);
    if (!cfg && kind != 5) {
        fprintf(stderr, "[ERROR] 未知配置 tile=%s reg=%s\n", tile.c_str(), reg.c_str());
        exit(2);
    }
    if (!kernel_compiled(kind)) {
        unsupported_kernel_exit(kernel_s.c_str(),
                                "本构建未编译该 kernel 类（cp.async 或 WMMA 探测/门控未启用，"
                                "逐 kernel 结论见 probe-result.json / KNOWN_ISSUES）");
    }

    gpuDeviceProp_t prop;
    GPU_CHECK(gpuGetDeviceProperties(&prop, 0));

    const bool use_half = (dtype == "fp16");
    if (use_half && kind != 5) {
        fprintf(stderr, "[ERROR] 本期 scalar 三级为 FP32 主线，fp16 仅用于 wmma_pipe\n");
        exit(2);
    }
    if (!use_half && kind == 5) {
        fprintf(stderr, "[ERROR] wmma_pipe 为 FP16 输入核，请 --dtype fp16\n");
        exit(2);
    }

    size_t a_bytes = (size_t)m * k * (use_half ? 2 : 4);
    size_t b_bytes = (size_t)k * n * (use_half ? 2 : 4);
    size_t c_bytes = (size_t)m * n * (use_half ? 2 : 4);

    char* dA;
    char* dB;
    char* dC;
    GPU_CHECK(gpuMalloc(&dA, a_bytes));
    GPU_CHECK(gpuMalloc(&dB, b_bytes));
    GPU_CHECK(gpuMalloc(&dC, c_bytes));
    {
        dim3 fb(256);
        if (use_half) {
#if !defined(DISABLE_FP16)
            fill_random_half<<<dim3((unsigned)(((size_t)m * k + 255) / 256)), fb>>>(
                (KERNEL_HALF_TYPE*)dA, (size_t)m * k, seed);
            check_launch("fill A");
            fill_random_half<<<dim3((unsigned)(((size_t)k * n + 255) / 256)), fb>>>(
                (KERNEL_HALF_TYPE*)dB, (size_t)k * n, seed + 1);
            check_launch("fill B");
#endif
        } else {
            fill_random_float<<<dim3((unsigned)(((size_t)m * k + 255) / 256)), fb>>>(
                (float*)dA, (size_t)m * k, seed);
            check_launch("fill A");
            fill_random_float<<<dim3((unsigned)(((size_t)k * n + 255) / 256)), fb>>>(
                (float*)dB, (size_t)k * n, seed + 1);
            check_launch("fill B");
        }
        GPU_CHECK(gpuDeviceSynchronize());
    }

    dim3 grid, block;
    unsigned nthreads_1d = 0;
    launch_geometry(kind, cfg ? *cfg : config_table()[7], m, n, k, grid, block, nthreads_1d);

    TimingStats st{};
    if (kind == 0) {
        st = time_launches(warmup, repeats, [&] {
            if (!launch_scalar<float>(0, *cfg, grid, block, (const float*)dA, (const float*)dB,
                                      (float*)dC, m, n, k)) {
                fprintf(stderr, "[ERROR] 配置 %s 未实例化\n", cfg->name);
                exit(2);
            }
            check_launch("sync launch");
        });
    } else if (kind == 1 || kind == 2 || kind == 3) {
        st = time_launches(warmup, repeats, [&] {
            if (!launch_scalar<float>(kind, *cfg, grid, block, (const float*)dA, (const float*)dB,
                                      (float*)dC, m, n, k)) {
                fprintf(stderr, "[ERROR] 配置 %s 未实例化（kernel=%s）\n", cfg->name,
                        kernel_name_of(kind));
                exit(2);
            }
            check_launch("cpasync launch");
        });
    } else if (kind == 4) {
        st = time_launches(warmup, repeats, [&] {
            if (!launch_warpspec(*cfg, grid, nthreads_1d, (const float*)dA, (const float*)dB,
                                 (float*)dC, m, n, k)) {
                fprintf(stderr, "[ERROR] 配置 %s 未实例化（warpspec）\n", cfg->name);
                exit(2);
            }
            check_launch("warpspec launch");
        });
    } else if (kind == 5) {
#if defined(HAVE_WMMA) && defined(HAVE_CP_ASYNC) && !defined(DISABLE_FP16)
        st = time_launches(warmup, repeats, [&] {
            if (!launch_wmma_pipe(grid, nthreads_1d, (const KERNEL_HALF_TYPE*)dA,
                                  (const KERNEL_HALF_TYPE*)dB, (KERNEL_HALF_TYPE*)dC, m, n, k)) {
                fprintf(stderr, "[ERROR] wmma_pipe 仅支持 M/N 为 64 倍数、K 为 16 倍数\n");
                exit(2);
            }
            check_launch("wmma_pipe launch");
        });
#else
        (void)nthreads_1d; (void)grid;
        unsupported_kernel_exit("wmma_pipe", "本构建无 WMMA+cp.async 组合（MUSA 边界或显式禁用）");
#endif
    } else if (kind == 6 || kind == 7) {
        st = time_launches(warmup, repeats, [&] {
            if (!launch_broken(kind, *cfg, grid, block, (const float*)dA, (const float*)dB,
                               (float*)dC, m, n, k)) {
                fprintf(stderr, "[ERROR] broken 核仅实例化 32x32x8/64x64x8 reg4x4\n");
                exit(2);
            }
            check_launch("broken launch");
        });
    }

    double flops = 2.0 * m * n * k;
    double bytes = ((double)m * k + (double)k * n + (double)m * n) * (use_half ? 2 : 4);
    int steps = (k + (kind == 5 ? 16 : cfg->tk) - 1) / (kind == 5 ? 16 : cfg->tk);

    std::string line;
    line += "{\"mode\":\"bench\"";
    line += ",\"backend_binary\":\"async_gemm_bench\"";
    line += ",\"episode\":20";
    line += ",\"dtype\":\"" + dtype + "\"";
    line += ",\"kernel\":\"" + std::string(kernel_name_of(kind)) + "\"";
    line += ",\"tile\":\"" + tile + "\"";
    line += ",\"reg\":\"" + reg + "\"";
    char tmp[512];
    snprintf(tmp, sizeof(tmp),
             ",\"m\":%d,\"n\":%d,\"k\":%d,\"k_steps\":%d,\"grid\":[%u,%u],\"block\":[%u,%u]", m, n,
             k, steps, grid.x, grid.y, block.x, block.y);
    line += tmp;
    snprintf(tmp, sizeof(tmp), ",\"warps_per_block\":%u",
             (block.x * block.y + 31) / 32);
    line += tmp;
    snprintf(tmp, sizeof(tmp), ",\"smem_bytes_static\":%zu,\"smem_buffers\":%s",
             smem_total_of(kind, cfg ? *cfg : config_table()[7]),
             (kind == 2 || kind == 3 || kind == 4 || kind == 6 || kind == 7) ? "2" : "1");
    line += tmp;
    snprintf(tmp, sizeof(tmp), ",\"flops\":%.0f,\"nominal_bytes\":%.0f", flops, bytes);
    line += tmp;
    snprintf(tmp, sizeof(tmp), ",\"warmup\":%d,\"repeats\":%d", st.warmup, st.repeats);
    line += tmp;
    line += ",\"p50_ms\":" + fnum(st.p50_ms);
    line += ",\"p95_ms\":" + fnum(st.p95_ms);
    line += ",\"mean_ms\":" + fnum(st.mean_ms);
    line += ",\"min_ms\":" + fnum(st.min_ms);
    line += ",\"samples_ms\":[";
    for (size_t i = 0; i < st.samples_ms.size(); ++i) {
        line += fnum(st.samples_ms[i]);
        if (i + 1 < st.samples_ms.size()) line += ",";
    }
    line += "]";
    snprintf(tmp, sizeof(tmp), ",\"tflops_p50\":%.4f", flops / (st.p50_ms / 1000.0) / 1e12);
    line += tmp;
    snprintf(tmp, sizeof(tmp), ",\"effective_gb_s_p50\":%.2f", bytes / (st.p50_ms / 1000.0) / 1e9);
    line += tmp;
    line += ",\"device_name\":\"" + std::string(prop.name) + "\"";
    snprintf(tmp, sizeof(tmp), ",\"sm_count\":%d,\"warp_size\":%d", prop.multiProcessorCount,
             prop.warpSize);
    line += tmp;
    line += "}";
    printf("%s\n", line.c_str());

    GPU_CHECK(gpuFree(dA));
    GPU_CHECK(gpuFree(dB));
    GPU_CHECK(gpuFree(dC));
}

// ---------------------------------------------------------------------------
// run 模式：单次执行（fixtures 正确性链；raw bytes 文件进出，第 17 期同口径）
// ---------------------------------------------------------------------------
static std::vector<char> read_file_bytes(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "[ERROR] 无法打开 %s\n", path.c_str());
        exit(2);
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<char> data((size_t)size);
    if (size > 0 && fread(data.data(), 1, (size_t)size, f) != (size_t)size) {
        fprintf(stderr, "[ERROR] 读取 %s 失败\n", path.c_str());
        exit(2);
    }
    fclose(f);
    return data;
}

static void write_file_bytes(const std::string& path, const void* data, size_t size) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "[ERROR] 无法写入 %s\n", path.c_str());
        exit(2);
    }
    fwrite(data, 1, size, f);
    fclose(f);
}

static void mode_run(const std::map<std::string, std::string>& args) {
    int m = std::stoi(args.at("--m"));
    int n = std::stoi(args.at("--n"));
    int k = std::stoi(args.at("--k"));
    std::string kernel_s = args.count("--kernel") ? args.at("--kernel") : "sync";
    std::string tile = args.count("--tile") ? args.at("--tile") : "32x32x8";
    std::string reg = args.count("--reg") ? args.at("--reg") : "4x4";
    std::string dtype = args.count("--dtype") ? args.at("--dtype") : "fp32";

    const int kind = kernel_kind_from(kernel_s);
    if (kind < 0) {
        fprintf(stderr, "[ERROR] 未知 kernel=%s\n", kernel_s.c_str());
        exit(2);
    }
    const Config* cfg = find_config(tile, reg);
    if (!cfg && kind != 5) {
        fprintf(stderr, "[ERROR] 未知配置 tile=%s reg=%s\n", tile.c_str(), reg.c_str());
        exit(2);
    }
    if (!kernel_compiled(kind)) {
        unsupported_kernel_exit(kernel_s.c_str(), "本构建未编译该 kernel 类");
    }
    const bool use_half = (dtype == "fp16");
    if (use_half && kind != 5) {
        fprintf(stderr, "[ERROR] 本期 scalar 三级为 FP32 主线，fp16 仅用于 wmma_pipe\n");
        exit(2);
    }
    if (!use_half && kind == 5) {
        fprintf(stderr, "[ERROR] wmma_pipe 为 FP16 输入核，请 --dtype fp16\n");
        exit(2);
    }

    std::vector<char> a_data = read_file_bytes(args.at("--in-a"));
    std::vector<char> b_data = read_file_bytes(args.at("--in-b"));
    size_t expect_ab = (size_t)m * k * (use_half ? 2 : 4);
    size_t expect_bb = (size_t)k * n * (use_half ? 2 : 4);
    if (a_data.size() != expect_ab || b_data.size() != expect_bb) {
        fprintf(stderr, "[ERROR] 输入尺寸不符：A=%zu（期望 %zu） B=%zu（期望 %zu）\n",
                a_data.size(), expect_ab, b_data.size(), expect_bb);
        exit(2);
    }

    char* dA;
    char* dB;
    char* dC;
    size_t total = (size_t)m * n;
    GPU_CHECK(gpuMalloc(&dA, expect_ab));
    GPU_CHECK(gpuMalloc(&dB, expect_bb));
    GPU_CHECK(gpuMalloc(&dC, total * (use_half ? 2 : 4)));
    GPU_CHECK(gpuMemcpy(dA, a_data.data(), expect_ab, gpuMemcpyHostToDevice));
    GPU_CHECK(gpuMemcpy(dB, b_data.data(), expect_bb, gpuMemcpyHostToDevice));

    dim3 grid, block;
    unsigned nthreads_1d = 0;
    launch_geometry(kind, cfg ? *cfg : config_table()[7], m, n, k, grid, block, nthreads_1d);

    if (kind == 0 || kind == 1 || kind == 2 || kind == 3) {
        if (!launch_scalar<float>(kind, *cfg, grid, block, (const float*)dA, (const float*)dB,
                                  (float*)dC, m, n, k)) {
            fprintf(stderr, "[ERROR] 配置 %s 未实例化（kernel=%s）\n", cfg->name,
                    kernel_name_of(kind));
            exit(2);
        }
    } else if (kind == 4) {
        if (!launch_warpspec(*cfg, grid, nthreads_1d, (const float*)dA, (const float*)dB,
                             (float*)dC, m, n, k)) {
            fprintf(stderr, "[ERROR] 配置 %s 未实例化（warpspec）\n", cfg->name);
            exit(2);
        }
    } else if (kind == 5) {
#if defined(HAVE_WMMA) && defined(HAVE_CP_ASYNC) && !defined(DISABLE_FP16)
        if (!use_half) {
            fprintf(stderr, "[ERROR] wmma_pipe 需要 --dtype fp16\n");
            exit(2);
        }
        if (!launch_wmma_pipe(grid, nthreads_1d, (const KERNEL_HALF_TYPE*)dA,
                              (const KERNEL_HALF_TYPE*)dB, (KERNEL_HALF_TYPE*)dC, m, n, k)) {
            fprintf(stderr, "[ERROR] wmma_pipe 仅支持 M/N%%64==0 且 K%%16==0（fixtures 请用整除形状）\n");
            exit(2);
        }
#else
        unsupported_kernel_exit("wmma_pipe", "本构建无 WMMA+cp.async 组合");
#endif
    } else {
        if (!launch_broken(kind, *cfg, grid, block, (const float*)dA, (const float*)dB,
                           (float*)dC, m, n, k)) {
            fprintf(stderr, "[ERROR] broken 核仅实例化 32x32x8/64x64x8 reg4x4\n");
            exit(2);
        }
    }
    check_launch("run launch");
    GPU_CHECK(gpuDeviceSynchronize());

    std::vector<char> out(total * (use_half ? 2 : 4));
    GPU_CHECK(gpuMemcpy(out.data(), dC, out.size(), gpuMemcpyDeviceToHost));
    write_file_bytes(args.at("--out-c"), out.data(), out.size());

    GPU_CHECK(gpuFree(dA));
    GPU_CHECK(gpuFree(dB));
    GPU_CHECK(gpuFree(dC));
}

// ---------------------------------------------------------------------------
// selftest 模式：host 三重循环 vs 各 kernel（FP32）+ wmma_pipe 设备内对照
// ragged 形状（17×19×13）强制走各核的同步回退路径——回退正确性在此锁定。
// ---------------------------------------------------------------------------
static void mode_selftest(const std::map<std::string, std::string>& args) {
    (void)args;
    struct Shape {
        int m, k, n;
    };
    std::vector<Shape> shapes = {{32, 24, 40}, {17, 19, 13}, {4, 8, 4}, {64, 64, 64}};
    int passed = 0, failed = 0;

    for (const auto& s : shapes) {
        size_t count_a = (size_t)s.m * s.k, count_b = (size_t)s.k * s.n;
        std::vector<float> hA(count_a), hB(count_b);
        unsigned long long seed = 20261118ULL + s.m * 131 + s.k * 17 + s.n;
        for (size_t i = 0; i < count_a; ++i)
            hA[i] = ((float)hash_u32(i, seed) / 2147483648.0f) - 1.0f;
        for (size_t i = 0; i < count_b; ++i)
            hB[i] = ((float)hash_u32(i, seed + 1) / 2147483648.0f) - 1.0f;
        std::vector<float> hRef((size_t)s.m * s.n, 0.0f);
        for (int i = 0; i < s.m; ++i)
            for (int p = 0; p < s.k; ++p) {
                float a = hA[(size_t)i * s.k + p];
                for (int j = 0; j < s.n; ++j) hRef[(size_t)i * s.n + j] += a * hB[(size_t)p * s.n + j];
            }

        float *dA, *dB, *dC;
        GPU_CHECK(gpuMalloc(&dA, count_a * 4));
        GPU_CHECK(gpuMalloc(&dB, count_b * 4));
        GPU_CHECK(gpuMalloc(&dC, (size_t)s.m * s.n * 4));
        GPU_CHECK(gpuMemcpy(dA, hA.data(), count_a * 4, gpuMemcpyHostToDevice));
        GPU_CHECK(gpuMemcpy(dB, hB.data(), count_b * 4, gpuMemcpyHostToDevice));

        for (int kind = 0; kind <= 4; ++kind) {  // sync/cpasync1/cpasync2/v16/warpspec
            if (!kernel_compiled(kind)) {
                printf("SELFTEST kernel=%s SKIP（本构建未编译该 kernel 类）\n",
                       kernel_name_of(kind));
                continue;
            }
            for (const auto& cfg : config_table()) {
                if (kind == 3 && !(std::string(cfg.name) == "32x32x8_reg4x4" ||
                                   std::string(cfg.name) == "64x64x8_reg4x4"))
                    continue;  // v16 只实例化两个配置
                dim3 grid, block;
                unsigned nth1d = 0;
                launch_geometry(kind, cfg, s.m, s.n, s.k, grid, block, nth1d);
                bool ok_launch = false;
                if (kind == 4) {
                    ok_launch = launch_warpspec(cfg, grid, nth1d, dA, dB, dC, s.m, s.n, s.k);
                } else {
                    ok_launch = launch_scalar<float>(kind, cfg, grid, block, dA, dB, dC, s.m, s.n,
                                                     s.k);
                }
                if (!ok_launch) {
                    printf("SELFTEST kernel=%s config=%s shape=%dx%dx%d SKIP\n",
                           kernel_name_of(kind), cfg.name, s.m, s.k, s.n);
                    continue;
                }
                check_launch("selftest launch");
                GPU_CHECK(gpuDeviceSynchronize());
                std::vector<float> out((size_t)s.m * s.n);
                GPU_CHECK(gpuMemcpy(out.data(), dC, out.size() * 4, gpuMemcpyDeviceToHost));
                float max_abs = 0;
                for (size_t i = 0; i < out.size(); ++i)
                    max_abs = std::max(max_abs, fabsf(out[i] - hRef[i]));
                bool ok = max_abs < 1e-3f;
                printf("SELFTEST kernel=%s config=%s shape=%dx%dx%d max_abs=%.3e %s\n",
                       kernel_name_of(kind), cfg.name, s.m, s.k, s.n, max_abs, ok ? "PASS" : "FAIL");
                ok ? passed++ : failed++;
                GPU_CHECK(gpuMemset(dC, 0, (size_t)s.m * s.n * 4));
            }
        }
        GPU_CHECK(gpuFree(dA));
        GPU_CHECK(gpuFree(dB));
        GPU_CHECK(gpuFree(dC));
    }

// wmma_pipe：fp16 输入无 host 真值路径（与第 18 期同口径），用设备内 sync
// 核（FP32 主线）作对照，容差覆盖 fp16 输入量化（K=64 量级）
#if defined(HAVE_WMMA) && defined(HAVE_CP_ASYNC) && !defined(DISABLE_FP16)
    {
        // fp16 输入无 host half 真值路径：fill_random_half 与 host hA 用同一
        // hash（同 seed 同值），host 三重循环 FP32 结果即真值；容差覆盖 fp16
        // 输入量化（|acc| ≤ K×4 量级）
        int m = 64, kk = 64, n = 64;
        size_t ca = (size_t)m * kk, cb = (size_t)kk * n;
        std::vector<float> hRef((size_t)m * n, 0.0f);
        unsigned long long seed = 20261118ULL + 991;
        for (int i = 0; i < m; ++i)
            for (int p = 0; p < kk; ++p) {
                float a = ((float)hash_u32((size_t)i * kk + p, seed) / 2147483648.0f) - 1.0f;
                for (int j = 0; j < n; ++j) {
                    float b = ((float)hash_u32((size_t)p * n + j, seed + 1) / 2147483648.0f) - 1.0f;
                    hRef[(size_t)i * n + j] += a * b;
                }
            }
        KERNEL_HALF_TYPE *dAh, *dBh, *dCh;
        GPU_CHECK(gpuMalloc(&dAh, ca * 2));
        GPU_CHECK(gpuMalloc(&dBh, cb * 2));
        GPU_CHECK(gpuMalloc(&dCh, (size_t)m * n * 2));
        dim3 fb(256);
        fill_random_half<<<dim3((unsigned)((ca + 255) / 256)), fb>>>(dAh, ca, seed);
        check_launch("selftest fill half A");
        fill_random_half<<<dim3((unsigned)((cb + 255) / 256)), fb>>>(dBh, cb, seed + 1);
        check_launch("selftest fill half B");

        dim3 wgrid((unsigned)((n + 63) / 64), (unsigned)((m + 63) / 64));
        launch_wmma_pipe(wgrid, 512, dAh, dBh, dCh, m, n, kk);
        check_launch("selftest wmma_pipe");
        GPU_CHECK(gpuDeviceSynchronize());
        std::vector<KERNEL_HALF_TYPE> outh((size_t)m * n);
        GPU_CHECK(gpuMemcpy(outh.data(), dCh, outh.size() * 2, gpuMemcpyDeviceToHost));
        float max_abs = 0, max_rel = 0;
        for (size_t i = 0; i < outh.size(); ++i) {
            float v = to_float_val(outh[i]);
            float d = fabsf(v - hRef[i]);
            max_abs = std::max(max_abs, d);
            max_rel = std::max(max_rel, d / std::max(fabsf(hRef[i]), 1.0f));
        }
        bool ok = (max_abs < 0.35f) && (max_rel < 0.06f);  // fp16 输入量化 + K=64 累加
        printf("SELFTEST kernel=wmma_pipe config=64x64x16 shape=64x64x64 max_abs=%.3e max_rel=%.3e %s\n",
               max_abs, max_rel, ok ? "PASS" : "FAIL");
        ok ? passed++ : failed++;
        GPU_CHECK(gpuFree(dAh));
        GPU_CHECK(gpuFree(dBh));
        GPU_CHECK(gpuFree(dCh));
    }
#endif

    printf("{\"mode\":\"selftest\",\"passed\":%d,\"failed\":%d}\n", passed, failed);
    if (failed > 0) exit(3);
}

// ---------------------------------------------------------------------------
// attrs 模式：全部实例的寄存器/共享内存占用（流水版 bookkeeping 寄存器开销证据）
// ---------------------------------------------------------------------------
struct AttrEntry {
    std::string name;
    const void* func;
    unsigned threads;
    size_t smem;
};

static std::vector<AttrEntry> collect_attr_entries() {
    std::vector<AttrEntry> v;
    const auto& table = config_table();
    for (const auto& c : table) {
        char nm[96];
#define ADD(kind_id, ptr)                                     \
    {                                                         \
        snprintf(nm, sizeof(nm), "%s/%s", kernel_name_of(kind_id), c.name); \
        v.push_back({nm, (const void*)ptr, 0, smem_total_of(kind_id, c)});   \
    }
#define ADD_SYNC(TM, TN, TK, RM, RN) \
    if (CONFIG_MATCH(c, TM, TN, TK, RM, RN)) ADD(0, (gemm_sync_kernel<float, float, TM, TN, TK, RM, RN, 0>))
        ADD_SYNC(16, 16, 16, 2, 2); ADD_SYNC(16, 16, 16, 4, 4);
        ADD_SYNC(32, 32, 8, 2, 2); ADD_SYNC(32, 32, 8, 4, 4);
        ADD_SYNC(32, 64, 8, 4, 4); ADD_SYNC(64, 32, 8, 4, 4);
        ADD_SYNC(64, 64, 8, 2, 2); ADD_SYNC(64, 64, 8, 4, 4);
#undef ADD_SYNC
#if defined(HAVE_CP_ASYNC)
#define ADD_CP1(TM, TN, TK, RM, RN) \
    if (CONFIG_MATCH(c, TM, TN, TK, RM, RN)) ADD(1, (gemm_cpasync1_kernel<float, TM, TN, TK, RM, RN>))
#define ADD_CP2(TM, TN, TK, RM, RN) \
    if (CONFIG_MATCH(c, TM, TN, TK, RM, RN)) ADD(2, (gemm_cpasync2_kernel<float, TM, TN, TK, RM, RN>))
        ADD_CP1(16, 16, 16, 2, 2); ADD_CP1(16, 16, 16, 4, 4);
        ADD_CP1(32, 32, 8, 2, 2); ADD_CP1(32, 32, 8, 4, 4);
        ADD_CP1(32, 64, 8, 4, 4); ADD_CP1(64, 32, 8, 4, 4);
        ADD_CP1(64, 64, 8, 2, 2); ADD_CP1(64, 64, 8, 4, 4);
        ADD_CP2(16, 16, 16, 2, 2); ADD_CP2(16, 16, 16, 4, 4);
        ADD_CP2(32, 32, 8, 2, 2); ADD_CP2(32, 32, 8, 4, 4);
        ADD_CP2(32, 64, 8, 4, 4); ADD_CP2(64, 32, 8, 4, 4);
        ADD_CP2(64, 64, 8, 2, 2); ADD_CP2(64, 64, 8, 4, 4);
        if (std::string(c.name) == "32x32x8_reg4x4")
            ADD(3, (gemm_cpasync2v16_kernel<float, 32, 32, 8, 4, 4>));
        if (std::string(c.name) == "64x64x8_reg4x4")
            ADD(3, (gemm_cpasync2v16_kernel<float, 64, 64, 8, 4, 4>));
        {   // warpspec
            snprintf(nm, sizeof(nm), "warpspec/%s", c.name);
            const void* f = nullptr;
            if (CONFIG_MATCH(c, 16, 16, 16, 2, 2)) f = (const void*)(gemm_warpspec_kernel<16, 16, 16, 2, 2>);
            if (CONFIG_MATCH(c, 16, 16, 16, 4, 4)) f = (const void*)(gemm_warpspec_kernel<16, 16, 16, 4, 4>);
            if (CONFIG_MATCH(c, 32, 32, 8, 2, 2)) f = (const void*)(gemm_warpspec_kernel<32, 32, 8, 2, 2>);
            if (CONFIG_MATCH(c, 32, 32, 8, 4, 4)) f = (const void*)(gemm_warpspec_kernel<32, 32, 8, 4, 4>);
            if (CONFIG_MATCH(c, 32, 64, 8, 4, 4)) f = (const void*)(gemm_warpspec_kernel<32, 64, 8, 4, 4>);
            if (CONFIG_MATCH(c, 64, 32, 8, 4, 4)) f = (const void*)(gemm_warpspec_kernel<64, 32, 8, 4, 4>);
            if (CONFIG_MATCH(c, 64, 64, 8, 4, 4)) f = (const void*)(gemm_warpspec_kernel<64, 64, 8, 4, 4>);
            if (f) v.push_back({nm, f, 0, smem_total_of(4, c)});
        }
        if (std::string(c.name) == "32x32x8_reg4x4") {
            ADD(6, (gemm_pipe2_broken_kernel<32, 32, 8, 4, 4, false>));
            ADD(7, (gemm_pipe2_broken_kernel<32, 32, 8, 4, 4, true>));
        }
#undef ADD_CP1
#undef ADD_CP2
#endif
#undef ADD
    }
#if defined(HAVE_WMMA) && defined(HAVE_CP_ASYNC) && !defined(DISABLE_FP16)
    {
        Config c{"64x64x16", 64, 64, 16, 16, 16};
        v.push_back({"wmma_pipe/64x64x16",
                     (const void*)(gemm_wmma_pipe_kernel<64, 64, 16>), 512, smem_total_of(5, c)});
    }
#endif
    return v;
}

static void mode_attrs(const std::map<std::string, std::string>& args) {
    (void)args;
    std::string json = "{\"mode\":\"attrs\",\"cp_async_enabled\":";
#if defined(HAVE_CP_ASYNC)
    json += "true";
#else
    json += "false";
#endif
#if defined(HAVE_WMMA)
    json += ",\"wmma_enabled\":true";
#else
    json += ",\"wmma_enabled\":false";
#endif
    json += ",\"configs\":[";
    bool first = true;
    for (const auto& e : collect_attr_entries()) {
        char buf[512];
#if !defined(DISABLE_FUNC_ATTRS)
        gpuFuncAttributes attr;
        gpuError_t err = gpuFuncGetAttributes(&attr, e.func);
        if (err == gpuSuccess) {
            snprintf(buf, sizeof(buf),
                     "%s{\"name\":\"%s\",\"num_regs\":%d,\"shared_bytes_static\":%zu,"
                     "\"shared_bytes_attr\":%zu,\"local_bytes\":%zu,\"max_threads_per_block\":%d}",
                     first ? "" : ",", e.name.c_str(), attr.numRegs, e.smem,
                     (size_t)attr.sharedSizeBytes, (size_t)attr.localSizeBytes,
                     attr.maxThreadsPerBlock);
        } else
#endif
        {
            snprintf(buf, sizeof(buf),
                     "%s{\"name\":\"%s\",\"num_regs\":null,\"shared_bytes_static\":%zu,"
                     "\"shared_bytes_attr\":null,\"local_bytes\":null,\"max_threads_per_block\":null}",
                     first ? "" : ",", e.name.c_str(), e.smem);
        }
        json += buf;
        first = false;
    }
    json += "]}";
    printf("%s\n", json.c_str());
}

// ---------------------------------------------------------------------------
// main：极简 key=value 参数解析（第 17 期同款）
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr,
                "用法: async_gemm_bench <selftest|attrs|run|bench> [--key value ...]\n");
        return 1;
    }
    std::string mode = argv[1];
    std::map<std::string, std::string> args;
    for (int i = 2; i + 1 < argc; i += 2) args[argv[i]] = argv[i + 1];

    if (mode == "selftest")
        mode_selftest(args);
    else if (mode == "attrs")
        mode_attrs(args);
    else if (mode == "run")
        mode_run(args);
    else if (mode == "bench")
        mode_bench(args);
    else {
        fprintf(stderr, "[ERROR] 未知模式 %s\n", mode.c_str());
        return 1;
    }
    gpuError_t err = gpuGetLastError();
    if (err != gpuSuccess) {
        fprintf(stderr, "[GPU-ERROR] 尾部遗留错误: %s\n", gpuGetErrorString(err));
        return 2;
    }
    return 0;
}
