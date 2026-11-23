// ============================================================================
// rmsnorm_kernel.cu —— 第 06 期 MUSA 手写 RMSNorm（第 02 期实机验证版原样复用；
//   本期在 ext 层实现三合同策略，本文件与第 02 期逐字一致，仅文件头注明复用）
//
// ✅ API 命名已实机核对（2026-11-21，/usr/local/musa/include 头文件 grep）：
//   1. <musa_runtime.h> 对应 <cuda_runtime.h> —— 存在 ✅；
//   2. <musa_fp16.h> 对应 <cuda_fp16.h> —— 存在 ✅；
//   3. musaError_t / musaGetErrorString / musaGetLastError 对应
//      cudaError_t / cudaGetErrorString / cudaGetLastError —— 存在 ✅；
//      （协议 §6 提示的 musMalloc/musMemcpy 等 mus 前缀实机不存在，
//      全部为 musa 前缀，已按实机修正）
//   4. musaStream_t 对应 cudaStream_t —— 存在 ✅。
//   Kernel 本体（__global__ / __shfl_xor_sync / __syncthreads / rsqrtf /
//   float4 / uint4 / <<<>>> 启动语法）预期零改动 —— mcc 编译结果为最终
//   验证（待编译核对项见 MUSA_PORT_NOTES.md #26/#27）。
//
// 组织方式与 CUDA 版逐条对应（见 CUDA 版文件头）：
//   每 block 一行 + grid-stride；__shfl_xor_sync warp 归约；shared 跨 warp
//   归约；FP32 累加；对齐时 16B 向量化（FP32: float4，FP16: uint4）；
//   块内 stride 循环支持 H=127~2048 非整 Tile；x 行缓存 shared。
// ============================================================================

#include <musa_runtime.h>  // ✅ 实机确认存在（对应 <cuda_runtime.h>）
#include <musa_fp16.h>     // ✅ 实机确认存在（对应 <cuda_fp16.h>）

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace mini_infra {

// ---------------------------------------------------------------------------
// 错误检查（host 侧）。CUDA→MUSA 机械替换：cudaError_t→musaError_t，
// cudaGetErrorString→musaGetErrorString（✅ 实机确认）。
// ---------------------------------------------------------------------------
#define RMSNORM_MUSA_CHECK(expr)                                       \
    do {                                                               \
        musaError_t _err = (expr);                                     \
        if (_err != musaSuccess) {                                     \
            throw std::runtime_error(std::string("MUSA error at ") +   \
                                     __FILE__ + ":" +                  \
                                     std::to_string(__LINE__) + " " +  \
                                     #expr + ": " +                    \
                                     musaGetErrorString(_err));         \
        }                                                              \
    } while (0)

// 合同：支持的最大 H（超出则 Pass 2 回退重读 global x，仍正确）。
constexpr int kMaxCacheElems = 2048;
// grid-stride 一次发射的最大 block 数（行数更多时循环接管）。
constexpr long kMaxGridBlocks = 1L << 20;

// ---------------------------------------------------------------------------
// 向量化访存适配层：FP32 用 float4（4 元素/16B），FP16 用 uint4（8 half/16B）。
// 与 CUDA 版逐字相同（预期零改动）。
// ---------------------------------------------------------------------------
template <typename T>
struct VecTraits;

template <>
struct VecTraits<float> {
    static constexpr int kWidth = 4;   // 每次向量化访存的元素数
    using Pack = float4;               // 16B 载荷
};

template <>
struct VecTraits<__half> {
    static constexpr int kWidth = 8;
    using Pack = uint4;                // 8 个 half 的 16B 载荷
};

// FP32 载荷 <-> 标量。
__device__ __forceinline__ float pack_get(const float4& p, int k) {
    return (&p.x)[k];
}

__device__ __forceinline__ void pack_set(float4& p, int k, float v) {
    (&p.x)[k] = v;
}

// FP16 载荷 <-> 标量（reinterpret 型双关：uint4 16B 对齐，8 half 连续排布；
// 比 union 双关对编译器更友好，mcc/nvcc 均接受）。
__device__ __forceinline__ float pack_get(const uint4& p, int k) {
    return __half2float(reinterpret_cast<const __half*>(&p)[k]);
}

// 值参数为 __half（CUDA 13 起无 __half→float 隐式转换，调用方经
// cast_from_float<T> 已转换到 T；float 路径走 float4 重载）。
__device__ __forceinline__ void pack_set(uint4& p, int k, __half v) {
    reinterpret_cast<__half*>(&p)[k] = v;
}

// dtype 适配：读入转 FP32 / 写回从 FP32 舍入到 T。
__device__ __forceinline__ float to_float(float v) { return v; }
__device__ __forceinline__ float to_float(__half v) { return __half2float(v); }

template <typename T>
__device__ __forceinline__ T cast_from_float(float v);

template <>
__device__ __forceinline__ float cast_from_float<float>(float v) { return v; }

template <>
__device__ __forceinline__ __half cast_from_float<__half>(float v) { return __float2half(v); }

// 核心模板：TILE = block 维度（编译期），VEC = 是否向量化路径（编译期）。
// 归约代码以 warpSize 为准（CUDA=32，MUSA=128，第 01 期实测）：蝶形步数、
// lane/warp 索引、跨 warp 部分和数量全部由 warpSize 推导 —— 硬编码 32 在
// MUSA 上语义错误（同一份 device 代码因此跨两平台成立，见迁移台账）。
template <typename T, int TILE, bool VEC>
__global__ void rmsnorm_kernel(const T* __restrict__ x, const T* __restrict__ w,
                               T* __restrict__ y, int rows, int h, float eps) {
    constexpr int kW = VecTraits<T>::kWidth;
    const int warp_size = warpSize;  // 设备内建：CUDA 32 / MUSA 128（待本机数值验证）
    __shared__ float s_x[kMaxCacheElems];  // 行缓存（H<=2048 时 Pass 2 免重读 x）
    __shared__ float s_warp[TILE / 32];    // 跨 warp 归约的部分和（上界：warpSize>=32）

    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        const T* xrow = x + static_cast<size_t>(row) * h;
        T* yrow = y + static_cast<size_t>(row) * h;

        // ---- Pass 1：读 x（向量化/标量），缓存进 shared，FP32 累加平方和 ----
        float acc = 0.0f;
        if (VEC) {
            // host 端已保证：h % kW == 0 且三个指针 16B 对齐。
            int j = threadIdx.x * kW;
            while (j < h) {
                typename VecTraits<T>::Pack p =
                    *reinterpret_cast<const typename VecTraits<T>::Pack*>(xrow + j);
                #pragma unroll
                for (int k = 0; k < kW; ++k) {
                    const float f = pack_get(p, k);
                    if (h <= kMaxCacheElems) s_x[j + k] = f;
                    acc += f * f;
                }
                j += TILE * kW;
            }
        } else {
            for (int j = threadIdx.x; j < h; j += TILE) {  // 块内 stride 循环（非整 Tile）
                const float f = to_float(xrow[j]);
                if (h <= kMaxCacheElems) s_x[j] = f;
                acc += f * f;
            }
        }

        // ---- warp 内蝶形归约：步数由 warpSize 推导（CUDA 5 步 / MUSA 7 步）----
        for (int off = warp_size / 2; off > 0; off >>= 1) {
            acc += __shfl_xor_sync(0xffffffffu, acc, off);
        }
        const int warp_count = (TILE + warp_size - 1) / warp_size;
        const int warp = threadIdx.x / warp_size;
        const int lane = threadIdx.x % warp_size;
        if (lane == 0) s_warp[warp] = acc;
        __syncthreads();

        // ---- 跨 warp 归约：warp 0 汇总各 warp 部分和并广播 ----
        if (warp == 0) {
            float v = (lane < warp_count) ? s_warp[lane] : 0.0f;
            for (int off = warp_size / 2; off > 0; off >>= 1) {
                v += __shfl_xor_sync(0xffffffffu, v, off);
            }
            if (lane == 0) s_warp[0] = v;
        }
        __syncthreads();
        const float total = s_warp[0];
        // scale = 1/sqrt(mean(x^2) + eps)，eps 在根号内（全零行输出全零，无 NaN）。
        const float scale = rsqrtf(total / static_cast<float>(h) + eps);

        // ---- Pass 2：y = x * scale * w（x 优先从 shared 读；w 每行重读，驻留 L2）----
        if (VEC) {
            int j = threadIdx.x * kW;
            while (j < h) {
                typename VecTraits<T>::Pack wv =
                    *reinterpret_cast<const typename VecTraits<T>::Pack*>(w + j);
                typename VecTraits<T>::Pack yv;
                #pragma unroll
                for (int k = 0; k < kW; ++k) {
                    const float xv = (h <= kMaxCacheElems) ? s_x[j + k]
                                                           : to_float(xrow[j + k]);
                    pack_set(yv, k, cast_from_float<T>(xv * scale * to_float(pack_get(wv, k))));
                }
                *reinterpret_cast<typename VecTraits<T>::Pack*>(yrow + j) = yv;
                j += TILE * kW;
            }
        } else {
            for (int j = threadIdx.x; j < h; j += TILE) {
                const float xv = (h <= kMaxCacheElems) ? s_x[j] : to_float(xrow[j]);
                yrow[j] = cast_from_float<T>(xv * scale * to_float(w[j]));
            }
        }
        // grid-stride 下本 block 可能继续处理下一行：先保证所有 warp 读完
        // s_x / s_warp，再允许下一行 Pass 1 覆盖它们。
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// host 端 dispatch：按 (T, tile, 可否向量化) 选择编译期特化。
// 与 CUDA 版对应；cudaStream_t→musaStream_t（✅ 实机确认）。
// ---------------------------------------------------------------------------
inline bool is_aligned16(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & 0xF) == 0;
}

// 本设备 warpSize（缓存）：蝶形归约要求 TILE 是 warpSize 整数倍 ——
// MUSA warpSize=128（第 01 期实测），TILE=64 在其上显式拒绝而非静默错误。
inline int device_warp_size() {
    static int cached = 0;
    if (cached == 0) {
        int device = 0;
        RMSNORM_MUSA_CHECK(musaGetDevice(&device));
        musaDeviceProp prop;
        RMSNORM_MUSA_CHECK(musaGetDeviceProperties(&prop, device));
        cached = prop.warpSize;  // 待数值验证：应为 128（第 01 期实测）
    }
    return cached;
}

template <typename T>
bool vec_feasible(const void* x, const void* w, const void* y, int h) {
    if (h % VecTraits<T>::kWidth != 0) return false;
    return is_aligned16(x) && is_aligned16(w) && is_aligned16(y);
}

template <typename T>
void launch_tile(const void* x, const void* w, void* y, int rows, int h, float eps,
                 int tile, musaStream_t stream) {  // ✅ 实机确认 musaStream_t
    const int warp_size = device_warp_size();
    if (tile < warp_size || tile % warp_size != 0) {
        throw std::runtime_error("TILE=" + std::to_string(tile) +
                                 " 必须是本设备 warpSize(" + std::to_string(warp_size) +
                                 ") 的整数倍（蝶形归约要求整 warp 覆盖；MUSA warpSize=128）");
    }
    const bool vec = vec_feasible<T>(x, w, y, h);
    const T* xt = static_cast<const T*>(x);
    const T* wt = static_cast<const T*>(w);
    T* yt = static_cast<T*>(y);
    dim3 block(tile);
    dim3 grid(static_cast<unsigned>(rows < kMaxGridBlocks ? rows : kMaxGridBlocks));

    #define RMSNORM_LAUNCH(T_NAME, TILE_V, VEC_V)                                   \
        do {                                                                        \
            rmsnorm_kernel<T_NAME, TILE_V, VEC_V>                                   \
                <<<grid, block, 0, stream>>>(xt, wt, yt, rows, h, eps);             \
        } while (0)
    #define RMSNORM_DISPATCH_VEC(T_NAME, TILE_V)                                    \
        do {                                                                        \
            if (vec) { RMSNORM_LAUNCH(T_NAME, TILE_V, true); }                      \
            else     { RMSNORM_LAUNCH(T_NAME, TILE_V, false); }                     \
        } while (0)
    switch (tile) {
        case 64:   RMSNORM_DISPATCH_VEC(T, 64);   break;
        case 128:  RMSNORM_DISPATCH_VEC(T, 128);  break;
        case 256:  RMSNORM_DISPATCH_VEC(T, 256);  break;
        case 512:  RMSNORM_DISPATCH_VEC(T, 512);  break;
        case 1024: RMSNORM_DISPATCH_VEC(T, 1024); break;
        default:
            throw std::runtime_error("unsupported TILE=" + std::to_string(tile) +
                                     "（编译期实例化集合：{64,128,256,512,1024}）");
    }
    #undef RMSNORM_LAUNCH
    #undef RMSNORM_DISPATCH_VEC
}

void launch_rmsnorm_fp32(const float* x, const float* w, float* y, int rows, int h,
                         float eps, int tile, musaStream_t stream) {
    launch_tile<float>(x, w, y, rows, h, eps, tile, stream);
    RMSNORM_MUSA_CHECK(musaGetLastError());  // ✅ 实机确认 musaGetLastError
}

void launch_rmsnorm_fp16(const void* x, const void* w, void* y, int rows, int h,
                         float eps, int tile, musaStream_t stream) {
    launch_tile<__half>(x, w, y, rows, h, eps, tile, stream);
    RMSNORM_MUSA_CHECK(musaGetLastError());
}


// dtype 分发辅助（torch 扩展用；kFloat/kHalf 布局一致，指针按字节传入）。
void launch_rmsnorm_any(const void* x, const void* w, void* y, bool is_fp16,
                        int rows, int h, float eps, int tile, musaStream_t stream) {
    if (is_fp16) {
        launch_rmsnorm_fp16(x, w, y, rows, h, eps, tile, stream);
    } else {
        launch_rmsnorm_fp32(static_cast<const float*>(x), static_cast<const float*>(w),
                            static_cast<float*>(y), rows, h, eps, tile, stream);
    }
}

}  // namespace mini_infra
