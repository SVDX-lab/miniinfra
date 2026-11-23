// ============================================================================
// rmsnorm_kernel.cu —— 第 05 期 MUSA 手写 RMSNorm：dtype 三路径版
// （CUDA 版逐文件迁移。✅ 头文件/API 已实机核对（2026-11-21，
//   grep /usr/local/musa/include）：musa_runtime.h / musa_fp16.h /
//   musa_bf16.h 存在；__mt_bfloat16 为 MUSA 的 BF16 设备类型（CUDA 为
//   __nv_bfloat16，语义改动条目）；__float2bfloat16_rn/__bfloat162float
//   同名存在；__hadd/__hmul（half 算术）与 __shfl_xor_sync(__half) 重载
//   存在（musa_fp16.h，__MUSA_ARCH__ ∈ (100, 220) 门控内，S4000 CC 2.2
//   编 mp_22 → 220 ≤ 220 成立）。
//
// 与 CUDA 版的全部差异（迁移台账口径）：
//   1. 头文件 cuda_runtime.h/cuda_fp16.h/cuda_bf16.h → musa_runtime.h/
//      musa_fp16.h/musa_bf16.h（机械替换）；
//   2. cudaError_t/cudaGetErrorString/cudaGetLastError/cudaGetDevice/
//      cudaGetDeviceProperties/cudaDeviceProp/cudaStream_t → musa 前缀
//      （机械替换）；
//   3. __nv_bfloat16 → __mt_bfloat16（语义改动：类型名，位布局同为
//      sign(1)+exp(8,bias127)+man(7)，IEEE 754-2008 binary16 的 bf16 侧
//      完全同构）；
//   4. __float2bfloat16 → __float2bfloat16_rn（显式 RNE：CUDA 的无后缀
//      版本即 RNE，MUSA 版拆出 _rn 后缀，转 RNE 语义显式化核对）。
//   Kernel 本体（__global__ / 模板 / 蝶形归约 / <<<>>> 启动语法）零改动。
// ============================================================================

#include <musa_runtime.h>
#include <musa_fp16.h>
#include <musa_bf16.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace mini_infra {

#define RMSNORM_CUDA_CHECK(expr)                                       \
    do {                                                               \
        musaError_t _err = (expr);                                     \
        if (_err != musaSuccess) {                                     \
            throw std::runtime_error(std::string("MUSA error at ") +   \
                                     __FILE__ + ":" +                  \
                                     std::to_string(__LINE__) + " " +  \
                                     #expr + ": " +                    \
                                     musaGetErrorString(_err));        \
        }                                                              \
    } while (0)

// 合同：支持的最大 H（H<=2048 时 Pass 2 从 shared 读 x，免重读 global）。
constexpr int kMaxCacheElems = 2048;
constexpr long kMaxGridBlocks = 1L << 20;

// ---------------------------------------------------------------------------
// 向量化访存适配层：FP32 float4（4 元素）；FP16/BF16 uint4（8 元素）。
// ---------------------------------------------------------------------------
template <typename T>
struct VecTraits;

template <>
struct VecTraits<float> {
    static constexpr int kWidth = 4;
    using Pack = float4;
};

template <>
struct VecTraits<__half> {
    static constexpr int kWidth = 8;
    using Pack = uint4;
};

template <>
struct VecTraits<__mt_bfloat16> {
    static constexpr int kWidth = 8;
    using Pack = uint4;
};

// dtype 转换适配：读入转 FP32 / 写回从 FP32 舍入到 T。
__device__ __forceinline__ float to_float(float v) { return v; }
__device__ __forceinline__ float to_float(__half v) { return __half2float(v); }
__device__ __forceinline__ float to_float(__mt_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__device__ __forceinline__ T cast_from_float(float v);

template <>
__device__ __forceinline__ float cast_from_float<float>(float v) { return v; }

template <>
__device__ __forceinline__ __half cast_from_float<__half>(float v) { return __float2half(v); }

template <>
__device__ __forceinline__ __mt_bfloat16 cast_from_float<__mt_bfloat16>(float v) {
    return __float2bfloat16_rn(v);  // 显式 RNE（见文件头差异条目 4）
}

// 向量化载荷 <-> 标量。
template <typename T>
__device__ __forceinline__ float pack_get(const typename VecTraits<T>::Pack& p, int k);

template <>
__device__ __forceinline__ float pack_get<float>(const float4& p, int k) {
    return (&p.x)[k];
}

template <>
__device__ __forceinline__ float pack_get<__half>(const uint4& p, int k) {
    return __half2float(reinterpret_cast<const __half*>(&p)[k]);
}

template <>
__device__ __forceinline__ float pack_get<__mt_bfloat16>(const uint4& p, int k) {
    return __bfloat162float(reinterpret_cast<const __mt_bfloat16*>(&p)[k]);
}

template <typename T>
__device__ __forceinline__ void pack_set(typename VecTraits<T>::Pack& p, int k, float v);

template <>
__device__ __forceinline__ void pack_set<float>(float4& p, int k, float v) {
    (&p.x)[k] = v;
}

template <>
__device__ __forceinline__ void pack_set<__half>(uint4& p, int k, float v) {
    reinterpret_cast<__half*>(&p)[k] = __float2half(v);
}

template <>
__device__ __forceinline__ void pack_set<__mt_bfloat16>(uint4& p, int k, float v) {
    reinterpret_cast<__mt_bfloat16*>(&p)[k] = __float2bfloat16_rn(v);
}

// ---------------------------------------------------------------------------
// 累加 dtype 策略（实验 4 的被测变量；语义与 CUDA 版一致）。
// ---------------------------------------------------------------------------
template <typename ACC>
struct AccTraits;

template <>
struct AccTraits<float> {
    using type = float;
    static constexpr const char* kName = "fp32";
    __device__ __forceinline__ static float zero() { return 0.0f; }
    __device__ __forceinline__ static float square_add(float acc, float f) {
        return acc + f * f;
    }
    __device__ __forceinline__ static float to_float_total(float acc) { return acc; }
};

template <>
struct AccTraits<__half> {
    using type = __half;
    static constexpr const char* kName = "fp16";
    __device__ __forceinline__ static __half zero() { return __float2half(0.0f); }
    __device__ __forceinline__ static __half square_add(__half acc, float f) {
        const __half fh = __float2half(f);
        return __hadd(acc, __hmul(fh, fh));
    }
    __device__ __forceinline__ static float to_float_total(__half acc) {
        return __half2float(acc);
    }
};

// ---------------------------------------------------------------------------
// 归约原语（与 CUDA 版一致：float / __half 各自走 __shfl_xor_sync 原生重载）。
// ---------------------------------------------------------------------------
__device__ __forceinline__ float shfl_xor(float v, int off) {
    return __shfl_xor_sync(0xffffffffu, v, off);
}

__device__ __forceinline__ __half shfl_xor(__half v, int off) {
    return __shfl_xor_sync(0xffffffffu, v, off);
}

template <typename ACC>
__device__ __forceinline__ ACC acc_combine(ACC a, ACC b);

template <>
__device__ __forceinline__ float acc_combine<float>(float a, float b) { return a + b; }

template <>
__device__ __forceinline__ __half acc_combine<__half>(__half a, __half b) {
    return __hadd(a, b);
}

template <typename ACC>
__device__ __forceinline__ ACC warp_butterfly_step(ACC v, int off) {
    return acc_combine<ACC>(v, shfl_xor(v, off));
}

// ---------------------------------------------------------------------------
// 核心模板：T × ACC × TILE × VEC 全部编译期参数（与 CUDA 版零改动）。
// ---------------------------------------------------------------------------
template <typename T, typename ACC, int TILE, bool VEC>
__global__ void rmsnorm_kernel(const T* __restrict__ x, const T* __restrict__ w,
                               T* __restrict__ y, int rows, int h, float eps) {
    constexpr int kW = VecTraits<T>::kWidth;
    const int warp_size = warpSize;  // 设备内建（S4000 实测值见台账：ch03=128 / ch04=32）
    __shared__ float s_x[kMaxCacheElems];
    __shared__ typename AccTraits<ACC>::type s_warp[TILE / 32];

    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        const T* xrow = x + static_cast<size_t>(row) * h;
        T* yrow = y + static_cast<size_t>(row) * h;

        // ---- Pass 1：读 x（转 FP32），平方和在 ACC 算术内累加 ----
        ACC acc = AccTraits<ACC>::zero();
        if (VEC) {
            int j = threadIdx.x * kW;
            while (j < h) {
                typename VecTraits<T>::Pack p =
                    *reinterpret_cast<const typename VecTraits<T>::Pack*>(xrow + j);
                #pragma unroll
                for (int k = 0; k < kW; ++k) {
                    const float f = pack_get<T>(p, k);
                    if (h <= kMaxCacheElems) s_x[j + k] = f;
                    acc = AccTraits<ACC>::square_add(acc, f);
                }
                j += TILE * kW;
            }
        } else {
            for (int j = threadIdx.x; j < h; j += TILE) {
                const float f = to_float(xrow[j]);
                if (h <= kMaxCacheElems) s_x[j] = f;
                acc = AccTraits<ACC>::square_add(acc, f);
            }
        }

        // ---- warp 内蝶形归约（ACC 形态一路保持）----
        for (int off = warp_size / 2; off > 0; off >>= 1) {
            acc = warp_butterfly_step<ACC>(acc, off);
        }
        const int warp_count = (TILE + warp_size - 1) / warp_size;
        const int warp = threadIdx.x / warp_size;
        const int lane = threadIdx.x % warp_size;
        if (lane == 0) s_warp[warp] = acc;
        __syncthreads();

        // ---- 跨 warp 归约：warp 0 在 ACC 算术内汇总并广播 ----
        if (warp == 0) {
            ACC v = (lane < warp_count) ? s_warp[lane] : AccTraits<ACC>::zero();
            for (int off = warp_size / 2; off > 0; off >>= 1) {
                v = warp_butterfly_step<ACC>(v, off);
            }
            if (lane == 0) s_warp[0] = v;
        }
        __syncthreads();
        const float total = AccTraits<ACC>::to_float_total(s_warp[0]);
        const float scale = rsqrtf(total / static_cast<float>(h) + eps);

        // ---- Pass 2：y = x * scale * w（FP32 计算，一次舍入写回 T）----
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
                    pack_set<T>(yv, k, xv * scale * to_float(pack_get<T>(wv, k)));
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
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// host 端 dispatch（与 CUDA 版一致）。
// ---------------------------------------------------------------------------
inline bool is_aligned16(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & 0xF) == 0;
}

inline int device_warp_size() {
    static int cached = 0;
    if (cached == 0) {
        int device = 0;
        RMSNORM_CUDA_CHECK(musaGetDevice(&device));
        musaDeviceProp prop;
        RMSNORM_CUDA_CHECK(musaGetDeviceProperties(&prop, device));
        cached = prop.warpSize;
    }
    return cached;
}

template <typename T>
bool vec_feasible(const void* x, const void* w, const void* y, int h) {
    if (h % VecTraits<T>::kWidth != 0) return false;
    return is_aligned16(x) && is_aligned16(w) && is_aligned16(y);
}

template <typename T, typename ACC>
void launch_tile_acc(const void* x, const void* w, void* y, int rows, int h, float eps,
                     int tile, musaStream_t stream) {
    const int warp_size = device_warp_size();
    if (tile < warp_size || tile % warp_size != 0) {
        throw std::runtime_error("TILE=" + std::to_string(tile) +
                                 " 必须是本设备 warpSize(" + std::to_string(warp_size) +
                                 ") 的整数倍（蝶形归约要求整 warp 覆盖）");
    }
    const bool vec = vec_feasible<T>(x, w, y, h);
    const T* xt = static_cast<const T*>(x);
    const T* wt = static_cast<const T*>(w);
    T* yt = static_cast<T*>(y);
    dim3 block(tile);
    dim3 grid(static_cast<unsigned>(rows < kMaxGridBlocks ? rows : kMaxGridBlocks));

    #define RMSNORM_LAUNCH(T_NAME, TILE_V, VEC_V)                                  \
        do {                                                                       \
            rmsnorm_kernel<T_NAME, ACC, TILE_V, VEC_V>                             \
                <<<grid, block, 0, stream>>>(xt, wt, yt, rows, h, eps);            \
        } while (0)
    #define RMSNORM_DISPATCH_VEC(T_NAME, TILE_V)                                   \
        do {                                                                       \
            if (vec) { RMSNORM_LAUNCH(T_NAME, TILE_V, true); }                     \
            else     { RMSNORM_LAUNCH(T_NAME, TILE_V, false); }                    \
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

template <typename ACC>
void launch_acc(const void* x, const void* w, void* y, int dtype_code, int rows, int h,
                float eps, int tile, musaStream_t stream) {
    switch (dtype_code) {
        case 0:
            launch_tile_acc<float, ACC>(x, w, y, rows, h, eps, tile, stream);
            break;
        case 1:
            launch_tile_acc<__half, ACC>(x, w, y, rows, h, eps, tile, stream);
            break;
        case 2:
            launch_tile_acc<__mt_bfloat16, ACC>(x, w, y, rows, h, eps, tile, stream);
            break;
        default:
            throw std::runtime_error("dtype_code 只支持 0=fp32 / 1=fp16 / 2=bf16");
    }
}

void launch_rmsnorm(const void* x, const void* w, void* y, int dtype_code, int acc_code,
                    int rows, int h, float eps, int tile, musaStream_t stream) {
    switch (acc_code) {
        case 0:
            launch_acc<float>(x, w, y, dtype_code, rows, h, eps, tile, stream);
            break;
        case 1:
            launch_acc<__half>(x, w, y, dtype_code, rows, h, eps, tile, stream);
            break;
        default:
            throw std::runtime_error("acc_code 只支持 0=fp32 / 1=fp16");
    }
    RMSNORM_CUDA_CHECK(musaGetLastError());
}

}  // namespace mini_infra
