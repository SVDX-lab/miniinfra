// ============================================================================
// fused_rmsnorm_kernels.cu —— 第 03 期 Residual Add + RMSNorm Kernel 库
//
// 组织（对照大纲 §13 第 03 期"实现"条目）：
//   分离路径 = add_kernel（第 1 步：t = r + h，物化中间 Tensor）
//            + rmsnorm_kernel_ep03（第 2 步：第 02 期 Kernel 原样复用，
//              见 rmsnorm_kernel_ep03.cu）—— 共 2 次 Kernel Launch；
//   融合路径 = fused_add_rmsnorm_kernel —— 单 Kernel：
//     Pass 1：读 r、读 h，t = r + h 在寄存器完成，更新后的 residual
//             原位写回 r（合同 in-place 语义），t 的 FP32 值缓存进
//             shared memory 并累加平方和；
//     归约：  与第 02 期逐字相同的 warpSize-generic 蝶形归约
//             （CUDA 5 步 / MUSA 7 步，由 warpSize 推导）；
//     Pass 2：y = t * rsqrt(mean(t^2) + eps) * w（t 优先从 shared 读；
//             H > 2048 时回退重读 global——此时 r buffer 已含 t，
//             与第 02 期"重读 x"逐字对应）。
//
// 数值语义（与分离路径逐位一致的刻意设计）：
//   t 先按 dtype T 舍入一次（t_T = T(t_f32)）——这正是分离路径"物化
//   中间 Tensor t"的舍入语义；融合路径把这次物化从显存搬进寄存器，
//   随后的一切（shared 缓存、平方和、scale、写 y）与第 02 期 kernel
//   对同一 t 的处理逐字相同。因此分离与融合的 y 在同 Tile 下**位级
//   一致**——本期单变量是"组织方式"，数值零差异由实验证实。
//
// Residual 更新语义（合同显式条目）：
//   in-place：调用后 r buffer 内容 = t = r + h（T 舍入）；y 独立输出。
//   调用方需保留原 r 时先克隆（vLLM fused_add_rmsnorm 同语义）。
//   out-of-place 变体只在 CPU 参考实现提供（语义对照，见合同）。
//
// 绑定（M-17 ctypes 直传模式）：本文件只做 Kernel 启动（默认流），
// 无 cudaMalloc/cudaGetDeviceProperties 等任何 Runtime API 调用
// ——显存分配/同步/计时全部留在 Python/torch 侧；extern "C" 导出
// C ABI，ctypes 通过 data_ptr() 直传设备指针。warpSize 由 Python 侧
// 用 warp_size_probe kernel 实测后以参数传入（零 Runtime API 的
// warpSize 查询替代方案）。
//
// 错误处理：extern "C" 启动器返回状态码（0=OK，非 0=参数非法），
// 不跨 C ABI 抛异常；Python 绑定按状态码抛 RuntimeError。
// ============================================================================

#include "rmsnorm_kernel_ep03.cu"  // 第 02 期 Kernel 原样复用（单编译单元）

#include <cstdint>

namespace mini_infra {

// ---------------------------------------------------------------------------
// 分离路径第 1 步：add kernel（t = r + h，grid-stride 全元素遍历）
// ---------------------------------------------------------------------------
template <typename T, bool VEC>
__global__ void add_kernel(const T* __restrict__ r, const T* __restrict__ h,
                           T* __restrict__ t, long long n) {
    constexpr int kW = VecTraits<T>::kWidth;
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (VEC) {
        // host 端已保证：n % kW == 0 且三指针 16B 对齐；按 16B 步进。
        const long long vec_n = n / kW;
        const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
        for (long long v = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
             v < vec_n; v += stride) {
            const long long j = v * kW;
            typename VecTraits<T>::Pack rv =
                *reinterpret_cast<const typename VecTraits<T>::Pack*>(r + j);
            typename VecTraits<T>::Pack hv =
                *reinterpret_cast<const typename VecTraits<T>::Pack*>(h + j);
            typename VecTraits<T>::Pack tv;
            #pragma unroll
            for (int k = 0; k < kW; ++k) {
                pack_set(tv, k,
                         cast_from_float<T>(to_float(pack_get(rv, k)) + to_float(pack_get(hv, k))));
            }
            *reinterpret_cast<typename VecTraits<T>::Pack*>(t + j) = tv;
        }
    } else {
        const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
        for (; i < n; i += stride) {
            t[i] = cast_from_float<T>(to_float(r[i]) + to_float(h[i]));
        }
    }
}

// ---------------------------------------------------------------------------
// 融合路径：单 Kernel（Pass 1 加法+缓存+原位 residual 写回 + 平方和；
// 归约与第 02 期逐字相同；Pass 2 权重缩放写 y）
// ---------------------------------------------------------------------------
template <typename T, int TILE, bool VEC>
__global__ void fused_add_rmsnorm_kernel(T* __restrict__ r, const T* __restrict__ h_in,
                                         const T* __restrict__ w, T* __restrict__ y,
                                         int rows, int h, float eps) {
    constexpr int kW = VecTraits<T>::kWidth;
    const int warp_size = warpSize;  // 设备内建（probe 实测）：CUDA 32 / MUSA builtin 32（C API 报 128，M-20）
    __shared__ float s_x[kMaxCacheElems];  // t 行缓存（H<=2048 时 Pass 2 免重读）
    __shared__ float s_warp[TILE / 32];    // 跨 warp 归约的部分和（上界：warpSize>=32）

    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        T* rrow = r + static_cast<size_t>(row) * h;        // in-out：更新后的 residual
        const T* hrow = h_in + static_cast<size_t>(row) * h;
        T* yrow = y + static_cast<size_t>(row) * h;

        // ---- Pass 1：t = r + h（FP32 加法、T 舍入一次），缓存 + 累加 + 原位写回 ----
        float acc = 0.0f;
        if (VEC) {
            // host 端已保证：h % kW == 0 且各指针 16B 对齐。
            int j = threadIdx.x * kW;
            while (j < h) {
                typename VecTraits<T>::Pack rv =
                    *reinterpret_cast<const typename VecTraits<T>::Pack*>(rrow + j);
                typename VecTraits<T>::Pack hv =
                    *reinterpret_cast<const typename VecTraits<T>::Pack*>(hrow + j);
                typename VecTraits<T>::Pack tv;  // 更新后的 residual（T 舍入）
                #pragma unroll
                for (int k = 0; k < kW; ++k) {
                    const float t_f32 = to_float(pack_get(rv, k)) + to_float(pack_get(hv, k));
                    const T t_T = cast_from_float<T>(t_f32);  // 中间 t 的物化舍入语义
                    const float t_use = to_float(t_T);
                    if (h <= kMaxCacheElems) s_x[j + k] = t_use;
                    acc += t_use * t_use;
                    pack_set(tv, k, t_T);
                }
                // 原位 residual 更新（合同 in-place 语义）：读在后写在同地址、
                // 同线程，块内无交叉访问（一行只归本 block）。
                *reinterpret_cast<typename VecTraits<T>::Pack*>(rrow + j) = tv;
                j += TILE * kW;
            }
        } else {
            for (int j = threadIdx.x; j < h; j += TILE) {  // 块内 stride 循环（非整 Tile）
                const float t_f32 = to_float(rrow[j]) + to_float(hrow[j]);
                const T t_T = cast_from_float<T>(t_f32);
                const float t_use = to_float(t_T);
                if (h <= kMaxCacheElems) s_x[j] = t_use;
                acc += t_use * t_use;
                rrow[j] = t_T;  // 原位 residual 更新
            }
        }

        // ---- warp 内蝶形归约：与第 02 期逐字相同（步数由 warpSize 推导）----
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
        const float scale = rsqrtf(total / static_cast<float>(h) + eps);

        // ---- Pass 2：y = t * scale * w（t 优先从 shared 读；H>2048 回退重读
        //      global——此时 r buffer 已含 t，与第 02 期"重读 x"逐字对应）----
        if (VEC) {
            int j = threadIdx.x * kW;
            while (j < h) {
                typename VecTraits<T>::Pack wv =
                    *reinterpret_cast<const typename VecTraits<T>::Pack*>(w + j);
                typename VecTraits<T>::Pack yv;
                #pragma unroll
                for (int k = 0; k < kW; ++k) {
                    const float tv = (h <= kMaxCacheElems) ? s_x[j + k] : to_float(rrow[j + k]);
                    pack_set(yv, k, cast_from_float<T>(tv * scale * to_float(pack_get(wv, k))));
                }
                *reinterpret_cast<typename VecTraits<T>::Pack*>(yrow + j) = yv;
                j += TILE * kW;
            }
        } else {
            for (int j = threadIdx.x; j < h; j += TILE) {
                const float tv = (h <= kMaxCacheElems) ? s_x[j] : to_float(rrow[j]);
                yrow[j] = cast_from_float<T>(tv * scale * to_float(w[j]));
            }
        }
        // grid-stride 下本 block 可能继续处理下一行：先保证所有 warp 读完
        // s_x / s_warp，再允许下一行 Pass 1 覆盖它们（与第 02 期逐字相同）。
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// warpSize 探针：Python 侧实测本设备 warpSize（替代 cudaGetDeviceProperties，
// M-17 ctypes 模式下保持 kernel 库零 Runtime API 调用）。
// ---------------------------------------------------------------------------
__global__ void warp_size_probe_kernel(int* out) {
    out[0] = static_cast<int>(warpSize);
}

// ---------------------------------------------------------------------------
// host 端公共校验（纯指针算术/参数检查，零 Runtime API）。状态码：
//   0=OK；1=tile 非法（非 {64,128,256,512,1024} 或非 warpSize 整数倍）；
//   2=H<=0；3=rows<0。
// is_aligned16 / kMaxGridBlocks / kMaxCacheElems 复用 rmsnorm_kernel_ep03.cu
// （单编译单元 #include，避免重复定义）。
// ---------------------------------------------------------------------------
template <typename T>
bool vec_feasible4(const void* r, const void* h, const void* w, const void* y, int h_dim) {
    if (h_dim % VecTraits<T>::kWidth != 0) return false;
    return is_aligned16(r) && is_aligned16(h) && is_aligned16(w) && is_aligned16(y);
}

int validate_tile(int tile, int warp_size) {
    if (tile != 64 && tile != 128 && tile != 256 && tile != 512 && tile != 1024) return 1;
    if (warp_size <= 0 || tile < warp_size || tile % warp_size != 0) return 1;
    return 0;
}

}  // namespace mini_infra

// ============================================================================
// extern "C" 启动器（C ABI；默认流；返回状态码，不抛异常）
// ============================================================================
using namespace mini_infra;

extern "C" int warp_size_probe(long long out_ptr) {
    int* out = reinterpret_cast<int*>(static_cast<uintptr_t>(out_ptr));
    warp_size_probe_kernel<<<1, 1>>>(out);
    return 0;
}

// 分离路径第 1 步：t = r + h（n = rows*H 元素数）
extern "C" int add_rows(long long r_ptr, long long h_ptr, long long t_ptr,
                        int is_fp16, long long n, int block_threads) {
    const void* r = reinterpret_cast<const void*>(static_cast<uintptr_t>(r_ptr));
    const void* h = reinterpret_cast<const void*>(static_cast<uintptr_t>(h_ptr));
    void* t = reinterpret_cast<void*>(static_cast<uintptr_t>(t_ptr));
    if (n < 0) return 3;
    if (n == 0) return 0;  // 空 Tensor：不发射 Kernel
    const int threads = block_threads > 0 ? block_threads : 256;
    const long long grid = (n + threads - 1) / threads;
    #define ADD_LAUNCH(T_NAME, VEC_V)                                                     \
        do {                                                                              \
            if (VEC_V) {                                                                  \
                add_kernel<T_NAME, true><<<static_cast<unsigned>(grid), threads>>>(        \
                    static_cast<const T_NAME*>(r), static_cast<const T_NAME*>(h),         \
                    static_cast<T_NAME*>(t), n);                                          \
            } else {                                                                      \
                add_kernel<T_NAME, false><<<static_cast<unsigned>(grid), threads>>>(       \
                    static_cast<const T_NAME*>(r), static_cast<const T_NAME*>(h),         \
                    static_cast<T_NAME*>(t), n);                                          \
            }                                                                             \
        } while (0)
    if (is_fp16) {
        if (n % 8 == 0 && is_aligned16(r) && is_aligned16(h) && is_aligned16(t)) {
            ADD_LAUNCH(__half, true);
        } else {
            ADD_LAUNCH(__half, false);
        }
    } else {
        if (n % 4 == 0 && is_aligned16(r) && is_aligned16(h) && is_aligned16(t)) {
            ADD_LAUNCH(float, true);
        } else {
            ADD_LAUNCH(float, false);
        }
    }
    #undef ADD_LAUNCH
    return 0;
}

// 分离路径第 2 步：第 02 期 RMSNorm Kernel（warp_size 由 Python 侧 probe 取得）
extern "C" int rmsnorm_rows(long long x_ptr, long long w_ptr, long long y_ptr,
                            int is_fp16, int rows, int h_dim, float eps,
                            int tile, int warp_size) {
    const void* x = reinterpret_cast<const void*>(static_cast<uintptr_t>(x_ptr));
    const void* w = reinterpret_cast<const void*>(static_cast<uintptr_t>(w_ptr));
    void* y = reinterpret_cast<void*>(static_cast<uintptr_t>(y_ptr));
    int status = validate_tile(tile, warp_size);
    if (status != 0) return status;
    if (h_dim <= 0) return 2;
    if (rows < 0) return 3;
    if (rows == 0) return 0;  // 空 Tensor：不发射 Kernel
    if (is_fp16) {
        launch_rmsnorm_fp16(x, w, y, rows, h_dim, eps, tile, 0, warp_size);
    } else {
        launch_rmsnorm_fp32(static_cast<const float*>(x), static_cast<const float*>(w),
                            static_cast<float*>(y), rows, h_dim, eps, tile, 0, warp_size);
    }
    return 0;
}

// 融合路径：单 Kernel，r 原位更新（in-place），y 独立输出
extern "C" int fused_add_rmsnorm_rows(long long r_ptr, long long h_ptr, long long w_ptr,
                                      long long y_ptr, int is_fp16, int rows, int h_dim,
                                      float eps, int tile, int warp_size) {
    void* r = reinterpret_cast<void*>(static_cast<uintptr_t>(r_ptr));
    const void* h = reinterpret_cast<const void*>(static_cast<uintptr_t>(h_ptr));
    const void* w = reinterpret_cast<const void*>(static_cast<uintptr_t>(w_ptr));
    void* y = reinterpret_cast<void*>(static_cast<uintptr_t>(y_ptr));
    int status = validate_tile(tile, warp_size);
    if (status != 0) return status;
    if (h_dim <= 0) return 2;
    if (rows < 0) return 3;
    if (rows == 0) return 0;  // 空 Tensor：不发射 Kernel

    const bool vec = is_fp16 ? vec_feasible4<__half>(r, h, w, y, h_dim)
                             : vec_feasible4<float>(r, h, w, y, h_dim);
    dim3 block(tile);
    dim3 grid(static_cast<unsigned>(rows < kMaxGridBlocks ? rows : kMaxGridBlocks));

    #define FUSED_LAUNCH(T_NAME, TILE_V, VEC_V)                                          \
        do {                                                                             \
            fused_add_rmsnorm_kernel<T_NAME, TILE_V, VEC_V>                              \
                <<<grid, block, 0, 0>>>(static_cast<T_NAME*>(r),                          \
                                        static_cast<const T_NAME*>(h),                    \
                                        static_cast<const T_NAME*>(w),                    \
                                        static_cast<T_NAME*>(y), rows, h_dim, eps);       \
        } while (0)
    #define FUSED_DISPATCH_VEC(T_NAME, TILE_V)                                           \
        do {                                                                             \
            if (vec) { FUSED_LAUNCH(T_NAME, TILE_V, true); }                             \
            else     { FUSED_LAUNCH(T_NAME, TILE_V, false); }                            \
        } while (0)
    #define FUSED_DISPATCH_TILE(T_NAME)                                                  \
        do {                                                                             \
            switch (tile) {                                                              \
                case 64:   FUSED_DISPATCH_VEC(T_NAME, 64);   break;                      \
                case 128:  FUSED_DISPATCH_VEC(T_NAME, 128);  break;                      \
                case 256:  FUSED_DISPATCH_VEC(T_NAME, 256);  break;                      \
                case 512:  FUSED_DISPATCH_VEC(T_NAME, 512);  break;                      \
                case 1024: FUSED_DISPATCH_VEC(T_NAME, 1024); break;                      \
                default: return 1;                                                       \
            }                                                                            \
        } while (0)
    if (is_fp16) {
        FUSED_DISPATCH_TILE(__half);
    } else {
        FUSED_DISPATCH_TILE(float);
    }
    #undef FUSED_LAUNCH
    #undef FUSED_DISPATCH_VEC
    #undef FUSED_DISPATCH_TILE
    return 0;
}
