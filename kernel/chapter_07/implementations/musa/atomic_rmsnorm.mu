// ============================================================================
// atomic_rmsnorm.cu —— 第 07 期 atomicAdd 交叉归约版 RMSNorm（device 代码）
//
// 本期唯一核心变量：归约的组织方式。对照对象是 rmsnorm_kernel_ep03.cu
// （确定性 warp/block 树形归约，第 02 期 kernel 原样）。本文件提供两档
// atomic 归约（大纲 §13 第 07 期"实现"条目）：
//
//   档 A shared atomic（rmsnorm_atomic_shared_kernel）：
//     每 block 一行（grid-stride），block 内每线程把局部平方和用
//     atomicAdd 累到 shared 累加器 s_acc（每行 1 个）。
//     - 竞争强度 = TILE 线程（同 block 多 warp）；
//     - 归约顺序 = 原子到达顺序（warp 调度/访存延迟决定，运行时变化）；
//     - Launch 数与确定性版相同（1）。
//
//   档 B global atomic（rmsnorm_sqsum_atomic_global_kernel + scale kernel）：
//     一行拆 splits 段，grid = rows × splits，每 block 处理一段，
//     每线程 atomicAdd 到全局累加器 g_acc[row]；随后第二个 kernel
//     读 g_acc 完成 scale（跨 block 无 grid-sync，必须两个 kernel）。
//     - 竞争强度 = splits × TILE 线程（跨 block 跨 SM）；
//     - 调用前 g_acc 必须清零（zero_acc kernel，1 次额外 launch）；
//     - Launch 数 = 3（zero + sqsum + scale）—— atomic 的实现复杂度代价；
//     - scale kernel 重读 global x（跨 kernel 无法保留 shared 行缓存）
//       —— 理论字节比确定性版多 1 次读 x，性能对账单独标注。
//
// 累加 dtype：ACC ∈ {float, __half} 模板化。ACC=__half 接第 05 期
// "卡死区"主题（ulp(acc) > 增量/2 后累加器静默冻结）：atomic 乱序到达
// 在 FP16 累加器下是否逃离卡死区是本期实验点（CPU 一阶模型已在
// exp_cpu_sum_order.py 给出"逃不掉"的先验）。half atomicAdd 需要硬件
// fp16 原子（CUDA sm_70+；MUSA 侧待实机核对，编译开关见 build.sh）。
//
// 数值语义（与确定性版唯一差异 = 平方和的归约顺序）：
//   Pass 1 数据搬移 / s_x 行缓存 / Pass 2 归一化写回与 ep03 kernel 逐字
//   相同（同 tile 同输入时 y 的差异只能来自 s_acc/g_acc 的累加顺序）。
//   ACC=__half 时局部累加也在 half（每步舍入），scale 除法/rsqrt 仍在
//   FP32（把"累加链 dtype"与"数学函数精度"分开归因，同第 05 期口径）。
//
// 确定性：本文件三个 kernel 都是**不确定**的（原子到达顺序运行时决定）
// —— 这正是本期对照实验的被测对象。
// ============================================================================

#include <cstdint>

namespace mini_infra {

// ACC=half 编译开关：half atomicAdd（fp16 原子）CUDA sm_70+ 原生提供；
// MUSA SDK 5.2.0 是否提供待实机核对 —— 编译失败时 build.sh 以
// -DEP08_NO_HALF_ATOMIC 重试（ACC=half 档记平台边界），fp32 累加档不受影响。
#if defined(EP08_NO_HALF_ATOMIC)
constexpr int kHalfAtomicEnabled = 0;
#else
constexpr int kHalfAtomicEnabled = 1;
#endif

constexpr int kMaxCacheElemsAtomic = 2048;  // 与 ep03 的 kMaxCacheElems 同值（shared 行缓存上限）

// ---------------------------------------------------------------------------
// 档 A：shared atomic 交叉归约（每 block 一行，每线程局部和原子累加）
// ---------------------------------------------------------------------------
template <typename T, int TILE, bool VEC, typename ACC>
__global__ void rmsnorm_atomic_shared_kernel(const T* __restrict__ x, const T* __restrict__ w,
                                             T* __restrict__ y, int rows, int h, float eps) {
    constexpr int kW = VecTraits<T>::kWidth;
    __shared__ ACC s_acc;                    // 行累加器（原子累加，顺序运行时定）
    __shared__ float s_x[kMaxCacheElemsAtomic];

    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        const T* xrow = x + static_cast<size_t>(row) * h;
        T* yrow = y + static_cast<size_t>(row) * h;

        // (A) 上一行的 atomicAdd 与 s_acc/s_x 读取全部完成后才能重置。
        __syncthreads();
        if (threadIdx.x == 0) s_acc = ACC(0);
        __syncthreads();                     // (B) 重置对全体可见

        // ---- Pass 1：数据搬移/行缓存与 ep03 逐字相同，局部累加进 ACC ----
        ACC local = ACC(0);
        if (VEC) {
            int j = threadIdx.x * kW;
            while (j < h) {
                typename VecTraits<T>::Pack p =
                    *reinterpret_cast<const typename VecTraits<T>::Pack*>(xrow + j);
                #pragma unroll
                for (int k = 0; k < kW; ++k) {
                    const float f = pack_get(p, k);
                    if (h <= kMaxCacheElemsAtomic) s_x[j + k] = f;
                    local = ACC(float(local) + f * f);
                }
                j += TILE * kW;
            }
        } else {
            for (int j = threadIdx.x; j < h; j += TILE) {
                const float f = to_float(xrow[j]);
                if (h <= kMaxCacheElemsAtomic) s_x[j] = f;
                local = ACC(float(local) + f * f);
            }
        }
        // ---- 交叉归约本体：每线程一次原子加（到达顺序 = 运行时）----
        atomicAdd(&s_acc, local);
        __syncthreads();                     // (C) 所有原子加完成后才能读总量

        const float total = float(s_acc);    // 读进寄存器；pass2 不再碰 s_acc
        const float scale = rsqrtf(total / static_cast<float>(h) + eps);

        // ---- Pass 2：与 ep03 逐字相同（差异只可能来自 total 的顺序）----
        if (VEC) {
            int j = threadIdx.x * kW;
            while (j < h) {
                typename VecTraits<T>::Pack wv =
                    *reinterpret_cast<const typename VecTraits<T>::Pack*>(w + j);
                typename VecTraits<T>::Pack yv;
                #pragma unroll
                for (int k = 0; k < kW; ++k) {
                    const float xv = (h <= kMaxCacheElemsAtomic) ? s_x[j + k]
                                                                 : to_float(xrow[j + k]);
                    pack_set(yv, k, cast_from_float<T>(xv * scale * to_float(pack_get(wv, k))));
                }
                *reinterpret_cast<typename VecTraits<T>::Pack*>(yrow + j) = yv;
                j += TILE * kW;
            }
        } else {
            for (int j = threadIdx.x; j < h; j += TILE) {
                const float xv = (h <= kMaxCacheElemsAtomic) ? s_x[j] : to_float(xrow[j]);
                yrow[j] = cast_from_float<T>(xv * scale * to_float(w[j]));
            }
        }
        // 尾部无需 __syncthreads：下一轮开头的 (A) 承担该职责。
    }
}

// ---------------------------------------------------------------------------
// 档 B 第 1 步：跨 block global atomic 平方和（grid = rows × splits）
// ---------------------------------------------------------------------------
template <typename T, int TILE, bool VEC, typename ACC>
__global__ void rmsnorm_sqsum_atomic_global_kernel(const T* __restrict__ x,
                                                   ACC* __restrict__ g_acc,
                                                   int rows, int h, int splits) {
    constexpr int kW = VecTraits<T>::kWidth;
    const int row = blockIdx.x / splits;
    const int seg = blockIdx.x % splits;
    // 均匀分段（含余数）：前 h % splits 段各多 1 个元素。
    const int base = h / splits;
    const int rem = h - base * splits;
    const int start = seg * base + (seg < rem ? seg : rem);
    const int end = start + base + (seg < rem ? 1 : 0);

    ACC local = ACC(0);
    if (VEC) {
        // host 端保证：start % kW == 0 且 end % kW == 0（splits 整除 h 且
        // 段长是 kW 倍数；否则走标量路径）。
        int j = start + threadIdx.x * kW;
        while (j < end) {
            typename VecTraits<T>::Pack p =
                *reinterpret_cast<const typename VecTraits<T>::Pack*>(x +
                    static_cast<size_t>(row) * h + j);
            #pragma unroll
            for (int k = 0; k < kW; ++k) {
                const float f = pack_get(p, k);
                local = ACC(float(local) + f * f);
            }
            j += TILE * kW;
        }
    } else {
        for (int j = start + threadIdx.x; j < end; j += TILE) {
            const float f = to_float(x[static_cast<size_t>(row) * h + j]);
            local = ACC(float(local) + f * f);
        }
    }
    // 跨 block 原子：splits × TILE 个线程竞争 g_acc[row]。
    atomicAdd(&g_acc[row], local);
}

// ---------------------------------------------------------------------------
// 档 B 第 2 步：scale kernel（读 g_acc，重读 global x，写 y；确定性）
// ---------------------------------------------------------------------------
template <typename T, bool VEC, typename ACC>
__global__ void rmsnorm_scale_global_kernel(const T* __restrict__ x, const T* __restrict__ w,
                                            T* __restrict__ y, const ACC* __restrict__ g_acc,
                                            int rows, int h, float eps) {
    constexpr int kW = VecTraits<T>::kWidth;
    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        // ACC=__half 时累加链在 half（卡死区语义），除法/rsqrt 统一 FP32。
        const float total = float(g_acc[row]);
        const float scale = rsqrtf(total / static_cast<float>(h) + eps);
        const T* xrow = x + static_cast<size_t>(row) * h;
        T* yrow = y + static_cast<size_t>(row) * h;
        if (VEC) {
            int j = threadIdx.x * kW;
            while (j < h) {
                typename VecTraits<T>::Pack wv =
                    *reinterpret_cast<const typename VecTraits<T>::Pack*>(w + j);
                typename VecTraits<T>::Pack yv;
                #pragma unroll
                for (int k = 0; k < kW; ++k) {
                    const float xv = to_float(xrow[j + k]);
                    pack_set(yv, k, cast_from_float<T>(xv * scale * to_float(pack_get(wv, k))));
                }
                *reinterpret_cast<typename VecTraits<T>::Pack*>(yrow + j) = yv;
                j += blockDim.x * kW;
            }
        } else {
            for (int j = threadIdx.x; j < h; j += blockDim.x) {
                yrow[j] = cast_from_float<T>(to_float(xrow[j]) * scale * to_float(w[j]));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// g_acc 清零（档 B 每次调用前必须执行；1 次 launch）
// ---------------------------------------------------------------------------
template <typename ACC>
__global__ void zero_acc_kernel(ACC* __restrict__ acc, long long n) {
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        acc[i] = ACC(0);
    }
}

}  // namespace mini_infra
