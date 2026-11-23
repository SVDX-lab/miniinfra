// ============================================================================
// kernels_lib.cu —— 第 07 期 Kernel 库（C ABI 共享库，单编译单元）
//
// M-17 ctypes 直传模式（第 03/16 期同款）：
//   - 单 TU：#include rmsnorm_kernel_ep03.cu（确定性版，ep03 原样）
//            + atomic_rmsnorm.cu（atomic 两档）；
//   - 零 Runtime API 调用（显存/同步/计时全部留 torch 侧）；warpSize 由
//     Python 侧 warp_size_probe kernel 实测后传参（M-20 口径）；
//   - extern "C" 启动器返回状态码（0=OK），不跨 C ABI 抛异常。
//
// 编译期实例化集合：
//   确定性版（ep03）:  T{fp32,fp16} × TILE{64,128,256,512,1024} × VEC{0,1}
//   atomic shared  :  T{fp32,fp16} × ACC{fp32,half} × TILE{128,256,512} × VEC{0,1}
//   atomic global  :  T{fp32,fp16} × ACC{fp32,half} × TILE{128,256,512} × VEC{0,1}
//                     + scale kernel T × ACC × VEC + zero kernel ACC
//   （EP08_NO_HALF_ATOMIC 宏关闭 ACC=half 实例 —— MUSA 侧 half atomicAdd
//    可用性待实机核对，编译失败时 build.sh 以该宏重试并记录平台边界）
//
// Launch 数契约（本期对照的结构性保证）：
//   rmsnorm_det_rows           = 1 launch（确定性）
//   rmsnorm_atomic_shared_rows = 1 launch（不确定）
//   rmsnorm_atomic_global_rows = 2 launches（sqsum + scale，不确定）
//   zero_acc                   = 1 launch（调用方在 atomic_global 前调用
//                                → 档 B 整链 = 3 launches）
// ============================================================================

#include "rmsnorm_kernel_ep03.cu"
#include "atomic_rmsnorm.cu"

#include <cstring>
#include <type_traits>

using namespace mini_infra;

namespace {

thread_local char g_status_detail[256];

void set_detail(const char* text) {
    std::strncpy(g_status_detail, text, sizeof(g_status_detail) - 1);
    g_status_detail[sizeof(g_status_detail) - 1] = '\0';
}

inline bool aligned16(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & 0xF) == 0;
}

template <typename T>
bool vec_ok_det(const void* x, const void* w, const void* y, int h) {
    if (h % VecTraits<T>::kWidth != 0) return false;
    return aligned16(x) && aligned16(w) && aligned16(y);
}

template <typename T>
bool vec_ok_global_sqsum(const void* x, int h, int splits) {
    if (h % VecTraits<T>::kWidth != 0) return false;
    if (h % splits != 0) return false;
    if ((h / splits) % VecTraits<T>::kWidth != 0) return false;
    return aligned16(x);
}

template <typename T>
bool vec_ok_scale(const void* x, const void* w, const void* y, int h) {
    if (h % VecTraits<T>::kWidth != 0) return false;
    return aligned16(x) && aligned16(w) && aligned16(y);
}

template <typename T> struct DtypeOf;
template <> struct DtypeOf<float> { static constexpr int kIsHalf = 0; };
template <> struct DtypeOf<__half> { static constexpr int kIsHalf = 1; };
// （预留：本期 dispatch 未引用；保留供后续按 dtype 元编程扩展）

// ---- 档 A dispatch：T × ACC × TILE × VEC 全特化展开 -----------------------
template <typename T, typename ACC>
int dispatch_shared(const void* x, const void* w, void* y, int rows, int h, float eps,
                    int tile, bool acc_selected) {
    if (!acc_selected) return 0;   // ACC 与请求不符时不发射（由调用方双调用）
    const unsigned grid = static_cast<unsigned>(rows < kMaxGridBlocks ? rows : kMaxGridBlocks);
    const bool vec = vec_ok_det<T>(x, w, y, h);
    switch (tile) {
        case 128:
            if (vec) rmsnorm_atomic_shared_kernel<T, 128, true, ACC>
                <<<grid, 128, 0, 0>>>(static_cast<const T*>(x), static_cast<const T*>(w),
                                      static_cast<T*>(y), rows, h, eps);
            else     rmsnorm_atomic_shared_kernel<T, 128, false, ACC>
                <<<grid, 128, 0, 0>>>(static_cast<const T*>(x), static_cast<const T*>(w),
                                      static_cast<T*>(y), rows, h, eps);
            return 0;
        case 256:
            if (vec) rmsnorm_atomic_shared_kernel<T, 256, true, ACC>
                <<<grid, 256, 0, 0>>>(static_cast<const T*>(x), static_cast<const T*>(w),
                                      static_cast<T*>(y), rows, h, eps);
            else     rmsnorm_atomic_shared_kernel<T, 256, false, ACC>
                <<<grid, 256, 0, 0>>>(static_cast<const T*>(x), static_cast<const T*>(w),
                                      static_cast<T*>(y), rows, h, eps);
            return 0;
        case 512:
            if (vec) rmsnorm_atomic_shared_kernel<T, 512, true, ACC>
                <<<grid, 512, 0, 0>>>(static_cast<const T*>(x), static_cast<const T*>(w),
                                      static_cast<T*>(y), rows, h, eps);
            else     rmsnorm_atomic_shared_kernel<T, 512, false, ACC>
                <<<grid, 512, 0, 0>>>(static_cast<const T*>(x), static_cast<const T*>(w),
                                      static_cast<T*>(y), rows, h, eps);
            return 0;
        default:
            set_detail("unsupported_tile");
            return 1;
    }
}

// ---- 档 B dispatch ---------------------------------------------------------
template <typename T, typename ACC>
int dispatch_global(const void* x, const void* w, void* g_acc, void* y, int rows, int h,
                    float eps, int tile, int splits, bool acc_selected) {
    if (!acc_selected) return 0;
    const long long grid_blocks = static_cast<long long>(rows) * splits;
    const unsigned grid = static_cast<unsigned>(
        grid_blocks < kMaxGridBlocks ? grid_blocks : kMaxGridBlocks);
    const unsigned scale_grid = static_cast<unsigned>(
        rows < kMaxGridBlocks ? rows : kMaxGridBlocks);
    const bool vec_sq = vec_ok_global_sqsum<T>(x, h, splits);
    const bool vec_sc = vec_ok_scale<T>(x, w, y, h);
    switch (tile) {
        case 128:
            if (vec_sq) rmsnorm_sqsum_atomic_global_kernel<T, 128, true, ACC>
                <<<grid, 128, 0, 0>>>(static_cast<const T*>(x), static_cast<ACC*>(g_acc), rows, h, splits);
            else        rmsnorm_sqsum_atomic_global_kernel<T, 128, false, ACC>
                <<<grid, 128, 0, 0>>>(static_cast<const T*>(x), static_cast<ACC*>(g_acc), rows, h, splits);
            if (vec_sc) rmsnorm_scale_global_kernel<T, true, ACC>
                <<<scale_grid, 256, 0, 0>>>(static_cast<const T*>(x), static_cast<const T*>(w),
                                            static_cast<T*>(y), static_cast<const ACC*>(g_acc), rows, h, eps);
            else        rmsnorm_scale_global_kernel<T, false, ACC>
                <<<scale_grid, 256, 0, 0>>>(static_cast<const T*>(x), static_cast<const T*>(w),
                                            static_cast<T*>(y), static_cast<const ACC*>(g_acc), rows, h, eps);
            return 0;
        case 256:
            if (vec_sq) rmsnorm_sqsum_atomic_global_kernel<T, 256, true, ACC>
                <<<grid, 256, 0, 0>>>(static_cast<const T*>(x), static_cast<ACC*>(g_acc), rows, h, splits);
            else        rmsnorm_sqsum_atomic_global_kernel<T, 256, false, ACC>
                <<<grid, 256, 0, 0>>>(static_cast<const T*>(x), static_cast<ACC*>(g_acc), rows, h, splits);
            if (vec_sc) rmsnorm_scale_global_kernel<T, true, ACC>
                <<<scale_grid, 256, 0, 0>>>(static_cast<const T*>(x), static_cast<const T*>(w),
                                            static_cast<T*>(y), static_cast<const ACC*>(g_acc), rows, h, eps);
            else        rmsnorm_scale_global_kernel<T, false, ACC>
                <<<scale_grid, 256, 0, 0>>>(static_cast<const T*>(x), static_cast<const T*>(w),
                                            static_cast<T*>(y), static_cast<const ACC*>(g_acc), rows, h, eps);
            return 0;
        case 512:
            if (vec_sq) rmsnorm_sqsum_atomic_global_kernel<T, 512, true, ACC>
                <<<grid, 512, 0, 0>>>(static_cast<const T*>(x), static_cast<ACC*>(g_acc), rows, h, splits);
            else        rmsnorm_sqsum_atomic_global_kernel<T, 512, false, ACC>
                <<<grid, 512, 0, 0>>>(static_cast<const T*>(x), static_cast<ACC*>(g_acc), rows, h, splits);
            if (vec_sc) rmsnorm_scale_global_kernel<T, true, ACC>
                <<<scale_grid, 256, 0, 0>>>(static_cast<const T*>(x), static_cast<const T*>(w),
                                            static_cast<T*>(y), static_cast<const ACC*>(g_acc), rows, h, eps);
            else        rmsnorm_scale_global_kernel<T, false, ACC>
                <<<scale_grid, 256, 0, 0>>>(static_cast<const T*>(x), static_cast<const T*>(w),
                                            static_cast<T*>(y), static_cast<const ACC*>(g_acc), rows, h, eps);
            return 0;
        default:
            set_detail("unsupported_tile");
            return 1;
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// 确定性版启动器（ep03 kernel，1 launch）
// ---------------------------------------------------------------------------
extern "C" int rmsnorm_det_rows(long long x_ptr, long long w_ptr, long long y_ptr,
                                int is_fp16, int rows, int h, float eps,
                                int tile, int warp_size) {
    g_status_detail[0] = '\0';
    if (h <= 0) { set_detail("h<=0"); return 2; }
    if (rows < 0) { set_detail("rows<0"); return 3; }
    if (rows == 0) return 0;
    if (tile < warp_size || tile % warp_size != 0) { set_detail("tile_not_warp_multiple"); return 1; }
    try {
        const void* x = reinterpret_cast<const void*>(static_cast<uintptr_t>(x_ptr));
        const void* w = reinterpret_cast<const void*>(static_cast<uintptr_t>(w_ptr));
        void* y = reinterpret_cast<void*>(static_cast<uintptr_t>(y_ptr));
        if (is_fp16) {
            launch_rmsnorm_fp16(x, w, y, rows, h, eps, tile, 0, warp_size);
        } else {
            launch_rmsnorm_fp32(static_cast<const float*>(x), static_cast<const float*>(w),
                                static_cast<float*>(y), rows, h, eps, tile, 0, warp_size);
        }
    } catch (const std::exception& error) {
        set_detail(error.what());
        return 10;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// 档 A：shared atomic 启动器（1 launch）
// ---------------------------------------------------------------------------
extern "C" int rmsnorm_atomic_shared_rows(long long x_ptr, long long w_ptr, long long y_ptr,
                                          int is_fp16, int acc_is_half, int rows, int h,
                                          float eps, int tile, int warp_size) {
    g_status_detail[0] = '\0';
    if (h <= 0) { set_detail("h<=0"); return 2; }
    if (rows < 0) { set_detail("rows<0"); return 3; }
    if (rows == 0) return 0;
    if (tile < warp_size || tile % warp_size != 0) { set_detail("tile_not_warp_multiple"); return 1; }
    if (acc_is_half && kHalfAtomicEnabled != 1) { set_detail("half_atomic_disabled"); return 4; }

    const void* x = reinterpret_cast<const void*>(static_cast<uintptr_t>(x_ptr));
    const void* w = reinterpret_cast<const void*>(static_cast<uintptr_t>(w_ptr));
    void* y = reinterpret_cast<void*>(static_cast<uintptr_t>(y_ptr));
    if (is_fp16) {
        const bool want_half = acc_is_half != 0;
        int status = dispatch_shared<__half, float>(x, w, y, rows, h, eps, tile, !want_half);
        if (status) return status;
#if !defined(EP08_NO_HALF_ATOMIC)
        status = dispatch_shared<__half, __half>(x, w, y, rows, h, eps, tile, want_half);
        if (status) return status;
#endif
    } else {
        const bool want_half = acc_is_half != 0;
        int status = dispatch_shared<float, float>(x, w, y, rows, h, eps, tile, !want_half);
        if (status) return status;
#if !defined(EP08_NO_HALF_ATOMIC)
        status = dispatch_shared<float, __half>(x, w, y, rows, h, eps, tile, want_half);
        if (status) return status;
#endif
    }
    return 0;
}

// ---------------------------------------------------------------------------
// 档 B：global atomic 启动器（2 launches：sqsum + scale；调用前须 zero_acc）
// ---------------------------------------------------------------------------
extern "C" int rmsnorm_atomic_global_rows(long long x_ptr, long long w_ptr,
                                          long long g_acc_ptr, long long y_ptr,
                                          int is_fp16, int acc_is_half, int rows, int h,
                                          float eps, int tile, int warp_size, int splits) {
    g_status_detail[0] = '\0';
    if (h <= 0) { set_detail("h<=0"); return 2; }
    if (rows < 0) { set_detail("rows<0"); return 3; }
    if (rows == 0) return 0;
    if (splits <= 0 || splits > 1024) { set_detail("splits_range"); return 5; }
    if (tile < warp_size || tile % warp_size != 0) { set_detail("tile_not_warp_multiple"); return 1; }
    if (acc_is_half && kHalfAtomicEnabled != 1) { set_detail("half_atomic_disabled"); return 4; }

    const void* x = reinterpret_cast<const void*>(static_cast<uintptr_t>(x_ptr));
    const void* w = reinterpret_cast<const void*>(static_cast<uintptr_t>(w_ptr));
    void* g_acc = reinterpret_cast<void*>(static_cast<uintptr_t>(g_acc_ptr));
    void* y = reinterpret_cast<void*>(static_cast<uintptr_t>(y_ptr));
    const bool want_half = acc_is_half != 0;
    if (is_fp16) {
        int status = dispatch_global<__half, float>(x, w, g_acc, y, rows, h, eps, tile,
                                                    splits, !want_half);
        if (status) return status;
#if !defined(EP08_NO_HALF_ATOMIC)
        status = dispatch_global<__half, __half>(x, w, g_acc, y, rows, h, eps, tile,
                                                 splits, want_half);
        if (status) return status;
#endif
    } else {
        int status = dispatch_global<float, float>(x, w, g_acc, y, rows, h, eps, tile,
                                                   splits, !want_half);
        if (status) return status;
#if !defined(EP08_NO_HALF_ATOMIC)
        status = dispatch_global<float, __half>(x, w, g_acc, y, rows, h, eps, tile,
                                                splits, want_half);
        if (status) return status;
#endif
    }
    return 0;
}

// ---------------------------------------------------------------------------
// g_acc 清零（档 B 前置；1 launch）
// ---------------------------------------------------------------------------
extern "C" int zero_acc(long long g_acc_ptr, int rows, int acc_is_half) {
    g_status_detail[0] = '\0';
    if (rows < 0) { set_detail("rows<0"); return 3; }
    if (rows == 0) return 0;
    void* g_acc = reinterpret_cast<void*>(static_cast<uintptr_t>(g_acc_ptr));
    const int threads = 256;
    const int blocks = static_cast<int>((rows + threads - 1) / threads);
    if (acc_is_half) {
#if defined(EP08_NO_HALF_ATOMIC)
        set_detail("half_atomic_disabled");
        return 4;
#else
        zero_acc_kernel<__half><<<blocks, threads, 0, 0>>>(static_cast<__half*>(g_acc), rows);
#endif
    } else {
        zero_acc_kernel<float><<<blocks, threads, 0, 0>>>(static_cast<float*>(g_acc), rows);
    }
    return 0;
}

// ---------------------------------------------------------------------------
// warpSize 探针（M-20：host guard 用实测值，不依赖 C API warpSize 口径）
// ---------------------------------------------------------------------------
__global__ void ep08_warp_size_probe_kernel(int* out) { *out = static_cast<int>(warpSize); }

extern "C" int warp_size_probe(long long out_ptr) {
    int* out = reinterpret_cast<int*>(static_cast<uintptr_t>(out_ptr));
    ep08_warp_size_probe_kernel<<<1, 1, 0, 0>>>(out);
    return 0;
}
