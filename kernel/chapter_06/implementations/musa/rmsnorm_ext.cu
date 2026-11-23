// ============================================================================
// rmsnorm_ext.cu —— 第 06 期 MUSA torch 扩展：三合同策略（CUDA 版逐文件迁移）
//
// 迁移面（继承第 02 期实机核对结论 M-08/M-09/M-10/M-27）：
//   - 设备类型判断 c10::DeviceType::CUDA → PrivateUse1（torch_musa 注册）；
//   - 流获取两变体（build_ext.py 依次尝试）：
//     变体 0 c10::musa::getCurrentMUSAStream()（可能与 torch 官方头
//     redefinition，M-10 族）；变体 1 默认流（torch_musa eager 单流等价）；
//   - guard c10::musa::MUSAGuard（变体 0）/无（变体 1）；
//   - 三策略的合同校验与 pybind 结构与 CUDA 版逐字一致（行为对照的前提：
//     同一策略语义在两个平台可逐项对照——这正是大纲第 06 期 MUSA 条目的要求）。
// ============================================================================

#include <torch/extension.h>

#include "rmsnorm_stride_kernel.cu"

// 流获取变体（build_ext.py 依次尝试；见上方说明）。
#if !defined(RMSNORM_MUSA_STREAM_VARIANT)
#define RMSNORM_MUSA_STREAM_VARIANT 0
#endif

#if RMSNORM_MUSA_STREAM_VARIANT == 0
#include <c10/musa/MUSAGuard.h>
#include <c10/musa/MUSAStream.h>
#define RMSNORM_CURRENT_STREAM() c10::musa::getCurrentMUSAStream()
#define RMSNORM_DEVICE_GUARD(device) const c10::musa::MUSAGuard device_guard(device);
#else
#define RMSNORM_CURRENT_STREAM() 0
#define RMSNORM_DEVICE_GUARD(device)
#endif

namespace {

void check_common(const at::Tensor& x, const at::Tensor& w) {
    TORCH_CHECK(x.dim() == 2, "x 必须是 [rows, H] 二维 Tensor，实际 dim=", x.dim());
    TORCH_CHECK(w.dim() == 1 && w.numel() == x.size(-1),
                "w 必须是 [H] 且与 x 的 H 一致：w.numel()=", w.numel(),
                " x.size(-1)=", x.size(-1));
    TORCH_CHECK(x.scalar_type() == w.scalar_type(),
                "x 与 w dtype 必须一致，实际 ", x.scalar_type(), " vs ", w.scalar_type());
    TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf,
                "本期合同 dtype ∈ {float32, float16}，实际 ", x.scalar_type());
    TORCH_CHECK(x.device() == w.device(), "x 与 w 必须在同一设备");
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1,
                "本扩展是 MUSA 实现，不接受 ", x.device(), " Tensor（显式报错，不静默回退）");
    TORCH_CHECK(x.size(1) > 0, "合同：H=0 为空 Tensor，显式报错（不支持静默处理）");
    TORCH_CHECK(w.is_contiguous(), "合同：w 必须连续（[H] 权重无非连续用例）");
}

void check_tile(int64_t tile) {
    TORCH_CHECK(tile == 64 || tile == 128 || tile == 256 || tile == 512 || tile == 1024,
                "tile 必须是编译期实例化集合 {64,128,256,512,1024} 之一，实际 ", tile);
}

at::Tensor rmsnorm_policy_a(const at::Tensor& x, const at::Tensor& w, double eps, int64_t tile) {
    check_common(x, w);
    check_tile(tile);
    TORCH_CHECK(x.is_contiguous(),
                "策略 A（reject）：合同只接受连续输入，实际 x.is_contiguous()=false，"
                "x.shape=", x.sizes(), " x.stride()=", x.strides(),
                "。处理选项：调用方先 x.contiguous()（一次性，可摊销），"
                "或改用本扩展的 rmsnorm_policy_b/c。非连续拒绝是显式合同行为。");
    RMSNORM_DEVICE_GUARD(x.device())
    auto y = at::empty({x.size(0), x.size(1)}, x.options());
    const int rows = static_cast<int>(x.size(0));
    if (rows == 0) {
        return y;
    }
    const int h = static_cast<int>(x.size(1));
    mini_infra::launch_rmsnorm_any(x.data_ptr(), w.data_ptr(), y.data_ptr(),
                                   x.scalar_type() == at::kHalf, rows, h,
                                   static_cast<float>(eps), static_cast<int>(tile),
                                   RMSNORM_CURRENT_STREAM());
    return y;
}

at::Tensor rmsnorm_policy_b(const at::Tensor& x, const at::Tensor& w, double eps, int64_t tile) {
    check_common(x, w);
    check_tile(tile);
    RMSNORM_DEVICE_GUARD(x.device())
    at::Tensor xc = x.is_contiguous() ? x : x.contiguous();  // 隐性拷贝点（非连续时）
    auto y = at::empty({xc.size(0), xc.size(1)}, xc.options());
    const int rows = static_cast<int>(xc.size(0));
    if (rows == 0) {
        return y;
    }
    const int h = static_cast<int>(xc.size(1));
    mini_infra::launch_rmsnorm_any(xc.data_ptr(), w.data_ptr(), y.data_ptr(),
                                   xc.scalar_type() == at::kHalf, rows, h,
                                   static_cast<float>(eps), static_cast<int>(tile),
                                   RMSNORM_CURRENT_STREAM());
    return y;
}

at::Tensor rmsnorm_policy_c(const at::Tensor& x, const at::Tensor& w, double eps, int64_t tile) {
    check_common(x, w);
    check_tile(tile);
    RMSNORM_DEVICE_GUARD(x.device())
    const int rows = static_cast<int>(x.size(0));
    const int h = static_cast<int>(x.size(1));
    auto y = at::empty({x.size(0), x.size(1)}, x.options());
    if (rows == 0) {
        return y;
    }
    const long stride_row = rows > 1 ? static_cast<long>(x.stride(0)) : 0L;
    const long stride_col = static_cast<long>(x.stride(1));
    mini_infra::launch_rmsnorm_strided_any(x.data_ptr(), w.data_ptr(), y.data_ptr(),
                                           x.scalar_type() == at::kHalf, rows, h,
                                           stride_row, stride_col,
                                           static_cast<float>(eps),
                                           static_cast<int>(tile),
                                           RMSNORM_CURRENT_STREAM());
    return y;
}

at::Tensor rmsnorm_layout_copy(const at::Tensor& x) {
    TORCH_CHECK(x.dim() == 2, "layout_copy 只面向 [rows, H] 二维 Tensor");
    RMSNORM_DEVICE_GUARD(x.device())
    return x.contiguous();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("rmsnorm_policy_a", &rmsnorm_policy_a, py::arg("x"), py::arg("w"),
               py::arg("eps") = 1e-6, py::arg("tile") = 256,
               "策略 A：连续 only，非连续显式报错");
    module.def("rmsnorm_policy_b", &rmsnorm_policy_b, py::arg("x"), py::arg("w"),
               py::arg("eps") = 1e-6, py::arg("tile") = 256,
               "策略 B：内部 .contiguous() 后走连续 kernel（隐性拷贝）");
    module.def("rmsnorm_policy_c", &rmsnorm_policy_c, py::arg("x"), py::arg("w"),
               py::arg("eps") = 1e-6, py::arg("tile") = 256,
               "策略 C：stride 寻址 kernel，零拷贝吃任意布局");
    module.def("layout_copy", &rmsnorm_layout_copy, py::arg("x"),
               "布局归一化拷贝（x.contiguous() 裸成本）");
}
