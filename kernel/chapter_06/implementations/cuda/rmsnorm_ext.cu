// ============================================================================
// rmsnorm_ext.cu —— 第 06 期 torch 扩展：同一 RMSNorm Kernel 的三种合同策略
//
//   策略 A（reject）     ：is_contiguous 检查，非连续输入 TORCH_CHECK 显式报错；
//   策略 B（auto-copy）  ：内部 x.contiguous() 物化行主序副本后走连续 kernel
//                          ——隐性拷贝发生在这一行（成本由 benchmark 量化）；
//   策略 C（strided）    ：stride 寻址 kernel 直接吃任意布局（零拷贝，访存
//                          模式由 stride 决定——教学点：合并访存的代价）。
//
// 另提供 layout_copy（x.contiguous() 单独测拷贝成本）。
// 公共校验（shape/dtype/device/空 Tensor）三策略一致；差别只有 stride 行为
// ——这正是算子合同"是否允许非连续 Tensor"条目的可执行形态。
//
// 合同补充条目（第 06 期新增，全策略一致）：
//   - 输出 y 恒为行主序连续 [rows, H]（三策略都是 at::empty({rows,H}) 新分配，
//     不继承输入布局——调用方永远拿到连续输出，不放大布局传染）；
//   - w 恒要求连续（[H] 权重向量没有非连续用例）；
//   - expand（stride=0）合法：stride_row=0 即所有行读同一 buffer 行（策略 C
//     支持且数值正确——广播输入的教学用例）。
// ============================================================================

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "rmsnorm_stride_kernel.cu"  // 内部再 include rmsnorm_kernel.cu（单编译单元）

namespace {

// 三策略共享的公共校验（与第 02 期合同一致，差别只在 stride 行为）。
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
    TORCH_CHECK(x.device().type() == c10::DeviceType::CUDA,
                "本扩展是 CUDA 实现，不接受 ", x.device(), " Tensor（显式报错，不静默回退）");
    TORCH_CHECK(x.size(1) > 0, "合同：H=0 为空 Tensor，显式报错（不支持静默处理）");
    TORCH_CHECK(w.is_contiguous(), "合同：w 必须连续（[H] 权重无非连续用例）");
}

void check_tile(int64_t tile) {
    TORCH_CHECK(tile == 64 || tile == 128 || tile == 256 || tile == 512 || tile == 1024,
                "tile 必须是编译期实例化集合 {64,128,256,512,1024} 之一，实际 ", tile);
}

void launch_common(const at::Tensor& x, const at::Tensor& w, at::Tensor& y,
                   double eps, int64_t tile) {
    const int rows = static_cast<int>(x.size(0));
    const int h = static_cast<int>(x.size(1));
    if (rows == 0) {
        return;  // rows=0：无行可算（合同条目），调用方已返回空 Tensor
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    mini_infra::launch_rmsnorm_any(x.data_ptr(), w.data_ptr(), y.data_ptr(),
                                   x.scalar_type() == at::kHalf, rows, h,
                                   static_cast<float>(eps), static_cast<int>(tile), stream);
}

// ---------------------------------------------------------------------------
// 策略 A（reject）：非连续输入显式报错——合同策略的可执行形态。
// ---------------------------------------------------------------------------
at::Tensor rmsnorm_policy_a(const at::Tensor& x, const at::Tensor& w, double eps, int64_t tile) {
    check_common(x, w);
    check_tile(tile);
    TORCH_CHECK(x.is_contiguous(),
                "策略 A（reject）：合同只接受连续输入，实际 x.is_contiguous()=false，"
                "x.shape=", x.sizes(), " x.stride()=", x.strides(),
                "。处理选项：调用方先 x.contiguous()（一次性，可摊销），"
                "或改用本扩展的 rmsnorm_policy_b/c。非连续拒绝是显式合同行为。");
    at::cuda::CUDAGuard device_guard(x.device());
    auto y = at::empty({x.size(0), x.size(1)}, x.options());
    launch_common(x, w, y, eps, tile);
    return y;
}

// ---------------------------------------------------------------------------
// 策略 B（auto-copy）：内部 .contiguous()——隐性拷贝在这两行之间发生。
// ---------------------------------------------------------------------------
at::Tensor rmsnorm_policy_b(const at::Tensor& x, const at::Tensor& w, double eps, int64_t tile) {
    check_common(x, w);
    check_tile(tile);
    at::cuda::CUDAGuard device_guard(x.device());
    at::Tensor xc = x.is_contiguous() ? x : x.contiguous();  // 隐性拷贝点（非连续时）
    auto y = at::empty({xc.size(0), xc.size(1)}, xc.options());
    launch_common(xc, w, y, eps, tile);
    return y;
}

// ---------------------------------------------------------------------------
// 策略 C（strided）：stride 寻址 kernel，零拷贝直接吃任意布局。
// ---------------------------------------------------------------------------
at::Tensor rmsnorm_policy_c(const at::Tensor& x, const at::Tensor& w, double eps, int64_t tile) {
    check_common(x, w);
    check_tile(tile);
    at::cuda::CUDAGuard device_guard(x.device());
    const int rows = static_cast<int>(x.size(0));
    const int h = static_cast<int>(x.size(1));
    auto y = at::empty({x.size(0), x.size(1)}, x.options());
    if (rows == 0) {
        return y;
    }
    // torch stride 以元素为单位，与 kernel 的 stride_row/stride_col 同单位。
    // rows==1 时 stride(0) 无意义（torch 可能报任意值），kernel 只用 row=0。
    const long stride_row = rows > 1 ? static_cast<long>(x.stride(0)) : 0L;
    const long stride_col = static_cast<long>(x.stride(1));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    mini_infra::launch_rmsnorm_strided_any(x.data_ptr(), w.data_ptr(), y.data_ptr(),
                                           x.scalar_type() == at::kHalf, rows, h,
                                           stride_row, stride_col,
                                           static_cast<float>(eps),
                                           static_cast<int>(tile), stream);
    return y;
}

// layout_copy：把布局归一化拷贝单独暴露（测 .contiguous() 的裸成本）。
at::Tensor rmsnorm_layout_copy(const at::Tensor& x) {
    TORCH_CHECK(x.dim() == 2, "layout_copy 只面向 [rows, H] 二维 Tensor");
    at::cuda::CUDAGuard device_guard(x.device());
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
