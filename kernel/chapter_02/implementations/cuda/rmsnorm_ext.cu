// ============================================================================
// rmsnorm_ext.cu —— 第 02 期 CUDA torch 扩展包装（选择：cpp_extension load +
// PYBIND11_MODULE；TORCH_LIBRARY 注册路径留给第 04 期"进入框架 Dispatch"）。
//
// 职责：
//   - 合同校验（Shape/dtype/连续性/空 Tensor），显式报错，绝不静默回退；
//   - 获取当前 CUDA Stream（与 torch 执行流对齐，避免隐式同步）；
//   - 分配输出 Tensor，调用 rmsnorm_kernel.cu 的 launch 函数。
// 编译：见 build.sh（torch.utils.cpp_extension.load，构建产物在服务器本地，
// 上传前清理）。
// ============================================================================

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "rmsnorm_kernel.cu"

namespace {

at::Tensor rmsnorm(const at::Tensor& x, const at::Tensor& w, double eps, int64_t tile) {
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
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous(),
                "合同：本期 Kernel 只接受连续输入（非连续布局是第 06 期主题）");
    TORCH_CHECK(x.size(1) > 0, "合同：H=0 为空 Tensor，显式报错（不支持静默处理）");
    TORCH_CHECK(tile == 64 || tile == 128 || tile == 256 || tile == 512 || tile == 1024,
                "tile 必须是编译期实例化集合 {64,128,256,512,1024} 之一，实际 ", tile);

    const at::cuda::CUDAGuard device_guard(x.device());
    auto y = at::empty_like(x);
    const int rows = static_cast<int>(x.size(0));
    const int h = static_cast<int>(x.size(1));
    if (rows == 0) {
        return y;  // rows=0：无行可算，返回空 Tensor（合同条目）
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    mini_infra::launch_rmsnorm_any(x.data_ptr(), w.data_ptr(), y.data_ptr(),
                                   x.scalar_type() == at::kHalf, rows, h,
                                   static_cast<float>(eps), static_cast<int>(tile), stream);
    return y;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("rmsnorm", &rmsnorm, py::arg("x"), py::arg("w"),
               py::arg("eps") = 1e-6, py::arg("tile") = 256,
               "第 02 期手写 RMSNorm：每 block 一行 + warp 归约 + shared 跨 warp 归约");
}
