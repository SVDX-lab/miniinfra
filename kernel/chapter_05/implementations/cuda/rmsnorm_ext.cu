// ============================================================================
// rmsnorm_ext.cu —— 第 05 期 CUDA torch 扩展包装（结构沿用第 02 期）。
//
// 对外接口：rmsnorm(x, w, eps=1e-6, tile=256, acc="fp32")
//   x/w: 同 dtype ∈ {float32, float16, bfloat16}，[rows,H] / [H]，CUDA 连续；
//   acc: 平方和累加 dtype ∈ {"fp32"（合同默认）, "fp16"（实验 4 对照实现）}；
//   返回与 x 同 dtype 的 y。dtype/设备/连续性全部显式检查（合同：不静默回退）。
// ============================================================================

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "rmsnorm_kernel.cu"

namespace {

at::Tensor rmsnorm(const at::Tensor& x, const at::Tensor& w, double eps, int64_t tile,
                   std::string acc) {
    TORCH_CHECK(x.dim() == 2, "x 必须是 [rows, H] 二维 Tensor，实际 dim=", x.dim());
    TORCH_CHECK(w.dim() == 1 && w.numel() == x.size(-1),
                "w 必须是 [H] 且与 x 的 H 一致：w.numel()=", w.numel(),
                " x.size(-1)=", x.size(-1));
    TORCH_CHECK(x.scalar_type() == w.scalar_type(),
                "x 与 w dtype 必须一致，实际 ", x.scalar_type(), " vs ", w.scalar_type());
    TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf ||
                    x.scalar_type() == at::kBFloat16,
                "本期合同 dtype ∈ {float32, float16, bfloat16}，实际 ", x.scalar_type());
    TORCH_CHECK(x.device().type() == c10::DeviceType::CUDA,
                "本扩展是 CUDA 实现，不接受 ", x.device(), " Tensor（显式报错，不静默回退）");
    TORCH_CHECK(x.device() == w.device(), "x 与 w 必须在同一设备");
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous(),
                "合同：本期 Kernel 只接受连续输入（非连续布局是第 06 期主题）");
    TORCH_CHECK(x.size(1) > 0, "合同：H=0 为空 Tensor，显式报错（不支持静默处理）");
    TORCH_CHECK(tile == 64 || tile == 128 || tile == 256 || tile == 512 || tile == 1024,
                "tile 必须是编译期实例化集合 {64,128,256,512,1024} 之一，实际 ", tile);
    TORCH_CHECK(acc == "fp32" || acc == "fp16",
                "acc 只支持 'fp32'（合同默认）或 'fp16'（实验对照），实际 ", acc);

    const c10::cuda::CUDAGuard device_guard(x.device());
    auto y = at::empty_like(x);
    const int rows = static_cast<int>(x.size(0));
    const int h = static_cast<int>(x.size(1));
    if (rows == 0) {
        return y;  // rows=0：无行可算，返回空 Tensor（合同条目）
    }
    int dtype_code = 0;
    if (x.scalar_type() == at::kHalf) dtype_code = 1;
    else if (x.scalar_type() == at::kBFloat16) dtype_code = 2;
    const int acc_code = (acc == "fp16") ? 1 : 0;
    mini_infra::launch_rmsnorm(x.data_ptr(), w.data_ptr(), y.data_ptr(), dtype_code,
                               acc_code, rows, h, static_cast<float>(eps),
                               static_cast<int>(tile),
                               at::cuda::getCurrentCUDAStream());
    return y;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("rmsnorm", &rmsnorm, py::arg("x"), py::arg("w"),
               py::arg("eps") = 1e-6, py::arg("tile") = 256, py::arg("acc") = "fp32",
               "第 05 期手写 RMSNorm：FP32/FP16/BF16 三条编译路径 + 累加 dtype 编译期可选");
}
