// ============================================================================
// rmsnorm_ext.cu —— 第 02 期 MUSA torch 扩展包装（CUDA 版 rmsnorm_ext.cu
// 的逐文件迁移；扩展机制：torch_musa.utils.musa_extension.load，见
// build_ext.py / build.sh；若该路径不通，退到 rmsnorm_bench 文件 IO 对比
// 路径并记录到迁移台账）。
//
// ✅ 实机核对（S4000，2026-11-21）：
//   1. torch_musa 设备类型：PrivateUse1（c10::DeviceType::PrivateUse1）；
//   2. musa_extension.load 只识别 .mu/.muh 后缀（_is_musa_file），且被
//      编译的 .mu 不能再 #include ".cu"（mcc 按后缀分派语言）——构建时
//      源复制为 .mu 并把 include 行同步改写（build_ext.py，语义改动）；
//   3. 流获取：c10::musa::MUSAStream.h 在 torch_musa 2.11 的
//      generated_cuda_compatible 头树中存在，但与 torch 官方头同译时
//      产生 redefinition（见 build/ext_build.log，平台边界记录）——
//      变体 0 尝试 c10::musa::getCurrentMUSAStream()；失败则变体 1
//      用默认流（torch_musa eager 单流场景等价，多流语义第 08 期）。
// ============================================================================

#include <torch/extension.h>

#include "rmsnorm_kernel.cu"

// 流获取变体（build_ext.py 依次尝试；见上方说明）。
#if !defined(RMSNORM_MUSA_STREAM_VARIANT)
#define RMSNORM_MUSA_STREAM_VARIANT 0
#endif

#if RMSNORM_MUSA_STREAM_VARIANT == 0
#include <c10/musa/MUSAGuard.h>    // 实机确认存在（torch_musa generated_cuda_compatible 头树）
#include <c10/musa/MUSAStream.h>   // 实机确认存在，但与 torch 官方头同译时 redefinition
#define RMSNORM_CURRENT_STREAM() c10::musa::getCurrentMUSAStream()
#define RMSNORM_DEVICE_GUARD(device) const c10::musa::MUSAGuard device_guard(device);
#else
// 变体 1：默认流。torch_musa eager 默认在单流上执行（第 01/01 期脚本未用
// side stream），扩展在此发射与框架当前流等价；多 stream 语义第 08 期。
#define RMSNORM_CURRENT_STREAM() 0
#define RMSNORM_DEVICE_GUARD(device)
#endif

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
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1,
                "本扩展是 MUSA 实现，不接受 ", x.device(), " Tensor（显式报错，不静默回退）");
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous(),
                "合同：本期 Kernel 只接受连续输入（非连续布局是第 06 期主题）");
    TORCH_CHECK(x.size(1) > 0, "合同：H=0 为空 Tensor，显式报错（不支持静默处理）");
    TORCH_CHECK(tile == 64 || tile == 128 || tile == 256 || tile == 512 || tile == 1024,
                "tile 必须是编译期实例化集合 {64,128,256,512,1024} 之一，实际 ", tile);

    RMSNORM_DEVICE_GUARD(x.device())
    auto y = at::empty_like(x);
    const int rows = static_cast<int>(x.size(0));
    const int h = static_cast<int>(x.size(1));
    if (rows == 0) {
        return y;  // rows=0：无行可算，返回空 Tensor（合同条目）
    }
    mini_infra::launch_rmsnorm_any(x.data_ptr(), w.data_ptr(), y.data_ptr(),
                                   x.scalar_type() == at::kHalf, rows, h,
                                   static_cast<float>(eps), static_cast<int>(tile),
                                   RMSNORM_CURRENT_STREAM());
    return y;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("rmsnorm", &rmsnorm, py::arg("x"), py::arg("w"),
               py::arg("eps") = 1e-6, py::arg("tile") = 256,
               "第 02 期手写 RMSNorm（MUSA 迁移版）：每 block 一行 + warp 归约 + shared 跨 warp 归约");
}
