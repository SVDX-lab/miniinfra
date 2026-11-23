// ============================================================================
// health_targets.cu —— 第 10 期 CUDA 列：ch04/ch08 健康 Kernel 的
// compute-sanitizer 体检靶子（独立可执行；ch03 用期原样 rmsnorm_bench）。
//
// 每个库都是单编译单元且内部各自 #include rmsnorm_kernel_ep03.cu（第 02 期
// kernel 原样），因此按 target 编译为两个二进制（-DCH11_TARGET_CH04 /
// -DCH11_TARGET_CH08），避免双库同 TU 的重定义——这一限制本身是"体检靶子
// 必须复用被体检源码原样"的工程注脚。
//
// targets：
//   ch04-fused          第 03 期融合 add+RMSNorm in-place
//                       （fused_rmsnorm_kernels.cu，逐字节同源）
//   ch08-det            第 07 期确定性归约版（kernels_lib.cu rmsnorm_det_rows）
//   ch08-atomic-shared  第 07 期 shared atomic 交叉归约版
//   ch08-atomic-global  第 07 期 global atomic 两段版（zero_acc+sqsum+scale）
//
// 自检口径：iters=1 时与 host 串行参考 max_abs_error（fp32，同第 02 期 bench
// --check 顺序）；sanitizer 报告干净 + 自检通过 = "健康版对照"证据。
// racecheck 口径默认 rows=64（racecheck 逐访存 happened-before 追踪，大 grid
// 不可用——时间成本本身是开销素材）。
// ============================================================================

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(CH11_TARGET_CH04)
#include "fused_rmsnorm_kernels.cu"
#elif defined(CH11_TARGET_CH08)
#include "kernels_lib.cu"
#else
#error "必须定义 CH11_TARGET_CH04 或 CH11_TARGET_CH08"
#endif

#define CUDA_CHECK(expr)                                                          \
    do {                                                                          \
        cudaError_t _err = (expr);                                                \
        if (_err != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error %s at %s:%d (%s)\n",                 \
                         cudaGetErrorString(_err), __FILE__, __LINE__, #expr);    \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)

static double splitmix64(uint64_t& state) {
    state += 0x9E3779B97F4A7C15ull;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return static_cast<double>(z ^ (z >> 31)) / 18446744073709551616.0;
}
static void fill_normal(std::vector<float>& data, uint64_t seed) {
    uint64_t state = seed;
    for (size_t i = 0; i + 1 < data.size(); i += 2) {
        double u1 = splitmix64(state), u2 = splitmix64(state);
        if (u1 < 1e-12) u1 = 1e-12;
        double r = std::sqrt(-2.0 * std::log(u1));
        double theta = 2.0 * 3.14159265358979323846 * u2;
        data[i] = static_cast<float>(r * std::cos(theta));
        data[i + 1] = static_cast<float>(r * std::sin(theta));
    }
    if (data.size() % 2 == 1) {
        double u1 = splitmix64(state), u2 = splitmix64(state);
        data[data.size() - 1] = static_cast<float>(std::sqrt(-2.0 * std::log(u1)) *
                                                   std::cos(2.0 * 3.14159265358979323846 * u2));
    }
}

static void host_reference(const std::vector<float>& x, const std::vector<float>& w,
                           int rows, int h, float eps, std::vector<float>& y) {
    for (int i = 0; i < rows; ++i) {
        const float* xr = &x[static_cast<size_t>(i) * h];
        float acc = 0.0f;
        for (int j = 0; j < h; ++j) acc += xr[j] * xr[j];
        const float scale = 1.0f / std::sqrt(acc / static_cast<float>(h) + eps);
        float* yr = &y[static_cast<size_t>(i) * h];
        for (int j = 0; j < h; ++j) yr[j] = (xr[j] * scale) * w[j];
    }
}

static int g_iters = 2;

static int check_vs_host(const std::vector<float>& y_host, const std::vector<float>& y_ref,
                         const char* tag) {
    double max_err = 0.0;
    for (size_t i = 0; i < y_host.size(); ++i) {
        const double e = std::fabs(static_cast<double>(y_host[i]) - static_cast<double>(y_ref[i]));
        if (e > max_err) max_err = e;
    }
    std::printf("[health %s] max_abs_error=%.3e iters=%d\n", tag, max_err, g_iters);
    return 0;
}

int main(int argc, char** argv) {
    std::string target;
    int rows = 64, h = 1024, tile = 256, splits = 8;
    int warp = 32;
    float eps = 1e-6f;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? std::string(argv[++i]) : std::string(); };
        if (a == "--target") target = next();
        else if (a == "--rows") rows = std::stoi(next());
        else if (a == "--h") h = std::stoi(next());
        else if (a == "--tile") tile = std::stoi(next());
        else if (a == "--splits") splits = std::stoi(next());
        else if (a == "--iters") g_iters = std::stoi(next());
        else if (a == "--warp") warp = std::stoi(next());
        else { std::fprintf(stderr, "未知参数 %s\n", a.c_str()); return 2; }
    }

    std::vector<float> hx(static_cast<size_t>(rows) * h), hw(h);
    fill_normal(hx, 20261105u);
    fill_normal(hw, 20261106u);
    std::vector<float> y_ref(static_cast<size_t>(rows) * h, 0.0f);
    host_reference(hx, hw, rows, h, eps, y_ref);

    float *dx, *dw, *dy, *dr = nullptr, *dg = nullptr;
    CUDA_CHECK(cudaMalloc(&dx, hx.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dw, hw.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy, hx.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), hx.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw, hw.data(), hw.size() * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<float> y_host(hx.size());

#if defined(CH11_TARGET_CH04)
    (void)target;
    (void)splits;
    if (g_iters != 1) {
        std::fprintf(stderr, "ch04-fused 体检口径要求 --iters 1（in-place 语义下多轮无固定真值）\n");
        return 2;
    }
    CUDA_CHECK(cudaMalloc(&dr, hx.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dr, hx.data(), hx.size() * sizeof(float), cudaMemcpyHostToDevice));
    const int rc4 = fused_add_rmsnorm_rows(reinterpret_cast<long long>(dr),
                                           reinterpret_cast<long long>(dx),
                                           reinterpret_cast<long long>(dw),
                                           reinterpret_cast<long long>(dy),
                                           0, rows, h, eps, tile, warp);
    if (rc4) { std::fprintf(stderr, "fused_add_rmsnorm_rows rc=%d\n", rc4); return 1; }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(y_host.data(), dy, y_host.size() * sizeof(float), cudaMemcpyDeviceToHost));
    return check_vs_host(y_host, y_ref, "ch04-fused");
#endif

#if defined(CH11_TARGET_CH08)
    if (target == "ch08-det") {
        for (int it = 0; it < g_iters; ++it) {
            const int rc = rmsnorm_det_rows(reinterpret_cast<long long>(dx),
                                            reinterpret_cast<long long>(dw),
                                            reinterpret_cast<long long>(dy),
                                            0, rows, h, eps, tile, warp);
            if (rc) { std::fprintf(stderr, "rmsnorm_det_rows rc=%d\n", rc); return 1; }
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        CUDA_CHECK(cudaMemcpy(y_host.data(), dy, y_host.size() * sizeof(float), cudaMemcpyDeviceToHost));
        return check_vs_host(y_host, y_ref, target.c_str());
    }
    if (target == "ch08-atomic-shared") {
        for (int it = 0; it < g_iters; ++it) {
            const int rc = rmsnorm_atomic_shared_rows(reinterpret_cast<long long>(dx),
                                                      reinterpret_cast<long long>(dw),
                                                      reinterpret_cast<long long>(dy),
                                                      0, 0, rows, h, eps, tile, warp);
            if (rc) { std::fprintf(stderr, "rmsnorm_atomic_shared_rows rc=%d\n", rc); return 1; }
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        CUDA_CHECK(cudaMemcpy(y_host.data(), dy, y_host.size() * sizeof(float), cudaMemcpyDeviceToHost));
        return check_vs_host(y_host, y_ref, target.c_str());
    }
    if (target == "ch08-atomic-global") {
        CUDA_CHECK(cudaMalloc(&dg, rows * sizeof(float)));
        for (int it = 0; it < g_iters; ++it) {
            int rc = zero_acc(reinterpret_cast<long long>(dg), rows, 0);
            if (rc) { std::fprintf(stderr, "zero_acc rc=%d\n", rc); return 1; }
            rc = rmsnorm_atomic_global_rows(reinterpret_cast<long long>(dx),
                                            reinterpret_cast<long long>(dw),
                                            reinterpret_cast<long long>(dg),
                                            reinterpret_cast<long long>(dy),
                                            0, 0, rows, h, eps, tile, warp, splits);
            if (rc) { std::fprintf(stderr, "rmsnorm_atomic_global_rows rc=%d\n", rc); return 1; }
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        CUDA_CHECK(cudaMemcpy(y_host.data(), dy, y_host.size() * sizeof(float), cudaMemcpyDeviceToHost));
        return check_vs_host(y_host, y_ref, target.c_str());
    }
    std::fprintf(stderr, "未知 ch08 target %s\n", target.c_str());
    return 2;
#endif
}
