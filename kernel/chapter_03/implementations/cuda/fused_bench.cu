// ============================================================================
// fused_bench.cu —— 第 03 期独立可执行基准 + 正确性自检（nsys 被测对象）。
//
// 结构沿用第 02 期 rmsnorm_bench.cu 骨架（splitmix64 输入、Event 对计时、
// 末尾一次同步、host 串行参考自检、stdout 逐行 JSON），新增：
//   --mode separated|fused|both：
//     separated = add kernel + 第 02 期 rmsnorm kernel（2 launch/迭代）；
//     fused     = 融合 kernel（1 launch/迭代，r 原位更新）。
//   理论字节（合同口径，fp16 itemsize=2 / fp32 itemsize=4）：
//     separated = (5*rows*H + H) × itemsize
//                 （读r+读h+写t）+（读t+读w+写y）；
//     fused     = (4*rows*H + H) × itemsize
//                 （读r+读h+读w+写y+写r原位更新）；
//     理论加速比上限 = separated_bytes / fused_bytes（大形状 → 5/4）。
//   自检同时核对 residual（t）与 y（融合模式在计时前用新鲜输入单次调用，
//   避免 in-place 语义把输入改掉后污染自检）。
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

extern "C" int warp_size_probe(long long out_ptr);
extern "C" int add_rows(long long r_ptr, long long h_ptr, long long t_ptr,
                        int is_fp16, long long n, int block_threads);
extern "C" int rmsnorm_rows(long long x_ptr, long long w_ptr, long long y_ptr,
                            int is_fp16, int rows, int h_dim, float eps,
                            int tile, int warp_size);
extern "C" int fused_add_rmsnorm_rows(long long r_ptr, long long h_ptr,
                                      long long w_ptr, long long y_ptr,
                                      int is_fp16, int rows, int h_dim,
                                      float eps, int tile, int warp_size);

namespace bench {

#define BENCH_CUDA_CHECK(expr)                                                          \
    do {                                                                                \
        cudaError_t _err = (expr);                                                      \
        if (_err != cudaSuccess) {                                                      \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_err),  \
                         __FILE__, __LINE__);                                           \
            std::exit(1);                                                               \
        }                                                                               \
    } while (0)

// 确定性输入生成（splitmix64 + Box-Muller），与第 02 期 bench 同一规则。
double splitmix64(uint64_t& state) {
    state += 0x9E3779B97F4A7C15ull;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return static_cast<double>(z ^ (z >> 31)) / 18446744073709551616.0;
}

void fill_normal(std::vector<float>& data, uint64_t seed) {
    uint64_t state = seed;
    for (size_t i = 0; i + 1 < data.size(); i += 2) {
        double u1 = splitmix64(state);
        double u2 = splitmix64(state);
        if (u1 < 1e-12) u1 = 1e-12;
        double r = std::sqrt(-2.0 * std::log(u1));
        double theta = 2.0 * 3.14159265358979323846 * u2;
        data[i] = static_cast<float>(r * std::cos(theta));
        data[i + 1] = static_cast<float>(r * std::sin(theta));
    }
    if (data.size() % 2 == 1) {
        double u1 = splitmix64(state);
        double u2 = splitmix64(state);
        if (u1 < 1e-12) u1 = 1e-12;
        data[data.size() - 1] = static_cast<float>(std::sqrt(-2.0 * std::log(u1)) *
                                                   std::cos(2.0 * 3.14159265358979323846 * u2));
    }
}

// host 串行参考：分离路径两步（与 CPU 真值同一 FP32 累加顺序）。
void naive_reference(const std::vector<float>& r, const std::vector<float>& h,
                     const std::vector<float>& w, int rows, int hh, float eps,
                     std::vector<float>& t_out, std::vector<float>& y_out) {
    for (int i = 0; i < rows; ++i) {
        const float* rr = &r[static_cast<size_t>(i) * hh];
        const float* hr = &h[static_cast<size_t>(i) * hh];
        float* tr = &t_out[static_cast<size_t>(i) * hh];
        float acc = 0.0f;
        for (int j = 0; j < hh; ++j) {
            tr[j] = rr[j] + hr[j];
            acc += tr[j] * tr[j];
        }
        const float scale = 1.0f / std::sqrt(acc / static_cast<float>(hh) + eps);
        float* yr = &y_out[static_cast<size_t>(i) * hh];
        for (int j = 0; j < hh; ++j) yr[j] = (tr[j] * scale) * w[j];
    }
}

double percentile(std::vector<double>& sorted, double q) {
    const size_t n = sorted.size();
    const size_t idx = static_cast<size_t>(std::min(
        static_cast<double>(n - 1), std::floor(q * static_cast<double>(n))));
    return sorted[idx];
}

struct Options {
    int h = 1024;
    int rows = 2048;
    std::string dtype = "fp16";
    std::string mode = "both";  // separated | fused | both
    std::vector<int> tiles = {256};
    float eps = 1e-6f;
    int warmup = 10;
    int repeats = 100;
    uint64_t seed = 20260401;
    double peak_gbs = 918.8;
    std::string json_file;
    bool check = true;
    int check_rows = 8;
};

void print_json_line(const std::string& device_name, const Options& opt, const std::string& mode,
                     int tile, bool vec, const std::vector<double>& samples_ms,
                     long long theoretical_bytes, int launches_per_iter,
                     double t_max_err, double t_mean_err, double y_max_err, double y_mean_err,
                     double y_cos, bool has_check) {
    std::vector<double> sorted(samples_ms);
    std::sort(sorted.begin(), sorted.end());
    double p50 = percentile(sorted, 0.50);
    double p95 = percentile(sorted, 0.95);
    double mean = 0.0;
    for (double v : samples_ms) mean += v;
    mean /= static_cast<double>(samples_ms.size());
    double effective = static_cast<double>(theoretical_bytes) / (p50 / 1000.0);
    double ratio = effective / (opt.peak_gbs * 1e9);

    char check_buf[320];
    std::snprintf(check_buf, sizeof(check_buf),
                  "{\"t_max_abs\":%.3e,\"t_mean_abs\":%.3e,\"y_max_abs\":%.3e,"
                  "\"y_mean_abs\":%.3e,\"y_cos\":%.8f}",
                  t_max_err, t_mean_err, y_max_err, y_mean_err, y_cos);
    char line[1024];
    std::snprintf(line, sizeof(line),
                  "{\"device\":\"%s\",\"mode\":\"%s\",\"h\":%d,\"rows\":%d,\"dtype\":\"%s\","
                  "\"tile\":%d,\"vec\":%s,\"warmup\":%d,\"repeats\":%zu,"
                  "\"launches_per_iter\":%d,\"p50_ms\":%.4f,\"p95_ms\":%.4f,\"mean_ms\":%.4f,"
                  "\"min_ms\":%.4f,\"theoretical_bytes\":%lld,"
                  "\"effective_bandwidth_gb_s\":%.2f,\"peak_bandwidth_gb_s\":%.1f,"
                  "\"achievement_ratio\":%.4f,\"check\":%s}",
                  device_name.c_str(), mode.c_str(), opt.h, opt.rows, opt.dtype.c_str(), tile,
                  vec ? "true" : "false", opt.warmup, samples_ms.size(), launches_per_iter,
                  p50, p95, mean, sorted.front(), theoretical_bytes, effective / 1e9,
                  opt.peak_gbs, ratio, has_check ? check_buf : "null");
    std::printf("%s\n", line);
    std::fflush(stdout);
    if (!opt.json_file.empty()) {
        std::ofstream file(opt.json_file, std::ios::app);
        file << line << "\n";
    }
}

// 一轮测量（单模式单 tile）：自检（新鲜输入）→ warmup → Event 对计时。
void run_mode(const Options& opt, const std::string& mode, int tile, int warp_size,
              const std::string& device_name, bool is_fp16, size_t itemsize,
              const std::vector<float>& r_host, const std::vector<float>& h_host,
              const std::vector<float>& w_host, void* d_r, void* d_h, void* d_w,
              void* d_t, void* d_y) {
    const int hh = opt.h;
    const long long n = static_cast<long long>(opt.rows) * hh;
    const long long theoretical_bytes =
        (mode == "separated" ? (5LL * n + hh) : (4LL * n + hh)) * static_cast<long long>(itemsize);
    const int launches = (mode == "separated") ? 2 : 1;

    // ---- 自检：新鲜输入上传，单次调用，前 check_rows 行 vs host 串行参考 ----
    double t_max = 0, t_mean = 0, y_max = 0, y_mean = 0, y_cos = 0;
    bool has_check = false;
    if (opt.check && opt.rows > 0) {
        const int cr = std::min(opt.rows, opt.check_rows);
        // r/h/w 每次上传新鲜值（r 可能被上次融合调用污染）。
        if (is_fp16) {
            std::vector<__half> buf(n), bw(hh);
            for (size_t i = 0; i < static_cast<size_t>(n); ++i) buf[i] = __float2half(r_host[i]);
            for (int i = 0; i < hh; ++i) bw[i] = __float2half(w_host[i]);
            BENCH_CUDA_CHECK(cudaMemcpy(d_r, buf.data(), n * 2, cudaMemcpyHostToDevice));
            for (size_t i = 0; i < static_cast<size_t>(n); ++i) buf[i] = __float2half(h_host[i]);
            BENCH_CUDA_CHECK(cudaMemcpy(d_h, buf.data(), n * 2, cudaMemcpyHostToDevice));
            BENCH_CUDA_CHECK(cudaMemcpy(d_w, bw.data(), static_cast<size_t>(hh) * 2,
                                        cudaMemcpyHostToDevice));
        } else {
            BENCH_CUDA_CHECK(cudaMemcpy(d_r, r_host.data(), static_cast<size_t>(n) * 4,
                                        cudaMemcpyHostToDevice));
            BENCH_CUDA_CHECK(cudaMemcpy(d_h, h_host.data(), static_cast<size_t>(n) * 4,
                                        cudaMemcpyHostToDevice));
            BENCH_CUDA_CHECK(cudaMemcpy(d_w, w_host.data(), static_cast<size_t>(hh) * 4,
                                        cudaMemcpyHostToDevice));
        }
        std::vector<float> t_ref(static_cast<size_t>(cr) * hh), y_ref(static_cast<size_t>(cr) * hh);
        naive_reference(r_host, h_host, w_host, cr, hh, opt.eps, t_ref, y_ref);

        if (mode == "separated") {
            add_rows(reinterpret_cast<long long>(d_r), reinterpret_cast<long long>(d_h),
                     reinterpret_cast<long long>(d_t), is_fp16 ? 1 : 0, n, 256);
            rmsnorm_rows(reinterpret_cast<long long>(d_t), reinterpret_cast<long long>(d_w),
                         reinterpret_cast<long long>(d_y), is_fp16 ? 1 : 0, opt.rows, hh,
                         opt.eps, tile, warp_size);
        } else {
            fused_add_rmsnorm_rows(reinterpret_cast<long long>(d_r),
                                   reinterpret_cast<long long>(d_h),
                                   reinterpret_cast<long long>(d_w),
                                   reinterpret_cast<long long>(d_y), is_fp16 ? 1 : 0,
                                   opt.rows, hh, opt.eps, tile, warp_size);
        }
        BENCH_CUDA_CHECK(cudaDeviceSynchronize());

        // 拉回 residual（separated 从 d_t；fused 从 d_r —— 原位更新语义）与 y。
        const void* residual_dev = (mode == "separated") ? d_t : d_r;
        std::vector<float> t_host(static_cast<size_t>(cr) * hh), y_host(static_cast<size_t>(cr) * hh);
        if (is_fp16) {
            std::vector<__half> buf(static_cast<size_t>(cr) * hh);
            BENCH_CUDA_CHECK(cudaMemcpy(buf.data(), residual_dev,
                                        static_cast<size_t>(cr) * hh * 2, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < buf.size(); ++i) t_host[i] = __half2float(buf[i]);
            BENCH_CUDA_CHECK(cudaMemcpy(buf.data(), d_y,
                                        static_cast<size_t>(cr) * hh * 2, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < buf.size(); ++i) y_host[i] = __half2float(buf[i]);
        } else {
            BENCH_CUDA_CHECK(cudaMemcpy(t_host.data(), residual_dev,
                                        static_cast<size_t>(cr) * 4, cudaMemcpyDeviceToHost));
            BENCH_CUDA_CHECK(cudaMemcpy(y_host.data(), d_y,
                                        static_cast<size_t>(cr) * 4, cudaMemcpyDeviceToHost));
        }
        for (int rr = 0; rr < cr; ++rr) {
            const size_t off = static_cast<size_t>(rr) * hh;
            double t_local_max = 0, t_local_sum = 0, y_local_max = 0, y_local_sum = 0;
            double dot = 0, na = 0, nb = 0;
            for (int j = 0; j < hh; ++j) {
                const double ta = t_host[off + j], tb = t_ref[off + j];
                const double ya = y_host[off + j], yb = y_ref[off + j];
                t_local_max = std::max(t_local_max, std::fabs(ta - tb));
                t_local_sum += std::fabs(ta - tb);
                y_local_max = std::max(y_local_max, std::fabs(ya - yb));
                y_local_sum += std::fabs(ya - yb);
                dot += ya * yb; na += ya * ya; nb += yb * yb;
            }
            t_max = std::max(t_max, t_local_max);
            t_mean = std::max(t_mean, t_local_sum / hh);
            y_max = std::max(y_max, y_local_max);
            y_mean = std::max(y_mean, y_local_sum / hh);
            y_cos = (rr == 0) ? dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30)
                              : std::min(y_cos, dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30));
        }
        has_check = true;
    }

    // ---- 计时：warmup → Event 对按次记录 → 末尾一次同步（同第 02 期口径）----
    auto launch_once = [&]() {
        if (mode == "separated") {
            add_rows(reinterpret_cast<long long>(d_r), reinterpret_cast<long long>(d_h),
                     reinterpret_cast<long long>(d_t), is_fp16 ? 1 : 0, n, 256);
            rmsnorm_rows(reinterpret_cast<long long>(d_t), reinterpret_cast<long long>(d_w),
                         reinterpret_cast<long long>(d_y), is_fp16 ? 1 : 0, opt.rows, hh,
                         opt.eps, tile, warp_size);
        } else {
            fused_add_rmsnorm_rows(reinterpret_cast<long long>(d_r),
                                   reinterpret_cast<long long>(d_h),
                                   reinterpret_cast<long long>(d_w),
                                   reinterpret_cast<long long>(d_y), is_fp16 ? 1 : 0,
                                   opt.rows, hh, opt.eps, tile, warp_size);
        }
    };
    std::vector<double> samples_ms;
    std::vector<cudaEvent_t> starts(opt.repeats), stops(opt.repeats);
    for (int i = 0; i < opt.repeats; ++i) {
        BENCH_CUDA_CHECK(cudaEventCreate(&starts[i]));
        BENCH_CUDA_CHECK(cudaEventCreate(&stops[i]));
    }
    for (int i = 0; i < opt.warmup; ++i) launch_once();
    BENCH_CUDA_CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < opt.repeats; ++i) {
        BENCH_CUDA_CHECK(cudaEventRecord(starts[i]));
        launch_once();
        BENCH_CUDA_CHECK(cudaEventRecord(stops[i]));
    }
    BENCH_CUDA_CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < opt.repeats; ++i) {
        float ms = 0.0f;
        BENCH_CUDA_CHECK(cudaEventElapsedTime(&ms, starts[i], stops[i]));
        samples_ms.push_back(static_cast<double>(ms));
        BENCH_CUDA_CHECK(cudaEventDestroy(starts[i]));
        BENCH_CUDA_CHECK(cudaEventDestroy(stops[i]));
    }

    const bool vec = (hh % (is_fp16 ? 8 : 4) == 0);
    print_json_line(device_name, opt, mode, tile, vec, samples_ms, theoretical_bytes, launches,
                    t_max, t_mean, y_max, y_mean, y_cos, has_check);
}

int run(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? std::string(argv[++i]) : std::string(); };
        if (a == "--h") opt.h = std::stoi(next());
        else if (a == "--rows") opt.rows = std::stoi(next());
        else if (a == "--dtype") opt.dtype = next();
        else if (a == "--mode") opt.mode = next();
        else if (a == "--tiles") {
            std::string list = next();
            opt.tiles.clear();
            size_t pos = 0;
            while (pos < list.size()) {
                size_t comma = list.find(',', pos);
                if (comma == std::string::npos) comma = list.size();
                opt.tiles.push_back(std::stoi(list.substr(pos, comma - pos)));
                pos = comma + 1;
            }
        }
        else if (a == "--eps") opt.eps = std::stof(next());
        else if (a == "--warmup") opt.warmup = std::stoi(next());
        else if (a == "--repeats") opt.repeats = std::stoi(next());
        else if (a == "--seed") opt.seed = std::stoull(next());
        else if (a == "--peak-gbs") opt.peak_gbs = std::stod(next());
        else if (a == "--check") { opt.check = (next() != "false"); }
        else if (a == "--check-rows") opt.check_rows = std::stoi(next());
        else if (a == "--json-file") opt.json_file = next();
        else {
            std::fprintf(stderr, "未知参数: %s\n", a.c_str());
            return 2;
        }
    }
    if (opt.h <= 0 || opt.rows < 0 || opt.tiles.empty()) {
        std::fprintf(stderr, "非法参数：h>0、rows>=0、tiles 非空\n");
        return 2;
    }
    if (opt.dtype != "fp32" && opt.dtype != "fp16") {
        std::fprintf(stderr, "dtype 只支持 fp32 / fp16\n");
        return 2;
    }
    if (opt.mode != "separated" && opt.mode != "fused" && opt.mode != "both") {
        std::fprintf(stderr, "mode 只支持 separated / fused / both\n");
        return 2;
    }

    int device = 0;
    BENCH_CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    BENCH_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::string device_name = prop.name;
    const int warp_size = prop.warpSize;

    const bool is_fp16 = (opt.dtype == "fp16");
    const size_t itemsize = is_fp16 ? 2 : 4;
    const size_t n = static_cast<size_t>(opt.rows) * opt.h;

    // ---- host 输入（splitmix64 确定性；r/h 双流错开）----
    std::vector<float> r_host(n), h_host(n), w_host(opt.h);
    fill_normal(r_host, opt.seed);
    fill_normal(h_host, opt.seed ^ 0x5F5F5F5F5F5F5F5Full);
    fill_normal(w_host, opt.seed + 7);

    // ---- 设备内存 ----
    void *d_r = nullptr, *d_h = nullptr, *d_w = nullptr, *d_t = nullptr, *d_y = nullptr;
    BENCH_CUDA_CHECK(cudaMalloc(&d_r, n * itemsize));
    BENCH_CUDA_CHECK(cudaMalloc(&d_h, n * itemsize));
    BENCH_CUDA_CHECK(cudaMalloc(&d_w, static_cast<size_t>(opt.h) * itemsize));
    BENCH_CUDA_CHECK(cudaMalloc(&d_t, n * itemsize));  // 分离路径中间 Tensor
    BENCH_CUDA_CHECK(cudaMalloc(&d_y, n * itemsize));

    std::vector<std::string> modes;
    if (opt.mode == "both") modes = {"separated", "fused"};
    else modes = {opt.mode};

    for (int tile : opt.tiles) {
        for (const std::string& mode : modes) {
            run_mode(opt, mode, tile, warp_size, device_name, is_fp16, itemsize,
                     r_host, h_host, w_host, d_r, d_h, d_w, d_t, d_y);
        }
    }

    BENCH_CUDA_CHECK(cudaFree(d_r));
    BENCH_CUDA_CHECK(cudaFree(d_h));
    BENCH_CUDA_CHECK(cudaFree(d_w));
    BENCH_CUDA_CHECK(cudaFree(d_t));
    BENCH_CUDA_CHECK(cudaFree(d_y));
    return 0;
}

}  // namespace bench

int main(int argc, char** argv) { return bench::run(argc, argv); }
