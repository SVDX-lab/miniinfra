// ============================================================================
// det_atomic_bench.cu —— 第 07 期独立可执行程序（nsys 被测对象 + 大形状通道）
//
// 两种模式：
//   --mode perf        与第 02 期 bench 同口径（Event 对 + 末尾一次同步，
//                      P50/P95/mean/min + 有效带宽达成率）；被测实现
//                      --impl det|shared|global，global 附带 zero（整链
//                      3 launches，与 Python ctypes 通道同一口径）。
//   --mode determinism 本期主模式：N 次重复运行（默认 100），统计
//                      (a) 与第一次运行位级相同的次数（device 端比较
//                          kernel，不回传全量数据）；
//                      (b) 与 host FP32 串行真值（naive_reference，C 速度
//                          现算——大形状真值通道）的最大/平均漂移，
//                          并统计 100 次的漂移带（min/max）；
//                      (c) 每次运行输出 SHA256（仅 rows*H <= 8M 时逐次
//                          回传哈希，更大形状只做 device 比较，口径在
//                          JSON 里标注 hash_scope）。
//
// FP16 输入的参照口径：host 参考用**量化后的 x**（half→float）串行累加
// ——把"输入舍入"与"归约顺序漂移"分开归因（输入舍入是确定性的一次性
// 误差，归约顺序才是本期的单变量）。
//
// 计时口径与第 02 期 bench 逐字相同（warmup -> Event 对 -> 一次 sync）。
// ============================================================================

#include "kernels_lib.mu"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

#define BENCH_CUDA_CHECK(expr)                                                          \
    do {                                                                                \
        musaError_t _err = (expr);                                                      \
        if (_err != musaSuccess) {                                                      \
            std::fprintf(stderr, "MUSA error %s at %s:%d\n", musaGetErrorString(_err),  \
                         __FILE__, __LINE__);                                           \
            std::exit(1);                                                               \
        }                                                                               \
    } while (0)

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
        data[data.size() - 1] = static_cast<float>(
            std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * 3.14159265358979323846 * u2));
    }
}

// host FP32 串行参考（真值口径：j=0..H-1 自然顺序累加；x 已按输入 dtype 量化）。
void naive_reference(const std::vector<float>& x, const std::vector<float>& w, int rows, int h,
                     float eps, std::vector<float>& y) {
    y.assign(static_cast<size_t>(rows) * h, 0.0f);
    for (int i = 0; i < rows; ++i) {
        const float* xr = &x[static_cast<size_t>(i) * h];
        float acc = 0.0f;
        for (int j = 0; j < h; ++j) acc += xr[j] * xr[j];
        const float scale = 1.0f / std::sqrt(acc / static_cast<float>(h) + eps);
        float* yr = &y[static_cast<size_t>(i) * h];
        for (int j = 0; j < h; ++j) yr[j] = (xr[j] * scale) * w[j];
    }
}

void quantize_to_half_and_back(std::vector<float>& data) {
    for (float& v : data) {
        __half hv = __float2half(v);
        v = __half2float(hv);
    }
}

double percentile(std::vector<double>& sorted, double q) {
    const size_t n = sorted.size();
    return sorted[static_cast<size_t>(std::min(
        static_cast<double>(n - 1), std::floor(q * static_cast<double>(n))))];
}

// device 端比较 kernel：按字节统计 y 与 y0 的位级差（byte 口径对 fp32/fp16
// 统一——位级相同 ⇔ 全部字节相同；第 07 期初版按 float 元素计数在 fp16
// 缓冲上越界读（illegal memory access），修正为字节口径）。
__global__ void count_diff_kernel(const unsigned char* a, const unsigned char* b,
                                  long long n_bytes, unsigned long long* counter) {
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n_bytes; i += stride) {
        if (a[i] != b[i]) atomicAdd(counter, 1ull);
    }
}

struct Options {
    std::string mode = "perf";            // perf | determinism
    std::string impl = "det";             // det | shared | global
    int h = 1024;
    int rows = 2048;
    std::string dtype = "fp16";
    std::string acc = "fp32";             // fp32 | half（仅 atomic 版）
    int tile = 256;
    int splits = 4;
    float eps = 1e-6f;
    int warmup = 10;
    int repeats = 100;                    // perf 事件的重复数；determinism 的运行次数
    uint64_t seed = 20260808;
    double peak_gbs = 1008.0;
    std::string json_file;
    std::string device_tag = "musa";
};

// 统一启动入口（含 global 版的 zero 前置 —— 与 Python ctypes 口径一致）。
int launch_impl(const Options& opt, const void* d_x, const void* d_w, void* d_acc, void* d_y,
                int is_fp16, int acc_is_half, int warp_size) {
    if (opt.impl == "det") {
        return rmsnorm_det_rows(reinterpret_cast<long long>(d_x),
                                reinterpret_cast<long long>(d_w),
                                reinterpret_cast<long long>(d_y),
                                is_fp16, opt.rows, opt.h, opt.eps, opt.tile, warp_size);
    }
    if (opt.impl == "shared") {
        return rmsnorm_atomic_shared_rows(reinterpret_cast<long long>(d_x),
                                          reinterpret_cast<long long>(d_w),
                                          reinterpret_cast<long long>(d_y),
                                          is_fp16, acc_is_half, opt.rows, opt.h, opt.eps,
                                          opt.tile, warp_size);
    }
    if (opt.impl == "global") {
        int status = zero_acc(reinterpret_cast<long long>(d_acc), opt.rows, acc_is_half);
        if (status) return status;
        return rmsnorm_atomic_global_rows(reinterpret_cast<long long>(d_x),
                                          reinterpret_cast<long long>(d_w),
                                          reinterpret_cast<long long>(d_acc),
                                          reinterpret_cast<long long>(d_y),
                                          is_fp16, acc_is_half, opt.rows, opt.h, opt.eps,
                                          opt.tile, warp_size, opt.splits);
    }
    return 6;
}

void write_line(const Options& opt, const std::string& line) {
    std::printf("%s\n", line.c_str());
    std::fflush(stdout);
    if (!opt.json_file.empty()) {
        std::ofstream file(opt.json_file, std::ios::app);
        file << line << "\n";
    }
}

int run(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? std::string(argv[++i]) : std::string(); };
        if (a == "--mode") opt.mode = next();
        else if (a == "--impl") opt.impl = next();
        else if (a == "--h") opt.h = std::stoi(next());
        else if (a == "--rows") opt.rows = std::stoi(next());
        else if (a == "--dtype") opt.dtype = next();
        else if (a == "--acc") opt.acc = next();
        else if (a == "--tile") opt.tile = std::stoi(next());
        else if (a == "--splits") opt.splits = std::stoi(next());
        else if (a == "--eps") opt.eps = std::stof(next());
        else if (a == "--warmup") opt.warmup = std::stoi(next());
        else if (a == "--repeats") opt.repeats = std::stoi(next());
        else if (a == "--seed") opt.seed = std::stoull(next());
        else if (a == "--peak-gbs") opt.peak_gbs = std::stod(next());
        else if (a == "--json-file") opt.json_file = next();
        else if (a == "--device-tag") opt.device_tag = next();
        else { std::fprintf(stderr, "未知参数: %s\n", a.c_str()); return 2; }
    }
    if (opt.h <= 0 || opt.rows <= 0) { std::fprintf(stderr, "非法 h/rows\n"); return 2; }
    if (opt.dtype != "fp32" && opt.dtype != "fp16") { std::fprintf(stderr, "dtype 非法\n"); return 2; }
    if (opt.impl != "det" && opt.impl != "shared" && opt.impl != "global") {
        std::fprintf(stderr, "impl 非法\n"); return 2;
    }

    int device = 0;
    BENCH_CUDA_CHECK(musaGetDevice(&device));
    musaDeviceProp prop;
    BENCH_CUDA_CHECK(musaGetDeviceProperties(&prop, device));
    std::string device_name = prop.name;

    // warpSize 探针（M-20：device 内建口径）。
    int* d_warp_probe = nullptr;
    BENCH_CUDA_CHECK(musaMalloc(&d_warp_probe, sizeof(int)));
    warp_size_probe(reinterpret_cast<long long>(d_warp_probe));
    BENCH_CUDA_CHECK(musaDeviceSynchronize());
    int warp_size = 0;
    BENCH_CUDA_CHECK(musaMemcpy(&warp_size, d_warp_probe, sizeof(int), musaMemcpyDeviceToHost));
    BENCH_CUDA_CHECK(musaFree(d_warp_probe));

    const bool is_fp16 = opt.dtype == "fp16";
    const bool acc_is_half = opt.acc == "half";
    if (acc_is_half && kHalfAtomicEnabled != 1 && opt.impl != "det") {
        std::fprintf(stderr, "ACC=half 被编译开关关闭（平台边界）\n");
        return 4;
    }
    const size_t itemsize = is_fp16 ? 2 : 4;
    const size_t total = static_cast<size_t>(opt.rows) * opt.h;
    const long long theoretical_bytes =
        (static_cast<long long>(opt.rows) * opt.h * 2 + opt.h) *
        static_cast<long long>(itemsize);

    // ---- host 输入（splitmix64 确定性生成）----
    std::vector<float> x_host(total), w_host(opt.h);
    fill_normal(x_host, opt.seed);
    fill_normal(w_host, opt.seed + 7);
    if (is_fp16) quantize_to_half_and_back(x_host);   // 参照与 kernel 看到同一量化输入
    if (is_fp16) quantize_to_half_and_back(w_host);

    // ---- 设备内存 ----
    void *d_x = nullptr, *d_w = nullptr, *d_y = nullptr, *d_y0 = nullptr, *d_acc = nullptr,
         *d_counter = nullptr;
    const size_t x_bytes = total * itemsize;
    const size_t w_bytes = static_cast<size_t>(opt.h) * itemsize;
    BENCH_CUDA_CHECK(musaMalloc(&d_x, x_bytes));
    BENCH_CUDA_CHECK(musaMalloc(&d_w, w_bytes));
    BENCH_CUDA_CHECK(musaMalloc(&d_y, x_bytes));
    BENCH_CUDA_CHECK(musaMalloc(&d_y0, x_bytes));
    BENCH_CUDA_CHECK(musaMalloc(&d_counter, sizeof(unsigned long long)));
    if (opt.impl == "global") {
        BENCH_CUDA_CHECK(musaMalloc(&d_acc, static_cast<size_t>(opt.rows) * (acc_is_half ? 2 : 4)));
    }
    {
        std::vector<__half> xh(total), wh(opt.h);
        for (size_t i = 0; i < total; ++i) xh[i] = __float2half(x_host[i]);
        for (int i = 0; i < opt.h; ++i) wh[i] = __float2half(w_host[i]);
        if (is_fp16) {
            BENCH_CUDA_CHECK(musaMemcpy(d_x, xh.data(), x_bytes, musaMemcpyHostToDevice));
            BENCH_CUDA_CHECK(musaMemcpy(d_w, wh.data(), w_bytes, musaMemcpyHostToDevice));
        } else {
            BENCH_CUDA_CHECK(musaMemcpy(d_x, x_host.data(), x_bytes, musaMemcpyHostToDevice));
            BENCH_CUDA_CHECK(musaMemcpy(d_w, w_host.data(), w_bytes, musaMemcpyHostToDevice));
        }
    }

    // ---- host 真值（FP32 串行，量化后输入）----
    std::vector<float> y_ref;
    naive_reference(x_host, w_host, opt.rows, opt.h, opt.eps, y_ref);

    // ---- 公共自检与 helper：launch N 次，比较、计时 ----
    auto launch_once = [&]() {
        const int status = launch_impl(opt, d_x, d_w, d_acc, d_y, is_fp16 ? 1 : 0,
                                       acc_is_half ? 1 : 0, warp_size);
        if (status != 0) {
            std::fprintf(stderr, "launch 失败 status=%d detail=%s\n", status, g_status_detail);
            std::exit(1);
        }
    };

    if (opt.mode == "perf") {
        std::vector<musaEvent_t> starts(opt.repeats), stops(opt.repeats);
        for (int i = 0; i < opt.repeats; ++i) {
            BENCH_CUDA_CHECK(musaEventCreate(&starts[i]));
            BENCH_CUDA_CHECK(musaEventCreate(&stops[i]));
        }
        for (int i = 0; i < opt.warmup; ++i) launch_once();
        BENCH_CUDA_CHECK(musaDeviceSynchronize());
        for (int i = 0; i < opt.repeats; ++i) {
            BENCH_CUDA_CHECK(musaEventRecord(starts[i]));
            launch_once();
            BENCH_CUDA_CHECK(musaEventRecord(stops[i]));
        }
        BENCH_CUDA_CHECK(musaDeviceSynchronize());
        std::vector<double> samples;
        for (int i = 0; i < opt.repeats; ++i) {
            float ms = 0.0f;
            BENCH_CUDA_CHECK(musaEventElapsedTime(&ms, starts[i], stops[i]));
            samples.push_back(static_cast<double>(ms));
            BENCH_CUDA_CHECK(musaEventDestroy(starts[i]));
            BENCH_CUDA_CHECK(musaEventDestroy(stops[i]));
        }
        std::sort(samples.begin(), samples.end());
        const double p50 = percentile(samples, 0.50);
        const double p95 = percentile(samples, 0.95);
        double mean = 0.0;
        for (double v : samples) mean += v;
        mean /= static_cast<double>(samples.size());
        const double effective = static_cast<double>(theoretical_bytes) / (p50 / 1000.0);
        const int launches = (opt.impl == "global") ? 3 : 1;
        char line[1024];
        std::snprintf(line, sizeof(line),
                      "{\"device\":\"%s\",\"backend\":\"%s\",\"mode\":\"perf\",\"impl\":\"%s\","
                      "\"acc\":\"%s\",\"h\":%d,\"rows\":%d,\"dtype\":\"%s\",\"tile\":%d,\"splits\":%d,"
                      "\"launches\":%d,\"warmup\":%d,\"repeats\":%d,"
                      "\"p50_ms\":%.4f,\"p95_ms\":%.4f,\"mean_ms\":%.4f,\"min_ms\":%.4f,"
                      "\"theoretical_bytes\":%lld,\"effective_bandwidth_gb_s\":%.2f,"
                      "\"peak_bandwidth_gb_s_spec\":%.1f,\"achievement_ratio\":%.4f,"
                      "\"warp_size_probe\":%d}",
                      device_name.c_str(), opt.device_tag.c_str(), opt.impl.c_str(), opt.acc.c_str(),
                      opt.h, opt.rows, opt.dtype.c_str(), opt.tile, opt.splits, launches,
                      opt.warmup, opt.repeats, p50, p95, mean, samples.front(),
                      theoretical_bytes, effective / 1e9, opt.peak_gbs,
                      effective / (opt.peak_gbs * 1e9), warp_size);
        write_line(opt, line);
    } else {
        // ---- determinism 模式：N 次重复，device 端位级比较 + host 真值漂移 ----
        // 第一次运行进 d_y0 作为参照；后续每次进 d_y 并 device 比较。
        std::vector<unsigned long long> diff_counts;
        diff_counts.reserve(opt.repeats);
        BENCH_CUDA_CHECK(musaMemset(d_counter, 0, sizeof(unsigned long long)));
        launch_once();
        BENCH_CUDA_CHECK(musaDeviceSynchronize());
        BENCH_CUDA_CHECK(musaMemcpy(d_y0, d_y, x_bytes, musaMemcpyDeviceToDevice));

        // run 0 的真值漂移（host 全量比较）。
        std::vector<float> y_run(total);
        {
            if (is_fp16) {
                std::vector<__half> yh(total);
                BENCH_CUDA_CHECK(musaMemcpy(yh.data(), d_y, x_bytes, musaMemcpyDeviceToHost));
                for (size_t i = 0; i < total; ++i) y_run[i] = __half2float(yh[i]);
            } else {
                BENCH_CUDA_CHECK(musaMemcpy(y_run.data(), d_y, x_bytes, musaMemcpyDeviceToHost));
            }
        }
        double run0_max_err = 0.0, run0_sum_err = 0.0;
        for (size_t i = 0; i < total; ++i) {
            const double e = std::fabs(static_cast<double>(y_run[i]) - y_ref[i]);
            run0_max_err = std::max(run0_max_err, e);
            run0_sum_err += e;
        }

        const bool host_hash_per_run = total <= 8u * 1024u * 1024u;
        double band_max = run0_max_err, band_min = run0_max_err;
        std::vector<std::string> hashes;
        // 简易 64 位滚动哈希（位级敏感；跨运行比较用途，SHA256 由 Python
        // 通道承担，此处口径标注 digest64）。
        auto hash_of = [&](const std::vector<float>& data) {
            unsigned long long hsh = 1469598103934665603ull;
            const unsigned char* bytes = reinterpret_cast<const unsigned char*>(data.data());
            for (size_t i = 0; i < data.size() * sizeof(float); ++i) {
                hsh = (hsh ^ bytes[i]) * 1099511628211ull;
            }
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%016llx", hsh);
            return std::string(buf);
        };
        if (host_hash_per_run) hashes.push_back(hash_of(y_run));
        int identical = 1;
        for (int run = 1; run < opt.repeats; ++run) {
            BENCH_CUDA_CHECK(musaMemset(d_counter, 0, sizeof(unsigned long long)));
            launch_once();
            BENCH_CUDA_CHECK(musaDeviceSynchronize());
            count_diff_kernel<<<256, 256>>>(static_cast<const unsigned char*>(d_y),
                                            static_cast<const unsigned char*>(d_y0),
                                            static_cast<long long>(x_bytes),
                                            reinterpret_cast<unsigned long long*>(d_counter));
            BENCH_CUDA_CHECK(musaDeviceSynchronize());
            unsigned long long count = 0;
            BENCH_CUDA_CHECK(musaMemcpy(&count, d_counter, sizeof(unsigned long long),
                                        musaMemcpyDeviceToHost));
            diff_counts.push_back(count);
            if (count == 0) ++identical;
            if (host_hash_per_run || total <= 64u * 1024u * 1024u) {
                // 漂移带：中等形状逐次回传比较；大形状只比较 vs first（真值
                // 漂移带退化为 vs first-run 位级差 + run0 真值漂移）。
                if (is_fp16) {
                    std::vector<__half> yh(total);
                    BENCH_CUDA_CHECK(musaMemcpy(yh.data(), d_y, x_bytes, musaMemcpyDeviceToHost));
                    for (size_t i = 0; i < total; ++i) y_run[i] = __half2float(yh[i]);
                } else {
                    BENCH_CUDA_CHECK(musaMemcpy(y_run.data(), d_y, x_bytes, musaMemcpyDeviceToHost));
                }
                double run_max = 0.0;
                for (size_t i = 0; i < total; ++i) {
                    const double e = std::fabs(static_cast<double>(y_run[i]) - y_ref[i]);
                    run_max = std::max(run_max, e);
                }
                band_max = std::max(band_max, run_max);
                band_min = std::min(band_min, run_max);
                if (host_hash_per_run) hashes.push_back(hash_of(y_run));
            }
        }
        // 唯一输出计数（host_hash_per_run 时按哈希；否则 null）。
        std::string unique_str = "null";
        if (host_hash_per_run) {
            std::sort(hashes.begin(), hashes.end());
            hashes.erase(std::unique(hashes.begin(), hashes.end()), hashes.end());
            unique_str = std::to_string(hashes.size());
        }
        unsigned long long worst_diff = 0;
        for (unsigned long long c : diff_counts) worst_diff = std::max(worst_diff, c);
        char line[1280];
        std::snprintf(line, sizeof(line),
                      "{\"device\":\"%s\",\"backend\":\"%s\",\"mode\":\"determinism\",\"impl\":\"%s\","
                      "\"acc\":\"%s\",\"h\":%d,\"rows\":%d,\"dtype\":\"%s\",\"tile\":%d,\"splits\":%d,"
                      "\"repeats\":%d,\"identical_vs_first\":%d,\"unique_outputs\":%s,"
                      "\"hash_scope\":\"%s\",\"worst_diff_bytes\":%llu,"
                      "\"run0_max_abs_error\":%.3e,\"run0_mean_abs_error\":%.3e,"
                      "\"drift_band_min\":%.3e,\"drift_band_max\":%.3e,"
                      "\"truth_impl\":\"host_serial_f32_quantized_input\",\"warp_size_probe\":%d}",
                      device_name.c_str(), opt.device_tag.c_str(), opt.impl.c_str(), opt.acc.c_str(),
                      opt.h, opt.rows, opt.dtype.c_str(), opt.tile, opt.splits, opt.repeats,
                      identical, unique_str.c_str(),
                      host_hash_per_run ? "per_run_digest64" : "first_run_only", worst_diff,
                      run0_max_err, run0_sum_err / static_cast<double>(total),
                      band_min, band_max, warp_size);
        write_line(opt, line);
    }

    BENCH_CUDA_CHECK(musaFree(d_x));
    BENCH_CUDA_CHECK(musaFree(d_w));
    BENCH_CUDA_CHECK(musaFree(d_y));
    BENCH_CUDA_CHECK(musaFree(d_y0));
    BENCH_CUDA_CHECK(musaFree(d_counter));
    if (d_acc) BENCH_CUDA_CHECK(musaFree(d_acc));
    return 0;
}

}  // namespace

int main(int argc, char** argv) { return run(argc, argv); }
