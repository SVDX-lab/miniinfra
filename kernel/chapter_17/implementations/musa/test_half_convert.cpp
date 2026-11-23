// 本地验证：naive_gemm.cu 中 MUSA 位级半精度转换 fallback 的正确性。
// （文件内容从 implementations/cuda/naive_gemm.cu 提取复制；源文件修改时需同步重跑本测试。）
// 验证方式：
//   1. half→float：穷举全部 65536 个 16 位位型，与 numpy.float16 视图逐位比对；
//   2. float→half：全部 half 可表示值往返 + 100 万随机 float + 边界值，与 numpy 对拍。
// 用法：g++ -O2 -o test_half_convert test_half_convert.cpp && ./test_half_convert
//       /home/aiuser/venvs/ml/bin/python verify_half_ref.py   （numpy 对拍部分）
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>

// ==== 以下从 naive_gemm.cu 复制（float_to_half_rne + half→float 内联逻辑）====
static unsigned short float_to_half_rne(float value) {
    unsigned int bits;
    memcpy(&bits, &value, sizeof(bits));
    unsigned int sign = (bits >> 16) & 0x8000u;
    unsigned int exp32 = (bits >> 23) & 0xFFu;
    unsigned int mant = bits & 0x7FFFFFu;
    unsigned short out;
    if (exp32 == 0xFF) {
        out = (unsigned short)(mant ? 0xFE00u : 0x7C00u);
    } else if (exp32 == 0 && mant == 0) {
        out = 0;
    } else {
        int e = (int)exp32 - 127;
        if (e > 15) {
            out = 0x7C00u;
        } else if (e >= -14) {
            unsigned int hm = mant >> 13;
            unsigned int rem = mant & 0x1FFFu;
            unsigned int h = ((unsigned int)(e + 15) << 10) | hm;
            if (rem > 0x1000u || (rem == 0x1000u && (hm & 1u))) h += 1;
            out = (unsigned short)h;
        } else if (e >= -25) {
            unsigned int full = 0x800000u | mant;
            int shift = -e - 1;
            if (shift > 24) {
                out = 0;
            } else {
                unsigned int s = full >> shift;
                unsigned int rem = full & ((1u << shift) - 1u);
                unsigned int half_point = 1u << (shift - 1);
                if (rem > half_point || (rem == half_point && (s & 1u))) s += 1;
                out = (unsigned short)s;
            }
        } else {
            out = 0;
        }
    }
    return (unsigned short)(sign | out);
}

static float half_bits_to_float(unsigned short h) {
    unsigned int sign = (unsigned int)(h & 0x8000u) << 16;
    unsigned int exp = (h >> 10) & 0x1Fu;
    unsigned int mant = h & 0x3FFu;
    unsigned int bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int e = -14;
            unsigned int m = mant;
            while ((m & 0x400u) == 0) { m <<= 1; e -= 1; }
            m &= 0x3FFu;
            bits = sign | ((unsigned int)(127 + e) << 23) | (m << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

int main() {
    // 阶段 1：half→float 穷举，输出 CSV 供 numpy 对拍
    FILE* f = fopen("half-table.csv", "w");
    for (unsigned int h = 0; h < 65536u; ++h) {
        float v = half_bits_to_float((unsigned short)h);
        unsigned int b;
        memcpy(&b, &v, 4);
        fprintf(f, "%u,%u\n", h, b);
    }
    fclose(f);

    // 阶段 2：float→half 测试集（全部 half 值转回 float 再转 half 必须幂等）+ 边界
    std::vector<float> tests;
    for (unsigned int h = 0; h < 65536u; ++h) tests.push_back(half_bits_to_float((unsigned short)h));
    // 边界：最大规格、临界值、rounding 边界
    float boundaries[] = {65504.0f, 65519.0f, 65520.0f, 65521.0f, 65536.0f, 65552.0f,
                          5.9604645e-08f, 2.9802322e-08f, 5.960464e-08f, 6.1035156e-05f,
                          6.0975556e-05f, 1.1754944e-38f, 1e-40f, 0.0f, -0.0f, 1.0f, -1.0f,
                          0.5f, 2048.0f, 2049.0f, 2050.0f};
    for (float v : boundaries) tests.push_back(v);
    unsigned long long seed = 20261118;
    auto lcg = [&]() {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        return (unsigned int)(seed >> 33);
    };
    for (int i = 0; i < 1000000; ++i) {
        unsigned int bits = lcg();
        float v;
        memcpy(&v, &bits, 4);
        tests.push_back(v);
    }

    FILE* g = fopen("f2h-tests.csv", "w");
    for (float v : tests) {
        unsigned short h = float_to_half_rne(v);
        // 幂等：float→half→float→half 必须稳定
        float back = half_bits_to_float(h);
        unsigned short h2 = float_to_half_rne(back);
        if (h != h2) {
            printf("FAIL 幂等: v=%g h=%04x h2=%04x\n", v, h, h2);
            return 1;
        }
        unsigned int b32;
        memcpy(&b32, &v, 4);
        fprintf(g, "%u,%u\n", b32, h);
    }
    fclose(g);
    printf("OK: half-table.csv (65536 行) + f2h-tests.csv (%zu 行) 已生成，等待 numpy 对拍\n",
           tests.size());
    return 0;
}
