// 第 15 期 stub 检查用假 cuda_fp16.h（最小 __half 语义）
#pragma once

struct __half {
    unsigned short x;
};

inline __half __float2half(float v) {
    __half h;
    h.x = (unsigned short)(int)(v * 128.0f);
    return h;
}

inline float __half2float(__half h) {
    return (float)(short)h.x / 128.0f;
}
