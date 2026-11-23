"""第 19 期 tile 配置表（Python 侧单源，与 naive_gemm_async.cu 的 config_table 严格一致）。

第 17 期 tile 配置族的第 19 期相关子集（ep18 的 11 配置里保留 8 个：
去掉 32x32x8_reg2x2 以外的 reg2x2 大 tile 与 reg8x8——后者 ep18 已证溢出/
占用率受损，不在本期"只改供给组织"的变量里）。
"""

from __future__ import annotations

# (name, TILE_M, TILE_N, TILE_K, REG_M, REG_N)
# block = (TILE_N/RN, TM/RM)（x=列快维，y=行）；warpspec 在此基础上 +32 线程（1D）
# smem  = (TM*(TK+1) + TK*(TN+1)) * 4 字节（+1 padding，第 17 期同款）
CONFIGS = [
    {"name": "16x16x16_reg2x2", "tm": 16, "tn": 16, "tk": 16, "rm": 2, "rn": 2},
    {"name": "16x16x16_reg4x4", "tm": 16, "tn": 16, "tk": 16, "rm": 4, "rn": 4},
    {"name": "32x32x8_reg2x2", "tm": 32, "tn": 32, "tk": 8, "rm": 2, "rn": 2},
    {"name": "32x32x8_reg4x4", "tm": 32, "tn": 32, "tk": 8, "rm": 4, "rn": 4},
    {"name": "32x64x8_reg4x4", "tm": 32, "tn": 64, "tk": 8, "rm": 4, "rn": 4},
    {"name": "64x32x8_reg4x4", "tm": 64, "tn": 32, "tk": 8, "rm": 4, "rn": 4},
    {"name": "64x64x8_reg2x2", "tm": 64, "tn": 64, "tk": 8, "rm": 2, "rn": 2},
    {"name": "64x64x8_reg4x4", "tm": 64, "tn": 64, "tk": 8, "rm": 4, "rn": 4},
]

# 16B 向量化变体（+4 padding）与 warp-spec / race 演示核的实例化子集
V16_CONFIGS = ["32x32x8_reg4x4", "64x64x8_reg4x4"]
BROKEN_CONFIGS = V16_CONFIGS
# warpspec 线程数 = 计算槽 + 32（生产者 warp）——64x64x8_reg2x2 计算槽已
# 1024，+32 超过每 block 上限，唯一排除项（smoke_test 锁定）
WARPSPEC_EXCLUDED = {"64x64x8_reg2x2"}

# kernel 级别（三级递进 + 扩展）
KERNEL_KINDS = {
    "sync": "a 同步 load 基线（= ep18 gemm_tiled_kernel 原样）",
    "cpasync1": "b cp.async 单缓冲（wait_group 0 后计算）",
    "cpasync2": "c cp.async 双缓冲（i+1 搬运与 i 计算 overlap）",
    "cpasync2v16": "c+ 16B 向量化 cp.async.cg 双缓冲（+4 padding）",
    "warpspec": "d 生产者 warp0 / 消费者其余 warp（cp.async 双缓冲）",
    "wmma_pipe": "组合：WMMA 64x64x16（FP16 输入 FP32 累加）× cp.async 双缓冲",
    "sync2_broken": "race 演示：同步双缓冲去尾屏障（不进基准表）",
    "cpasync2_broken": "race 演示：cp.async 双缓冲去尾屏障（不进基准表）",
}

SCALAR_LEVELS = ["sync", "cpasync1", "cpasync2"]  # 三级递进主对照


def parse_config(name: str) -> dict:
    for config in CONFIGS:
        if config["name"] == name:
            return dict(config)
    raise ValueError(f"未知配置: {name}（可选: {[c['name'] for c in CONFIGS]}）")


def block_dims(config: dict) -> tuple[int, int]:
    """(block.x, block.y) = (TN/RN, TM/RM)。"""
    return config["tn"] // config["rn"], config["tm"] // config["rm"]


def grid_dims(config: dict, m: int, n: int) -> tuple[int, int]:
    """(grid.x, grid.y) = (ceil(N/TN), ceil(M/TM))。"""
    return -(-n // config["tn"]), -(-m // config["tm"])


def smem_bytes(config: dict, pad: int = 1) -> int:
    """单缓冲静态共享内存字节数（与 kernel 中 As[TM][TK+pad]+Bs[TK][TN+pad] 一致）。"""
    return (config["tm"] * (config["tk"] + pad) + config["tk"] * (config["tn"] + pad)) * 4


def smem_bytes_double(config: dict, pad: int = 1) -> int:
    """双缓冲 = 2×单缓冲。"""
    return 2 * smem_bytes(config, pad)


def kernel_supports_config(kernel: str, config_name: str) -> bool:
    if kernel in ("sync", "cpasync1", "cpasync2"):
        return True
    if kernel == "warpspec":
        return config_name not in WARPSPEC_EXCLUDED
    if kernel in ("cpasync2v16", "sync2_broken", "cpasync2_broken"):
        return config_name in V16_CONFIGS
    if kernel == "wmma_pipe":
        return False  # 固定 64x64x16，不走本配置表
    return False


# MUSA 静态共享内存上限（迁移台账 M-43，ep20 双缓冲 smem 预算的硬约束）
MUSA_STATIC_SMEM_LIMIT = 28672
