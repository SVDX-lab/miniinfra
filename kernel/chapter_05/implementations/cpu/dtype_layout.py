"""第 05 期 dtype 位布局工具 —— FP16 与 BF16 的 sign/exponent/mantissa 分解。

第 05 期的第一个实验是"位布局对照"：数值表示不只是"精度"二字，而是
位域划分（sign 1 bit + exponent + mantissa）决定的四件事：
  1. 舍入步长（ulp）：mantissa 越多，同指数区间内可分辨的值越密；
  2. 指数范围：exponent 位数 + 阶码偏置决定不溢出/不下溢的量级范围；
  3. subnormal（非规格化数）下边界：渐进下溢的缓冲区有多宽；
  4. 溢出行为：超出 max 后统一变成 inf（IEEE 溢出语义）。

两种格式的位域（IEEE 754 binary16 / brain float 16，本模块全部实拆验证）：
  FP16: sign(1) + exponent(5, bias=15)  + mantissa(10 stored) => 有效精度 11 位
  BF16: sign(1) + exponent(8, bias=127) + mantissa(7 stored)  => 有效精度 8 位
（大纲"尾数 10 vs 8 位"用的是有效精度口径：fp16 10+1=11 位、bf16 7+1=8 位；
 存储的 mantissa 位数为 10 vs 7 —— 两套口径都写进对照表，避免歧义。）

位分解通过 torch CPU 的 view(torch.int16)（fp16/bf16 都是 2 字节）拿到
原始位型，再用整数运算拆位 —— 不靠任何"文档说法"，全部实拆。
torch CPU 原生支持 float16/bfloat16（本地 venv torch 2.6.0+cpu 已验证）。
"""

from __future__ import annotations

import struct

import numpy as np
import torch

# 位域参数（本模块用整数运算实拆，这两个表是拆位依据与断言依据）
FORMATS = {
    "fp16": {"sign_bits": 1, "exp_bits": 5, "man_bits": 10, "bias": 15},
    "bf16": {"sign_bits": 1, "exp_bits": 8, "man_bits": 7, "bias": 127},
}


def _bits16(value: float, dtype_name: str) -> int:
    """把一个 Python float 经 torch 转入目标 16 位格式，返回无符号位型。"""
    t = torch.tensor([float(value)])
    t = t.to(torch.float16 if dtype_name == "fp16" else torch.bfloat16)
    return int(t.view(torch.int16).item() & 0xFFFF)


def decompose_bits16(bits: int, dtype_name: str) -> dict:
    """16 位浮点位型 -> sign / exponent / mantissa 字段与数值分类。

    分类规则（IEEE 754 通用）：
      exponent 全 0 + mantissa 0        -> zero（±0）
      exponent 全 0 + mantissa != 0     -> subnormal（值 = (-1)^s * mantissa/2^k * 2^-bias+1）
      exponent 全 1 + mantissa 0        -> inf
      exponent 全 1 + mantissa != 0     -> nan
      其余                              -> normal（值 = (-1)^s * 1.mantissa * 2^(e-bias)）
    """
    spec = FORMATS[dtype_name]
    sign = (bits >> 15) & 0x1
    exponent = (bits >> spec["man_bits"]) & ((1 << spec["exp_bits"]) - 1)
    mantissa = bits & ((1 << spec["man_bits"]) - 1)
    exp_max = (1 << spec["exp_bits"]) - 1
    if exponent == 0 and mantissa == 0:
        cls = "zero"
    elif exponent == 0:
        cls = "subnormal"
    elif exponent == exp_max and mantissa == 0:
        cls = "inf"
    elif exponent == exp_max:
        cls = "nan"
    else:
        cls = "normal"
    # 数值重建（从位型，不从 float 路径）——与 torch 转换值互相印证
    if cls == "normal":
        value = (1.0 + mantissa / (1 << spec["man_bits"])) * 2.0 ** (exponent - spec["bias"])
    elif cls == "subnormal":
        value = (mantissa / (1 << spec["man_bits"])) * 2.0 ** (1 - spec["bias"])
    elif cls == "inf":
        value = float("inf") if sign == 0 else float("-inf")
    elif cls == "nan":
        value = float("nan")
    else:
        value = 0.0
    return {
        "bits": bits,
        "bits_str": f"{bits:016b}",
        "sign": sign,
        "exponent_raw": exponent,
        "exponent_unbiased": (exponent - spec["bias"]) if cls == "normal" else None,
        "mantissa": mantissa,
        "class": cls,
        "value": value,
        "dtype": dtype_name,
    }


def convert(value: float, dtype_name: str) -> dict:
    """float 值 -> 目标 16 位格式（torch 原生 RNE 转换）-> 位分解 + 回转误差。

    返回：转换后的位分解、位级分类、相对舍入误差（对 0 用绝对误差）。
    """
    spec = FORMATS[dtype_name]
    torch_dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
    t = torch.tensor([float(value)], dtype=torch.float32)
    roundtrip = t.to(torch_dtype).float().item()
    bits = _bits16(value, dtype_name)
    dec = decompose_bits16(bits, dtype_name)
    err = abs(roundtrip - float(value))
    denom = max(abs(float(value)), 1e-45)
    dec.update({
        "input_value": float(value),
        "roundtrip_value": roundtrip,
        "abs_roundtrip_error": err,
        "rel_roundtrip_error": err / denom,
    })
    return dec


def ulp_at(power: int, dtype_name: str) -> float:
    """区间 [2^power, 2^(power+1)) 内的 ulp（尾数位数决定）。

    normal 数的 ulp = 2^(power - mantissa_bits)；两个格式同指数区间的
    ulp 之比恒为 2^((man_fp16 - man_bf16)) = 2^3 = 8 倍。
    """
    spec = FORMATS[dtype_name]
    return 2.0 ** (power - spec["man_bits"])


def max_normal(dtype_name: str) -> float:
    """最大 normal 数：(2 - 2^-man) * 2^(2^(exp_bits)-1-bias)。"""
    spec = FORMATS[dtype_name]
    e = (1 << spec["exp_bits"]) - 2 - spec["bias"]
    return (1.0 - 2.0 ** (-spec["man_bits"])) * 2.0 ** e * 2.0


def min_normal(dtype_name: str) -> float:
    spec = FORMATS[dtype_name]
    return 2.0 ** (1 - spec["bias"])


def min_subnormal(dtype_name: str) -> float:
    spec = FORMATS[dtype_name]
    return 2.0 ** (1 - spec["bias"] - spec["man_bits"])


def format_boundaries() -> dict:
    """两格式关键边界值（实拆核对：位型重算值 vs torch 转换值一致）。"""
    out = {}
    for name in ("fp16", "bf16"):
        spec = FORMATS[name]
        torch_dtype = torch.float16 if name == "fp16" else torch.bfloat16
        # 用全 1 尾数构造 max normal：s=0 e=max-1 m=all-ones
        man_all = (1 << spec["man_bits"]) - 1
        exp_max1 = (1 << spec["exp_bits"]) - 2
        bits_max = (exp_max1 << spec["man_bits"]) | man_all
        dec_max = decompose_bits16(bits_max, name)
        torch_max = torch.finfo(torch_dtype).max
        # min normal: e=1 m=0
        dec_min = decompose_bits16(1 << spec["man_bits"], name)
        # min subnormal: e=0 m=1
        dec_sub = decompose_bits16(1, name)
        out[name] = {
            "max_normal_bits_recomputed": dec_max["value"],
            "max_normal_torch_finfo": torch_max,
            "max_normal_match": bool(dec_max["value"] == torch_max),
            "min_normal": dec_min["value"],
            "min_normal_torch_finfo": torch.finfo(torch_dtype).tiny,
            "min_subnormal": dec_sub["value"],
            "exp_bits": spec["exp_bits"],
            "man_bits_stored": spec["man_bits"],
            "man_bits_effective": spec["man_bits"] + 1,
            "exponent_bias": spec["bias"],
        }
    return out


def value_table(values: list[float]) -> list[dict]:
    """构造值表：每个值给出 FP32 / FP16 / BF16 三方分解与误差 —— EXPERIMENT_LOG 对照表数据源。"""
    rows = []
    for v in values:
        row = {"value_repr": repr(float(v)), "value": float(v)}
        fp32 = np.float32(v)
        row["fp32_repr"] = f"{struct.unpack('>I', struct.pack('>f', float(fp32)))[0]:032b}"
        row["fp32_abs_error_to_f64"] = abs(float(fp32) - float(v))
        for name in ("fp16", "bf16"):
            dec = convert(v, name)
            row[f"{name}_bits"] = dec["bits_str"]
            row[f"{name}_class"] = dec["class"]
            row[f"{name}_roundtrip"] = dec["roundtrip_value"]
            row[f"{name}_rel_error"] = dec["rel_roundtrip_error"]
        rows.append(row)
    return rows


def random_roundtrip_stats(rng: np.random.Generator, lo: float, hi: float,
                           n: int = 100000) -> dict:
    """区间 [lo, hi) 均匀采样 n 个 float32，统计 FP32->FP16/BF16->FP32 的相对舍入误差。

    理论上限：fp16 = 2^-11 ≈ 4.88e-4（半区间舍入），bf16 = 2^-8 ≈ 3.91e-3；
    实测 max 应接近但不超上限（RNE 的性质），比值恒约 8×。
    """
    x = (lo + (hi - lo) * rng.random(n)).astype(np.float32)
    out = {"range": [lo, hi], "n": n}
    for name in ("fp16", "bf16"):
        torch_dtype = torch.float16 if name == "fp16" else torch.bfloat16
        t = torch.from_numpy(x)
        rt = t.to(torch_dtype).float().numpy()
        rel = np.abs(rt - x) / np.abs(x)
        out[name] = {
            "max_rel_error": float(rel.max()),
            "mean_rel_error": float(rel.mean()),
            "theoretical_max_rel": 2.0 ** -(FORMATS[name]["man_bits"] + 1),
        }
    return out


def integer_precision_boundaries() -> list[dict]:
    """整数可分辨边界：fp16 在 2^11=2048 以上丢整数精度，bf16 在 2^8=256 以上丢。"""
    cases = []
    for n in (255, 256, 257, 511, 1023, 2047, 2048, 2049, 4095, 4097, 65535, 65537):
        fp16 = float(np.float16(np.float32(n)))
        bf16_t = torch.tensor([float(n)]).to(torch.bfloat16).float().item()
        cases.append({
            "integer": n,
            "fp16_roundtrip": fp16,
            "fp16_exact": fp16 == float(n),
            "bf16_roundtrip": bf16_t,
            "bf16_exact": bf16_t == float(n),
        })
    return cases
