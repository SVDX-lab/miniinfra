#!/usr/bin/env python3
"""第 02 期估算题演练（自动核版）：Qwen3-0.6B 权重显存 / KV Cache / RMSNorm 访存。

手算过程写在 EXPERIMENT_LOG.md；本脚本把每一步做成可复核的计算，并在有
模型文件的环境里（--model-dir）用文件系统数值（safetensors 头逐张量求和、
文件大小、config.json）校核手算。

口径：
- 权重显存 = 参数量 × dtype 字节（FP16=2，FP32=4）；
- 每 Token KV Cache = 2(K+V) × layers × kv_heads × head_dim × dtype 字节；
- RMSNorm 每 Token 理论访存 = 每次 3H×itemsize（读 x + 读 w + 写 y），
  Qwen3-0.6B 每 Token 经过 57 次 RMSNorm（28×2 层内 + 1 final）。

用法：
    python estimate_qwen3.py --model-dir ../models/qwen3-0.6b
    python estimate_qwen3.py                # 无模型时只做公式计算
输出：results/estimation-check.json
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

# Qwen3-0.6B 结构常数（手算口径；--model-dir 时与 config.json 逐项核对）
CONFIG = {
    "hidden_size": 1024,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 3072,
    "vocab_size": 151936,
    "tie_word_embeddings": True,
    "rms_norm_eps": 1e-6,
}

SCHEMA_VERSION = "kernel-chapter02.estimation.v1"


def param_formula(config: dict) -> dict:
    """按结构公式逐项数参数（不考虑 lm_head 复制）。"""
    h = config["hidden_size"]
    layers = config["num_hidden_layers"]
    q_dim = config["num_attention_heads"] * config["head_dim"]
    kv_dim = config["num_key_value_heads"] * config["head_dim"]
    i = config["intermediate_size"]
    vocab = config["vocab_size"]
    per_layer = {
        "input_layernorm": h,
        "q_proj": h * q_dim,
        "k_proj": h * kv_dim,
        "v_proj": h * kv_dim,
        "o_proj": q_dim * h,
        "post_attention_layernorm": h,
        "gate_proj": h * i,
        "up_proj": h * i,
        "down_proj": i * h,
    }
    per_layer_total = sum(per_layer.values())
    embed = vocab * h
    final_norm = h
    total = per_layer_total * layers + embed + final_norm
    return {
        "per_layer": per_layer,
        "per_layer_total": per_layer_total,
        "embed_tokens": embed,
        "final_norm": final_norm,
        "total_params_model_in_memory": total,
        "note": "lm_head 与 embed_tokens tied，模型常驻参数不含第二份",
    }


def kv_cache_per_token(config: dict, itemsize: int = 2) -> dict:
    per_token = 2 * config["num_hidden_layers"] * config["num_key_value_heads"] * config["head_dim"] * itemsize
    return {
        "bytes_per_token": per_token,
        "kiB_per_token": round(per_token / 1024, 1),
        "formula": "2(K+V) x layers(28) x kv_heads(8) x head_dim(128) x 2B(fp16)",
        "context_2048_bytes": per_token * 2048,
        "context_2048_MiB": round(per_token * 2048 / 1024 / 1024, 1),
        "context_32768_bytes": per_token * 32768,
        "context_32768_GiB": round(per_token * 32768 / 1024 / 1024 / 1024, 2),
    }


def rmsnorm_per_token(config: dict, itemsize: int = 2) -> dict:
    h = config["hidden_size"]
    norms_per_layer = 2
    norms_total = norms_per_layer * config["num_hidden_layers"] + 1
    per_call = 3 * h * itemsize
    per_token = norms_total * per_call
    return {
        "norms_per_token": norms_total,
        "bytes_per_norm_call": per_call,
        "bytes_per_token_incl_weight_reads": per_token,
        "bytes_per_token_marginal_weight_cached": norms_total * 2 * h * itemsize,
        "formula": "57 x (读 x 2048B + 读 w 2048B + 写 y 2048B) = 350208B",
        "composite_unfused_note": "组合表达式未融合链路流量约为该值的 ~9.5 倍（见 roofline.py）",
    }


def parse_safetensors_header(path: Path) -> dict:
    """纯 Python 解析 safetensors 头（8 字节小端 u64 长度 + JSON），逐张量字节求和。"""
    with path.open("rb") as file:
        (header_len,) = struct.unpack("<Q", file.read(8))
        header = json.loads(file.read(header_len).decode("utf-8"))
    dtype_items = {"F16": 2, "F32": 4, "BF16": 2, "I8": 1, "U8": 1}
    tensors = {}
    total = 0
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        shape = meta["shape"]
        n = 1
        for dim in shape:
            n *= dim
        items = dtype_items.get(meta["dtype"], 0)
        size = n * items
        tensors[name] = {"shape": shape, "dtype": meta["dtype"], "params": n, "bytes": size}
        total += size
    return {
        "header_len": header_len,
        "tensor_count": len(tensors),
        "total_params_in_file": sum(t["params"] for t in tensors.values()),
        "total_bytes_in_file": total,
        "file_size_on_disk": path.stat().st_size,
        "header_plus_consistency": path.stat().st_size == total + 8 + header_len,
        "has_lm_head_copy": "lm_head.weight" in tensors,
        "tensors": tensors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=None)
    arguments = parser.parse_args()

    payload: dict = {
        "schema_version": SCHEMA_VERSION,
        "config_used": CONFIG,
        "config_source": "手算口径（--model-dir 提供时与 config.json 核对）",
    }
    payload["param_formula"] = param_formula(CONFIG)
    payload["kv_cache_per_token"] = kv_cache_per_token(CONFIG)
    payload["rmsnorm_per_token"] = rmsnorm_per_token(CONFIG)

    config = CONFIG
    if arguments.model_dir:
        model_dir = Path(arguments.model_dir)
        config_path = model_dir / "config.json"
        if config_path.exists():
            actual = json.loads(config_path.read_text())
            checks = {}
            for key in ["hidden_size", "num_hidden_layers", "num_attention_heads",
                        "num_key_value_heads", "head_dim", "intermediate_size", "vocab_size",
                        "tie_word_embeddings", "rms_norm_eps"]:
                if key in actual:
                    checks[key] = {"formula": config[key], "config_json": actual[key],
                                   "match": actual[key] == config[key]}
            payload["config_json_check"] = checks
            mismatches = [k for k, v in checks.items() if not v["match"]]
            if mismatches:
                print(f"[estimation] config.json 与手算口径不一致的字段: {mismatches}（以实际为准修正手算）")

        safetensors_path = model_dir / "model.safetensors"
        if safetensors_path.exists():
            file_check = parse_safetensors_header(safetensors_path)
            payload["safetensors_check"] = file_check
            in_memory = payload["param_formula"]["total_params_model_in_memory"]
            per_file = file_check["total_params_in_file"]
            payload["weight_memory"] = {
                "model_in_memory_params": in_memory,
                "model_in_memory_fp16_bytes": in_memory * 2,
                "model_in_memory_fp16_GiB": round(in_memory * 2 / 1024**3, 3),
                "file_params_incl_lm_head_copy": per_file,
                "file_bytes_sum": file_check["total_bytes_in_file"],
                "file_size_on_disk_bytes": file_check["file_size_on_disk"],
                "file_size_GiB": round(file_check["file_size_on_disk"] / 1024**3, 3),
                "explanation": "safetensors 文件比常驻权重多一份 lm_head.weight 复制（tied embedding 时很多 release 仍保存）",
            }
            print(f"[estimation] 权重显存(常驻, fp16) = {payload['weight_memory']['model_in_memory_fp16_bytes']:,} B "
                  f"= {payload['weight_memory']['model_in_memory_fp16_GiB']} GiB")
            print(f"[estimation] safetensors 文件 = {file_check['file_size_on_disk']:,} B "
                  f"(逐张量求和 {file_check['total_bytes_in_file']:,} B, 一致={file_check['header_plus_consistency']})")

    print(f"[estimation] KV Cache/Token = {payload['kv_cache_per_token']['bytes_per_token']:,} B "
          f"({payload['kv_cache_per_token']['kiB_per_token']} KiB)；2048 ctx = "
          f"{payload['kv_cache_per_token']['context_2048_MiB']} MiB")
    print(f"[estimation] RMSNorm/Token = {payload['rmsnorm_per_token']['bytes_per_token_incl_weight_reads']:,} B")

    out = Path("results") / "estimation-check.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[estimation] 已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
