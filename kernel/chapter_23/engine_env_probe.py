# -*- coding: utf-8 -*-
"""实验 1：引擎环境盘点（MUSA 列）。

不初始化引擎，只做静态探测：
- vllm / torch / torch_musa / vllm-omni 版本锁定（§14 版本耦合矩阵素材）
- platform 解析机制与 platforms/musa.py 关键行为（支持面、streams 缺口）
- vllm 自定义算子注册机制（_custom_ops / torch.ops._C 可解析面）
- musa_tools 目录（ep15 streams 缺口的 vendor 侧素材）
产出 results/env-probe-musa.json
"""
import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine24_common import dump_json, utcnow  # noqa: E402


def sh(cmd, timeout=60):
    try:
        out = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return {"rc": out.returncode, "stdout": out.stdout[-4000:], "stderr": out.stderr[-2000:]}
    except Exception as e:  # noqa: BLE001
        return {"rc": -1, "error": repr(e)}


def read_text(path, limit=200000):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(limit)
    except Exception as e:  # noqa: BLE001
        return None


def grep(text, pattern, ctx=0, maxhit=40):
    if text is None:
        return None
    hits = []
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if re.search(pattern, ln):
            hits.append(f"{i+1}: {ln.strip()[:240]}")
            if len(hits) >= maxhit:
                break
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/env-probe-musa.json")
    args = ap.parse_args()

    R = {"schema_version": "kernel-chapter24.envprobe.v1", "created_at_utc": utcnow()}

    # ---- 1. 版本锁定 ----
    import torch
    R["versions"] = {"python": sys.version.split()[0], "torch": torch.__version__}
    try:
        import torch_musa
        R["versions"]["torch_musa"] = getattr(torch_musa, "__version__", "n/a")
        R["versions"]["torch_musa_path"] = os.path.dirname(torch_musa.__file__)
    except Exception as e:  # noqa: BLE001
        R["versions"]["torch_musa"] = f"IMPORT_FAIL: {e}"
    try:
        import vllm
        R["versions"]["vllm"] = vllm.__version__
        R["versions"]["vllm_path"] = os.path.dirname(vllm.__file__)
    except Exception as e:  # noqa: BLE001
        R["versions"]["vllm"] = f"IMPORT_FAIL: {e}"
        dump_json(R, args.out)
        print("vllm import failed, partial dump written")
        return
    try:
        import vllm_omni
        R["versions"]["vllm_omni"] = getattr(vllm_omni, "__version__", "import-ok-no-version")
        R["versions"]["vllm_omni_path"] = os.path.dirname(vllm_omni.__file__)
    except Exception as e:  # noqa: BLE001
        R["versions"]["vllm_omni"] = f"IMPORT_FAIL: {type(e).__name__}: {e}"
    R["versions"]["musa_sdk"] = sh("ls /usr/local/musa && cat /usr/local/musa/version 2>/dev/null | head -5")
    R["versions"]["driver"] = sh("mthreads-gmi -q 2>/dev/null | head -40 || mthreads-gmi 2>&1 | head -20")
    R["versions"]["pip_vllm"] = sh("pip list 2>/dev/null | grep -i -E 'vllm|torch|mudnn|triton'")

    # ---- 2. platform 解析机制 ----
    try:
        from vllm.platforms import current_platform, PlatformEnum
        R["platform"] = {
            "class": f"{type(current_platform).__module__}.{type(current_platform).__name__}",
            "enum": str(getattr(current_platform, "_enum", getattr(PlatformEnum, "MUSA", "n/a"))),
            "device_name": str(getattr(current_platform, "device_type", None)),
            "is_cuda": getattr(current_platform, "is_cuda", lambda: None)(),
            "is_musa_like": hasattr(current_platform, "device_type"),
            "supported_quant_apis": str(getattr(current_platform, "supported_quant_apis", None)),
            "get_default_compile_config": bool(hasattr(current_platform, "get_default_compile_config")),
        }
    except Exception as e:  # noqa: BLE001
        R["platform"] = {"error": repr(e)}

    vroot = R["versions"].get("vllm_path")
    if vroot:
        # ---- 3. platform 注册机制源码证据 ----
        pinit = read_text(os.path.join(vroot, "platforms", "__init__.py"))
        R["platform_registry"] = {
            "resolve_mechanism": grep(pinit, r"resolve|plugin|_init_musa|importlib|register", maxhit=30),
        }
        pmusa = read_text(os.path.join(vroot, "platforms", "musa.py"))
        R["platform_musa_py"] = {
            "size_lines": len(pmusa.splitlines()) if pmusa else 0,
            "class_def": grep(pmusa, r"^class |MUSAPlatform", maxhit=5),
            "supports_fp8": grep(pmusa, r"supports_fp8|has_device_capability", maxhit=10),
            "streams": grep(pmusa, r"streams|Stream", maxhit=20),
            "graph": grep(pmusa, r"[gG]raph", maxhit=30),
            "custom_ops": grep(pmusa, r"custom_op|register|_C\b|direct_register", maxhit=40),
            "attention_backend": grep(pmusa, r"attention_backend|backend_name|TRITON|FLASH", maxhit=20),
            "quant": grep(pmusa, r"quant|fp8|int8", maxhit=20),
            "device_name_probe": grep(pmusa, r"device_name|mthreads|gmi", maxhit=10),
        }
        # ---- 4. musa_tools（vendor shim 素材）----
        mt = os.path.join(vroot, "musa_tools")
        if os.path.isdir(mt):
            R["musa_tools"] = {"listing": sorted(os.listdir(mt))}
            for fn in sorted(os.listdir(mt)):
                if fn.endswith(".py"):
                    txt = read_text(os.path.join(mt, fn), 60000)
                    R["musa_tools"][f"{fn}_head"] = (txt[:3000] if txt else None)
        # ---- 5. 自定义算子注册机制 ----
        cust = read_text(os.path.join(vroot, "_custom_ops.py"), 120000)
        R["custom_ops"] = {
            "has_native_call": grep(cust, r"def native_", maxhit=20),
            "register_musa": grep(cust, r"musa|MUSA", maxhit=40),
            "rms_norm_def": grep(cust, r"def rms_norm|_C::rms_norm", maxhit=10),
            "head_60": cust.splitlines()[:60] if cust else None,
        }
        # torch.ops._C 实际可解析面（import vllm._custom_ops 触发注册）
        try:
            import vllm._custom_ops as vco  # noqa: F401
            import torch
            names = ["rms_norm", "fused_add_rms_norm", "rotary_embedding",
                     "reshape_and_cache_flash", "paged_attention_v1",
                     "fused_moe", "silu_and_mul", "topk_sigmoid"]
            R["torch_ops_C_available"] = {
                n: hasattr(torch.ops._C, n) for n in names
            }
        except Exception as e:  # noqa: BLE001
            R["torch_ops_C_available"] = {"error": repr(e)}
        # layernorm 分发路径（RMSNorm 在 MUSA 上走哪条实现）
        ln = read_text(os.path.join(vroot, "model_executor", "layers", "layernorm.py"), 80000)
        R["layernorm_dispatch"] = {
            "forward_def": grep(ln, r"def forward_(native|cuda|musa|triton)", maxhit=20),
            "platform_check": grep(ln, r"current_platform|is_cuda|musa", maxhit=30),
            "rms_head": grep(ln, r"class RMSNorm", maxhit=5),
        }
        # rotaty embedding 分发
        re_ = read_text(os.path.join(vroot, "model_executor", "layers", "rotary_embedding.py"), 80000)
        R["rotary_dispatch"] = {
            "dynamic_import": grep(re_, r"def get_rope|_ROPE_INIT|is_neox", maxhit=15),
            "custom_call": grep(re_, r"rotary_embedding\(|torch.ops|_C\.", maxhit=20),
        }
        # attention backend 解析（decode 步算子序列定位用）
        ab = read_text(os.path.join(vroot, "v1", "attention", "backends", "__init__.py"), 40000)
        R["attention_backend_registry"] = grep(ab, r"musa|MUSA|TRITON|FLASH_ATTN|backend", maxhit=40)
        # compilation config / graph 默认（Graph-Paged 共存注记素材）
        comp = read_text(os.path.join(vroot, "config", "compilation.py"), 60000)
        if comp is None:
            comp = read_text(os.path.join(vroot, "v1", "worker", "gpu_model_runner.py"), 100000)
        R["graph_defaults"] = {
            "compilation_config_graph_mode": grep(comp, r"cudagraph|CUDAGraph|graph_mode|FULL_AND_PIECEWISE|PIECEWISE", maxhit=40),
        }
        gmr = read_text(os.path.join(vroot, "v1", "worker", "gpu_model_runner.py"), 150000)
        R["graph_runner"] = {
            "capture_loop": grep(gmr, r"cudagraph|capture|graph_pool|replay", maxhit=50),
            "paged_kv": grep(gmr, r"kv_cache|block_table|slot_mapping|paged", maxhit=40),
        }
        # streams 缺口素材（ep15 的 torch_musa.streams 缺失 vs vllm 如何绕开）
        R["streams_material"] = {
            "torch_musa_streams_exists": sh(
                "python3 -c \"import torch_musa; print(hasattr(torch_musa, 'streams'))\""
            ),
            "vllm_cuda_stream_usage": sh(
                f"grep -rn 'torch.cuda.Stream\\|current_stream\\|set_device_and_stream' "
                f"{vroot}/v1/worker/ {vroot}/platforms/musa.py 2>/dev/null | head -20"
            ),
        }

    dump_json(R, args.out)
    print(f"env probe written -> {args.out}")
    # 控制台摘要（关键版本一行）
    v = R.get("versions", {})
    print(
        "SUMMARY vllm=%s torch=%s torch_musa=%s omni=%s platform=%s"
        % (
            v.get("vllm"),
            v.get("torch"),
            v.get("torch_musa"),
            v.get("vllm_omni"),
            R.get("platform", {}).get("class"),
        )
    )


if __name__ == "__main__":
    main()
