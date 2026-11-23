#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 30 期 实验 1：代际边界实证（4090 / Ada sm_89）。

四条证据通道：
  A. 独立 flash-attn 包：wheel 可得性 -> import -> FA4 路径调用尝试（记录架构门控原文）；
  B. 官方 flash-attention sdist 源码考古（不编译）：grep FA4 架构门控与预编译 arch 列表；
  C. 已装栈（vllm/flashinfer，ep24 装机）源码扫描：FA4/Blackwell/sm100/tcgen05 门控字符串；
  D. torch 2.14 SDPA 后端同形状对照（ch21 分母形状 S=8192 B=4 H=16 D=128 fp16 causal，
     ch21 锚点 148 TF = 86% FP16 峰值），flash/cudnn/efficient/math 逐后端正确性+性能。

用法：python3 fa4_generation_probe.py [--skip-sdist]
输出：results/fa4-generation-probe-cuda.json + results/sdpa-backend-compare-cuda.json
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

SCHEMA = "kernel-chapter31.fa4probe.v1"
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.normpath(os.path.join(HERE, "..", "..", "results"))

# ch21 SDPA 分母锚点形状（chapter_20 EXPERIMENT_LOG §7.3：S=8192 B=4 16 头 D=128 fp16 causal
# = 1.1 TFLOP / 7.46ms = 148 TF = 86% FP16 峰值）
ANCHOR = dict(B=4, S=8192, HQ=16, HKV=16, D=128, dtype="fp16", causal=True)
SMALL = dict(B=4, S=2048, HQ=16, HKV=16, D=128, dtype="fp16", causal=True)


def utcnow():
    return datetime.now(timezone.utc).strftime("%FT%TZ")


def run(cmd, timeout=180, cwd=None):
    """跑子进程，返回 (rc, stdout_tail, stderr_tail)。"""
    try:
        p = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True,
                           text=True, timeout=timeout, cwd=cwd)
        tail = lambda s: "\n".join(s.splitlines()[-25:]) if s else ""
        return p.returncode, tail(p.stdout), tail(p.stderr)
    except subprocess.TimeoutExpired as e:
        tail = lambda s: "\n".join((s or "").splitlines()[-10:])
        return 124, tail(getattr(e, "stdout", "")), "TIMEOUT after %ss\n%s" % (timeout, tail(getattr(e, "stderr", "")))
    except Exception as e:  # noqa: BLE001
        return 125, "", repr(e)


def pip_install(pkg, timeout=240):
    """只收预编译 wheel（--only-binary），失败即快速记录，绝不触发源码编译。"""
    return run([sys.executable, "-m", "pip", "install", "--only-binary=:all:",
                "--no-cache-dir", "--no-deps", pkg], timeout=timeout)


def tail_lines(s, n=8):
    ls = (s or "").splitlines()
    return ls[-n:]


# ---------------------------------------------------------------------------
def channel_a_flash_attn(res):
    """独立 flash-attn 包：wheel 可得性 -> import -> FA 调用尝试。"""
    a = {"channel": "A_standalone_flash_attn"}
    rc, out, err = run([sys.executable, "-c", "import flash_attn; print(flash_attn.__version__)"], timeout=60)
    a["preinstalled"] = (rc == 0)
    if rc == 0:
        a["version"] = out.strip().splitlines()[-1] if out.strip() else "?"
    else:
        a["import_error_tail"] = tail_lines(err)
        t0 = time.time()
        rc2, out2, err2 = pip_install("flash-attn", timeout=300)
        a["pip_only_binary_rc"] = rc2
        a["pip_only_binary_seconds"] = round(time.time() - t0, 1)
        a["pip_tail"] = {"stdout": tail_lines(out2, 15), "stderr": tail_lines(err2, 15)}
        if rc2 != 0:
            joined = (out2 + "\n" + err2).lower()
            a["failure_class"] = ("no_wheel" if ("only binary" in joined or "no matching distribution" in joined
                                                  or "could not find" in joined)
                                  else "other")
        else:
            a["version"] = "?"
    # 无论装没装上，都尝试 import + FA4 路径调用
    rc, out, err = run([sys.executable, "-c", "import flash_attn; print(flash_attn.__version__)"], timeout=60)
    a["post_import"] = {"rc": rc, "version": out.strip() if rc == 0 else None,
                        "error_tail": tail_lines(err) if rc != 0 else []}
    if rc == 0:
        probe = r"""
import json, traceback
info = {}
try:
    import flash_attn, inspect, pkgutil
    info["version"] = getattr(flash_attn, "__version__", "?")
    info["modules"] = sorted(m.name for m in pkgutil.iter_modules(flash_attn.__path__))
    fa3_fa4 = {}
    for modname in ("flash_attn.flash_attn_interface", "flash_attn_interface",
                    "flash_attn.flash_attn_v4", "flash_attn.flash4"):
        try:
            mod = __import__(modname, fromlist=["*"])
            fns = [n for n in dir(mod) if "flash_attn" in n.lower() or "func" in n.lower()]
            fa3_fa4[modname] = fns[:20]
            for gate in ("is_hopper", "is_blackwell", "is_sm100", "check_cuda_arch"):
                if hasattr(mod, gate):
                    fa3_fa4[modname + "." + gate] = inspect.getsource(getattr(mod, gate))[:400]
        except Exception as e:
            fa3_fa4[modname] = "IMPORT_FAIL: %r" % (e,)
    info["fa_modules"] = fa3_fa4
    # 尝试最可能的 FA 高代际入口（Hopper/Blackwell gated），同锚点小形状调用
    call_log = []
    import torch
    B, S, H, D = 2, 512, 16, 128
    q, k, v = (torch.randn(B, S, H, D, dtype=torch.float16, device="cuda") for _ in range(3))
    for modname, fnname in (("flash_attn.flash_attn_interface", "flash_attn_func"),
                            ("flash_attn_interface", "flash_attn_func"),
                            ("flash_attn.flash_attn_interface", "flash_attn_with_kvcache")):
        try:
            mod = __import__(modname, fromlist=["*"])
            fn = getattr(mod, fnname, None)
            if fn is None:
                call_log.append({"entry": modname + "." + fnname, "result": "ABSENT"})
                continue
            out = fn(q, k, v, causal=True)
            torch.cuda.synchronize()
            call_log.append({"entry": modname + "." + fnname,
                             "result": "OK out=" + str(tuple(out.shape)) + " " + str(out.dtype)})
        except Exception as e:
            call_log.append({"entry": modname + "." + fnname, "result": "FAIL",
                             "error": repr(e)[:600]})
    info["call_log"] = call_log
except Exception:
    info["fatal"] = traceback.format_exc()[-800:]
print(json.dumps(info))
"""
        rc, out, err = run([sys.executable, "-c", probe], timeout=180)
        a["introspect_rc"] = rc
        if rc == 0:
            try:
                a["introspect"] = json.loads(out.strip().splitlines()[-1])
            except Exception:
                a["introspect_raw"] = tail_lines(out, 10)
        else:
            a["introspect_error_tail"] = tail_lines(err, 15)
    res["channel_A"] = a


# ---------------------------------------------------------------------------
def channel_b_sdist(res, skip=False):
    """官方 sdist 源码考古（不编译）：FA4 架构门控 + 预编译 arch 列表。"""
    b = {"channel": "B_official_sdist_source_scan", "skipped": skip}
    if skip:
        res["channel_B"] = b
        return
    if shutil.which("pip") is None and not hasattr(sys, "real_prefix"):
        pass
    tmpd = tempfile.mkdtemp(prefix="fa4sdist_")
    try:
        rc, out, err = run([sys.executable, "-m", "pip", "download", "flash-attn", "--no-deps",
                            "--no-binary", ":all:", "--no-cache-dir", "-d", tmpd], timeout=420)
        b["download_rc"] = rc
        b["download_tail"] = {"stdout": tail_lines(out, 10), "stderr": tail_lines(err, 10)}
        sdists = [f for f in os.listdir(tmpd) if f.endswith((".tar.gz", ".zip"))]
        if sdists:
            import tarfile
            b["sdist"] = sdists[0]
            ext = os.path.join(tmpd, "x")
            os.makedirs(ext, exist_ok=True)
            try:
                with tarfile.open(os.path.join(tmpd, sdists[0])) as tf:
                    tf.extractall(ext)  # noqa: S202 受控目录
            except Exception as e:  # noqa: BLE001
                b["extract_fail"] = repr(e)
                res["channel_B"] = b
                return
            pats = re.compile(r"(fa4|FA4|flash4|Flash4|sm_?100|sm_?103|blackwell|Blackwell|tcgen05|tmem|10\.0a?)")
            arch_pats = re.compile(r"(sm8[09]|sm9[0]|sm10[03]|sm_89|compute_89|gencode|CC_LIST|cuda_arch)")
            hits, arch_hits = [], []
            for root, _dirs, files in os.walk(ext):
                for f in files:
                    if not f.endswith((".py", ".h", ".cu", ".cuh", ".hpp", "setup", ".cfg", ".toml")):
                        continue
                    p = os.path.join(root, f)
                    try:
                        if os.path.getsize(p) > 3_000_000:
                            continue
                        with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                            for i, line in enumerate(fh, 1):
                                if pats.search(line):
                                    hits.append({"file": p.replace(ext + "/", ""), "line": i,
                                                 "text": line.strip()[:220]})
                                if arch_pats.search(line) and len(arch_hits) < 25:
                                    arch_hits.append({"file": p.replace(ext + "/", ""), "line": i,
                                                      "text": line.strip()[:220]})
                    except OSError:
                        continue
                    if len(hits) >= 60:
                        break
                if len(hits) >= 60:
                    break
            b["gate_hits"] = hits
            b["arch_hits"] = arch_hits
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)
    res["channel_B"] = b


# ---------------------------------------------------------------------------
def channel_c_installed_stack(res):
    """已装栈源码扫描：vllm / flashinfer 的 FA4/Blackwell 门控字符串。"""
    c = {"channel": "C_installed_stack_gate_scan"}
    rc, out, err = run([sys.executable, "-c",
                        "import importlib.util as u; print(u.find_spec('vllm').origin if u.find_spec('vllm') else 'NA');"
                        "print(u.find_spec('flashinfer').origin if u.find_spec('flashinfer') else 'NA')"], timeout=60)
    roots = {}
    for line in (out or "").splitlines():
        if line and line != "NA":
            roots[os.path.dirname(line)] = line
    c["roots"] = {k: v for k, v in roots.items()}
    pats = re.compile(r"(fa4|FA4|flash4|sm_?100|sm_?103|blackwell|Blackwell|tcgen05|warp_specializ)"
                      r"|(capability\s*\(?\)\s*>=?\s*\(?\s*1?0|is_sm\d{3})")
    scan = {}
    for root, origin in roots.items():
        hits = []
        for dirpath, dirnames, files in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in (".git", "build", "dist-info", "tests", "__pycache__")]
            for f in files:
                if not f.endswith(".py"):
                    continue
                p = os.path.join(dirpath, f)
                try:
                    if os.path.getsize(p) > 1_500_000:
                        continue
                    with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                        for i, line in enumerate(fh, 1):
                            if pats.search(line):
                                hits.append({"file": p.replace(root + "/", ""), "line": i,
                                             "text": line.strip()[:220]})
                                if len(hits) >= 40:
                                    break
                except OSError:
                    continue
                if len(hits) >= 40:
                    break
            if len(hits) >= 40:
                break
        scan[os.path.basename(root) or root] = {"count": len(hits), "hits": hits}
    c["scan"] = scan
    res["channel_C"] = c


# ---------------------------------------------------------------------------
def channel_d_device_facts(res):
    d = {"channel": "D_device_facts"}
    rc, out, _ = run("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader", timeout=30)
    d["nvidia_smi"] = out.strip()
    rc, out, _ = run("nvcc --version", timeout=30)
    d["nvcc"] = tail_lines(out, 4)
    probe = r"""
import json, torch
info = {
    "torch": torch.__version__,
    "torch_cuda": getattr(torch.version, "cuda", None),
    "device": torch.cuda.get_device_name(0),
    "capability": list(torch.cuda.get_device_capability(0)),
    "sdpa_flags": {
        "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
        "cudnn_sdp_enabled": getattr(torch.backends.cuda, "cudnn_sdp_enabled", lambda: "NA")(),
    },
    "sdp_backends": [n for n in dir(__import__("torch.nn.attention", fromlist=["SDPBackend"]).SDPBackend)
                     if not n.startswith("_")] if hasattr(__import__("torch.nn.attention", fromlist=["SDPBackend"]), "SDPBackend") else [],
    "flex_attention": hasattr(__import__("torch.nn.attention.flex_attention", fromlist=["flex_attention"]), "flex_attention"),
    "triton": __import__("triton").__version__,
}
print(json.dumps(info))
"""
    rc, out, err = run([sys.executable, "-c", probe], timeout=120)
    d["probe_rc"] = rc
    if rc == 0:
        d.update(json.loads(out.strip().splitlines()[-1]))
    else:
        d["error_tail"] = tail_lines(err, 12)
    res["channel_D"] = d


# ---------------------------------------------------------------------------
def _make_qkv(shape, device="cuda"):
    import torch
    g = torch.Generator(device=device).manual_seed(3131)
    B, S, H, D = shape["B"], shape["S"], shape["HQ"], shape["D"]
    q = torch.randn(B, H, S, D, generator=g, device=device, dtype=torch.float16)
    k = torch.randn(B, H, S, D, generator=g, device=device, dtype=torch.float16)
    v = torch.randn(B, H, S, D, generator=g, device=device, dtype=torch.float16)
    return q, k, v


def _math_ref(q, k, v, causal):
    """fp32 数学参考（softmax(QK^T/sqrt(D))V）——按 (b,h) 分块防 OOM
    （整张 fp32 S×S @ S=8192 B=4 H=16 = 17.2GB，4090 24G 单块必 OOM——首跑实录）。"""
    import torch
    B, H, S, D = q.shape
    outs = []
    scale = float(D) ** 0.5
    for b in range(B):
        for h in range(H):
            qf = q[b, h].float()
            scores = qf @ k[b, h].float().transpose(-2, -1) / scale
            if causal:
                mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=scores.device), diagonal=1)
                scores = scores.masked_fill(mask, float("-inf"))
            outs.append((torch.softmax(scores, dim=-1) @ v[b, h].float()).half())
    return torch.stack(outs).view(B, H, S, D)


def sdpa_compare(res):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel  # torch >= 2.3
    out = {"schema_version": "kernel-chapter31.sdpa-backend-compare.v1", "ts": utcnow(),
           "note": "ch21 §7.3 分母锚点复测（torch 已被 ep24 从 2.12.1+cu130 升级到当前版本，版本差异如实记录）",
           "shapes": []}
    backends = [("FLASH_ATTENTION", SDPBackend.FLASH_ATTENTION),
                ("CUDNN_ATTENTION", SDPBackend.CUDNN_ATTENTION),
                ("EFFICIENT_ATTENTION", SDPBackend.EFFICIENT_ATTENTION),
                ("MATH", SDPBackend.MATH)]
    for shape in (ANCHOR, SMALL):
        entry = {"shape": shape, "backends": []}
        q, k, v = _make_qkv(shape)
        ref = _math_ref(q, k, v, shape["causal"]).half()
        flops = 4 * shape["B"] * shape["HQ"] * shape["S"] * shape["S"] * shape["D"]
        if shape["causal"]:
            flops /= 2.0  # ch21 口径：causal 半三角
        for name, be in backends:
            rec = {"backend": name}
            try:
                with sdpa_kernel([be]):
                    for _ in range(5):
                        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=shape["causal"])
                    torch.cuda.synchronize()
                    ts = []
                    for _ in range(20):
                        s0, e0 = torch.cuda.Event(True), torch.cuda.Event(True)
                        s0.record()
                        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=shape["causal"])
                        e0.record()
                        torch.cuda.synchronize()
                        ts.append(s0.elapsed_time(e0))
                ts.sort()
                p50 = ts[len(ts) // 2]
                err = (o.float() - ref.float()).abs()
                denom = ref.float().abs().clamp_min(1e-3)
                rec.update({
                    "ok": True,
                    "p50_ms": round(p50, 4),
                    "p95_ms": round(ts[int(len(ts) * 0.95) - 1], 4),
                    "tflops_p50": round(flops / (p50 * 1e-3) / 1e12, 1),
                    "max_abs_err": float(err.max()),
                    "mean_abs_err": float(err.mean()),
                    "cosine": float(torch.nn.functional.cosine_similarity(
                        o.float().flatten(), ref.float().flatten(), dim=0)),
                    "nan_inf": int(torch.isnan(o).sum() + torch.isinf(o).sum()),
                })
            except Exception as e:  # noqa: BLE001
                rec.update({"ok": False, "error": repr(e)[:500]})
            entry["backends"].append(rec)
        del q, k, v, ref
        torch.cuda.empty_cache()
        out["shapes"].append(entry)
    res["sdpa_compare"] = out
    with open(os.path.join(RESULTS, "sdpa-backend-compare-cuda.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sdist", action="store_true", help="跳过通道 B（sdist 下载）")
    ap.add_argument("--skip-pip", action="store_true", help="跳过通道 A 的 pip 安装尝试")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    res = {"schema_version": SCHEMA, "ts": utcnow(), "episode": 31,
           "purpose": "FA4 generational boundary on Ada sm_89: attempt FA4 channels, record arch gates, "
                      "fallback same-shape SDPA backend comparison"}
    channel_d_device_facts(res)
    channel_c_installed_stack(res)
    channel_a_flash_attn(res)
    if args.skip_pip:
        res.get("channel_A", {})["pip_attempted"] = False
    channel_b_sdist(res, skip=args.skip_sdist)
    path = os.path.join(RESULTS, "fa4-generation-probe-cuda.json")
    with open(path, "w") as f:  # 先落盘通道证据，SDPA 崩溃不丢通道结果
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("WROTE(channels)", path)
    sdpa_compare(res)
    with open(path, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("WROTE", path)
    # 摘要打到 stdout 供会话日志
    a = res.get("channel_A", {})
    print("[A] preinstalled=%s pip_rc=%s failure=%s" % (a.get("preinstalled"), a.get("pip_only_binary_rc"), a.get("failure_class")))
    for shp in res.get("sdpa_compare", {}).get("shapes", []):
        for b in shp["backends"]:
            print("[SDPA] S=%s %s ok=%s p50=%s tflops=%s" % (shp["shape"]["S"], b["backend"], b.get("ok"), b.get("p50_ms"), b.get("tflops_p50")))


if __name__ == "__main__":
    main()
