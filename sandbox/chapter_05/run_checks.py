"""第 05 期检查：布局隔离、导出白名单、符号链接逃逸、并发写、拷贝量与耗时。"""
from __future__ import annotations

import base64
import json
import os
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chapter_05.sandbox_service import SandboxService, make_handler, CHAPTER, sha256_file
from common import dockerops, images, results, stats
from http.server import ThreadingHTTPServer

CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}", flush=True)


def api(method, url, body=None):
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(url, data=data, method=method)
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def dec(b64):
    return base64.b64decode(b64).decode(errors="replace")


def main() -> int:
    dockerops.cleanup_chapter(CHAPTER)
    svc = SandboxService()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(svc))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    input_sha = svc.shared_input["input_dat_sha256"]

    # ---- structured：两会话共享输入，写入互不影响 ----
    _, sa = api("POST", f"{base}/sessions", {"layout": "structured"})
    _, sb = api("POST", f"{base}/sessions", {"layout": "structured"})
    A, B = sa["session_id"], sb["session_id"]
    _, e = api("POST", f"{base}/sessions/{A}/exec",
               {"cmd": ["sh", "-c", "cat /inputs/input.txt; sha256sum /inputs/input.dat | cut -c1-16"]})
    out_a = dec(e["stdout_b64"])
    _, e = api("POST", f"{base}/sessions/{B}/exec",
               {"cmd": ["sh", "-c", "echo mine > /workspace/note.txt; "
                                 "echo other > /inputs/input.txt 2>&1 || echo WRITE_DENIED"]})
    out_b = dec(e["stdout_b64"])
    _, e = api("POST", f"{base}/sessions/{B}/exec", {"cmd": ["cat", "/inputs/input.txt"]})
    input_b_after = dec(e["stdout_b64"])
    _, e = api("POST", f"{base}/sessions/{A}/exec",
               {"cmd": ["sh", "-c", "test -f /workspace/note.txt && echo LEAKED || echo ISOLATED"]})
    iso = dec(e["stdout_b64"]).strip()
    check("structured：/inputs 只读（写入被拒）", "WRITE_DENIED" in out_b, out_b.strip())
    check("structured：B 的写入未影响共享输入", "seed=" in input_b_after)
    check("structured：会话间工作区隔离", iso == "ISOLATED")
    check("structured：输入可读且哈希与控制面一致",
          input_sha[:16] in out_a, out_a.strip().splitlines()[-1])

    # ---- flat：两会话各自拷贝输入，同样隔离 ----
    _, fa = api("POST", f"{base}/sessions", {"layout": "flat"})
    _, fb = api("POST", f"{base}/sessions", {"layout": "flat"})
    FA, FB = fa["session_id"], fb["session_id"]
    api("POST", f"{base}/sessions/{FA}/exec",
        {"cmd": ["sh", "-c", "echo A > /workspace/inputs/input.txt"]})
    _, e = api("POST", f"{base}/sessions/{FB}/exec",
               {"cmd": ["sh", "-c", "head -1 /workspace/inputs/input.txt"]})
    check("flat：各自拷贝互不影响", dec(e["stdout_b64"]).startswith("seed="),
          dec(e["stdout_b64"]).strip())
    check("flat prep（拷贝 32MiB）慢于 structured（零拷贝）",
          fa["prep_ms"] > sa["prep_ms"] * 3,
          f"flat={fa['prep_ms']:.0f}ms structured={sa['prep_ms']:.0f}ms")

    # ---- 导出白名单与符号链接逃逸 ----
    api("POST", f"{base}/sessions/{A}/exec",
        {"cmd": ["sh", "-c",
                 "dd if=/dev/urandom of=/workspace/artifact.bin bs=1M count=16 2>/dev/null; "
                 "echo result > /workspace/result.txt; "
                 "ln -s /etc/passwd /workspace/pwn_link; "
                 "ln -s ../../etc /workspace/rel_link"]})
    _, ex = api("POST", f"{base}/sessions/{A}/export",
                {"paths": ["result.txt", "artifact.bin", "pwn_link", "rel_link", "../escape"]})
    names_ok = [m["path"] for m in ex["manifest"]]
    rejected = [r["path"] for r in ex["rejected"]]
    check("导出只包含允许的普通文件",
          names_ok == ["result.txt", "artifact.bin"], str(names_ok))
    check("符号链接与相对路径逃逸被拒绝",
          set(rejected) == {"pwn_link", "rel_link", "../escape"}, str(rejected))
    art = next(m for m in ex["manifest"] if m["path"] == "artifact.bin")
    host_sha = sha256_file(os.path.join(ex["export_dir"], "artifact.bin"))
    check("导出文件校验和一致", art["sha256"] == host_sha and art["bytes"] == 16 * 1024 * 1024,
          f"16MiB export_ms={ex['export_ms']}")

    # ---- 并发写同名文件：两会话导出内容各自正确 ----
    api("POST", f"{base}/sessions/{A}/exec",
        {"cmd": ["sh", "-c", "printf 'AAAA%.0s' $(seq 1 1000) > /workspace/out.txt"]})
    api("POST", f"{base}/sessions/{B}/exec",
        {"cmd": ["sh", "-c", "printf 'BBBB%.0s' $(seq 1 1000) > /workspace/out.txt"]})
    _, exa = api("POST", f"{base}/sessions/{A}/export", {"paths": ["out.txt"]})
    _, exb = api("POST", f"{base}/sessions/{B}/export", {"paths": ["out.txt"]})
    ha = exa["manifest"][0]["sha256"]
    hb = exb["manifest"][0]["sha256"]
    check("并发写同名文件：校验和不同且各自完整",
          ha != hb and exa["manifest"][0]["bytes"] == exb["manifest"][0]["bytes"] == 4000)

    # ---- 清理 ----
    for sid in (A, B, FA, FB):
        api("POST", f"{base}/sessions/{sid}/stop")
    left = dockerops.ps_ids(f"{dockerops.LABEL_CHAPTER}={CHAPTER}", include_stopped=False)
    check("实验后容器归零", len(left) == 0)

    svc.shutdown()
    srv.shutdown()
    passed = sum(1 for _, ok, _ in CHECKS if ok)
    doc = {"checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in CHECKS],
           "passed": passed, "total": len(CHECKS),
           "prep_ms": {"flat_32mib_copy": round(fa["prep_ms"], 1),
                       "structured_zero_copy": round(sa["prep_ms"], 1)},
           "export_16mib_ms": ex["export_ms"]}
    path = results.write_result(
        os.path.dirname(os.path.abspath(__file__)), "layout_ab", doc,
        extra_meta={"image": images.PINNED["alpine"], "cache_state": "warm",
                    "input_bytes": svc.shared_input["bytes"]})
    print(f"\n{passed}/{len(CHECKS)} passed → {path}")
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
