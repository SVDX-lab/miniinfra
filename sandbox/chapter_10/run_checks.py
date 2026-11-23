"""第 10 期实验：环境组合（每任务解包 vs 版本固定只读层/派生镜像）。"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import dockerops, images, results, stats

CHAPTER = "c10"
BASE = images.PINNED["alpine"]
TOOLS_BYTES = 8 * 1024 * 1024

CONSTRAINTS = ["--network", "none", "--read-only", "--user", "65534:65534",
               "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
               "--memory", "256m", "--pids-limit", "64",
               "--tmpfs", "/tmp:rw,size=16m,noexec"]

CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}", flush=True)


def sh(argv, timeout=300):
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


def make_tools(version: str, out_dir: str) -> str:
    """生成确定性工具包目录并打包 tar.gz，返回包路径与源目录。"""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "tool.sh"), "w") as f:
        f.write(f"#!/bin/sh\necho tools-{version}\nsha256sum /opt/tools/data.bin | cut -c1-16\n")
    os.chmod(os.path.join(out_dir, "tool.sh"), 0o755)
    dat = os.path.join(out_dir, "data.bin")
    if not os.path.exists(dat) or os.path.getsize(dat) != TOOLS_BYTES:
        block = bytes(range(256)) * (TOOLS_BYTES // 256)
        with open(dat, "wb") as f:
            f.write(block)
    tar = f"{os.path.dirname(out_dir)}/tools_{version}.tar.gz"
    subprocess.run(["tar", "czf", tar, "-C", os.path.dirname(out_dir),
                    os.path.basename(out_dir)], check=True)
    return tar


def run_task(image: str, prep_cmd: list[str] | None, tar_bind: str | None = "",
             name_suffix: str = "", tools_dir: str | None = None) -> dict:
    """跑一个任务容器：可选解包，然后执行 tool.sh。返回耗时与输出。

    解包组把每任务的宿主目录 bind 到 /opt/tools（rw）后解压——
    只读 rootfs 下这就是"每任务解包进任务卷"的语义。
    """
    name = f"mis-{CHAPTER}-task-{name_suffix}-{int(time.time()*100000)%10000000}"
    mounts = []
    if tar_bind:
        mounts += ["-v", tar_bind]
    if tools_dir:
        os.makedirs(tools_dir, exist_ok=True)
        os.chmod(tools_dir, 0o777)
        mounts += ["-v", f"{tools_dir}:/opt/tools"]
    t_total = time.perf_counter()
    r = sh(["docker", "run", "-d", "--name", name,
            f"--label={dockerops.LABEL_CHAPTER}={CHAPTER}", *CONSTRAINTS, *mounts,
            image, "sleep", "infinity"])
    assert r.returncode == 0, r.stderr
    prep_ms = 0.0
    if prep_cmd:
        t0 = time.perf_counter()
        rp = sh(["docker", "exec", name, *prep_cmd])
        prep_ms = (time.perf_counter() - t0) * 1000.0
        assert rp.returncode == 0, rp.stderr
    t0 = time.perf_counter()
    re_ = sh(["docker", "exec", name, "/opt/tools/tool.sh"])
    ready_ms = (time.perf_counter() - t0) * 1000.0
    out = re_.stdout.strip()
    sh(["docker", "rm", "-f", name])
    if tools_dir:
        shutil.rmtree(tools_dir, ignore_errors=True)
    return {"prep_ms": round(prep_ms, 1), "tool_ms": round(ready_ms, 1),
            "total_ms": round((time.perf_counter() - t_total) * 1000, 1),
            "out": out}


def main() -> int:
    dockerops.cleanup_chapter(CHAPTER)
    data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_local")
    os.makedirs(data, exist_ok=True)

    src_sha = hashlib.sha256(open(os.path.join(data, "tools_v1", "data.bin"), "rb").read()) \
        .hexdigest()[:16] if os.path.exists(os.path.join(data, "tools_v1", "data.bin")) else None
    make_tools("v1", os.path.join(data, "tools_v1"))
    tar_v1 = make_tools("v1", os.path.join(data, "tools_v1"))  # 确保包存在
    src_sha = hashlib.sha256(open(os.path.join(data, "tools_v1", "data.bin"), "rb").read()) \
        .hexdigest()[:16]

    # 只读层：构建派生镜像（v1）
    build_ctx = os.path.join(data, "build_v1")
    os.makedirs(build_ctx, exist_ok=True)
    shutil.copytree(os.path.join(data, "tools_v1"), os.path.join(build_ctx, "tools"),
                    dirs_exist_ok=True)
    with open(os.path.join(build_ctx, "Dockerfile"), "w") as f:
        f.write(f"FROM {BASE}\nCOPY tools /opt/tools\nLABEL mis.chapter={CHAPTER}\n")
    t0 = time.perf_counter()
    r = sh(["docker", "build", "-t", "mis-c10-tools:v1", build_ctx])
    assert r.returncode == 0, r.stderr
    build_v1_ms = (time.perf_counter() - t0) * 1000.0

    # ---- 5 任务对照：解包组（每任务解包进任务卷 /opt/tools）----
    unpack_tasks = []
    for i in range(5):
        unpack_tasks.append(run_task(
            BASE,
            ["sh", "-c", "tar -xzf /tmp/tools.tar.gz -C /opt/tools --strip-components=1"],
            tar_bind=f"{tar_v1}:/tmp/tools.tar.gz:ro",
            name_suffix=f"u{i}",
            tools_dir=os.path.join(data, "task_vol", f"u{i}")))

    # ---- 5 任务对照：只读层组（派生镜像直接启动）----
    layer_tasks = [run_task("mis-c10-tools:v1", None, name_suffix=f"l{i}")
                   for i in range(5)]

    uprep = stats.summary([t["prep_ms"] for t in unpack_tasks])
    lprep = stats.summary([t["prep_ms"] for t in layer_tasks])
    check("解包组每任务都要解包（prep p50>50ms）", uprep["p50"] > 50,
          f"p50={uprep['p50']:.0f}ms ×5 任务")
    check("只读层组任务准备为零", all(t["prep_ms"] == 0.0 for t in layer_tasks))
    check("两组工具版本与内容 digest 一致",
          all(t["out"].splitlines()[0] == "tools-v1" and t["out"].splitlines()[1] == src_sha
              for t in unpack_tasks + layer_tasks),
          f"期望 sha16={src_sha}")

    # ---- 升级 v1→v2 ----
    make_tools("v2", os.path.join(data, "tools_v2"))
    tar_v2 = make_tools("v2", os.path.join(data, "tools_v2"))
    src_sha2 = hashlib.sha256(open(os.path.join(data, "tools_v2", "data.bin"), "rb").read()) \
        .hexdigest()[:16]

    # 解包组升级：0 次构建，下一任务解 v2 包
    t_up_unpack = run_task(BASE,
                           ["sh", "-c", "tar -xzf /tmp/tools.tar.gz -C /opt/tools --strip-components=1"],
                           tar_bind=f"{tar_v2}:/tmp/tools.tar.gz:ro", name_suffix="u_v2",
                           tools_dir=os.path.join(data, "task_vol", "u_v2"))
    # 只读层组升级：1 次构建，任务 0 准备
    build_ctx2 = os.path.join(data, "build_v2")
    os.makedirs(build_ctx2, exist_ok=True)
    shutil.copytree(os.path.join(data, "tools_v2"), os.path.join(build_ctx2, "tools"),
                    dirs_exist_ok=True)
    with open(os.path.join(build_ctx2, "Dockerfile"), "w") as f:
        f.write(f"FROM {BASE}\nCOPY tools /opt/tools\nLABEL mis.chapter={CHAPTER}\n")
    t0 = time.perf_counter()
    r = sh(["docker", "build", "-t", "mis-c10-tools:v2", build_ctx2])
    assert r.returncode == 0, r.stderr
    build_v2_ms = (time.perf_counter() - t0) * 1000.0
    t_up_layer = run_task("mis-c10-tools:v2", None, name_suffix="l_v2")

    check("升级：解包组 0 次构建但每任务仍有解包成本，输出 v2",
          t_up_unpack["out"].splitlines()[0] == "tools-v2" and t_up_unpack["prep_ms"] > 50,
          f"prep={t_up_unpack['prep_ms']}ms")
    check("升级：只读层组 1 次构建后任务 0 准备输出 v2",
          t_up_layer["out"].splitlines()[0] == "tools-v2" and t_up_layer["prep_ms"] == 0.0
          and t_up_layer["out"].splitlines()[1] == src_sha2,
          f"build={build_v2_ms:.0f}ms")

    # 写入量对比：镜像层 vs 任务内解包
    r = sh(["docker", "image", "inspect", BASE, "--format", "{{.Size}}"])
    base_size = int(r.stdout.strip())
    r = sh(["docker", "image", "inspect", "mis-c10-tools:v2", "--format", "{{.Size}}"])
    layer_size = int(r.stdout.strip()) - base_size
    unpack_writes_total = TOOLS_BYTES * 6  # 5 任务 + 1 次升级

    check("只读层写入一次性（层大小≈工具包），解包组累计写入随任务线性",
          0 < layer_size < TOOLS_BYTES * 2 and unpack_writes_total == TOOLS_BYTES * 6,
          f"layer={layer_bytes_str(layer_size)} 解包累计≈{unpack_writes_total//1048576}MiB")

    dockerops.cleanup_chapter(CHAPTER)
    sh(["docker", "rmi", "mis-c10-tools:v1", "mis-c10-tools:v2"])
    check("实验后容器归零", len(dockerops.ps_ids(f"{dockerops.LABEL_CHAPTER}={CHAPTER}")) == 0)

    passed = sum(1 for _, ok, _ in CHECKS if ok)
    doc = {"checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in CHECKS],
           "passed": passed, "total": len(CHECKS),
           "unpack_prep_ms_per_task": uprep,
           "layer_prep_ms_per_task": lprep,
           "build_ms": {"v1": round(build_v1_ms, 1), "v2": round(build_v2_ms, 1)},
           "writes": {"layer_bytes_once": layer_size,
                      "unpack_bytes_total_6_tasks": unpack_writes_total},
           "task_raw": {"unpack": unpack_tasks, "layer": layer_tasks}}
    path = results.write_result(
        os.path.dirname(os.path.abspath(__file__)), "env_composition_ab", doc,
        extra_meta={"base_image": BASE, "tools_bytes": TOOLS_BYTES, "tasks_per_group": 5})
    print(f"\n{passed}/{len(CHECKS)} passed → {path}")
    return 0 if passed == len(CHECKS) else 1


def layer_bytes_str(n: int) -> str:
    return f"{n/1048576:.1f}MiB"


if __name__ == "__main__":
    sys.exit(main())
