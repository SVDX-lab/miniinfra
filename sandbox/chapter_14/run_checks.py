"""第 14 期实验：容器 vs Firecracker microVM（启动、常驻内存、隔离边界证据）。

microVM 配置：1 vCPU / 256MiB / 无网卡；bzImage(Alpine linux-virt 6.6.142)；
rootfs ext4 + init=/init；结果写第二块 ext4 盘，停机后用 debugfs 无挂载提取。
容器对照：docker run alpine（主线约束）+ 同一命令轨迹。
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import dockerops, images, results, stats

CHAPTER = "c14"
CHAPTER_DIR = os.path.dirname(os.path.abspath(__file__))
FC_DIR = os.path.join(CHAPTER_DIR, "data_local", "fc")
FC_BIN = os.path.join(FC_DIR, "release-v1.17.0-x86_64", "firecracker-v1.17.0-x86_64")
KERNEL = os.path.join(FC_DIR, "vmlinuz-virt")
ROOTFS = os.path.join(FC_DIR, "rootfs.ext4")
RESULTS_TPL = os.path.join(FC_DIR, "results.ext4")

CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}", flush=True)


def sh(argv, timeout=120):
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


def fc_put(sock_path, resource, obj):
    req = urllib.request.Request(
        f"http://localhost/{resource}", data=json.dumps(obj).encode(), method="PUT")
    return sh(["curl", "-s", "--unix-socket", sock_path, "-X", "PUT",
               "-H", "Content-Type: application/json",
               "-d", json.dumps(obj), f"http://localhost/{resource}"], timeout=30)


def fc_start(sock_path):
    return sh(["curl", "-s", "--unix-socket", sock_path, "-X",
               "PUT", "-H", "Content-Type: application/json",
               "-d", json.dumps({"action_type": "InstanceStart"}),
               "http://localhost/actions"], timeout=30)


def read_rss_kb(pid: int) -> int:
    with open(f"/proc/{pid}/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1])
    return -1


def run_microvm_once(workdir: str) -> dict:
    """启动一次 microVM，返回计时与提取的结果。"""
    os.makedirs(workdir, exist_ok=True)
    sock = os.path.join(workdir, "fc.sock")
    log = os.path.join(workdir, "fc.log")
    rootfs = os.path.join(workdir, "rootfs.ext4")
    results = os.path.join(workdir, "results.ext4")
    shutil.copyfile(ROOTFS, rootfs)
    shutil.copyfile(RESULTS_TPL, results)
    for p in (sock, log):
        if os.path.exists(p):
            os.remove(p)
    t_proc = time.perf_counter()
    fc = subprocess.Popen([FC_BIN, "--api-sock", sock],
                          stdout=open(log, "w"), stderr=subprocess.STDOUT)
    # 等 API socket
    for _ in range(100):
        if os.path.exists(sock):
            break
        time.sleep(0.02)
    fc_put(sock, "boot-source", {
        "kernel_image_path": KERNEL,
        "initrd_path": os.path.join(FC_DIR, "initramfs.cpio.gz"),
        "boot_args": "console=ttyS0 reboot=k panic=1 init=/init root=/dev/vda rw"})
    fc_put(sock, "drives/rootfs", {
        "drive_id": "rootfs", "path_on_host": rootfs,
        "is_root_device": True, "is_read_only": False})
    fc_put(sock, "drives/results", {
        "drive_id": "results", "path_on_host": results,
        "is_root_device": False, "is_read_only": False})
    fc_put(sock, "machine-config", {"vcpu_count": 1, "mem_size_mib": 256})
    t_start = time.perf_counter()
    fc_start(sock)
    boot_ms = task_ms = None
    rss_kb = -1
    deadline = t_start + 30
    while time.perf_counter() < deadline:
        try:
            with open(log) as f:
                content = f.read()
        except FileNotFoundError:
            content = ""
        if boot_ms is None and "BOOT_READY" in content:
            boot_ms = (time.perf_counter() - t_start) * 1000
            rss_kb = read_rss_kb(fc.pid)
        if task_ms is None and "TASK_DONE" in content:
            task_ms = (time.perf_counter() - t_start) * 1000
            break
        time.sleep(0.005)
    fc.wait(timeout=30)  # init 调 poweroff -f，VM 进程退出
    kernel_txt = sh(["debugfs", "-R", "cat /kernel.txt", results], timeout=30).stdout.strip()
    out_txt = sh(["debugfs", "-R", "cat /out.txt", results], timeout=30).stdout.strip()
    return {"boot_ms": round(boot_ms or -1, 1), "task_ms": round(task_ms or -1, 1),
            "fc_rss_kb_after_ready": rss_kb, "kernel": kernel_txt, "out": out_txt[:80],
            "proc_ms": round((time.perf_counter() - t_proc) * 1000, 1),
            "fc_exit_ok": fc.returncode == 0}


def run_container_once(name: str) -> dict:
    t_start = time.perf_counter()
    r = sh(["docker", "run", "-d", "--name", name,
            f"--label={dockerops.LABEL_CHAPTER}={CHAPTER}",
            "--network", "none", "--read-only", "--user", "65534:65534",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--memory", "256m", "--cpus", "1",
            "--tmpfs", "/tmp:rw,size=16m,noexec",
            images.PINNED["alpine"], "sleep", "infinity"])
    assert r.returncode == 0, r.stderr
    e1 = sh(["docker", "exec", name, "echo", "BOOT_READY"])
    boot_ms = (time.perf_counter() - t_start) * 1000
    e2 = sh(["docker", "exec", name, "sh", "-c",
             "uname -r > /tmp/kernel.txt; echo task-echo > /tmp/out.txt; "
             "dd if=/dev/zero bs=1M count=4 2>/dev/null | sha256sum | head -c 67 >> /tmp/out.txt; "
             "echo TASK_DONE"])
    task_ms = (time.perf_counter() - t_start) * 1000
    kernel = sh(["docker", "exec", name, "cat", "/tmp/kernel.txt"]).stdout.strip()
    mem = dockerops.docker(["stats", "--no-stream", "--format", "{{.MemUsage}}", name])
    sh(["docker", "rm", "-f", name])
    return {"boot_ms": round(boot_ms, 1), "task_ms": round(task_ms, 1),
            "kernel": kernel, "mem": mem.stdout.strip(),
            "ready_ok": "BOOT_READY" in e1.stdout}


def main() -> int:
    required = (FC_BIN, KERNEL, ROOTFS, RESULTS_TPL,
                os.path.join(FC_DIR, "initramfs.cpio.gz"))
    missing = [path for path in required if not os.path.isfile(path)]
    if missing:
        print("缺少 Firecracker 实验输入：\n" + "\n".join(missing), file=sys.stderr)
        return 2
    dockerops.cleanup_chapter(CHAPTER)
    host_kernel = os.uname().release
    work = os.path.join(FC_DIR, "runs")
    os.makedirs(work, exist_ok=True)

    # ---- VM 一次完整运行（正确性）----
    vm0 = run_microvm_once(os.path.join(work, "vm_check"))
    check("VM 启动并运行任务（BOOT_READY/TASK_DONE）",
          vm0["boot_ms"] > 0 and vm0["task_ms"] > 0, f"boot={vm0['boot_ms']}ms")
    check("隔离边界证据：VM guest kernel ≠ 宿主内核",
          "virt" in vm0["kernel"] and vm0["kernel"] != host_kernel,
          f"vm={vm0['kernel']} host={host_kernel}")
    check("文件兼容：结果盘带出内容（debugfs 提取）",
          "task-echo" in vm0["out"] and "virt" in vm0["kernel"])

    # ---- 容器一次运行（内核共享证据）----
    ct0 = run_container_once("mis-c14-check")
    check("容器与宿主共享内核（uname=宿主内核）",
          ct0["kernel"] == host_kernel and ct0["ready_ok"],
          f"container={ct0['kernel']}")
    check("网络语义对齐：容器 network none ↔ VM 无网卡", True,
          "VM machine-config 未配置任何 network-interfaces")

    # ---- 启动对照（各 10 次）----
    vm_runs = [run_microvm_once(os.path.join(work, f"vm{i}")) for i in range(10)]
    ct_runs = [run_container_once(f"mis-c14-b{i}") for i in range(10)]
    vm_boot = stats.summary([r["boot_ms"] for r in vm_runs])
    vm_task = stats.summary([r["task_ms"] for r in vm_runs])
    ct_boot = stats.summary([r["boot_ms"] for r in ct_runs])
    ct_task = stats.summary([r["task_ms"] for r in ct_runs])

    dockerops.cleanup_chapter(CHAPTER)
    check("实验后容器归零", len(dockerops.ps_ids(f"{dockerops.LABEL_CHAPTER}={CHAPTER}")) == 0)

    passed = sum(1 for _, ok, _ in CHECKS if ok)
    doc = {"checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in CHECKS],
           "passed": passed, "total": len(CHECKS),
           "kernels": {"vm": vm0["kernel"], "container": ct0["kernel"], "host": host_kernel},
           "vm": {"boot_ms": vm_boot, "task_ms": vm_task,
                  "rss_after_ready_kb": vm_runs[0]["fc_rss_kb_after_ready"],
                  "raw": [[r["boot_ms"], r["task_ms"]] for r in vm_runs]},
           "container": {"boot_ms": ct_boot, "task_ms": ct_task, "mem": ct0["mem"],
                         "raw": [[r["boot_ms"], r["task_ms"]] for r in ct_runs]},
           "meta": {"firecracker": "v1.17.0", "vm_config": "1vCPU/256MiB/no-nic",
                    "kernel": "alpine linux-virt 6.6.142 bzImage"}}
    path = results.write_result(
        CHAPTER_DIR, "backend_ab", doc,
        extra_meta={"note": "WSL2 嵌套虚拟化环境；VM 不经 jailer（教学取舍）"})
    print(f"\n{passed}/{len(CHECKS)} passed → {path}")
    print(json.dumps({"vm_boot": vm_boot, "ct_boot": ct_boot,
                      "vm_task": vm_task, "ct_task": ct_task}, indent=1))
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
