#!/usr/bin/env python3
"""smoke_test.py —— 第 10 期边界行为验证（在编译机上跑，无 torch 依赖）。

检查点（全部基于裸跑 binary 的可观察行为）：
  S1 defect_suite：good 版三 case 全过（自检 max_abs_error≈0 / 输出符合期望）；
  S2 defect_suite oob/race bad：程序正常退出但损坏证据在位
     （race bad 5 次运行 unique>1；oob bad mismatches=0=“未爆雷但错”的教学形态）；
  S3 defect_suite uninit bad：正常退出（值取决于脏内存，只验证不崩溃）；
  S4 defect_suite sync bad：裸跑死锁 → timeout=124（预期挂死）；
  S5 printf_trap 四模式可运行；
  S6 health_ch04 / health_ch08 各 target 自检可运行且 max_abs_error 记录在案。
"""
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
BUILD = HERE / "implementations" / "cuda" / "build"
TIMEOUT = 60


def run(cmd: list[str], timeout: int = TIMEOUT) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr)
    except subprocess.TimeoutExpired as exc:
        return 124, (exc.stdout or "") + (exc.stderr or "")


def main() -> int:
    if not BUILD.exists():
        print("build/ 不存在——先跑 implementations/cuda/build.sh", file=sys.stderr)
        return 2
    failures: list[str] = []
    ds = str(BUILD / "defect_suite")

    # S1 good 三 case
    for case, expect in [
        ("oob", "mismatches=0"),
        ("race", "unique_outputs_5rep=1"),
        ("uninit", "nonzero_outputs=0"),
    ]:
        rc, out = run([ds, "--case", case, "--variant", "good"])
        ok = rc == 0 and expect in out
        print(f"S1 {case}/good rc={rc} expect[{expect}] -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"S1-{case}")

    # S2 bad：正常退出 + 损坏证据
    rc, out = run([ds, "--case", "race", "--variant", "bad"])
    ok = rc == 0 and "unique_outputs_5rep=5" in out
    print(f"S2 race/bad rc={rc} unique=5 -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append("S2-race-bad")
    rc, out = run([ds, "--case", "oob", "--variant", "bad"])
    ok = rc == 0 and "mismatches=0" in out
    print(f"S2 oob/bad rc={rc}（未爆雷形态） -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append("S2-oob-bad")

    # S3 uninit bad：不崩溃即过（输出值依赖脏内存）
    rc, out = run([ds, "--case", "uninit", "--variant", "bad"])
    print(f"S3 uninit/bad rc={rc} -> {'PASS' if rc == 0 else 'FAIL'}")
    if rc != 0:
        failures.append("S3-uninit-bad")

    # S4 sync bad：死锁（timeout 124）
    rc, _ = run([ds, "--case", "sync", "--variant", "bad"], timeout=30)
    print(f"S4 sync/bad rc={rc}（124=死锁被截断，预期） -> {'PASS' if rc == 124 else 'FAIL'}")
    if rc != 124:
        failures.append("S4-sync-bad")

    # S5 printf 四模式
    pt = str(BUILD / "printf_trap")
    for args in (
        ["--mode", "order"],
        ["--mode", "trunc", "--blocks", "64", "--block", "64", "--per-thread", "1"],
        ["--mode", "frag", "--blocks", "2", "--block", "32"],
    ):
        rc, out = run([pt, *args])
        ok = rc == 0 and len(out) > 0
        print(f"S5 printf {' '.join(args)} rc={rc} -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append("S5-printf")

    # S6 健康靶自检
    for binary, args in [
        ("health_ch04", ["--target", "ch04-fused", "--rows", "64", "--h", "1024",
                         "--tile", "256", "--iters", "1"]),
        ("health_ch08", ["--target", "ch08-det", "--rows", "64", "--h", "1024",
                         "--tile", "256", "--iters", "1"]),
        ("health_ch08", ["--target", "ch08-atomic-shared", "--rows", "64", "--h", "1024",
                         "--tile", "256", "--iters", "1"]),
        ("health_ch08", ["--target", "ch08-atomic-global", "--rows", "64", "--h", "1024",
                         "--tile", "256", "--iters", "1", "--splits", "8"]),
    ]:
        rc, out = run([str(BUILD / binary), *args])
        ok = rc == 0 and "max_abs_error=" in out
        print(f"S6 {binary} {' '.join(args[1:4])} rc={rc} -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append("S6")

    print(f"\n{'全部通过' if not failures else '失败: ' + ','.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
