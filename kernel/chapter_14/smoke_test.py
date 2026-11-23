"""第 14 期 smoke test：编译路径的边界行为验证（含 functionalization 探测）。

第 01/02/03 期的边界项（零行、NaN 传播、H=0、空 Tensor）在编译路径下
全部复验——"编译器不改变语义"必须逐项证明而不是假设。新增编译特有项：
- in-place 输入突变经 torch.compile 后调用方可见性（functionalization）；
- 非连续输入触发 guard 重编译；
- dtype 变化（fp64）触发重编译且数值正确；
- 确定性：同一编译实例重复调用位级一致；
- dynamic=True 单次编译跨多 shape。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from implementations.cpu.compiled_objects import (
    add_rmsnorm_inplace_style,
    add_rmsnorm_separated,
    rmsnorm_expr,
)
from implementations.cpu.reference_serial import separated_add_rmsnorm_serial

RESULT_LINES: list[tuple[bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULT_LINES.append((ok, f"{name}{' | ' + detail if detail else ''}"))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" | {detail}" if detail else ""))


def main() -> None:
    torch.set_num_threads(min(6, torch.get_num_threads()))
    compiled_add = torch.compile(add_rmsnorm_separated)
    compiled_rms = torch.compile(rmsnorm_expr)

    # 1. 手算例（2×4，第 03 期同款）编译路径 vs 串行真值
    r = np.array([[1.0, 2.0, 3.0, 4.0], [-1.0, 0.0, 1.0, -2.0]], dtype=np.float32)
    h = np.array([[1.0, 0.0, -1.0, 0.0], [1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    w = np.ones(4, dtype=np.float32)
    t_true, y_true = separated_add_rmsnorm_serial(r, h, w)
    t_c, y_c = compiled_add(torch.from_numpy(r), torch.from_numpy(h), torch.from_numpy(w))
    check(
        "手算例编译路径位级/容差",
        np.allclose(y_c.numpy(), y_true, atol=1e-6)
        and np.array_equal(t_c.numpy(), t_true),
        f"t bitwise={np.array_equal(t_c.numpy(), t_true)}, "
        f"y max={np.abs(y_c.numpy() - y_true).max():.2e}",
    )

    # 2. 双零行 → residual=0、y=0，无 NaN
    z = torch.zeros(2, 8, dtype=torch.float32)
    wz = torch.ones(8, dtype=torch.float32)
    tz, yz = compiled_add(z.clone(), z.clone(), wz)
    check(
        "双零行（编译路径）",
        bool((yz == 0).all()) and bool(not torch.isnan(yz).any()),
    )

    # 3. NaN 传播语义：NaN 元素污染平方和 → 该行 y 全 NaN；其他行不受影响
    rng = np.random.default_rng(15)
    rn = torch.from_numpy(rng.standard_normal((3, 8), dtype=np.float32))
    hn = torch.from_numpy(rng.standard_normal((3, 8), dtype=np.float32))
    rn[1, 2] = float("nan")
    tn, yn = compiled_add(rn.clone(), hn, torch.ones(8))
    check(
        "NaN 行传播（编译路径）",
        bool(torch.isnan(yn[1]).all())
        and bool(not torch.isnan(yn[0]).any())
        and bool(not torch.isnan(yn[2]).any())
        and bool(torch.isnan(tn[1, 2]) and not torch.isnan(tn[1, 0])),
    )

    # 4. in-place 输入突变（functionalization）：调用后 r buffer == r+h 位级
    r_ip = torch.from_numpy(rng.standard_normal((4, 8), dtype=np.float32))
    r_orig = r_ip.clone()
    h_ip = torch.from_numpy(rng.standard_normal((4, 8), dtype=np.float32))
    compiled_ip = torch.compile(add_rmsnorm_inplace_style)
    r_out, y_ip = compiled_ip(r_ip, h_ip, torch.ones(8))
    t_np, y_np_true = separated_add_rmsnorm_serial(
        r_orig.numpy(), h_ip.numpy(), np.ones(8, dtype=np.float32)
    )
    check(
        "in-place 突变经编译后调用方可见（functionalization）",
        bool(r_out is r_ip) and np.array_equal(r_ip.numpy(), t_np),
        f"same_buffer={r_out is r_ip}, "
        f"y max={np.abs(y_ip.numpy() - y_np_true).max():.2e}",
    )

    # 5. H=0 显式报错：eager 与编译路径都要 raise（合同语义由 trace 期 raise 保留）
    try:
        add_rmsnorm_separated(torch.zeros(2, 0), torch.zeros(2, 0), torch.zeros(0))
        eager_raised = "none"
    except Exception as error:  # noqa: BLE001
        eager_raised = type(error).__name__
    try:
        compiled_add(torch.zeros(2, 0), torch.zeros(2, 0), torch.zeros(0))
        compiled_raised = "none"
    except Exception as error:  # noqa: BLE001
        compiled_raised = type(error).__name__
    check(
        "H=0 显式报错（eager 与编译路径一致）",
        eager_raised == "ValueError" and compiled_raised == "ValueError",
        f"eager={eager_raised}, compiled={compiled_raised}"
        "（首次以 H=0 调用 = 编译期 trace 直接抛出）",
    )

    # 6. rows=0 空 Tensor
    t0, y0 = compiled_add(torch.zeros(0, 8), torch.zeros(0, 8), torch.ones(8))
    check("rows=0 空 Tensor（编译路径）", t0.shape == (0, 8) and y0.shape == (0, 8))

    # 7. dtype 变化 → 重编译且数值正确（guard on dtype）。
    # 注意源码语义：.float() 上转换是函数体的一部分（ch01 RMSNorm 口径），
    # fp64 输入的 variance 数学仍是 FP32（生成代码 kernel 签名
    # double→float buffer），因此对照容差 = FP32 合同容差 1e-5，
    # 差异来源 = 归约顺序（向量化和 vs eager pairwise 和）。
    r64 = torch.from_numpy(rng.standard_normal((4, 8))).double()
    h64 = torch.from_numpy(rng.standard_normal((4, 8))).double()
    w64 = torch.ones(8, dtype=torch.float64)
    t64, y64 = compiled_add(r64, h64, w64)
    y64_ref = rmsnorm_expr((r64 + h64), w64)
    check(
        "fp64 输入重编译数值正确（FP32 内部语义，容差 1e-5）",
        bool(torch.allclose(y64, y64_ref, rtol=1e-5, atol=1e-5)),
        f"max={float((y64 - y64_ref).abs().max()):.2e}"
        "（≈1e-7 = FP32 归约顺序噪声，经 double 输出保留）",
    )

    # 8. 非连续输入 → guard 重编译（不崩、结果正确）
    big = torch.from_numpy(rng.standard_normal((4, 16), dtype=np.float32))
    r_nc = big[:, ::2]
    h_nc = torch.from_numpy(rng.standard_normal((4, 8), dtype=np.float32))
    t_nc, y_nc = compiled_add(r_nc, h_nc, torch.ones(8))
    t_ref, y_ref = add_rmsnorm_separated(r_nc.contiguous(), h_nc, torch.ones(8))
    check(
        "非连续输入 guard 重编译",
        bool(torch.allclose(y_nc, y_ref, atol=1e-6)) and bool(torch.allclose(t_nc, t_ref)),
    )

    # 9. 确定性：同编译实例重复调用位级一致
    r_d = torch.from_numpy(rng.standard_normal((8, 8), dtype=np.float32))
    h_d = torch.from_numpy(rng.standard_normal((8, 8), dtype=np.float32))
    wd = torch.ones(8)
    a1 = compiled_add(r_d, h_d, wd)
    a2 = compiled_add(r_d, h_d, wd)
    check(
        "重复调用位级一致",
        bool(torch.equal(a1[0], a2[0]) and torch.equal(a1[1], a2[1])),
    )

    # 10. dynamic=True：跨 shape 复用（≤2 张图：size-1 特化图 + 动态图）。
    # 子进程隔离——torch 2.6 的 automatic dynamic frame state 跨 reset 存活，
    # 同进程先前编译会污染 dynamic=True 行为（KI-15-01）。
    import subprocess
    import sys

    code = (
        "import torch, numpy as np\n"
        "from implementations.cpu.compiled_objects import add_rmsnorm_separated\n"
        "c = torch.compile(add_rmsnorm_separated, dynamic=True)\n"
        "rng = np.random.default_rng(15)\n"
        "shapes = []\n"
        "for rows in (1, 7, 128, 2048):\n"
        "    r = torch.from_numpy(rng.standard_normal((rows, 8), dtype=np.float32))\n"
        "    h = torch.from_numpy(rng.standard_normal((rows, 8), dtype=np.float32))\n"
        "    t, y = c(r, h, torch.ones(8))\n"
        "    shapes.append(tuple(t.shape))\n"
        "print('DYN-RESULT', shapes,\n"
        "      torch._dynamo.utils.counters['stats'].get('unique_graphs', 0))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parent), check=False,
    )
    dyn_ok = "DYN-RESULT" in done.stdout and "[(1, 8), (7, 8), (128, 8), (2048, 8)]" in done.stdout
    graphs_final = -1
    if dyn_ok:
        graphs_final = int(done.stdout.strip().split("]")[-1].strip().split(" ")[0])
    check(
        "dynamic=True 跨 shape ≤2 张图（size-1 特化 + 动态图，子进程隔离）",
        dyn_ok and 0 < graphs_final <= 2,
        f"unique_graphs={graphs_final}",
    )

    # 11. 编译路径 RMSNorm 与第 01 期 RMSNorm.forward 数学等价（随机大样本）
    xr = torch.from_numpy(rng.standard_normal((64, 1024), dtype=np.float32))
    wr = torch.from_numpy(rng.standard_normal(1024, dtype=np.float32))
    y_ours = compiled_rms(xr, wr)
    y_ref = (wr * ((xr.float() * torch.rsqrt(xr.float().pow(2).mean(-1, keepdim=True) + 1e-6)).to(xr.dtype)))
    check("RMSNorm 编译路径 == eager 参考表达式", bool(torch.allclose(y_ours, y_ref, atol=1e-6)))

    passed = sum(1 for ok, _ in RESULT_LINES if ok)
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "smoke-results-cpu.txt").write_text(
        "\n".join(f"{'PASS' if ok else 'FAIL'} | {line}" for ok, line in RESULT_LINES) + "\n",
        encoding="utf-8",
    )
    print(f"smoke: {passed}/{len(RESULT_LINES)} PASS")
    if passed != len(RESULT_LINES):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
