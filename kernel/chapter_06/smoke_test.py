#!/usr/bin/env python3
"""第 06 期冒烟测试：stride 语义、三策略边界行为、错误输入行为快照（CPU 本地全跑）。

边界集（合同 error_behavior 条目）：
  - stride 公式 vs numpy as_strided 逐元素一致；
  - 手算例（2×4，第 02 期同款）四种布局全对；
  - 策略 A：非连续输入显式报错（报错全文快照存 results/error-snapshot-cpu.json）；
  - 策略 B/C：转置视图、行切片视图数值正确；
  - expand（stride=0）广播视图策略 C 正确；
  - H=0 显式报错；rows=0 返回空；
  - torch CPU stride/.contiguous() 行为探针（transpose-back 陷阱收录）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "implementations" / "cpu"))

from stride_layout import (  # noqa: E402
    RMS_NORM_EPS,
    StridedTensor,
    layout_demonstration,
    naive_rmsnorm_strided,
    policy_a_reject,
    policy_b_autocopy,
    policy_c_strided,
    torch_stride_probe,
    verify_stride_semantics,
)

RESULTS = HERE / "results"
CHECKS: list[dict] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    CHECKS.append({"name": name, "ok": bool(condition), "detail": detail})
    status = "PASS" if condition else "FAIL"
    print(f"[smoke] {status}  {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        raise SystemExit(f"smoke_test 失败：{name} {detail}")


def hand_computed_example() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """第 02 期同款手算例：x=[[1,2,3,4],[-1,0,1,-2]]，w=1，eps=1e-6。"""
    x = np.array([[1.0, 2.0, 3.0, 4.0], [-1.0, 0.0, 1.0, -2.0]], dtype=np.float32)
    weight = np.ones(4, dtype=np.float32)
    expected = np.array(
        [
            [0.365148, 0.730297, 1.095445, 1.460593],
            [-0.816497, 0.0, 0.816497, -1.632993],
        ],
        dtype=np.float32,
    )
    return x, weight, expected


def main() -> int:
    RESULTS.mkdir(exist_ok=True)

    # 1. stride 公式 vs numpy 官方语义。
    report = verify_stride_semantics()
    check("offset_formula_vs_numpy_as_strided", report["all_equal"],
          json.dumps(report["layouts"], ensure_ascii=False))
    check("manual_offset_row_major", report["manual_offsets"] == {"(0,0)": 0, "(1,0)": 7, "(2,3)": 17},
          str(report["manual_offsets"]))
    check("manual_offset_col_major", report["manual_offsets_colmajor"] == {"(0,0)": 0, "(1,0)": 1, "(2,3)": 17},
          str(report["manual_offsets_colmajor"]))

    # 2. 手算例：四种布局 × 三策略（策略 A 只接受行主序）。
    x, w, expected = hand_computed_example()
    layouts = {
        "row_major": StridedTensor.row_major(x),
        "column_major": StridedTensor.column_major(x),
        "transposed_view": StridedTensor.transposed_view(x),
        "row_sliced_view": StridedTensor.row_sliced_view(x, pad=3),
    }
    for name, st in layouts.items():
        y = naive_rmsnorm_strided(st, w)
        err = float(np.abs(y.buffer.reshape(x.shape) - expected).max())
        check(f"hand_example_strided/{name}", err <= 1e-6, f"max_abs={err:.2e}")
        y_b = policy_b_autocopy(st, w)
        err_b = float(np.abs(y_b.buffer.reshape(x.shape) - expected).max())
        check(f"hand_example_policy_b/{name}", err_b <= 1e-6, f"max_abs={err_b:.2e}")
        y_c = policy_c_strided(st, w)
        err_c = float(np.abs(y_c.buffer.reshape(x.shape) - expected).max())
        check(f"hand_example_policy_c/{name}", err_c <= 1e-6, f"max_abs={err_c:.2e}")
    check("hand_example_policy_a_row_major_ok",
          policy_a_reject(layouts["row_major"], w).buffer.reshape(x.shape).max() > 0)

    # 3. 策略 A 拒绝非连续输入（错误快照）。
    error_snapshots = {}
    for name in ("transposed_view", "row_sliced_view"):
        try:
            policy_a_reject(layouts[name], w)
            check(f"policy_a_reject/{name}", False, "未按合同报错")
        except ValueError as error:
            error_snapshots[f"cpu_policy_a_{name}"] = {
                "exception": type(error).__name__,
                "message": str(error),
            }
            check(f"policy_a_reject/{name}", "is_contiguous=False" in str(error),
                  str(error)[:80])

    # 4. H=0 与 rows=0 边界。
    try:
        naive_rmsnorm_strided(StridedTensor.row_major(np.zeros((2, 0), dtype=np.float32)),
                              np.zeros(0, dtype=np.float32))
        check("H0_reject", False, "H=0 未按合同报错")
    except ValueError as error:
        check("H0_reject", "H=0" in str(error), str(error)[:60])
    y_empty = naive_rmsnorm_strided(StridedTensor.row_major(np.zeros((0, 8), dtype=np.float32)),
                                    np.zeros(8, dtype=np.float32))
    check("rows0_empty", y_empty.buffer.size == 0)

    # 5. expand（stride=0）广播视图：策略 C 支持且正确；策略 A/B 的行为记录。
    base = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    ones = np.ones((1, 4), dtype=np.float32)
    y_base = naive_rmsnorm_strided(StridedTensor.row_major(base), ones[0])
    broadcast = StridedTensor(np.tile(base, 4).reshape(-1)[:4], (4, 4), (0, 1))
    # 广播视图的 buffer 只需覆盖一行：物理 offset = 0*stride0 + j*stride1。
    y_broadcast = naive_rmsnorm_strided(broadcast, ones[0])
    expanded_ok = all(
        float(np.abs(y_broadcast.buffer.reshape(4, 4)[i]
                     - y_base.buffer.reshape(1, 4)[0]).max()) < 1e-6
        for i in range(4))
    check("expand_stride0_policy_c", expanded_ok,
          f"stride={broadcast.strides}（stride_row=0：所有行读同一物理行）")

    # 6. torch CPU stride/.contiguous() 探针（三平台同脚本对照的 CPU 列）。
    try:
        probe = torch_stride_probe()
        check("torch_tv_is_not_contiguous", probe["transposed_view"]["is_contiguous"] is False,
              f"stride={probe['transposed_view']['stride']}")
        check("torch_tv_values_equal", probe["transposed_view"]["same_values_as_x"] is True)
        check("torch_transpose_back_trap",
              probe["transpose_back_trap"]["xt_is_contiguous"] is True
              and probe["transpose_back_trap"]["xt_contiguous_t_stride"]
              == probe["transposed_view"]["stride"],
              "tv.t() 连续 => .contiguous() 零拷贝返回 self；.t() 回来 stride 不变"
              "（陷阱：transpose-back 不改变 [rows,H] 的物理布局）")
        check("torch_contiguous_normalizes",
              probe["contiguous_results"]["tv_contig_is_contiguous"] is True
              and probe["contiguous_results"]["tv_contig_equal_x"] is True)
        error_snapshots["torch_probe"] = probe
    except ImportError:
        check("torch_stride_probe", True, "本机无 torch，探针在服务器侧执行")

    (RESULTS / "error-snapshot-cpu.json").write_text(
        json.dumps({"schema_version": "kernel-chapter07.error-snapshot.v1",
                    "backend": "cpu",
                    "errors": error_snapshots,
                    "checks": CHECKS}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"[smoke] 全部 {len(CHECKS)} 项通过；错误快照与探针已存 results/error-snapshot-cpu.json")

    # 7. 布局演示表（打印供 EXPERIMENT_LOG 引用）。
    demo = layout_demonstration(rows=3, hidden=5)
    for name, info in demo.items():
        print(f"[smoke] 布局演示 {name}: strides={info['strides']} "
              f"offsets(0,0)={info['offsets']['(0,0)']} (1,0)={info['offsets']['(1,0)']} "
              f"(2,4)={info['offsets']['(2,4)']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
