#!/usr/bin/env python3
"""Chapter 43 fault injection: kill one rank, resume from the last checkpoint.

End-to-end scenario over subprocess torchrun launches of ``train_worker.py``:

1. REFERENCE run: ``--steps`` steps, no faults — the trajectory any recovery
   must reproduce.
2. FAULTY run: saves a checkpoint every ``--save-every`` steps; the rank
   ``--fault-rank`` hard-exits (``os._exit(70)``, no distributed cleanup)
   right after the step ``--fault-at-step`` checkpoint is durable.  torchrun
   reports the failure and tears down the remaining ranks; the fault script
   ASSERTS the launch actually failed (a surviving launch would mean the
   kill did not happen).
3. RESUME run: a fresh launch with ``--resume-dir`` pointing at the killed
   run's checkpoint; it restores the full state and finishes the remaining
   steps.

Continuity: the resumed run's per-step global loss, LR and next-batch
preview must equal the reference on the steps after the kill, and the final
parameter checksums must agree (bitwise on the deterministic CPU path).
Defaults to the tiny model on CPU gloo (safe to run on any server);
``--real-weights`` reruns the identical scenario on GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from experiment_utils import environment_record, write_json

FAULT_EXIT_CODE = 70


def torchrun(nproc: int, worker_args: list[str]) -> subprocess.CompletedProcess:
    from ddp_core import free_port

    command = [
        sys.executable, "-m", "torch.distributed.run", f"--nproc-per-node={nproc}",
        # 默认 29500 在共享服务器上可能被遗留进程占用；显式挑一个空闲端口。
        "--master-port", str(free_port()),
        "train_worker.py", *worker_args,
    ]
    env = dict(os.environ)
    env.setdefault("LOGLEVEL", "ERROR")
    return subprocess.run(command, env=env, capture_output=True, text=True)


def load_records(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--fault-at-step", type=int, default=2, help="0-based step whose checkpoint precedes the kill")
    parser.add_argument("--fault-rank", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--save-modes", default="centralized,sharded")
    parser.add_argument("--resume-mode", default="sharded", choices=("centralized", "sharded", "dcp"))
    parser.add_argument("--global-batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--real-weights", action="store_true")
    parser.add_argument("--data", type=Path, default=Path("data/chapter43_train.jsonl"))
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--workdir", type=Path, default=Path("fault-inject-run"))
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("fault-results.local.json"))
    args = parser.parse_args()

    if not 0 <= args.fault_at_step < args.steps - 1:
        raise SystemExit("--fault-at-step must leave at least one step after the kill")
    args.workdir.mkdir(parents=True, exist_ok=True)
    tiny = not args.real_weights
    data_flags = ["--tiny"] if tiny else ["--data", str(args.data), "--max-length", str(args.max_length)]
    common = ["--global-batch", str(args.global_batch), "--seed", str(args.seed), "--steps", str(args.steps)]

    # ---- Phase 1: reference ----
    reference_path = args.workdir / "reference-records.local.json"
    completed = torchrun(args.nproc_per_node, [*data_flags, *common, "--output", str(reference_path)])
    if completed.returncode != 0:
        print(completed.stdout, completed.stderr)
        raise SystemExit("reference run failed")
    reference = load_records(reference_path)

    # ---- Phase 2: faulty run (rank dies after its checkpoint) ----
    save_dir = args.workdir / "checkpoints"
    completed = torchrun(args.nproc_per_node, [
        *data_flags, *common,
        "--save-dir", str(save_dir),
        "--save-every", str(args.save_every),
        "--save-modes", args.save_modes,
        "--fault-at-step", str(args.fault_at_step),
        "--fault-rank", str(args.fault_rank),
    ])
    if completed.returncode == 0:
        raise SystemExit("the faulty run finished cleanly — the kill did not happen")
    faulty_returncode = completed.returncode
    print(f"faulty run failed as intended (returncode={faulty_returncode})")

    # ---- Phase 3: resume from the last checkpoint before the kill ----
    checkpoint_root = save_dir / f"step-{args.fault_at_step + 1:05d}" / args.resume_mode
    resume_path = args.workdir / "resume-records.local.json"
    completed = torchrun(args.nproc_per_node, [
        *data_flags, *common,
        "--resume-dir", str(checkpoint_root), "--resume-mode", args.resume_mode,
        "--output", str(resume_path),
    ])
    if completed.returncode != 0:
        print(completed.stdout, completed.stderr)
        raise SystemExit("resume run failed")
    resumed = load_records(resume_path)

    # ---- continuity: steps after the kill must match the reference ----
    reference_by_step = {step["step"]: step for step in reference["per_rank"][0]["steps"]}
    resumed_steps = resumed["per_rank"][0]["steps"]
    assert resumed_steps[0]["step"] == args.fault_at_step + 1, resumed_steps[0]
    worst_loss = 0.0
    for step in resumed_steps:
        expected = reference_by_step[step["step"]]
        worst_loss = max(worst_loss, abs(step["loss"] - expected["loss"]))
        assert step["learning_rate"] == expected["learning_rate"], (step, expected)
        assert step["next_batch_preview"] == expected["next_batch_preview"], (step, expected)
        assert step["rng_fingerprint"] == expected["rng_fingerprint"], (step, expected)
    ref_checksum = reference["final_model_checksum"][0]
    res_checksum = resumed["final_model_checksum"][0]
    checksum_error = abs(ref_checksum - res_checksum) / max(abs(ref_checksum), 1e-12)
    checks = {
        "kill_happened": True,
        "resume_start_step": resumed_steps[0]["step"],
        "steps_compared": [step["step"] for step in resumed_steps],
        "worst_loss_absolute_error": worst_loss,
        "checksum_relative_error": checksum_error,
        "progress_match": resumed["final_progress"] == reference["final_progress"],
        "loss_bar": 1e-7,
        "pass": bool(worst_loss < 1e-7 and checksum_error == 0.0
                     and resumed["final_progress"] == reference["final_progress"]),
    }

    payload = environment_record(world_size=args.nproc_per_node, dp_degree=args.nproc_per_node)
    payload.update({
        "scenario": "fault-inject-resume",
        "fault_at_step": args.fault_at_step,
        "fault_rank": args.fault_rank,
        "fault_exit_code": FAULT_EXIT_CODE,
        "faulty_launch_returncode": faulty_returncode,
        "resume_mode": args.resume_mode,
        "checkpoint": str(checkpoint_root),
        "tiny": tiny,
        "checks": checks,
        "pass": checks["pass"],
    })
    write_json(args.output, payload)
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    if not checks["pass"]:
        raise SystemExit("fault-injection continuity checks failed")
    if not args.keep:
        shutil.rmtree(args.workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
