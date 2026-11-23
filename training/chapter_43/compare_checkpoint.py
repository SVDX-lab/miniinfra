#!/usr/bin/env python3
"""Chapter 43 continuity verification: resume must reproduce the reference.

Orchestrates the chapter-43 worker through subprocesses of torchrun:

- Default (no ``--resume-from``): Phase 1 trains ``--steps`` steps and saves
  checkpoints at step ``--save-at`` and at the final step, in every mode of
  ``--save-modes`` (centralized / sharded / dcp).  Phase 2 restores from the
  step-``--save-at`` checkpoint of each mode in a FRESH worker set and
  continues to ``--steps``.  The resumed trajectory must stitch onto the
  reference: identical per-step global loss, LR, next-batch preview, RNG
  fingerprint, and final parameter checksum.
- ``--resume-from DIR --reference PATH``: only the restore phase, under the
  launcher's world size — this is the RESHARDING case (save on 4 ranks,
  resume on 2).  Per-rank records are NOT comparable (the data sharding
  changes with world size); the GLOBAL loss trajectory, LRs and previews
  are, within resharding bars.

The reference and resumed runs execute the identical deterministic plan, so
same-world-size differences must be exactly zero (bitwise); the bars exist
for the resharding case, where only the FP32 reduction ORDER changes.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from experiment_utils import environment_record, write_json


LOSS_ABS_BAR = 1e-7
LOSS_RESHARD_REL_BAR = 1e-6
CHECKSUM_RESHARD_REL_BAR = 1e-5


def check_rng_state_machine() -> None:
    """capture -> perturb -> restore must return the generator exactly."""

    torch.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)
    from checkpointing import capture_rng, restore_rng, rng_fingerprint

    before = rng_fingerprint()
    state = capture_rng()
    torch.rand(8)
    random.random()
    np.random.random()
    middle = rng_fingerprint()
    restore_rng(state)
    after = rng_fingerprint()
    assert before != middle, "perturbation did not advance the RNG"
    assert before == after, f"RNG restore mismatch: {before} != {after}"
    # numpy round-trip
    np_before = np.random.get_state()[1].copy()
    restore_rng(capture_rng())
    assert np.array_equal(np_before, np.random.get_state()[1]), "numpy RNG mismatch"


def torchrun(nproc: int, worker_args: list[str], env_extra: dict | None = None) -> subprocess.CompletedProcess:
    from ddp_core import free_port

    command = [
        sys.executable, "-m", "torch.distributed.run", f"--nproc-per-node={nproc}",
        # 默认 29500 在共享服务器上可能被遗留进程占用；显式挑一个空闲端口。
        "--master-port", str(free_port()),
        "train_worker.py", *worker_args,
    ]
    env = dict(os.environ)
    env.setdefault("LOGLEVEL", "ERROR")
    if env_extra:
        env.update(env_extra)
    return subprocess.run(command, env=env, capture_output=True, text=True)


def load_records(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def trajectory_of(records: dict) -> list[dict]:
    """Global per-step trajectory (rank 0 carries the gathered fields)."""

    return sorted(
        ({"step": step["step"], "loss": step["loss"], "learning_rate": step["learning_rate"],
          "rng_fingerprint": step["rng_fingerprint"], "next_batch_preview": step["next_batch_preview"]}
         for step in records["per_rank"][0]["steps"]),
        key=lambda item: item["step"],
    )


def compare(reference: dict, resumed: dict, resharded: bool) -> dict:
    reference_trajectory = trajectory_of(reference)
    resumed_trajectory = trajectory_of(resumed)
    overlap = [step["step"] for step in resumed_trajectory]
    assert overlap, "resumed run has no steps"
    reference_by_step = {step["step"]: step for step in reference_trajectory}

    worst_loss = 0.0
    for step in resumed_trajectory:
        expected = reference_by_step[step["step"]]
        error = abs(step["loss"] - expected["loss"])
        if resharded:
            error = error / max(abs(expected["loss"]), 1e-8)
        worst_loss = max(worst_loss, error)
        assert step["learning_rate"] == expected["learning_rate"], (step, expected)
        assert step["next_batch_preview"] == expected["next_batch_preview"], (step, expected)
        if not resharded:
            assert step["rng_fingerprint"] == expected["rng_fingerprint"], (step, expected)

    ref_checksum = reference["final_model_checksum"][0]
    res_checksum = resumed["final_model_checksum"][0]
    checksum_error = abs(ref_checksum - res_checksum) / max(abs(ref_checksum), 1e-12)
    loss_ok = worst_loss < (LOSS_RESHARD_REL_BAR if resharded else LOSS_ABS_BAR)
    checksum_ok = checksum_error <= (CHECKSUM_RESHARD_REL_BAR if resharded else 0.0)
    return {
        "steps_compared": overlap,
        "worst_loss_error": worst_loss,
        "loss_bar": LOSS_RESHARD_REL_BAR if resharded else LOSS_ABS_BAR,
        "checksum_relative_error": checksum_error,
        "checksum_bar": CHECKSUM_RESHARD_REL_BAR if resharded else 0.0,
        "progress_match": resumed["final_progress"] == reference["final_progress"],
        "rng_compared": not resharded,
        "pass": bool(loss_ok and checksum_ok and resumed["final_progress"] == reference["final_progress"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--save-at", type=int, default=3, help="checkpoint step the resume starts from")
    parser.add_argument("--save-modes", default="centralized,sharded,dcp")
    parser.add_argument("--global-batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--real-weights", action="store_true", help="Qwen3-0.6B on GPU instead of the tiny model")
    parser.add_argument("--data", type=Path, default=Path("data/chapter43_train.jsonl"))
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--resume-from", type=Path, default=None, help="restore-only phase (resharding case)")
    parser.add_argument("--resume-mode", default="auto", choices=("auto", "centralized", "sharded", "dcp"))
    parser.add_argument("--reference", type=Path, default=None, help="reference records for --resume-from")
    parser.add_argument("--workdir", type=Path, default=Path("checkpoint-continuity-run"))
    parser.add_argument("--keep", action="store_true", help="keep the working directory")
    parser.add_argument("--output", type=Path, default=Path("checkpoint-continuity.local.json"))
    args = parser.parse_args()

    check_rng_state_machine()
    print("RNG state machine round-trip passed")

    tiny = not args.real_weights
    data_flags = ["--tiny"] if tiny else ["--data", str(args.data), "--max-length", str(args.max_length)]
    common = ["--global-batch", str(args.global_batch), "--seed", str(args.seed), "--steps", str(args.steps)]

    if args.resume_from:
        assert args.reference is not None, "--resume-from needs --reference"
        args.workdir.mkdir(parents=True, exist_ok=True)
        output = args.workdir / "resume-records.local.json"
        completed = torchrun(args.nproc_per_node, [
            *data_flags, *common,
            "--resume-dir", str(args.resume_from), "--resume-mode", args.resume_mode,
            "--output", str(output),
        ])
        if completed.returncode != 0:
            print(completed.stdout, completed.stderr)
            raise SystemExit("resume run failed")
        report = compare(load_records(args.reference), load_records(output), resharded=True)
        payload = environment_record(world_size=args.nproc_per_node, dp_degree=args.nproc_per_node)
        payload.update({
            "scenario": "reshard-resume",
            "resume_from": str(args.resume_from),
            "resume_mode": args.resume_mode,
            "saved_world_size": load_records(args.reference)["world_size"],
            "resume_world_size": args.nproc_per_node,
            "checks": report,
            "pass": report["pass"],
        })
        write_json(args.output, payload)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["pass"]:
            raise SystemExit("continuity checks failed")
        return

    if args.save_at >= args.steps:
        raise SystemExit("--save-at must be < --steps")
    args.workdir.mkdir(parents=True, exist_ok=True)
    modes = [mode for mode in args.save_modes.split(",") if mode]

    reference_path = args.workdir / "reference-records.local.json"
    completed = torchrun(args.nproc_per_node, [
        *data_flags, *common,
        "--save-dir", str(args.workdir / "checkpoints"),
        "--save-every", str(args.save_at),
        "--save-modes", ",".join(modes),
        "--output", str(reference_path),
    ])
    if completed.returncode != 0:
        print(completed.stdout, completed.stderr)
        raise SystemExit("reference run failed")
    reference = load_records(reference_path)

    reports = {}
    for mode in modes:
        checkpoint_dir = args.workdir / "checkpoints" / f"step-{args.save_at:05d}" / mode
        output = args.workdir / f"resume-{mode}-records.local.json"
        completed = torchrun(args.nproc_per_node, [
            *data_flags, *common,
            "--resume-dir", str(checkpoint_dir), "--resume-mode", mode,
            "--output", str(output),
        ])
        if completed.returncode != 0:
            print(completed.stdout, completed.stderr)
            raise SystemExit(f"resume run ({mode}) failed")
        reports[mode] = compare(reference, load_records(output), resharded=False)

    payload = environment_record(world_size=args.nproc_per_node, dp_degree=args.nproc_per_node)
    payload.update({
        "scenario": "same-world-size-resume",
        "world_size": args.nproc_per_node,
        "steps": args.steps,
        "save_at": args.save_at,
        "reference_records": str(reference_path),
        "modes": {mode: {"checks": report, "pass": report["pass"]} for mode, report in reports.items()},
        "pass": all(report["pass"] for report in reports.values()),
    })
    write_json(args.output, payload)
    print(json.dumps({mode: report["pass"] for mode, report in reports.items()}, ensure_ascii=False))
    print(f"reference records kept at {reference_path} (use with --resume-from for the resharding case)")
    if not payload["pass"]:
        raise SystemExit("continuity checks failed")
    if not args.keep:
        shutil.rmtree(args.workdir / "checkpoints", ignore_errors=True)


if __name__ == "__main__":
    main()
