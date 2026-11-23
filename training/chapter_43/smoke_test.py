#!/usr/bin/env python3
"""CPU-only Chapter 43 checks: both save modes, resharding, RNG state machine.

All rounds run on CPU gloo with the tiny synthetic model:

- Unit checks (single process): RNG capture/perturb/restore round-trip for
  python/NumPy/torch, and a safetensors export round-trip of the tiny model.
- Round A ``continuity`` (2 ranks): train 2 steps, save at step 2 in ALL
  modes (centralized / sharded / dcp), continue 2 steps recording the
  reference trajectory; then rebuild FRESH model + optimizer, restore from
  each mode, and redo the last 2 steps.  The restored runs must reproduce
  the reference bitwise (loss, LR, RNG fingerprint, final checksum) on the
  deterministic CPU path.
- Round B ``reshard`` (save on 4 ranks, load on 2): builds the checkpoint
  with a 4-rank gloo group (one training step), then a 2-rank group loads
  both sharded and centralized checkpoints.  Restored parameters must equal
  the recorded checksums exactly (replica selection), with the DDP-progress
  and sampler fields intact — this is the 4-save -> 2-load resharding case;
  the symmetric 4-load direction follows from interchangeability.
"""

from __future__ import annotations

import io
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from checkpointing import (
    capture_rng,
    dcp_available,
    dcp_load_checkpoint,
    dcp_save_checkpoint,
    load_centralized,
    load_sharded,
    model_checksum,
    read_manifest,
    restore_rng,
    rng_fingerprint,
    save_centralized,
    save_sharded,
    sync_scheduler_lr,
)
from ddp_core import build_global_batch, ddp_train_step
from qwen3_model import Qwen3ForCausalLM
from train_worker import (
    TINY_CONFIG,
    build_batch_plan,
    build_optimizer_and_scheduler,
    synthetic_samples,
)


SEED = 11
GLOBAL_BATCH = 4
TOTAL_STEPS = 4
SAVE_AT = 2
STASH = Path("chapter43-smoke-stash.pt")


def check_rng_state_machine() -> None:
    torch.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)
    before = rng_fingerprint()
    state = capture_rng()
    torch.rand(8)
    random.random()
    np.random.random()
    assert before != rng_fingerprint(), "perturbation did not advance the RNG"
    restore_rng(state)
    assert before == rng_fingerprint(), "torch RNG restore mismatch"
    numpy_before = np.random.get_state()[1].copy()
    restore_rng(capture_rng())
    assert np.array_equal(numpy_before, np.random.get_state()[1]), "numpy RNG restore mismatch"
    python_before = random.getstate()
    restore_rng(capture_rng())
    assert python_before == random.getstate(), "python RNG restore mismatch"

    # The saved form must restore too: every distributed mode hands restore_rng
    # a torch.load'ed state (numpy stays a 5-tuple, python a 3-tuple).
    buffer = io.BytesIO()
    torch.save(capture_rng(), buffer)
    buffer.seek(0)
    torch.rand(8)
    random.random()
    np.random.random()
    restore_rng(torch.load(buffer, map_location="cpu", weights_only=False))
    assert before == rng_fingerprint(), "saved torch RNG state restore mismatch"
    assert python_before == random.getstate(), "saved python RNG state restore mismatch"
    assert np.array_equal(numpy_before, np.random.get_state()[1]), "saved numpy RNG state restore mismatch"

    # A JSON-style round trip de-tuples both states into plain lists; the
    # restorer must normalize them back (setstate/set_state reject raw lists).
    state = capture_rng()
    version, vector, gauss = state["python"]
    state["python"] = [version, [int(k) for k in vector], gauss]
    kind, keys, pos, has_gauss, cached = state["numpy"]
    state["numpy"] = [str(kind), [int(k) for k in keys], int(pos), int(has_gauss), float(cached)]
    torch.rand(8)
    random.random()
    np.random.random()
    restore_rng(state)
    assert before == rng_fingerprint(), "list-form torch RNG state restore mismatch"
    assert python_before == random.getstate(), "list-form python RNG state restore mismatch"
    assert np.array_equal(numpy_before, np.random.get_state()[1]), "list-form numpy RNG state restore mismatch"


def check_export_roundtrip(workdir: Path) -> None:
    from checkpointing import export_inference_weights
    from safetensors.torch import load_file

    torch.manual_seed(SEED)
    model = Qwen3ForCausalLM(TINY_CONFIG)
    target = workdir / "export" / "tiny.safetensors"
    metrics = export_inference_weights(model, target, {"model_id": "tiny-synthetic"})
    reloaded = load_file(str(target))
    expected = dict(model.state_dict())
    assert set(reloaded) == set(expected), "exported keys differ"
    for key, value in expected.items():
        assert torch.equal(reloaded[key], value), key
    assert metrics["bytes"] > 0


def _setup(rank: int, world_size: int, port: int) -> torch.device:
    if "RANK" not in os.environ:
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    return torch.device("cpu")


def _build_train_state(rank: int, world_size: int, device: torch.device, seed: int):
    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(TINY_CONFIG).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, learning_rate=1e-3,
                                                         warmup_steps=2, total_steps=TOTAL_STEPS)
    return model, optimizer, scheduler


def _train_steps(model, optimizer, scheduler, plan, samples, start: int, end: int,
                 rank: int, world_size: int) -> list[dict]:
    records = []
    for step in range(start, end):
        members = plan[step]
        batch, accounting = build_global_batch(samples, members, rank, world_size, pad_token_id=0)
        metrics = ddp_train_step(model, optimizer, batch, accounting["loss_scale"],
                                 torch.device("cpu"), 1.0, world_size, rank, collect_telemetry=False)
        scheduler.step()
        records.append({
            "step": step,
            "loss": metrics["global_loss"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "rng_fingerprint": rng_fingerprint(),
        })
    return records


def _worker_continuity(rank: int, world_size: int, port: int, workdir: str) -> None:
    device = _setup(rank, world_size, port)
    try:
        assert dcp_available(), "torch.distributed.checkpoint must exist under the pinned torch"
        save_root = Path(workdir) / "step-00002"
        samples = synthetic_samples()
        plan = build_batch_plan(len(samples), GLOBAL_BATCH, SEED)
        topology = {"world_size": world_size, "dp_size": world_size, "tp_size": 1, "pp_size": 1,
                    "backend": "gloo", "rank": rank}
        config = {"model_id": "tiny-synthetic"}

        model, optimizer, scheduler = _build_train_state(rank, world_size, device, SEED)
        _train_steps(model, optimizer, scheduler, plan, samples, 0, SAVE_AT, rank, world_size)
        common = (model, optimizer, scheduler,
                  {"global_step": SAVE_AT, "consumed_supervised_tokens": 0.0, "consumed_samples": 8},
                  {"plan_seed": SEED, "cursor": SAVE_AT}, config, topology)
        save_centralized(save_root / "centralized", *common, rank=rank, world_size=world_size)
        save_sharded(save_root / "sharded", *common, rank=rank, world_size=world_size)
        dcp_save_checkpoint(save_root / "dcp", model, optimizer)
        # DCP stores only tensors; write the same worker-meta sidecar the
        # chapter worker uses (progress/sampler/scheduler/per-rank RNG).
        gathered_rng = [None] * world_size
        dist.all_gather_object(gathered_rng, capture_rng())
        if rank == 0:
            torch.save({
                "progress": dict(common[3]),
                "sampler": dict(common[4]),
                "scheduler": scheduler.state_dict(),
                "rng_per_rank": gathered_rng,
                "topology": dict(topology),
            }, save_root / "dcp" / "worker-meta.pt")
        dist.barrier()

        reference = _train_steps(model, optimizer, scheduler, plan, samples, SAVE_AT, TOTAL_STEPS,
                                 rank, world_size)
        reference_checksum = model_checksum(model)

        for mode in ("centralized", "sharded", "dcp"):
            torch.manual_seed(999)  # diverge the RNG on purpose: the restore must overwrite it
            fresh_model, fresh_optimizer, fresh_scheduler = _build_train_state(
                rank, world_size, device, SEED + 500)  # wrong weights on purpose
            if mode == "centralized":
                load_centralized(save_root / "centralized" / "checkpoint.pt", fresh_model,
                                 fresh_optimizer, fresh_scheduler, device, rank)
            elif mode == "sharded":
                load_sharded(save_root / "sharded", fresh_model, fresh_optimizer,
                             fresh_scheduler, device, rank, world_size)
            else:
                dcp_load_checkpoint(save_root / "dcp", fresh_model, fresh_optimizer)
                payload = torch.load(save_root / "dcp" / "worker-meta.pt", map_location="cpu",
                                     weights_only=False)
                scheduler_state = payload["scheduler"]
                fresh_scheduler.load_state_dict(scheduler_state)
                sync_scheduler_lr(fresh_optimizer, fresh_scheduler)
                restore_rng(payload["rng_per_rank"][rank % len(payload["rng_per_rank"])])
            resumed = _train_steps(fresh_model, fresh_optimizer, fresh_scheduler, plan, samples,
                                   SAVE_AT, TOTAL_STEPS, rank, world_size)
            for expected, actual in zip(reference, resumed):
                assert actual["loss"] == expected["loss"], (mode, expected, actual)
                assert actual["learning_rate"] == expected["learning_rate"], (mode, expected, actual)
                assert actual["rng_fingerprint"] == expected["rng_fingerprint"], (mode, expected, actual)
            assert model_checksum(fresh_model) == reference_checksum, mode
        if rank == 0:
            print(f"round A (continuity, {world_size} ranks): centralized/sharded/dcp resume bitwise OK")
    finally:
        dist.barrier()
        dist.destroy_process_group()


def _worker_save4(rank: int, world_size: int, port: int, workdir: str) -> None:
    device = _setup(rank, world_size, port)
    try:
        samples = synthetic_samples()
        plan = build_batch_plan(len(samples), GLOBAL_BATCH, SEED)
        model, optimizer, scheduler = _build_train_state(rank, world_size, device, SEED)
        _train_steps(model, optimizer, scheduler, plan, samples, 0, 1, rank, world_size)
        topology = {"world_size": world_size, "dp_size": world_size, "tp_size": 1, "pp_size": 1,
                    "backend": "gloo", "rank": rank}
        common = (model, optimizer, scheduler,
                  {"global_step": 1, "consumed_supervised_tokens": 0.0, "consumed_samples": 4},
                  {"plan_seed": SEED, "cursor": 1}, {"model_id": "tiny-synthetic"}, topology)
        save_root = Path(workdir) / "w4-step-00001"
        save_centralized(save_root / "centralized", *common, rank=rank, world_size=world_size)
        save_sharded(save_root / "sharded", *common, rank=rank, world_size=world_size)
        if rank == 0:
            manifest = read_manifest(save_root / "sharded")
            torch.save({
                "model_checksum": manifest["shards"][0]["model_checksum"],
                "progress": manifest["progress"],
                "sampler": manifest["sampler"],
            }, STASH)
    finally:
        dist.barrier()
        dist.destroy_process_group()


def _worker_load2(rank: int, world_size: int, port: int, workdir: str) -> None:
    device = _setup(rank, world_size, port)
    try:
        stash = torch.load(STASH, map_location="cpu", weights_only=False)
        save_root = Path(workdir) / "w4-step-00001"
        # Build with a DIFFERENT seed: only a successful restore can match.
        model, optimizer, scheduler = _build_train_state(rank, world_size, device, SEED + 1234)
        payload = load_sharded(save_root / "sharded", model, optimizer, scheduler, device,
                               rank, world_size)
        assert payload["resharded"] and payload["saved_world_size"] == 4, payload
        assert model_checksum(model) == stash["model_checksum"], "sharded reshard checksum mismatch"
        assert payload["progress"] == stash["progress"], payload
        assert payload["sampler"] == stash["sampler"], payload

        model2, optimizer2, scheduler2 = _build_train_state(rank, world_size, device, SEED + 5678)
        load_centralized(save_root / "centralized" / "checkpoint.pt", model2, optimizer2,
                         scheduler2, device, rank)
        assert model_checksum(model2) == stash["model_checksum"], "centralized reshard checksum mismatch"
        if rank == 0:
            print("round B (reshard): 4-rank save -> 2-rank load OK (sharded + centralized)")
    finally:
        dist.barrier()
        dist.destroy_process_group()


def main() -> None:
    import socket

    workdir = Path("chapter43-smoke-run")
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)
    check_rng_state_machine()
    check_export_roundtrip(workdir)
    print("unit checks passed: RNG state machine, safetensors export round-trip")

    def free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            return int(probe.getsockname()[1])

    port = free_port()
    mp.spawn(_worker_continuity, args=(2, port, str(workdir)), nprocs=2, join=True)
    mp.spawn(_worker_save4, args=(4, free_port(), str(workdir)), nprocs=4, join=True)
    mp.spawn(_worker_load2, args=(2, free_port(), str(workdir)), nprocs=2, join=True)
    STASH.unlink(missing_ok=True)
    shutil.rmtree(workdir, ignore_errors=True)
    print("chapter 28 smoke passed: save modes, resume continuity, resharding, RNG state machine")


if __name__ == "__main__":
    main()
