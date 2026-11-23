"""Chapter 28 core: compiling the training step and measuring what it costs.

``compile_model`` wraps a model with ``torch.compile``.  The loss, gradient
clip and optimizer stay outside the compiled region, so the measured variable
is exactly the execution strategy of forward+backward.

The chapter's numbers come from three measurement helpers:

- ``timed_first_step`` returns the wall-clock split between the first call
  (Dynamo tracing + Inductor codegen + autotuning) and the steady state.
- ``dynamo_stats`` reads ``torch._dynamo.utils.counters`` to report graph
  counts, recompiles and graph-break reasons after a run.
- ``explain_graph_breaks`` uses ``torch._dynamo.explain`` - tracing only, no
  compilation - to enumerate where a non-compiled-friendly model would break.

Recompilation is the chapter's failure mode: every new batch shape is a new
graph.  ``RecompileTracker`` watches the counters while a caller feeds
variable-length batches so the recompile storm becomes a measured number
instead of an anecdote.
"""

from __future__ import annotations

import time
from typing import Any

import torch


def compile_model(model: torch.nn.Module, dynamic: bool = False) -> torch.nn.Module:
    """torch.compile with Inductor defaults; dynamic only for the stress arm."""

    return torch.compile(model, dynamic=dynamic)


def timed_first_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step_fn,
    batch,
) -> dict[str, Any]:
    """Run the first (compiling) step and one steady-state step.

    ``step_fn(model, batch)`` must run one full training step and return its
    metrics dict.  The returned split separates compile time from steady
    state; the caller decides whether the first step belongs in the average.
    """

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    first_start = time.perf_counter()
    first_metrics = step_fn(model, batch)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    first_ms = (time.perf_counter() - first_start) * 1000

    steady_start = time.perf_counter()
    step_fn(model, batch)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    steady_ms = (time.perf_counter() - steady_start) * 1000
    return {"first_step_ms": first_ms, "steady_step_ms": steady_ms, "first_metrics": first_metrics}


def dynamo_stats() -> dict[str, Any]:
    """Snapshot the dynamo counters that matter for this chapter."""

    from torch._dynamo.utils import counters

    stats = dict(counters["stats"])
    graph_breaks = dict(counters["graph_break"])
    return {
        "unique_graphs": int(stats.get("unique_graphs", 0)),
        "recompiles": int(stats.get("recompiles", 0)),
        "graph_breaks": {key: int(value) for key, value in graph_breaks.items()},
    }


def reset_dynamo_stats() -> None:
    from torch._dynamo.utils import counters

    counters.clear()


def explain_graph_breaks(model: torch.nn.Module, example_inputs: dict) -> dict[str, Any]:
    """Trace-only explanation of graph breaks; no compilation happens."""

    from torch._dynamo import explain

    explanation = explain(model)(example_inputs["input_ids"], attention_mask=example_inputs["attention_mask"])
    return {
        "graph_count": explanation.graph_count,
        "graph_break_count": explanation.graph_break_count,
        "break_reasons": [
            str(reason)[:200] for reason in explanation.break_reasons[:10]
        ],
    }


class RecompileTracker:
    """Diff the dynamo counters around a block of training steps."""

    def __enter__(self):
        reset_dynamo_stats()
        self._start = dynamo_stats()
        return self

    def __exit__(self, *_exc):
        self.end = dynamo_stats()
        self.compiled_graphs = self.end["unique_graphs"] - self._start["unique_graphs"]
        self.recompiles = self.end["recompiles"] - self._start["recompiles"]
        self.graph_breaks = self.end["graph_breaks"]
        return False
