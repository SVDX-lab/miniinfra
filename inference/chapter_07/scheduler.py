"""第 07 期 Chunked Prefill 调度器。

两条路径使用相同 FCFS、Token Budget 和 Decode 保护规则：

- full：Prompt 必须在一个 Prefill iteration 完成，队头单请求可超软预算；
- chunked：Prompt 可跨 iteration 推进，每轮 padded Prefill 成本不超过硬预算。
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    token_ids: tuple
    max_new_tokens: int
    arrival_ms: float = 0.0

    def validate(self):
        if not self.request_id:
            raise ValueError("request_id 不能为空")
        if not self.token_ids:
            raise ValueError("Prompt Token 不能为空")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens 必须大于 0")
        if self.arrival_ms < 0:
            raise ValueError("arrival_ms 不能小于 0")


@dataclass
class RequestState:
    spec: RequestSpec
    status: str = "waiting"
    generated: list = field(default_factory=list)
    slot_index: int | None = None
    cache_length: int = 0
    prefill_cursor: int = 0
    admitted_ms: float | None = None
    prefill_started_ms: float | None = None
    prefill_completed_ms: float | None = None
    first_token_ms: float | None = None
    completion_ms: float | None = None
    token_times_ms: list = field(default_factory=list)


def make_request_specs(token_sequences, max_new_tokens, arrival_times_ms=None):
    sequences = [tuple(int(token) for token in sequence) for sequence in token_sequences]
    if isinstance(max_new_tokens, int):
        budgets = [max_new_tokens] * len(sequences)
    else:
        budgets = list(max_new_tokens)
    arrivals = (
        [0.0] * len(sequences)
        if arrival_times_ms is None else list(arrival_times_ms)
    )
    if len(budgets) != len(sequences) or len(arrivals) != len(sequences):
        raise ValueError("Prompt、输出预算和到达时间的数量必须一致")
    specs = [
        RequestSpec(str(index), sequence, int(budget), float(arrival))
        for index, (sequence, budget, arrival) in enumerate(
            zip(sequences, budgets, arrivals)
        )
    ]
    for spec in specs:
        spec.validate()
    if len({spec.request_id for spec in specs}) != len(specs):
        raise ValueError("request_id 必须唯一")
    return specs


@dataclass(frozen=True)
class SchedulerConfig:
    mode: str
    max_running_requests: int
    token_budget: int

    def validate(self):
        if self.mode not in ("full", "chunked"):
            raise ValueError("mode 必须是 full 或 chunked")
        if self.max_running_requests < 1:
            raise ValueError("max_running_requests 必须大于 0")
        if self.token_budget < self.max_running_requests:
            raise ValueError(
                "token_budget 不能小于 max_running_requests，"
                "否则无法为每个运行名额分配至少一个 Token"
            )


@dataclass(frozen=True)
class PrefillPlan:
    state: RequestState
    start: int
    end: int

    @property
    def token_count(self):
        return self.end - self.start


@dataclass(frozen=True)
class SchedulerOutput:
    phase: str
    selected: tuple
    prefill_plans: tuple = ()
    scheduled_tokens: int = 0
    logical_tokens: int = 0
    token_budget: int | None = None
    oversize_singleton: bool = False

    @property
    def request_ids(self):
        return [state.spec.request_id for state in self.selected]


class ChunkedPrefillScheduler:
    """在每个 iteration 选择一次完整/分块 Prefill 或一次 Decode。"""

    def __init__(self, config):
        config.validate()
        self.config = config
        self.last_phase = None
        self.iteration_count = 0

    @staticmethod
    def _padded_cost(lengths):
        return len(lengths) * max(lengths) if lengths else 0

    def _full_plans(self, ready, free_slots):
        selected = []
        oversize = False
        for state in ready[:free_slots]:
            candidate = selected + [state]
            lengths = [len(item.spec.token_ids) for item in candidate]
            if self._padded_cost(lengths) <= self.config.token_budget:
                selected.append(state)
                continue
            if not selected:
                selected.append(state)
                oversize = True
            break
        plans = [
            PrefillPlan(state, 0, len(state.spec.token_ids))
            for state in selected
        ]
        return plans, oversize

    def _chunked_plans(self, candidates):
        if not candidates:
            return []
        # token_budget >= max_running_requests，因此所有已占用名额至少能推进
        # 一个 Token。每个候选使用相同上限，实际尾 Chunk 可以更短。
        chunk_cap = self.config.token_budget // len(candidates)
        plans = []
        for state in candidates:
            start = state.prefill_cursor
            remaining = len(state.spec.token_ids) - start
            if remaining < 1:
                raise RuntimeError("prefilling 请求没有剩余 Prompt Token")
            plans.append(PrefillPlan(state, start, start + min(remaining, chunk_cap)))
        cost = self._padded_cost([plan.token_count for plan in plans])
        if cost > self.config.token_budget:
            raise RuntimeError("Chunked Prefill 生成了超出硬预算的计划")
        return plans

    def schedule(self, waiting, prefilling, running, clock_ms):
        ready = [state for state in waiting if state.spec.arrival_ms <= clock_ms]
        occupied = len(prefilling) + len(running)
        free_slots = self.config.max_running_requests - occupied
        if free_slots < 0:
            raise RuntimeError("运行请求数超过 max_running_requests")

        # 与第 06 期相同：已有 Decode 请求时，Prefill 后至少插入一次 Decode。
        if running and self.last_phase == "prefill":
            return self._decode_output(running)

        if self.config.mode == "full":
            if prefilling:
                raise RuntimeError("full 模式不应存在部分 Prefill 请求")
            if ready and free_slots:
                plans, oversize = self._full_plans(ready, free_slots)
                if plans:
                    return self._prefill_output(plans, oversize)
        else:
            candidates = list(prefilling) + ready[:free_slots]
            if candidates:
                return self._prefill_output(self._chunked_plans(candidates), False)

        if running:
            return self._decode_output(running)
        return None

    def _prefill_output(self, plans, oversize):
        plans = tuple(plans)
        lengths = [plan.token_count for plan in plans]
        self.last_phase = "prefill"
        self.iteration_count += 1
        return SchedulerOutput(
            phase="prefill",
            selected=tuple(plan.state for plan in plans),
            prefill_plans=plans,
            scheduled_tokens=self._padded_cost(lengths),
            logical_tokens=sum(lengths),
            token_budget=self.config.token_budget,
            oversize_singleton=oversize,
        )

    def _decode_output(self, running):
        selected = tuple(running)
        self.last_phase = "decode"
        self.iteration_count += 1
        return SchedulerOutput(
            phase="decode",
            selected=selected,
            scheduled_tokens=len(selected),
            logical_tokens=len(selected),
            token_budget=self.config.token_budget,
        )
