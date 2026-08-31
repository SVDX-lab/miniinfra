"""第 08 期独立使用的 Chunked Prefill 调度器。

Prefix Cache disabled/enabled 两条路径共用完全相同的 FCFS、硬 Token Budget
和 Decode 保护规则。调度器只读取 ``prefill_cursor``，不参与缓存匹配。
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
    prefix_hit_tokens: int = 0
    prefix_lookup_ms: float = 0.0
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
    return specs


@dataclass(frozen=True)
class SchedulerConfig:
    max_running_requests: int
    token_budget: int

    def validate(self):
        if self.max_running_requests < 1:
            raise ValueError("max_running_requests 必须大于 0")
        if self.token_budget < self.max_running_requests:
            raise ValueError("token_budget 不能小于 max_running_requests")


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

    @property
    def request_ids(self):
        return [state.spec.request_id for state in self.selected]


class ChunkedPrefillScheduler:
    def __init__(self, config):
        config.validate()
        self.config = config
        self.last_phase = None

    @staticmethod
    def _padded_cost(lengths):
        return len(lengths) * max(lengths) if lengths else 0

    def _chunked_plans(self, candidates):
        chunk_cap = self.config.token_budget // len(candidates)
        plans = []
        for state in candidates:
            start = state.prefill_cursor
            remaining = len(state.spec.token_ids) - start
            if remaining < 1:
                raise RuntimeError("Prefill 请求没有剩余 Prompt Token")
            plans.append(PrefillPlan(state, start, start + min(remaining, chunk_cap)))
        if self._padded_cost([plan.token_count for plan in plans]) > self.config.token_budget:
            raise RuntimeError("Chunked Prefill 生成了超出硬预算的计划")
        return plans

    def schedule(self, waiting, prefilling, running, clock_ms):
        ready = [state for state in waiting if state.spec.arrival_ms <= clock_ms]
        free_slots = self.config.max_running_requests - len(prefilling) - len(running)
        if free_slots < 0:
            raise RuntimeError("运行请求数超过 max_running_requests")
        if running and self.last_phase == "prefill":
            return self._decode_output(running)
        candidates = list(prefilling) + ready[:free_slots]
        if candidates:
            plans = self._chunked_plans(candidates)
            lengths = [plan.token_count for plan in plans]
            self.last_phase = "prefill"
            return SchedulerOutput(
                "prefill", tuple(plan.state for plan in plans), tuple(plans),
                self._padded_cost(lengths), sum(lengths), self.config.token_budget,
            )
        if running:
            return self._decode_output(running)
        return None

    def _decode_output(self, running):
        self.last_phase = "decode"
        return SchedulerOutput(
            "decode", tuple(running), scheduled_tokens=len(running),
            logical_tokens=len(running), token_budget=self.config.token_budget,
        )
