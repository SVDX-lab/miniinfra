"""第 06 期迭代级请求调度器。

调度器只做决策，不执行模型。baseline 保留 Prefill 优先策略；budgeted
策略用软 Token Budget 限制一次 Batched Prefill 的规模，并在已有 Decode
请求时避免连续两个 Prefill iteration。
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
    admitted_ms: float | None = None
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
    policy: str
    max_running_requests: int
    token_budget: int | None = None

    def validate(self):
        if self.policy not in ("baseline", "budgeted"):
            raise ValueError("policy 必须是 baseline 或 budgeted")
        if self.max_running_requests < 1:
            raise ValueError("max_running_requests 必须大于 0")
        if self.policy == "baseline" and self.token_budget is not None:
            raise ValueError("baseline 不接受 token_budget")
        if self.policy == "budgeted":
            if self.token_budget is None or self.token_budget < 1:
                raise ValueError("budgeted 策略需要正整数 token_budget")
            if self.token_budget < self.max_running_requests:
                raise ValueError(
                    "token_budget 不能小于 max_running_requests，"
                    "否则一次完整 Decode iteration 会超预算"
                )


@dataclass(frozen=True)
class SchedulerOutput:
    phase: str
    selected: tuple
    scheduled_tokens: int
    logical_tokens: int
    token_budget: int | None
    oversize_singleton: bool = False

    @property
    def request_ids(self):
        return [state.spec.request_id for state in self.selected]


class IterationScheduler:
    """在每个 iteration 选择一次 Prefill batch 或一次 Decode batch。"""

    def __init__(self, config):
        config.validate()
        self.config = config
        self.last_phase = None
        self.iteration_count = 0

    def _prefill_cost(self, states):
        if not states:
            return 0
        # 当前教学实现使用左 Padding 的 Batched Prefill，因此按实际送入
        # 模型的 padded token 数扣预算，而不是只计算逻辑 Prompt Token。
        return len(states) * max(len(state.spec.token_ids) for state in states)

    def _select_budgeted_prefill(self, ready, free_slots):
        selected = []
        budget = self.config.token_budget
        oversize = False
        for state in ready[:free_slots]:
            candidate = selected + [state]
            candidate_cost = self._prefill_cost(candidate)
            if candidate_cost <= budget:
                selected.append(state)
                continue
            if not selected:
                # 没有 Chunked Prefill 时，单个长 Prompt 无法拆分。允许队头
                # 请求单独超预算执行，避免它永久阻塞 FCFS 队列。
                selected.append(state)
                oversize = True
            break
        return selected, oversize

    def schedule(self, waiting, running, clock_ms):
        ready = [
            state for state in waiting if state.spec.arrival_ms <= clock_ms
        ]
        free_slots = self.config.max_running_requests - len(running)

        if self.config.policy == "baseline":
            if ready and free_slots:
                selected = ready[:free_slots]
                return self._output_prefill(selected, False)
            if running:
                return self._output_decode(running)
            return None

        # 一旦已有请求在 Decode，budgeted 策略不允许两个 Prefill
        # iteration 连续出现。它限制 burst 对既有请求 ITL 的影响。
        if running and self.last_phase == "prefill":
            return self._output_decode(running)
        if ready and free_slots:
            selected, oversize = self._select_budgeted_prefill(ready, free_slots)
            if selected:
                return self._output_prefill(selected, oversize)
        if running:
            return self._output_decode(running)
        return None

    def _output_prefill(self, selected, oversize):
        selected = tuple(selected)
        self.last_phase = "prefill"
        self.iteration_count += 1
        return SchedulerOutput(
            phase="prefill",
            selected=selected,
            scheduled_tokens=self._prefill_cost(selected),
            logical_tokens=sum(len(state.spec.token_ids) for state in selected),
            token_budget=self.config.token_budget,
            oversize_singleton=oversize,
        )

    def _output_decode(self, running):
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
