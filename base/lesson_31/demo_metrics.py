from dataclasses import dataclass


@dataclass(frozen=True)
class Request:
    submit: int
    tokens: tuple[int, ...]
    finish: int

    def __post_init__(self):
        times = (self.submit, *self.tokens, self.finish)
        if any(a > b for a, b in zip(times, times[1:])):
            raise ValueError("timestamps must be nondecreasing")

    def metrics(self):
        ttft = self.tokens[0] - self.submit if self.tokens else None
        intervals = [b - a for a, b in zip(self.tokens, self.tokens[1:])]
        tpot = sum(intervals) / len(intervals) if intervals else None
        return ttft, intervals, tpot, self.finish - self.submit


def summarize(requests, start, end):
    if end <= start or any(not start <= r.submit <= r.finish <= end for r in requests):
        raise ValueError("expected complete requests inside a positive observation window")
    token_count = sum(len(r.tokens) for r in requests)
    changes = {}
    for r in requests:
        changes[r.submit] = changes.get(r.submit, 0) + 1
        changes[r.finish] = changes.get(r.finish, 0) - 1
    active = peak = 0
    for time in sorted(changes):
        active += changes[time]
        peak = max(peak, active)
    seconds = (end - start) / 1_000
    return token_count, token_count / seconds, len(requests) / seconds, peak


def main():
    a = Request(submit=0, tokens=(10, 16, 22), finish=25)
    b = Request(submit=4, tokens=(14, 24), finish=28)
    for name, request in (("A", a), ("B", b)):
        ttft, itl, tpot, e2e = request.metrics()
        print(f"{name}: TTFT={ttft} ms, ITL={itl} ms, TPOT={tpot} ms, E2E={e2e} ms")
    count, tps, rps, peak = summarize((a, b), 0, 30)
    print(f"window=30 ms, output tokens={count}, tokens/s={tps:.3f}, requests/s={rps:.3f}, peak={peak}")
    assert a.metrics() == (10, [6, 6], 6.0, 25)
    assert b.metrics() == (10, [10], 10.0, 24)
    assert peak == 2 and count == 5
    single = Request(submit=0, tokens=(7,), finish=9).metrics()
    assert single[2] is None
    print(f"single-token TPOT={single[2]}")


if __name__ == "__main__":
    main()
