import argparse
from statistics import median
from time import perf_counter_ns

import torch


def record_pairs(operations, *, warmup, repeats, measure):
    if len(operations) != 2 or warmup < 0 or repeats < 1:
        raise ValueError("expected A/B, nonnegative warm-up and positive repeats")
    for _ in range(warmup):
        for operation in operations:
            operation()
    rows = []
    for pair in range(repeats):
        order = (0, 1) if pair % 2 == 0 else (1, 0)
        for position, option in enumerate(order, start=1):
            rows.append((pair + 1, position, "AB"[option], measure(operations[option])))
    return rows


def micros(operation):
    start = perf_counter_ns()
    operation()
    return (perf_counter_ns() - start) / 1_000


def main():
    parser = argparse.ArgumentParser(description="Keep raw CPU A/B samples")
    parser.add_argument("--measure", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    x = torch.arange(4096, dtype=torch.int64)
    buffer = torch.empty_like(x)

    def allocate():
        return x * 2 + 1

    def reuse():
        torch.mul(x, 2, out=buffer)
        return buffer.add_(1)

    torch.testing.assert_close(allocate(), reuse(), atol=0, rtol=0)
    if args.measure:
        take = micros
        print("source=CPU wall clock")
    else:
        data = iter((8, 6, 7, 9, 10, 5))

        def take(operation):
            operation()
            return next(data)

        print("source=fixed teaching samples")
    print("dtype=int64, shape=(4096,), threads=1, warm-up=3 per option")
    rows = record_pairs((allocate, reuse), warmup=3, repeats=3, measure=take)
    print("pair,position,option,elapsed_us")
    for pair, position, option, elapsed in rows:
        print(f"{pair},{position},{option},{elapsed:.3f}")
    for name in "AB":
        samples = [elapsed for _, _, option, elapsed in rows if option == name]
        print(f"{name}: n={len(samples)}, median={median(samples):.3f} us")


if __name__ == "__main__":
    main()
