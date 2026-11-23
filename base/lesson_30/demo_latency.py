import math
from fractions import Fraction


def percentile(samples, fraction):
    if not samples or not all(math.isfinite(value) and value >= 0 for value in samples):
        raise ValueError("expected at least one finite nonnegative latency")
    if not 0 < fraction <= 1:
        raise ValueError("percentile fraction must be in (0, 1]")
    ordered = sorted(samples)
    return ordered[math.ceil(Fraction(str(fraction)) * len(ordered)) - 1]


def main():
    steady = [10] * 20
    tail = [90, *([5] * 18), 20]
    for name, samples in (("steady", steady), ("tail", tail)):
        mean = sum(samples) / len(samples)
        print(f"{name}: sorted={sorted(samples)} ms")
        print(f"n={len(samples)}, mean={mean:.1f} ms, "
              f"P50={percentile(samples, 0.50)} ms, P95={percentile(samples, 0.95)} ms")
    assert sum(steady) == sum(tail) == 200
    assert percentile(tail, 0.5) == 5 and percentile(tail, 0.95) == 20
    assert percentile(tail, 1) == 90
    print("nearest ranks for n=20: P50 uses rank 10; P95 uses rank 19")


if __name__ == "__main__":
    main()
