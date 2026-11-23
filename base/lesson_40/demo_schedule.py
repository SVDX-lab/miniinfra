import math


def learning_rate(step, *, mode, peak=0.2, warmup=2, total=6):
    if not 1 <= step <= total or not 0 < warmup < total:
        raise ValueError("require 1 <= step <= total and 0 < warmup < total")
    if mode not in ("step", "cosine"):
        raise ValueError("mode must be step or cosine")
    if step <= warmup:
        return peak * step / warmup
    if mode == "step":
        return peak * 0.5 ** ((step - warmup - 1) // 2)
    fraction = (step - warmup) / (total - warmup)
    return peak * (1 + math.cos(math.pi * fraction)) / 2


def main():
    for mode in ("step", "cosine"):
        rates = []
        weight = 1.
        for step in range(1, 7):
            rate = learning_rate(step, mode=mode)
            weight -= rate  # A fixed gradient of one isolates the schedule.
            rates.append(round(rate, 6))
        assert rates[:2] == [0.1, 0.2]
        assert math.isclose(weight, {"step": 0.1, "cosine": 0.4}[mode], abs_tol=1e-12)
        print(f"{mode}: steps=1..6, lr={rates}; final w={weight:.6f}")
    print("one LR per update; warm-up occupies steps 1 and 2")


if __name__ == "__main__":
    main()
