import argparse
from time import perf_counter

import torch


def verify(result, count):
    expected = torch.full_like(result, count)
    torch.testing.assert_close(result, expected, atol=0, rtol=0)


def cpu_smoke():
    x = torch.zeros(4, dtype=torch.float32)
    for _ in range(3):
        x.add_(1)
    verify(x, 3)
    print("CPU smoke: CUDA stream and Event timings are omitted")
    print(f"3 ordered additions: {x.tolist()}")
    print("value check: pass")


def cuda_timing(device, steps):
    # Work and both Events belong to this one stream.
    stream = torch.cuda.Stream(device=device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        x = torch.zeros(1 << 20, device=device, dtype=torch.float32)
        x.add_(1)
        x.zero_()
    torch.cuda.synchronize(device)

    start.record(stream)
    host_start = perf_counter()
    with torch.cuda.stream(stream):
        for _ in range(steps):
            x.add_(1)
    host_submit_ms = (perf_counter() - host_start) * 1_000
    end.record(stream)
    end.synchronize()
    elapsed_ms = start.elapsed_time(end)
    verify(x, steps)
    return host_submit_ms, elapsed_ms


def main():
    parser = argparse.ArgumentParser(description="Time one CUDA stream with Events")
    parser.add_argument("--cpu-smoke", action="store_true")
    args = parser.parse_args()
    if args.cpu_smoke:
        cpu_smoke()
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Use --cpu-smoke for the value check.")
    device = torch.device("cuda:0")
    steps = 64
    host_ms, event_ms = cuda_timing(device, steps)
    print(f"device={torch.cuda.get_device_name(device)}, torch={torch.__version__}, "
          f"torch CUDA build={torch.version.cuda}")
    print(f"shape=(1048576,), dtype=float32, iterations={steps}, streams=1")
    print(f"host submission={host_ms:.3f} ms")
    print(f"stream Event interval={event_ms:.3f} ms")
    print("value check: pass (all entries equal 64)")


if __name__ == "__main__":
    main()
