import argparse
from time import perf_counter

import torch


MIB = 1 << 20


def cpu_smoke():
    source = torch.arange(4, dtype=torch.float32)
    destination = torch.empty_like(source)
    destination.copy_(source)
    torch.testing.assert_close(destination, source, atol=0, rtol=0)
    print("CPU smoke: values and byte count only")
    print(f"shape={tuple(source.shape)}, bytes={source.numel() * source.element_size()}")
    print(f"destination={destination.tolist()}")


def measure(source, destination, non_blocking, iterations):
    if (source.device.type != "cpu" or destination.device.type != "cuda"
            or source.dtype != destination.dtype or source.shape != destination.shape):
        raise ValueError("expected matching CPU source and CUDA destination")
    if not source.is_contiguous() or not destination.is_contiguous():
        raise ValueError("this demo requires contiguous buffers")
    device = destination.device
    stream = torch.cuda.current_stream(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    destination.copy_(source, non_blocking=non_blocking)
    torch.cuda.synchronize(device)

    start.record(stream)
    host_start = perf_counter()
    for _ in range(iterations):
        destination.copy_(source, non_blocking=non_blocking)
    host_ms = (perf_counter() - host_start) * 1_000
    end.record(stream)
    end.synchronize()
    stream_ms = start.elapsed_time(end)
    torch.testing.assert_close(destination.cpu(), source, atol=0, rtol=0)
    return host_ms, stream_ms


def main():
    parser = argparse.ArgumentParser(description="Compare pageable and pinned host buffers")
    parser.add_argument("--cpu-smoke", action="store_true")
    args = parser.parse_args()
    if args.cpu_smoke:
        cpu_smoke()
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Use --cpu-smoke for the value check.")
    device = torch.device("cuda:0")
    with torch.cuda.device(device):
        pageable = torch.full((4 * MIB,), 1.0, dtype=torch.float32)
        pinned = pageable.pin_memory()
        destination = torch.empty_like(pinned, device=device)
        iterations = 8
        payload = pageable.numel() * pageable.element_size()
        print(f"device={torch.cuda.get_device_name(device)}, torch={torch.__version__}, "
              f"torch CUDA build={torch.version.cuda}")
        print(f"shape={tuple(pageable.shape)}, dtype=float32, bytes={payload}, copies={iterations}")
        for source in (pageable, pinned):
            for non_blocking in (False, True):
                host_ms, stream_ms = measure(source, destination, non_blocking, iterations)
                if stream_ms <= 0:
                    raise RuntimeError("Event interval is too short to estimate bandwidth")
                gbs = payload * iterations / (stream_ms * 1_000_000)
                print(f"pinned={source.is_pinned()} non_blocking={non_blocking}: "
                      f"host={host_ms:.3f} ms, Event={stream_ms:.3f} ms, {gbs:.3f} GB/s")
        print("copied values: pass")


if __name__ == "__main__":
    main()
