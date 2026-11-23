import argparse

import torch


MIB = 1 << 20


def cpu_smoke():
    x = torch.empty(4, 1024, dtype=torch.float32)
    size = x.numel() * x.element_size()
    assert size == 16_384
    free_blocks_mib = (3, 3)
    request_mib = 5
    fits = max(free_blocks_mib) >= request_mib
    assert sum(free_blocks_mib) >= request_mib and not fits
    print("CPU smoke: CUDA allocator counters are omitted")
    print(f"shape={tuple(x.shape)}, FP32 payload={size} bytes")
    print(f"illustration: free={sum(free_blocks_mib)} MiB, "
          f"largest={max(free_blocks_mib)} MiB, request={request_mib} MiB, fits={fits}")


def snapshot(device):
    torch.cuda.synchronize(device)
    return torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device)


def cuda_lifetime(device):
    if torch.cuda.get_allocator_backend() != "native":
        raise RuntimeError("this demo requires PyTorch's native CUDA allocator")
    torch.cuda.empty_cache()
    baseline = snapshot(device)
    x = torch.empty(4 * MIB, device=device, dtype=torch.float32)
    payload = x.numel() * x.element_size()
    live = snapshot(device)
    del x
    freed = snapshot(device)
    torch.cuda.empty_cache()
    trimmed = snapshot(device)
    assert payload == 16 * MIB
    assert live[0] - baseline[0] == payload
    assert freed[0] == trimmed[0] == baseline[0]
    assert freed[1] >= trimmed[1]
    return payload, (baseline, live, freed, trimmed)


def main():
    parser = argparse.ArgumentParser(description="Compare live allocation and cached segments")
    parser.add_argument("--cpu-smoke", action="store_true")
    args = parser.parse_args()
    if args.cpu_smoke:
        cpu_smoke()
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Use --cpu-smoke for the byte count.")
    device = torch.device("cuda:0")
    payload, rows = cuda_lifetime(device)
    print(f"device={torch.cuda.get_device_name(device)}, torch={torch.__version__}, "
          f"torch CUDA build={torch.version.cuda}, backend=native")
    print(f"Tensor payload={payload} bytes")
    for label, (allocated, reserved) in zip(("baseline", "live", "freed", "trimmed"), rows):
        print(f"{label}: allocated={allocated} bytes, reserved={reserved} bytes")
    print("counter checks: pass")


if __name__ == "__main__":
    main()
