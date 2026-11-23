import argparse

import torch


def transform(x):
    return (x * 2 + 1).clamp_min(0)


@torch.compiler.disable
def python_boundary(x):
    return x + 1


def split_function(x):
    return python_boundary(x * 2).clamp_min(0)


def check_graph_break(x):
    strict = torch.compile(split_function, backend="eager", fullgraph=True)
    try:
        strict(x)
    except torch._dynamo.exc.Unsupported:
        print("fullgraph diagnostic: expected graph break")
    else:
        raise AssertionError("strict capture unexpectedly accepted the disabled function")


def event_ms(operation, x, iterations):
    torch.cuda.synchronize(x.device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        operation(x)
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def main():
    parser = argparse.ArgumentParser(description="Compare eager and compiled pointwise operations")
    parser.add_argument("--cpu-smoke", action="store_true")
    args = parser.parse_args()
    x = torch.tensor([-2, -0.5, 0, 2], dtype=torch.float32)
    expected = torch.tensor([0, 0, 1, 5], dtype=x.dtype)
    y = transform(x)
    torch.testing.assert_close(y, expected, atol=0, rtol=0)
    print(f"CPU FP32 example: {y.tolist()}")
    print("ideal logical traffic per FP32 element: eager=24 bytes, fused=8 bytes")
    check_graph_break(x)

    if args.cpu_smoke:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Inductor run; use --cpu-smoke for the example")
    x = torch.linspace(-2, 2, 1 << 20, device="cuda:0", dtype=torch.float32)
    compiled = torch.compile(transform, backend="inductor", fullgraph=True)
    expected = transform(x)
    actual = compiled(x)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    for operation in (transform, compiled):
        for _ in range(4):
            operation(x)
    steps = 32
    print(f"device={torch.cuda.get_device_name(x.device)}, torch={torch.__version__}, "
          f"build CUDA={torch.version.cuda}")
    print(f"shape={tuple(x.shape)}, dtype=float32, iterations={steps}")
    print("eager / compiled values: match (atol=1e-6, rtol=1e-6)")
    for name, operation in (("eager", transform), ("compiled", compiled)):
        print(f"{name} CUDA Event batch: {event_ms(operation, x, steps):.3f} ms")


if __name__ == "__main__":
    main()
