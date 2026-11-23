import math

import torch


def activations(x):
    relu = x.clamp_min(0)
    gelu = x * (1 + torch.erf(x / math.sqrt(2))) / 2
    silu = x * x.sigmoid()
    return relu, gelu, silu


def swiglu(x, gate_weight, up_weight, down_weight):
    if x.ndim != 3 or gate_weight.ndim != 2 or gate_weight.shape[0] == 0:
        raise ValueError("expected X [B,S,D] and a nonempty gate matrix")
    width = x.shape[-1]
    intermediate = gate_weight.shape[0]
    if (gate_weight.shape != (intermediate, width)
            or up_weight.shape != gate_weight.shape
            or down_weight.shape != (width, intermediate)):
        raise ValueError("expected gate/up [I,D] and down [D,I]")
    if not x.is_floating_point() or any(
        weight.dtype != x.dtype or weight.device != x.device
        for weight in (gate_weight, up_weight, down_weight)
    ):
        raise ValueError("expected one floating dtype and one device")

    gate = x @ gate_weight.T
    up = x @ up_weight.T
    product = (gate * gate.sigmoid()) * up
    output = product @ down_weight.T
    return gate, up, product, output


def rounded(tensor):
    return [round(value, 6) for value in tensor.flatten().tolist()]


def main():
    options = dict(dtype=torch.float64, device="cpu")
    sample = torch.tensor([-1, 0, 1], **options)
    for name, values in zip(("ReLU", "GELU", "SiLU"), activations(sample)):
        print(f"{name}(-1,0,1): {rounded(values)}")
    x = torch.tensor([[[1, 2]]], **options)
    ln3 = math.log(3)
    gate_weight = torch.tensor([[0, 0], [ln3, 0], [-ln3, 0]], **options)
    up_weight = torch.tensor([[0, 1], [2, 1], [0, 2]], **options)
    down_weight = torch.tensor([[1, 1, 1], [0, 1, -1]], **options)

    gate, up, product, output = swiglu(x, gate_weight, up_weight, down_weight)
    print(f"X -> gate/up -> Y: {tuple(x.shape)} -> {tuple(gate.shape)} -> {tuple(output.shape)}")
    print(f"gate: {rounded(gate)}; up: {rounded(up)}")
    print(f"SiLU(gate) * up: {rounded(product)}")
    print(f"Y: {rounded(output)}")
    torch.testing.assert_close(
        output, torch.tensor([[[2 * ln3, 4 * ln3]]], **options), rtol=0, atol=1e-12
    )
    print("hand calculation: pass (atol=1e-12, rtol=0)")


if __name__ == "__main__":
    main()
