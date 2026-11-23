from io import BytesIO

import torch
from torch import nn


class Affine(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([2., -1.], dtype=torch.float32))
        self.register_buffer("shift", torch.tensor([1., 3.], dtype=torch.float32))
        self.register_buffer("scratch", torch.zeros(2), persistent=False)

    def forward(self, x):
        if x.shape != self.scale.shape or x.dtype != self.scale.dtype or x.device != self.scale.device:
            raise ValueError("x must match the model's shape, dtype and device")
        return self.scale * x + self.shift


def main():
    model = Affine()
    print(f"parameters: {list(dict(model.named_parameters()))}")
    print(f"buffers: {list(dict(model.named_buffers()))}")
    print(f"state_dict: {list(model.state_dict())}")
    print(f"parameter elements: {sum(value.numel() for value in model.parameters())}")

    checkpoint = BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = Affine()
    with torch.no_grad():
        restored.scale.zero_()
        restored.scratch.fill_(9)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restored.load_state_dict(state, strict=True)
    restored.to(device="cpu", dtype=torch.float64)
    x = torch.tensor([3., 4.], dtype=torch.float64)
    y = restored(x)
    torch.testing.assert_close(y, torch.tensor([7., -1.], dtype=y.dtype), rtol=0, atol=0)
    assert restored.scratch.tolist() == [9., 9.]
    print(f"restored: shape={tuple(y.shape)}, dtype={y.dtype}, device={y.device}, y={y.tolist()}")
    print(f"non-persistent scratch after load: {restored.scratch.tolist()}")


if __name__ == "__main__":
    main()
