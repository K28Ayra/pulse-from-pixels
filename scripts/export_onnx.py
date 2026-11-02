"""Export FactorizePhys (PURE checkpoint) to ONNX for the webcam app, which has no torch.

    envs/rppg-toolbox/bin/python scripts/export_onnx.py

Writes models/factorizephys.onnx and checks it against the phase 5 output on a real dev crop.
"""

import sys
from pathlib import Path

import numpy as np
import onnxruntime
import torch

sys.path.insert(0, "scripts")
from run_deep import CHUNK, load_model, read_crop  # noqa: E402

TARGET = Path("models") / "factorizephys.onnx"
CHECK_CROP = Path("data/ubfc_rppg/derived/subject1_crop.mkv")


class PulseOnly(torch.nn.Module):
    """The model returns several things, the app only needs the pulse."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[0]


def sliced_diff(x, dim):
    n = x.shape[dim]
    return x.narrow(dim, 1, n - 1) - x.narrow(dim, 0, n - 1)


def main():
    model = PulseOnly(load_model(torch.device("cpu")))
    example = torch.rand(1, 3, CHUNK + 1, 72, 72) * 255
    # ONNX has no aten::diff, and the model calls it once on the input frames
    original_diff = torch.diff
    torch.diff = sliced_diff
    try:
        torch.onnx.export(
            model, example, str(TARGET),
            input_names=["frames"], output_names=["pulse"], opset_version=17,
        )
    finally:
        torch.diff = original_diff
    print("wrote", TARGET, TARGET.stat().st_size // 1024, "KB")

    if not CHECK_CROP.exists():
        print("no dev crop, not checked")
        return
    # compared with the phase 5 output of the same crop (torch on GPU), first chunk
    frames = read_crop(CHECK_CROP, 30.0, False)[:CHUNK + 1]
    x = frames.astype(np.float32).transpose(3, 0, 1, 2)[None]
    session = onnxruntime.InferenceSession(str(TARGET), providers=["CPUExecutionProvider"])
    got = session.run(None, {"frames": x})[0][0]
    got = (got - got.mean()) / (got.std() + 1e-9)
    expected = np.load(str(CHECK_CROP).replace("_crop.mkv", "_deep.npy"))[:CHUNK]
    print("max difference", float(np.abs(expected - got).max()))


if __name__ == "__main__":
    main()
