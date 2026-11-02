"""FactorizePhys (PURE checkpoint) on the saved 72x72 crops, and its CPU speed.

    envs/rppg-toolbox/bin/python scripts/run_deep.py speed
    envs/rppg-toolbox/bin/python scripts/run_deep.py run

Needs the toolbox env. Writes <recording>_deep.npy next to each crop, one value per
frame for UBFC-rPPG and on a 30 Hz grid for UBFC-Phys, which runs at 35 fps. Frames
after the last full chunk are lost. Stress-variant crops (<name>_<variant>_crop.mkv)
are run too. Existing outputs are skipped: delete them to run again.
Only dev people unless names are given after "run".
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

TOOLBOX = Path("tools") / "rPPG-Toolbox"
sys.path.insert(0, str(TOOLBOX))
from neural_methods.model.FactorizePhys.FactorizePhys import FactorizePhys  # noqa: E402

CHECKPOINT = TOOLBOX / "final_model_release" / "PURE_FactorizePhys_FSAM_Res.pth"
CHUNK = 160
FS = 30.0
BATCH = 16


def load_model(device):
    md_config = {
        "FRAME_NUM": CHUNK,
        "MD_TYPE": "NMF",
        "MD_FSAM": False,
        "MD_TRANSFORM": "T_KAB",
        "MD_S": 1,
        "MD_R": 1,
        "MD_STEPS": 4,
        "MD_INFERENCE": True,
        "MD_RESIDUAL": True,
    }
    model = FactorizePhys(frames=CHUNK, md_config=md_config, in_channels=3, dropout=0.1, device=device)
    state = torch.load(CHECKPOINT, map_location=device)
    clean = {}
    for key, value in state.items():
        clean[key.replace("module.", "")] = value
    result = model.load_state_dict(clean, strict=False)
    print("missing", len(result.missing_keys), "unexpected", len(result.unexpected_keys), flush=True)
    model.to(device)
    model.eval()
    return model


def read_crop(path, fps, resample):
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    frames = np.array(frames)
    if not resample:
        return frames
    # the model expects about 30 Hz, so the nearest frame is taken on a 30 Hz grid
    times = np.arange(len(frames)) / fps
    grid = np.arange(0.0, times[-1], 1.0 / FS)
    picks = np.clip(np.round(grid * fps).astype(int), 0, len(frames) - 1)
    return frames[picks]


def pulse_of(model, frames, device):
    n_chunks = len(frames) // CHUNK
    parts = []
    for first in range(0, n_chunks, BATCH):
        batch = []
        for c in range(first, min(first + BATCH, n_chunks)):
            chunk = frames[c * CHUNK:(c + 1) * CHUNK + 1]
            if len(chunk) == CHUNK:
                chunk = np.concatenate([chunk, chunk[-1:]])
            batch.append(chunk)
        x = np.array(batch, dtype=np.float32).transpose(0, 4, 1, 2, 3)
        with torch.no_grad():
            output = model(torch.from_numpy(x).to(device))[0]
        output = output.cpu().numpy()
        for row in output:
            parts.append((row - row.mean()) / (row.std() + 1e-9))
    return np.concatenate(parts)


def crops_of(folder, stem):
    """The raw crop and every stress-variant crop of one recording."""
    found = [folder / (stem + "_crop.mkv")]
    found += sorted(folder.glob(stem + "_*_crop.mkv"))
    return [path for path in found if path.exists()]


def recordings(names):
    jobs = []
    rppg = Path("data/ubfc_rppg/derived")
    split = json.loads(Path("splits/ubfc_rppg.json").read_text())
    for name in split["dev"]:
        if names and name not in names:
            continue
        times = np.load(rppg / (name + ".npz"))["times"]
        fps = (len(times) - 1) / (times[-1] - times[0])
        for path in crops_of(rppg, name):
            jobs.append((path, fps, False))
    phys = Path("data/ubfc_phys/derived")
    split = json.loads(Path("splits/ubfc_phys.json").read_text())
    for person in split["dev"]:
        if names and person not in names:
            continue
        for task in ["T1", "T2", "T3"]:
            stem = person + "_" + task
            fps = float(np.load(phys / (stem + "_raw.npz"))["fps"])
            for path in crops_of(phys, stem):
                jobs.append((path, fps, True))
    return jobs


def run(names):
    device = torch.device("cuda:0")
    model = load_model(device)
    for path, fps, resample in recordings(names):
        target = Path(str(path).replace("_crop.mkv", "_deep.npy"))
        if target.exists():
            continue
        frames = read_crop(path, fps, resample)
        pulse = pulse_of(model, frames, device)
        np.save(target, pulse.astype(np.float32))
        print(path.name, len(frames), "frames,", len(pulse), "samples", flush=True)


def speed():
    device = torch.device("cpu")
    n_params = None
    for threads in [1, 2, 4, 8]:
        torch.set_num_threads(threads)
        model = load_model(device)
        n_params = sum(p.numel() for p in model.parameters())
        x = torch.rand(1, 3, CHUNK + 1, 72, 72) * 255
        times = []
        for i in range(6):
            started = time.perf_counter()
            with torch.no_grad():
                model(x)
            times.append(time.perf_counter() - started)
        seconds = np.median(times[1:])
        print(threads, "threads:", round(seconds, 3), "s per chunk,", round(CHUNK / seconds, 1), "fps", flush=True)
    print("parameters", n_params)


if __name__ == "__main__":
    if sys.argv[1] == "speed":
        speed()
    else:
        run(sys.argv[2:])
