import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# Runs in its own process, because the audit hook can't be removed again.
# The model and the face finder are stand-ins, so this needs no model files:
# the "model" returns the mean green of each crop, and the frames pulse at 72 bpm.
GUARDED_RUN = r"""
import os, socket, sys, urllib.request
import numpy as np
from pulsecam.config import load_config
from pulsecam.live import CHUNK, LiveHeartRate
from pulsecam.privacy import block_writes_and_network

class GreenModel:
    def run(self, names, inputs):
        frames = inputs["frames"]
        return [frames[:, 1, :CHUNK].mean(axis=(2, 3))]

angles = np.linspace(0, 2 * np.pi, 478)
landmarks = np.stack([320 + 80 * np.cos(angles), 240 + 100 * np.sin(angles)], axis=1)

cfg = load_config(sys.argv[1], live_mode=True)
heart = LiveHeartRate(cfg, lambda rgb, t: landmarks, model=GreenModel())
block_writes_and_network()

first = None
last_hr = None
for n in range(15 * 30):
    t = n / 30.0
    frame = np.full((480, 640, 3), 120.0)
    frame[:, :, 1] += 3 * np.sin(2 * np.pi * 1.2 * t)
    reading = heart.push(frame.astype(np.uint8), t)
    if reading is not None:
        last_hr = reading["hr"]
        if first is None:
            first = t

refused = 0
for attempt in [
    lambda: open("x.txt", "w"),
    lambda: os.open("y.txt", os.O_WRONLY | os.O_CREAT),
    lambda: os.mkdir("folder"),
    lambda: socket.socket(),
    lambda: urllib.request.urlopen("http://127.0.0.1:9"),
]:
    try:
        attempt()
    except PermissionError:
        refused += 1
print("first", first, "hr", last_hr, "refused", refused)
"""


def test_live_mode_writes_nothing_and_opens_no_sockets(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(REPO))
    result = subprocess.run(
        [sys.executable, "-c", GUARDED_RUN, str(REPO / "configs" / "default.yaml")],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr
    words = result.stdout.split()
    first = float(words[words.index("first") + 1])
    hr = float(words[words.index("hr") + 1])
    refused = int(words[words.index("refused") + 1])
    assert first <= 10
    assert abs(hr - 72) <= 2
    assert refused == 5
    assert list(tmp_path.iterdir()) == []


def files_under(folder, skip):
    found = set()
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if d not in skip]
        for name in files:
            found.add(os.path.join(root, name))
    return found


VIDEO = REPO / "data" / "ubfc_rppg" / "DATASET_2" / "subject1" / "vid.avi"
NEEDED = [REPO / "models" / "face_landmarker.task", REPO / "models" / "factorizephys.onnx", VIDEO]


@pytest.mark.skipif(not all(p.exists() for p in NEEDED), reason="needs the models and UBFC-rPPG subject1")
def test_real_app_leaves_no_files():
    skip = {"envs", "data", "tools", ".git", ".cache", "__pycache__"}
    temp = Path(tempfile.gettempdir())
    before = files_under(REPO, skip) | set(str(p) for p in temp.iterdir())
    result = subprocess.run(
        [sys.executable, "scripts/live.py", "--video", str(VIDEO), "--no-window"],
        cwd=REPO, capture_output=True, text=True, timeout=600,
    )
    after = files_under(REPO, skip) | set(str(p) for p in temp.iterdir())
    assert result.returncode == 0, result.stderr[-2000:]
    assert "bpm" in result.stdout
    assert after - before == set()
