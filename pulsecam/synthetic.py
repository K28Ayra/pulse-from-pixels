"""Fake signals and videos with a known heart rate, for tests."""

from pathlib import Path

import cv2
import numpy as np


def fake_ppg(hr_bpm, fs, seconds, noise=0.05, seed=0):
    """A PPG-like wave: fundamental plus a weaker second harmonic."""
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    t = np.arange(n) / fs
    f = hr_bpm / 60.0
    wave = np.sin(2 * np.pi * f * t)
    wave = wave + 0.3 * np.sin(2 * np.pi * 2 * f * t + 0.5)
    wave = wave + noise * rng.standard_normal(n)
    return t, wave


def fake_face_video(path, hr_bpm, fps=30, seconds=4, size=(64, 48)):
    """Write a small video of a skin-coloured oval whose colour pulses at hr_bpm."""
    width, height = size
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    n_frames = int(seconds * fps)
    f = hr_bpm / 60.0
    center = (width // 2, height // 2)
    axes = (width // 4, height // 3)

    for i in range(n_frames):
        t = i / fps
        pulse = np.sin(2 * np.pi * f * t)
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        # BGR skin tone; green moves most, like real blood volume changes
        blue = int(120 + 2 * pulse)
        green = int(150 + 4 * pulse)
        red = int(200 + 1 * pulse)
        cv2.ellipse(frame, center, axes, 0, 0, 360, (blue, green, red), -1)
        writer.write(frame)

    writer.release()
    return n_frames


def fake_ubfc_subject(root, name, hr_bpm=72, fps=30, seconds=4):
    """Create <root>/DATASET_2/<name>/ with vid.avi and ground_truth.txt."""
    folder = Path(root) / "DATASET_2" / name
    folder.mkdir(parents=True, exist_ok=True)

    n_frames = fake_face_video(folder / "vid.avi", hr_bpm, fps, seconds)

    times, ppg = fake_ppg(hr_bpm, fps, seconds)
    hr = np.full(len(times), float(hr_bpm))
    rows = np.vstack([ppg, hr, times])
    np.savetxt(folder / "ground_truth.txt", rows)
    return n_frames
