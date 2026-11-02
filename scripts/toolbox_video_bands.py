"""Video-level MAE of the deep baselines with the band of this toolbox commit
(0.6-3.3 Hz) and with the band the toolbox paper used (0.75-2.5 Hz).

    envs/rppg-toolbox/bin/python scripts/toolbox_video_bands.py

Uses the toolbox's own detrend and FFT code on its saved test outputs.
Writes outputs/phase1/toolbox_video_bands.csv (one row per model and subject).
"""

import csv
import glob
import os
import pickle
import sys

import numpy as np
from scipy import signal

sys.path.insert(0, os.path.join("tools", "rPPG-Toolbox"))

from evaluation.post_process import _calculate_fft_hr, _detrend  # noqa: E402

JOBS = ["PhysNet", "TS-CAN", "EfficientPhys", "FactorizePhys"]
BANDS = {"commit": (0.6, 3.3), "paper": (0.75, 2.5)}


def joined(chunks):
    pieces = []
    for index in sorted(chunks):
        pieces.append(np.asarray(chunks[index].cpu()).ravel())
    return np.concatenate(pieces)


def video_hr(x, fs, diff, band):
    if diff:
        x = np.cumsum(x)
    x = _detrend(x, 100)
    b, a = signal.butter(1, [band[0] / fs * 2, band[1] / fs * 2], btype="bandpass")
    x = signal.filtfilt(b, a, np.double(x))
    return _calculate_fft_hr(x, fs=fs, low_pass=band[0], high_pass=band[1])


def main():
    rows = []
    for job in JOBS:
        pattern = os.path.join("outputs", "phase1", "toolbox", job, "**", "*_outputs.pickle")
        paths = glob.glob(pattern, recursive=True)
        if not paths:
            print(job, "no saved outputs")
            continue
        with open(paths[0], "rb") as handle:
            saved = pickle.load(handle)
        diff = saved["label_type"] == "DiffNormalized"
        fs = saved["fs"]
        for name in saved["predictions"]:
            pred = joined(saved["predictions"][name])
            label = joined(saved["labels"][name])
            row = {"model": job, "subject": name}
            for band_name, band in BANDS.items():
                row["pred_" + band_name] = video_hr(pred, fs, diff, band)
                row["label_" + band_name] = video_hr(label, fs, diff, band)
            rows.append(row)
        for band_name in BANDS:
            errors = [abs(r["pred_" + band_name] - r["label_" + band_name]) for r in rows if r["model"] == job]
            print(job, band_name, "band MAE", round(float(np.mean(errors)), 2))

    out_path = os.path.join("outputs", "phase1", "toolbox_video_bands.csv")
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
