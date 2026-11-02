"""Mean colour of each skin patch in every UBFC-rPPG frame.

    python scripts/extract_patches.py

Uses the landmarks saved by scripts/extract_faces.py, so MediaPipe doesn't run
again. Writes data/ubfc_rppg/derived/<subject>_patches.npz with rgb of shape
(frames, patches, 3) and the patch names. Like extract_faces.py it never reads
the PPG, so it runs on all subjects.
"""

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from pulsecam.config import load_config
from pulsecam.datasets import iter_frames, list_ubfc_subjects
from pulsecam.face import PATCHES, mean_rgb_in_polygon

N_WORKERS = 8


def extract_subject(args):
    root, name = args
    derived = np.load(root / "derived" / (name + ".npz"))
    landmarks = derived["landmarks"].astype(float)
    found = derived["found"]
    names = list(PATCHES)

    rgb = np.full((len(found), len(names), 3), np.nan, dtype=np.float32)
    video = root / "DATASET_2" / name / "vid.avi"
    for i, frame in enumerate(iter_frames(video)):
        if i >= len(found) or not found[i]:
            continue
        for j, patch in enumerate(names):
            outline = landmarks[i][PATCHES[patch]]
            rgb[i, j] = mean_rgb_in_polygon(frame, outline)

    out_path = root / "derived" / (name + "_patches.npz")
    np.savez_compressed(out_path, rgb=rgb, names=np.array(names))
    return name


def main():
    cfg = load_config()
    root = Path(cfg["data_dir"]) / "ubfc_rppg"
    names = sys.argv[1:]
    if not names:
        names = list_ubfc_subjects(root)
    jobs = [(root, name) for name in names]
    with Pool(N_WORKERS) as pool:
        for name in pool.imap_unordered(extract_subject, jobs):
            print(name, flush=True)


if __name__ == "__main__":
    main()
