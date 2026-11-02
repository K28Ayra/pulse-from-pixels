"""Face landmarks and whole-face mean colour for every UBFC-rPPG frame.

    python scripts/extract_faces.py            # all subjects, 8 at a time
    python scripts/extract_faces.py subject5   # just one

Writes data/ubfc_rppg/derived/<subject>.npz with times, landmarks (pixels,
float16), found and face_rgb (mean colour inside the face oval, NaN where no
face). Subjects that already have a file are skipped. Only the video and the
timestamps are read, never the PPG, so running it on test subjects is fine.
"""

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from pulsecam.config import load_config
from pulsecam.datasets import iter_frames, list_ubfc_subjects, load_ubfc_subject
from pulsecam.face import FACE_OVAL, FaceTracker, mean_rgb_in_polygon

N_WORKERS = 8


def extract_subject(args):
    root, name, out_path = args
    started = time.time()
    subject = load_ubfc_subject(root, name)
    times = subject["ppg_times"] - subject["ppg_times"][0]

    tracker = FaceTracker()
    all_points = []
    found = []
    face_rgb = []
    for i, frame in enumerate(iter_frames(subject["video_path"])):
        points = tracker.find(frame, times[i])
        if points is None:
            all_points.append(np.full((478, 2), np.nan))
            found.append(False)
            face_rgb.append(np.full(3, np.nan))
            continue
        all_points.append(points)
        found.append(True)
        oval = points[FACE_OVAL]
        face_rgb.append(mean_rgb_in_polygon(frame, oval))
    tracker.close()

    n_frames = len(found)
    np.savez_compressed(
        out_path,
        times=times[:n_frames],
        landmarks=np.array(all_points, dtype=np.float16),
        found=np.array(found),
        face_rgb=np.array(face_rgb, dtype=np.float32),
    )
    seconds = time.time() - started
    found_pct = 100 * np.mean(found)
    return name, n_frames, found_pct, seconds


def main():
    cfg = load_config()
    root = Path(cfg["data_dir"]) / "ubfc_rppg"
    out_dir = root / "derived"
    out_dir.mkdir(exist_ok=True)

    names = sys.argv[1:]
    if not names:
        names = list_ubfc_subjects(root)

    jobs = []
    for name in names:
        out_path = out_dir / (name + ".npz")
        if out_path.exists():
            continue
        jobs.append((root, name, out_path))
    print(len(jobs), "subjects to do")

    with Pool(N_WORKERS) as pool:
        for name, n_frames, found_pct, seconds in pool.imap_unordered(extract_subject, jobs):
            print(name, n_frames, "frames, face in", round(found_pct, 2), "%,", round(seconds), "s", flush=True)


if __name__ == "__main__":
    main()
