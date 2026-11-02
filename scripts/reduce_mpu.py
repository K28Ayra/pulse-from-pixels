"""Landmarks, patch colour and the 72x72 face crop for the MPU-rPPG sample videos.

    python scripts/reduce_mpu.py

Writes data/mpu_rppg_sample/derived/<name>.npz (times, landmarks, found, rgb,
names, fps) and <name>_crop.mkv. Only the video is read, never the PPG. The
videos are 60 fps; scripts/run_deep.py takes every other frame for the model.
"""

from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from pulsecam.config import load_config
from pulsecam.datasets import MPU_FPS, iter_frames, list_mpu_recordings
from pulsecam.face import BOX_EVERY, PATCHES, FaceTracker, crop_box, face_crop, mean_rgb_in_polygon

ROOT = Path("data") / "mpu_rppg_sample"


def reduce_one(args):
    name, video, size = args
    out_dir = ROOT / "derived"
    writer = cv2.VideoWriter(str(out_dir / (name + "_crop.mkv")), cv2.VideoWriter_fourcc(*"FFV1"), MPU_FPS, (size, size))
    tracker = FaceTracker()
    names = list(PATCHES)
    all_points = []
    found = []
    rgb = []
    last_crop = np.zeros((size, size, 3), dtype=np.uint8)
    box = None
    for i, frame in enumerate(iter_frames(video)):
        points = tracker.find(frame, i / MPU_FPS)
        row = np.full((len(names), 3), np.nan, dtype=np.float32)
        if points is None:
            all_points.append(np.full((478, 2), np.nan))
            found.append(False)
        else:
            all_points.append(points)
            found.append(True)
            for j, patch in enumerate(names):
                row[j] = mean_rgb_in_polygon(frame, points[PATCHES[patch]])
            if box is None or i % BOX_EVERY == 0:
                box = crop_box(points)
            last_crop = face_crop(frame, box, size)
        rgb.append(row)
        writer.write(cv2.cvtColor(last_crop, cv2.COLOR_RGB2BGR))
    tracker.close()
    writer.release()
    np.savez_compressed(
        out_dir / (name + ".npz"),
        times=np.arange(len(found)) / MPU_FPS,
        landmarks=np.array(all_points, dtype=np.float16),
        found=np.array(found),
        rgb=np.array(rgb),
        names=np.array(names),
        fps=MPU_FPS,
    )
    return name, len(found), 100 * np.mean(found)


def main():
    cfg = load_config()
    (ROOT / "derived").mkdir(exist_ok=True)
    jobs = []
    for name, video, _ in list_mpu_recordings(ROOT):
        if not (ROOT / "derived" / (name + ".npz")).exists():
            jobs.append((name, video, cfg["crop_size"]))
    with Pool(len(jobs) or 1) as pool:
        for name, n_frames, found_pct in pool.imap_unordered(reduce_one, jobs):
            print(name, n_frames, "frames, face in", round(found_pct, 2), "%", flush=True)


if __name__ == "__main__":
    main()
