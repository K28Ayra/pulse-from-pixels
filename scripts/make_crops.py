"""Lossless 72x72 face crops from the raw video and the saved landmarks.

    python scripts/make_crops.py            # dev people of both datasets
"""

import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from pulsecam.config import load_config
from pulsecam.datasets import UBFC_PHYS_TASKS, iter_frames
from reduce_ubfc_phys import BOX_EVERY, crop_box, face_crop


def write_crop(args):
    video, landmarks_path, out_path, size = args
    saved = np.load(landmarks_path)
    landmarks = saved["landmarks"].astype(np.float64)
    found = saved["found"]
    if "times" in saved.files:
        times = saved["times"]
        fps = (len(times) - 1) / (times[-1] - times[0])
    else:
        fps = float(saved["fps"])
    fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (size, size))
    last_crop = np.zeros((size, size, 3), dtype=np.uint8)
    box = None
    n = 0
    for i, frame in enumerate(iter_frames(video)):
        if i >= len(found):
            break
        if found[i]:
            if box is None or i % BOX_EVERY == 0:
                box = crop_box(landmarks[i])
            last_crop = face_crop(frame, box, size)
        writer.write(cv2.cvtColor(last_crop, cv2.COLOR_RGB2BGR))
        n += 1
    writer.release()
    return out_path.name, n


def main():
    cfg = load_config()
    size = cfg["crop_size"]
    jobs = []
    root = Path(cfg["data_dir"]) / "ubfc_rppg"
    for name in json.loads(Path("splits/ubfc_rppg.json").read_text())["dev"]:
        video = root / "DATASET_2" / name / "vid.avi"
        derived = root / "derived"
        jobs.append((video, derived / (name + ".npz"), derived / (name + "_crop.mkv"), size))
    base = Path(cfg["data_dir"]) / "ubfc_phys"
    derived = base / "derived"
    for person in json.loads(Path("splits/ubfc_phys.json").read_text())["dev"]:
        for task in UBFC_PHYS_TASKS:
            stem = person + "_" + task
            video = base / "raw" / person / ("vid_" + stem + ".avi")
            if video.exists():
                jobs.append((video, derived / (stem + "_raw.npz"), derived / (stem + "_crop.mkv"), size))
    with Pool(12) as pool:
        for name, n in pool.imap_unordered(write_crop, jobs):
            print(name, n, "frames", flush=True)


if __name__ == "__main__":
    main()
