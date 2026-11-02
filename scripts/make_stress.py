"""Stress variants for phase 6 on dev people.

    python scripts/make_stress.py

UBFC-Phys already has landmarks and patch colour for every variant (from
reduce_ubfc_phys.py), so only the 72x72 crops are made, from those landmarks.
UBFC-rPPG gets everything: the face is found again in each variant, as the
live app would have to. Writes <name>_<variant>.npz and <name>_<variant>_crop.mkv
in the derived folders and skips what exists. Same x264 bitrates for both
datasets, so UBFC-rPPG (640x480) gets more bits per pixel than UBFC-Phys.
"""

import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from pulsecam.config import load_config
from pulsecam.datasets import UBFC_PHYS_TASKS
from pulsecam.face import BOX_EVERY, PATCHES, FaceTracker, crop_box, face_crop, mean_rgb_in_polygon
from reduce_ubfc_phys import crop_name, variant_frames, variant_names

N_WORKERS = 20


def crop_writer(path, fps, size):
    fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    return cv2.VideoWriter(str(path), fourcc, fps, (size, size))


def phys_crops(args):
    """Crops of one UBFC-Phys variant from its saved landmarks."""
    derived, video, stem, variant, cfg = args
    saved = np.load(derived / (stem + "_" + variant + ".npz"))
    landmarks = saved["landmarks"].astype(np.float64)
    found = saved["found"]
    size = cfg["crop_size"]
    temp_video = derived / ("tmp_" + stem + "_" + variant + ".mp4")
    person, task = stem.split("_")
    frames = variant_frames(video, variant, cfg, person + task + variant, temp_video)
    writer = crop_writer(derived / crop_name(stem, variant), float(saved["fps"]), size)
    last_crop = np.zeros((size, size, 3), dtype=np.uint8)
    box = None
    n = 0
    for i, frame in enumerate(frames):
        if i >= len(found):
            break
        if found[i]:
            if box is None or i % BOX_EVERY == 0:
                box = crop_box(landmarks[i])
            last_crop = face_crop(frame, box, size)
        writer.write(cv2.cvtColor(last_crop, cv2.COLOR_RGB2BGR))
        n += 1
    writer.release()
    if temp_video.exists():
        temp_video.unlink()
    return stem, variant, n, len(found)


def rppg_variant(args):
    """Landmarks, patch colour and crops of one UBFC-rPPG variant."""
    derived, video, name, variant, cfg = args
    times = np.load(derived / (name + ".npz"))["times"]
    fps = (len(times) - 1) / (times[-1] - times[0])
    size = cfg["crop_size"]
    temp_video = derived / ("tmp_" + name + "_" + variant + ".mp4")
    frames = variant_frames(video, variant, cfg, name + variant, temp_video, passthrough=True)
    writer = crop_writer(derived / crop_name(name, variant), fps, size)
    tracker = FaceTracker()
    names = list(PATCHES)
    all_points = []
    found = []
    rgb = []
    last_crop = np.zeros((size, size, 3), dtype=np.uint8)
    box = None
    for i, frame in enumerate(frames):
        if i >= len(times):
            break
        points = tracker.find(frame, times[i] - times[0])
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
    if temp_video.exists():
        temp_video.unlink()
    n = len(found)
    np.savez_compressed(
        derived / (name + "_" + variant + ".npz"),
        times=times[:n],
        landmarks=np.array(all_points, dtype=np.float16),
        found=np.array(found),
        rgb=np.array(rgb),
        names=np.array(names),
    )
    return name, variant, n, len(times)


def main():
    cfg = load_config()
    variants = [v for v in variant_names(cfg) if v != "raw"]
    rppg_jobs = []
    root = Path(cfg["data_dir"]) / "ubfc_rppg"
    for name in json.loads(Path("splits/ubfc_rppg.json").read_text())["dev"]:
        for variant in variants:
            if (root / "derived" / (name + "_" + variant + ".npz")).exists():
                continue
            video = root / "DATASET_2" / name / "vid.avi"
            rppg_jobs.append((root / "derived", video, name, variant, cfg))
    phys_jobs = []
    base = Path(cfg["data_dir"]) / "ubfc_phys"
    for person in json.loads(Path("splits/ubfc_phys.json").read_text())["dev"]:
        for task in UBFC_PHYS_TASKS:
            stem = person + "_" + task
            for variant in variants:
                if (base / "derived" / crop_name(stem, variant)).exists():
                    continue
                video = base / "raw" / person / ("vid_" + stem + ".avi")
                phys_jobs.append((base / "derived", video, stem, variant, cfg))
    print(len(rppg_jobs), "UBFC-rPPG jobs,", len(phys_jobs), "UBFC-Phys jobs", flush=True)
    with Pool(N_WORKERS) as pool:
        for result in pool.imap_unordered(rppg_variant, rppg_jobs):
            print(*result, flush=True)
        for result in pool.imap_unordered(phys_crops, phys_jobs):
            print(*result, flush=True)


if __name__ == "__main__":
    main()
