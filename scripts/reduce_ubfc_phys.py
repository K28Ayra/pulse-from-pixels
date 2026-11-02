"""Reduce UBFC-Phys people to landmarks, patch traces and a small face crop.

    python scripts/reduce_ubfc_phys.py s1 [s2 ...]

Reads data/ubfc_phys/raw/<person>/ (vid_<person>_T1.avi, bvp_<person>_T1.csv, ...)
and writes data/ubfc_phys/derived/:
  <person>_<task>_<variant>.npz  landmarks, found, rgb (frames, patches, 3), names, fps
  <person>_<task>_bvp.npy        the contact BVP, 64 Hz
  <person>_<task>[_<variant>]_crop.mkv   lossless 72x72 face crop of each variant

Variants are "raw", the x264 bitrates and the lighting gains from the config.
Landmarks are found again in every variant, as the live app would have to.
Nothing under raw/ is deleted here.
"""

import subprocess
import sys
import zlib
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from pulsecam.config import load_config
from pulsecam.datasets import UBFC_PHYS_TASKS, iter_frames, load_ubfc_phys_task
from pulsecam.face import BOX_EVERY, PATCHES, FaceTracker, crop_box, face_crop, mean_rgb_in_polygon

N_WORKERS = 12


def variant_names(cfg):
    stress = cfg["stress_variants"]
    names = ["raw"]
    for kbps in stress["bitrates_kbps"]:
        names.append("x264_" + str(kbps) + "k")
    for gain in stress["light_gains"]:
        names.append("light_" + str(int(round(gain * 100))))
    return names


def encode_x264(source, target, kbps, passthrough=False):
    rate = str(kbps) + "k"
    command = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source)]
    if passthrough:
        # keeps every frame; UBFC-rPPG timestamps are uneven and would be evened out otherwise
        command += ["-fps_mode", "passthrough"]
    command += [
        "-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p",
        "-b:v", rate, "-maxrate", rate, "-bufsize", str(2 * kbps) + "k",
        str(target),
    ]
    subprocess.run(command, check=True)


def dimmed(frames, gain, noise, seed):
    rng = np.random.default_rng(seed)
    for frame in frames:
        values = frame.astype(np.float32) * gain
        values = values + noise * rng.standard_normal(frame.shape).astype(np.float32)
        values = np.clip(np.round(values), 0, 255)
        yield values.astype(np.uint8)


def variant_frames(video, variant, cfg, seed_text, temp_video, passthrough=False):
    """Frames of one stress variant. temp_video is used for the x264 copy and
    must be deleted by the caller."""
    stress = cfg["stress_variants"]
    if variant.startswith("x264_"):
        kbps = int(variant[len("x264_"):-1])
        encode_x264(video, temp_video, kbps, passthrough)
        return iter_frames(temp_video)
    if variant.startswith("light_"):
        gain = int(variant[len("light_"):]) / 100
        seed = zlib.crc32(seed_text.encode())
        return dimmed(iter_frames(video), gain, stress["light_noise"], seed)
    return iter_frames(video)


def crop_name(stem, variant):
    if variant == "raw":
        return stem + "_crop.mkv"
    return stem + "_" + variant + "_crop.mkv"


def reduce_job(args):
    root, out_dir, person, task, variant, cfg = args
    data = load_ubfc_phys_task(root, person, task)
    fps = data["fps"]
    temp_video = out_dir / ("tmp_" + person + "_" + task + "_" + variant + ".mp4")
    frames = variant_frames(data["video_path"], variant, cfg, person + task + variant, temp_video)

    size = cfg["crop_size"]
    crop_path = out_dir / crop_name(person + "_" + task, variant)
    fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    writer = cv2.VideoWriter(str(crop_path), fourcc, fps, (size, size))
    if variant == "raw":
        np.save(out_dir / (person + "_" + task + "_bvp.npy"), data["ppg"])

    tracker = FaceTracker()
    names = list(PATCHES)
    all_points = []
    found = []
    rgb = []
    last_crop = np.zeros((size, size, 3), dtype=np.uint8)
    box = None
    for i, frame in enumerate(frames):
        points = tracker.find(frame, i / fps)
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

    out_path = out_dir / (person + "_" + task + "_" + variant + ".npz")
    np.savez_compressed(
        out_path,
        landmarks=np.array(all_points, dtype=np.float16),
        found=np.array(found),
        rgb=np.array(rgb),
        names=np.array(names),
        fps=fps,
    )
    return person, task, variant, int(np.sum(found)), len(found)


def main():
    cfg = load_config()
    base = Path(cfg["data_dir"]) / "ubfc_phys"
    root = base / "raw"
    out_dir = base / "derived"
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for person in sys.argv[1:]:
        for task in UBFC_PHYS_TASKS:
            for variant in variant_names(cfg):
                done = out_dir / (person + "_" + task + "_" + variant + ".npz")
                if done.exists():
                    continue
                jobs.append((root, out_dir, person, task, variant, cfg))

    with Pool(N_WORKERS) as pool:
        for person, task, variant, n_found, n_frames in pool.imap_unordered(reduce_job, jobs):
            share = 100 * n_found / max(n_frames, 1)
            print(person, task, variant, n_frames, "frames, face in", round(share, 1), "%", flush=True)


if __name__ == "__main__":
    main()
