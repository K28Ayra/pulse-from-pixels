"""Readers for the datasets. Each reader returns plain numpy arrays."""

from pathlib import Path

import cv2
import numpy as np

# From Agreement.xlsx in the UBFC-rPPG download: these people's faces must
# not appear in any publication, presentation or report. Subject 21 is not
# in DATASET_2 at all but is listed for completeness.
UBFC_NO_FACE_IMAGES = ["subject21", "subject27"]


def ubfc_subject_number(name):
    return int(name.replace("subject", ""))


def list_ubfc_subjects(root):
    """Subject folder names under <root>/DATASET_2, sorted by number."""
    folder = Path(root) / "DATASET_2"
    names = []
    for path in folder.iterdir():
        if path.is_dir() and path.name.startswith("subject"):
            names.append(path.name)
    names.sort(key=ubfc_subject_number)
    return names


def read_ubfc_ground_truth(path):
    """ground_truth.txt has three lines: PPG, HR from the oximeter, time in seconds."""
    rows = np.loadtxt(path)
    ppg = rows[0]
    hr = rows[1]
    times = rows[2]
    return times, ppg, hr


def load_ubfc_subject(root, name):
    folder = Path(root) / "DATASET_2" / name
    times, ppg, device_hr = read_ubfc_ground_truth(folder / "ground_truth.txt")
    subject = {
        "name": name,
        "video_path": folder / "vid.avi",
        "ppg_times": times,
        "ppg": ppg,
        "device_hr": device_hr,
    }
    return subject


def video_info(path):
    """fps, frame count, width and height as the container reports them.

    The frame count from the header is not always right for AVI files, so
    count_frames() is used when it matters.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError("cannot open video: " + str(path))
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def iter_frames(path):
    """Yield RGB frames one at a time, so a long video never sits in memory."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError("cannot open video: " + str(path))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield rgb
    cap.release()


def count_frames(path):
    n = 0
    for _ in iter_frames(path):
        n += 1
    return n


# UBFC-Phys: one folder per person, three tasks (T1 rest, T2 speech, T3 arithmetic).
# The BVP comes from an Empatica E4 at 64 Hz, one value per line, no timestamps.
UBFC_PHYS_TASKS = ["T1", "T2", "T3"]
UBFC_PHYS_BVP_FS = 64.0


def load_ubfc_phys_task(root, name, task):
    """Video path, fps and BVP with times in seconds for one person and task.

    The dataset has no frame timestamps, so frames are taken as evenly spaced
    at the container fps, starting at the same moment as the BVP.
    """
    folder = Path(root) / name
    video_path = folder / ("vid_" + name + "_" + task + ".avi")
    bvp_path = folder / ("bvp_" + name + "_" + task + ".csv")
    if not video_path.exists() or not bvp_path.exists():
        raise FileNotFoundError("missing video or bvp for " + name + " " + task + " in " + str(folder))
    bvp = np.loadtxt(bvp_path, delimiter=",", ndmin=1)
    times = np.arange(len(bvp)) / UBFC_PHYS_BVP_FS
    info = video_info(video_path)
    task_data = {
        "name": name,
        "task": task,
        "video_path": video_path,
        "fps": info["fps"],
        "ppg_times": times,
        "ppg": bvp,
    }
    return task_data


MPU_FPS = 60.0


def csv_rows(path):
    with open(path) as f:
        return sum(1 for line in f) - 1


def list_mpu_recordings(root):
    """(name, video, csv) for each MPU-rPPG sample video. The file numbers don't
    match, so a video is paired with the csv that has one row per frame."""
    root = Path(root)
    by_rows = {}
    for path in sorted(root.glob("*_Output.csv")):
        by_rows[csv_rows(path)] = path
    recordings = []
    for video in sorted(root.glob("*_Output.mp4")):
        n_frames = count_frames(video)
        if n_frames in by_rows:
            name = "mpu" + video.name.split("_")[0]
            recordings.append((name, video, by_rows[n_frames]))
    return recordings


def load_mpu_recording(video_path, csv_path):
    """Video path and the finger PPG, one sample per frame at 60 fps.

    The csv columns are Count, PPG, HR, SPO2; HR is the oximeter's own number.
    """
    table = np.loadtxt(csv_path, delimiter=",", skiprows=1, ndmin=2)
    return {
        "video_path": Path(video_path),
        "fps": MPU_FPS,
        "ppg_times": np.arange(len(table)) / MPU_FPS,
        "ppg": table[:, 1],
        "device_hr": table[:, 2],
    }
