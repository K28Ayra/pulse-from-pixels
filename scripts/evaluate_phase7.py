"""Phase 7: the live code on recorded video, and how fast it is.

    python scripts/evaluate_phase7.py replay     # dev UBFC-rPPG through pulsecam/live.py
    taskset -c 0-3 python scripts/evaluate_phase7.py speed   # 4 cores, like a laptop

speed runs one video twice. First as fast as it goes with the model inline, which
gives the time per stage. Then paced at 30 fps with the model in its thread, as
in the app, which gives the per-frame delay a viewer would see.

replay feeds every dev video frame by frame, with its real timestamps, exactly as
the app would get them from a camera, and scores the readings against the contact
reference and against the phase 5 offline numbers (outputs/phase6/windows.csv, raw).
Writes to outputs/phase7/.
"""

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pulsecam.config import load_config
from pulsecam.datasets import iter_frames, load_ubfc_subject
from pulsecam.face import FaceTracker
from pulsecam.live import LiveHeartRate, load_model
from pulsecam.reference import band_spectrum, reference_hr
from pulsecam.splits import load_split
from pulsecam.tracking import viterbi_track

sys.path.insert(0, "scripts")
from live import draw  # noqa: E402

OUT = Path("outputs/phase7")
ROOT = Path("data/ubfc_rppg")
N_WORKERS = 15
LAGS = [0, 2, 5]
SPEED_VIDEO = "subject1"


def replay_one(name):
    cfg = load_config(live_mode=True)
    subject = load_ubfc_subject(ROOT, name)
    times = subject["ppg_times"] - subject["ppg_times"][0]
    face = FaceTracker()
    model = load_model(cfg["live"]["model"], 1)
    heart = LiveHeartRate(cfg, face.find, model)
    rows = []
    for i, frame in enumerate(iter_frames(subject["video_path"])):
        if i >= len(times):
            break
        reading = heart.push(frame, times[i])
        if reading is None:
            continue
        end, length, spectrum = heart.recent_spectra[-1]
        reading["person"] = name
        reading["video_time_s"] = heart.start_time + reading["time_s"]
        reading["window_start_s"] = heart.start_time + reading["window_end_s"] - reading["window_s"]
        # the spectrum of the newest window, for re-tracking with other lags
        reading["new_start_s"] = heart.start_time + end - length
        reading["new_length_s"] = length
        reading["spectrum"] = spectrum
        rows.append(reading)
    face.close()
    return rows


def reference_of(name, cfg):
    subject = load_ubfc_subject(ROOT, name)
    ref_cfg = cfg["reference"]
    ref = reference_hr(
        subject["ppg_times"], subject["ppg"],
        length_s=cfg["window"]["length_s"], step_s=cfg["window"]["step_s"],
        band_bpm=cfg["hr_band_bpm"], min_beats=ref_cfg["min_beats"],
        max_ibi_cv=ref_cfg["max_ibi_cv"], max_disagree_bpm=ref_cfg["max_disagree_bpm"],
        max_gap_s=ref_cfg["max_gap_s"],
    )
    return pd.DataFrame({"t": ref["start_s"], "ref_hr": ref["hr_bpm"], "ok": ref["ok"]})


def score(table):
    error = (table["hr"] - table["ref_hr"]).abs()
    per_person = error.groupby(table["person"]).mean()
    return {
        "windows": len(table),
        "mae": error.mean(),
        "mean_person_mae": per_person.mean(),
        "within_5": 100 * (error <= 5).mean(),
    }


def replay():
    cfg = load_config(live_mode=True)
    names = load_split("splits/ubfc_rppg.json")["dev"]
    with Pool(N_WORKERS) as pool:
        results = pool.map(replay_one, names)

    # the bpm axis is the same for every window length (zero padding to 60 s)
    bpm, _ = band_spectrum(np.zeros(300), 30.0, cfg["hr_band_bpm"])
    tracking = cfg["deep_tracking"]
    full_windows = []
    first = []
    retracked = []
    for name, rows in zip(names, results):
        ref = reference_of(name, cfg)
        good = ref[ref["ok"]]
        readings = pd.DataFrame([{k: v for k, v in r.items() if k != "spectrum"} for r in rows])
        first_ok = good.iloc[0] if len(good) and good.iloc[0]["t"] == 0 else None
        first.append({
            "person": name,
            "first_reading_s": readings["video_time_s"].iloc[0],
            "first_hr": readings["hr"].iloc[0],
            "ref_first_10s": np.nan if first_ok is None else first_ok["ref_hr"],
        })

        # steady state: full-length windows that line up with the reference windows
        readings["t"] = readings["window_start_s"].round(2)
        full = readings[np.isclose(readings["window_s"], cfg["window"]["length_s"])]
        full = full.merge(good[["t", "ref_hr"]], on="t")
        full_windows.append(full)

        spectra = np.array([r["spectrum"] for r in rows])
        starts = np.array([r["new_start_s"] for r in rows]).round(2)
        lengths = np.array([r["new_length_s"] for r in rows])
        for lag in LAGS:
            hr = viterbi_track(bpm, spectra ** tracking["sharpen"], sigma_bpm=tracking["sigma_bpm"],
                               lag=lag, jump=tracking["jump"])
            part = pd.DataFrame({"person": name, "lag": lag, "t": starts, "length": lengths, "hr": hr})
            part = part[np.isclose(part["length"], cfg["window"]["length_s"])].merge(good[["t", "ref_hr"]], on="t")
            retracked.append(part)

    full = pd.concat(full_windows)
    first = pd.DataFrame(first)
    retracked = pd.concat(retracked)

    offline = pd.read_csv("outputs/phase6/windows.csv")
    offline = offline[(offline["dataset"] == "ubfc_rppg") & (offline["variant"] == "raw")]
    offline = offline[["person", "t", "hr_deep"]].rename(columns={"hr_deep": "hr"})
    offline["t"] = offline["t"].round(2)
    common = full.merge(offline[["person", "t"]], on=["person", "t"])

    rows = []
    rows.append({"what": "live, as shown", **score(common)})
    for lag in LAGS:
        part = retracked[retracked["lag"] == lag].merge(common[["person", "t"]], on=["person", "t"])
        rows.append({"what": "live spectra, lag " + str(lag), **score(part)})
    off = offline.merge(common[["person", "t", "ref_hr"]], on=["person", "t"])
    rows.append({"what": "phase 5 offline, lag 5", **score(off)})
    summary = pd.DataFrame(rows)

    first["first_error"] = (first["first_hr"] - first["ref_first_10s"]).abs()
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "replay.csv", index=False)
    first.to_csv(OUT / "first_reading.csv", index=False)
    full.to_csv(OUT / "replay_windows.csv", index=False)
    print(summary.round(3).to_string(index=False))
    print()
    print("first reading after", first["first_reading_s"].min(), "to", first["first_reading_s"].max(), "s")
    print("first reading error vs first 10 s reference: median", round(first["first_error"].median(), 2),
          "mean", round(first["first_error"].mean(), 2), "max", round(first["first_error"].max(), 2))


class Timed:
    """Wraps a function and adds up the seconds spent in it."""

    def __init__(self, function):
        self.function = function
        self.seconds = 0.0
        self.calls = 0

    def __call__(self, *args, **kwargs):
        started = time.perf_counter()
        result = self.function(*args, **kwargs)
        self.seconds += time.perf_counter() - started
        self.calls += 1
        return result


def speed():
    cfg = load_config(live_mode=True)
    video = ROOT / "DATASET_2" / SPEED_VIDEO / "vid.avi"
    face = FaceTracker()
    find = Timed(face.find)
    model = load_model(cfg["live"]["model"], cfg["live"]["threads"])
    run = Timed(model.run)
    model_wrapper = type("Model", (), {"run": staticmethod(run)})
    heart = LiveHeartRate(cfg, find, model_wrapper)

    capture = cv2.VideoCapture(str(video))
    stages = {"read": 0.0, "convert": 0.0, "push": 0.0, "draw": 0.0}
    n = 0
    reading = None
    total_started = time.perf_counter()
    while True:
        started = time.perf_counter()
        ok, frame = capture.read()
        stages["read"] += time.perf_counter() - started
        if not ok:
            break
        started = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stages["convert"] += time.perf_counter() - started
        started = time.perf_counter()
        new = heart.push(rgb, n / 30.0)
        stages["push"] += time.perf_counter() - started
        if new is not None:
            reading = new
        started = time.perf_counter()
        draw(frame, heart, reading, 30.0)
        stages["draw"] += time.perf_counter() - started
        n += 1
    total = time.perf_counter() - total_started
    face.close()

    rows = [{"stage": name, "ms_per_frame": 1000 * seconds / n} for name, seconds in stages.items()]
    rows.append({"stage": "  of push: face landmarks", "ms_per_frame": 1000 * find.seconds / n})
    rows.append({"stage": "  of push: model", "ms_per_frame": 1000 * run.seconds / n})
    rows.append({"stage": "all", "ms_per_frame": 1000 * total / n})
    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "speed.csv", index=False)
    print(table.round(2).to_string(index=False))
    print(n, "frames,", round(n / total, 1), "fps")
    print("face landmarks", round(1000 * find.seconds / find.calls, 1), "ms per call,", find.calls, "calls")
    print("model", round(1000 * run.seconds / run.calls, 1), "ms per call,", run.calls, "calls")

    paced = paced_run(cfg, video)
    paced.to_csv(OUT / "speed_paced.csv", index=False)
    late = paced["frame_ms"] > 1000 / 30
    print("paced at 30 fps, model in a thread:", len(paced), "frames,",
          "median", round(paced["frame_ms"].median(), 1), "ms,",
          "99th percentile", round(paced["frame_ms"].quantile(0.99), 1), "ms,",
          "max", round(paced["frame_ms"].max(), 1), "ms,",
          int(late.sum()), "frames over 33 ms,",
          int(paced["reading"].sum()), "readings")


def paced_run(cfg, video):
    face = FaceTracker()
    heart = LiveHeartRate(cfg, face.find, background=True)
    capture = cv2.VideoCapture(str(video))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    rows = []
    reading = None
    started = time.perf_counter()
    for n, frame in enumerate(frames):
        due = started + n / 30.0
        while time.perf_counter() < due:
            time.sleep(0.001)
        begin = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        new = heart.push(rgb, n / 30.0)
        if new is not None:
            reading = new
        draw(frame, heart, reading, 30.0)
        rows.append({"frame": n, "frame_ms": 1000 * (time.perf_counter() - begin), "reading": new is not None})
    face.close()
    return pd.DataFrame(rows)


if __name__ == "__main__":
    if sys.argv[1] == "speed":
        speed()
    else:
        replay()
