"""Sanity checks on UBFC-rPPG before any method touches it.

For every subject: does the video open, how many frames, does the PPG cover
the same time span, and does my reference HR from the PPG agree with the HR
the oximeter itself reported. No rPPG method runs here, so this is safe to
do on test subjects too.

    python scripts/check_ubfc.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from pulsecam.config import load_config
from pulsecam.datasets import count_frames, list_ubfc_subjects, load_ubfc_subject, video_info
from pulsecam.reference import reference_hr


# The HR the oximeter displays is averaged and runs about 3 s behind the
# pulse (on the first five subjects the gap to my PPG-based HR was smallest
# at a 3 s shift), so it is compared 3 s later.
DEVICE_LAG_S = 3


def device_hr_per_window(times, device_hr, starts, length_s, lag_s):
    times = times - times[0]
    # subject11's oximeter HR drops to 1 bpm at times, those readings are invalid
    valid = device_hr >= 30
    means = np.full(len(starts), np.nan)
    for i, start in enumerate(starts):
        first = start + lag_s
        inside = (times >= first) & (times < first + length_s) & valid
        enough = inside.sum() >= 0.8 * length_s * 25
        if enough:
            means[i] = np.mean(device_hr[inside])
    return means


def check_subject(root, name, cfg):
    subject = load_ubfc_subject(root, name)
    info = video_info(subject["video_path"])
    n_frames = count_frames(subject["video_path"])

    times = subject["ppg_times"]
    ppg_span = times[-1] - times[0]
    video_span = n_frames / info["fps"]

    win = cfg["window"]
    ref_cfg = cfg["reference"]
    ref = reference_hr(
        times,
        subject["ppg"],
        length_s=win["length_s"],
        step_s=win["step_s"],
        band_bpm=cfg["hr_band_bpm"],
        min_beats=ref_cfg["min_beats"],
        max_ibi_cv=ref_cfg["max_ibi_cv"],
        max_disagree_bpm=ref_cfg["max_disagree_bpm"],
        max_gap_s=ref_cfg["max_gap_s"],
    )
    starts = ref["start_s"]
    length = win["length_s"]
    dev_now = device_hr_per_window(times, subject["device_hr"], starts, length, 0)
    dev_late = device_hr_per_window(times, subject["device_hr"], starts, length, DEVICE_LAG_S)

    ok = ref["ok"]
    hr = ref["hr_bpm"]
    diff_now = np.abs(hr[ok] - dev_now[ok])
    diff_late = np.abs(hr[ok] - dev_late[ok])
    diff_spec = np.abs(hr[ok] - ref["spectral_hr_bpm"][ok])

    row = {
        "subject": name,
        "fps_header": info["fps"],
        "fps_from_ppg_times": (len(times) - 1) / ppg_span,
        "frames_header": info["n_frames"],
        "frames_counted": n_frames,
        "ppg_samples": len(times),
        "frame_minus_ppg": n_frames - len(times),
        "video_s": video_span,
        "ppg_s": ppg_span,
        "width": info["width"],
        "height": info["height"],
        "windows": len(ok),
        "windows_ok": int(ok.sum()),
        "ref_hr_min": float(np.nanmin(hr)),
        "ref_hr_max": float(np.nanmax(hr)),
        "ref_vs_device_mae": float(np.nanmean(diff_now)),
        "ref_vs_device_3s_mae": float(np.nanmean(diff_late)),
        "peak_vs_spectral_mae": float(np.mean(diff_spec)),
    }
    return row


def main():
    cfg = load_config()
    root = Path(cfg["data_dir"]) / "ubfc_rppg"
    out_dir = Path(cfg["output_dir"]) / "phase0"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in list_ubfc_subjects(root):
        folder = root / "DATASET_2" / name
        if not (folder / "vid.avi").exists():
            print(name, "no vid.avi yet, skipped")
            continue
        row = check_subject(root, name, cfg)
        print(name, row["frames_counted"], "frames,",
              row["windows_ok"], "/", row["windows"], "windows ok,",
              "vs device", round(row["ref_vs_device_mae"], 2),
              "(3 s later:", round(row["ref_vs_device_3s_mae"], 2), ")")
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "ubfc_check.csv", index=False)
    print()
    print("subjects:", len(table))
    print("frames == ppg samples:", int((table["frame_minus_ppg"] == 0).sum()))
    print("fps range:", table["fps_from_ppg_times"].min().round(2), "-",
          table["fps_from_ppg_times"].max().round(2))
    print("windows ok:", table["windows_ok"].sum(), "/", table["windows"].sum())
    print("ref vs device MAE:", round(table["ref_vs_device_mae"].mean(), 2),
          " 3 s later:", round(table["ref_vs_device_3s_mae"].mean(), 2))
    print("peak vs spectral MAE:", round(table["peak_vs_spectral_mae"].mean(), 2))


if __name__ == "__main__":
    main()
