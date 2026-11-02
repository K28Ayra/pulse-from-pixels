"""Window-level evaluation of all phase 1 baselines on the UBFC-rPPG dev subjects.

    python scripts/evaluate_phase1.py

Needs data/ubfc_rppg/derived (scripts/extract_faces.py), and for the toolbox
rows outputs/phase1/toolbox_bvp (scripts/toolbox_bvp.py) and the deep models'
saved outputs (scripts/run_toolbox.py). Missing inputs are skipped.

Every method is scored the same way: spectral peak of its pulse signal in
each 10 s window, using the real frame timestamps, against the PPG reference
of the same window. Only windows where the reference is ok count.
"""

import glob
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal, sparse
from scipy.sparse.linalg import spsolve

from pulsecam import metrics
from pulsecam.config import load_config
from pulsecam.datasets import load_ubfc_subject
from pulsecam.reference import reference_hr
from pulsecam.rppg import run_method, windowed_pulse_hr

OUT = Path("outputs") / "phase1"
DEEP_JOBS = ["PhysNet", "TS-CAN", "EfficientPhys", "FactorizePhys"]


def fill_gaps(rgb):
    """Linear interpolation over frames where no face was found."""
    filled = rgb.copy()
    index = np.arange(len(rgb))
    missing = np.isnan(rgb[:, 0])
    if missing.all():
        return filled
    for channel in range(3):
        known = ~missing
        filled[missing, channel] = np.interp(index[missing], index[known], rgb[known, channel])
    return filled


def frame_rate(times):
    return (len(times) - 1) / (times[-1] - times[0])


def load_deep_outputs(job):
    """subject name -> pulse signal, or {} if the job hasn't run."""
    paths = glob.glob(str(OUT / "toolbox" / job / "**" / "*_outputs.pickle"), recursive=True)
    if not paths:
        return {}
    with open(paths[0], "rb") as handle:
        saved = pickle.load(handle)

    signals = {}
    for name, chunks in saved["predictions"].items():
        pieces = []
        for index in sorted(chunks):
            piece = np.asarray(chunks[index].cpu()).ravel()
            pieces.append(piece)
        pulse = np.concatenate(pieces)
        # these models predict the frame-to-frame change of the pulse
        if saved["label_type"] == "DiffNormalized":
            pulse = np.cumsum(pulse)
        signals[name] = pulse
    return signals


def pulse_signals(name, derived_dir, deep):
    """method label -> pulse signal, one value per frame from frame 0."""
    signals = {}

    derived_path = derived_dir / (name + ".npz")
    if derived_path.exists():
        derived = np.load(derived_path)
        rgb = fill_gaps(derived["face_rgb"].astype(float))
        fs = frame_rate(derived["times"])
        for method in ["GREEN", "CHROM", "POS"]:
            signals[method + " mine, face oval"] = run_method(method, rgb, fs)

    toolbox_path = OUT / "toolbox_bvp" / (name + ".npz")
    if toolbox_path.exists():
        toolbox = np.load(toolbox_path)
        for method in ["GREEN", "ICA", "CHROM", "LGI", "PBV", "OMIT", "POS"]:
            signals[method + " toolbox"] = np.asarray(toolbox[method]).ravel()
        # the toolbox assumes 30 fps for every video, so does this row
        signals["POS mine, toolbox crop"] = run_method("POS", toolbox["mean_rgb"], 30.0)

    for job in DEEP_JOBS:
        if name in deep[job]:
            signals[job + " (PURE)"] = deep[job][name]
    return signals


def toolbox_post_processing(pulse, fs):
    """What the toolbox's POS_WANG does after the POS step: smoothness-priors
    detrend (lambda 100, Tarvainen 2002) and a first-order 0.75-3 Hz filter."""
    n = len(pulse)
    ones = np.ones(n - 2)
    second_diff = sparse.diags([ones, -2 * ones, ones], [0, 1, 2], shape=(n - 2, n))
    system = sparse.identity(n) + 100 ** 2 * (second_diff.T @ second_diff)
    trend = spsolve(system.tocsc(), pulse)
    detrended = pulse - trend
    b, a = signal.butter(1, [0.75 / fs * 2, 3 / fs * 2], btype="bandpass")
    return signal.filtfilt(b, a, detrended)


def pos_agreement(name):
    """Correlation of my POS and the toolbox POS on the same crop and with the
    same post-processing, so only the POS step itself is compared."""
    toolbox = np.load(OUT / "toolbox_bvp" / (name + ".npz"))
    theirs = np.asarray(toolbox["POS"]).ravel()
    mine = run_method("POS", toolbox["mean_rgb"], 30.0)
    mine = toolbox_post_processing(mine, 30.0)
    # filter edges and the first POS window are not comparable
    edge = int(2 * 30)
    r = np.corrcoef(mine[edge:-edge], theirs[edge:-edge])[0, 1]
    return float(r)


def summarise(windows):
    summaries = []
    for label, used in windows.groupby("method", sort=False):
        pred = used["pred_hr"].to_numpy()
        ref_hr = used["ref_hr"].to_numpy()
        row = {"method": label, "subjects": used["subject"].nunique()}
        row.update(metrics.summary(pred, ref_hr))
        low, high = metrics.bootstrap_ci(pred, ref_hr, used["subject"].to_numpy())
        row["mae_ci_low"] = low
        row["mae_ci_high"] = high
        summaries.append(row)
    return pd.DataFrame(summaries).sort_values("mae")


def main():
    cfg = load_config()
    root = Path(cfg["data_dir"]) / "ubfc_rppg"
    derived_dir = root / "derived"
    split = json.loads(Path("splits/ubfc_rppg.json").read_text())
    win = cfg["window"]
    ref_cfg = cfg["reference"]
    band = cfg["hr_band_bpm"]

    deep = {}
    for job in DEEP_JOBS:
        deep[job] = load_deep_outputs(job)

    rows = []
    agreement = []
    for name in split["dev"]:
        subject = load_ubfc_subject(root, name)
        times = subject["ppg_times"] - subject["ppg_times"][0]
        ref = reference_hr(
            subject["ppg_times"],
            subject["ppg"],
            length_s=win["length_s"],
            step_s=win["step_s"],
            band_bpm=band,
            min_beats=ref_cfg["min_beats"],
            max_ibi_cv=ref_cfg["max_ibi_cv"],
            max_disagree_bpm=ref_cfg["max_disagree_bpm"],
            max_gap_s=ref_cfg["max_gap_s"],
        )
        starts = ref["start_s"]

        signals = pulse_signals(name, derived_dir, deep)
        for label, pulse in signals.items():
            n = min(len(pulse), len(times))
            pred = windowed_pulse_hr(times[:n], pulse[:n], starts, win["length_s"], band)
            for i in range(len(starts)):
                rows.append({
                    "subject": name,
                    "method": label,
                    "start_s": starts[i],
                    "ref_hr": ref["hr_bpm"][i],
                    "ref_ok": bool(ref["ok"][i]),
                    "pred_hr": pred[i],
                })

        if (OUT / "toolbox_bvp" / (name + ".npz")).exists():
            r = pos_agreement(name)
            agreement.append({"subject": name, "pos_r": r})
        print(name, len(signals), "methods", flush=True)

    windows = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    windows.to_csv(OUT / "windows.csv", index=False)

    usable = windows[windows["ref_ok"] & windows["pred_hr"].notna()]
    table = summarise(usable)
    table.to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(table.round(2).to_string(index=False))

    # the deep models drop the frames after their last full chunk, so they
    # have fewer windows; this table only keeps windows every method has
    key = ["subject", "start_s"]
    n_methods = usable["method"].nunique()
    counts = usable.groupby(key)["method"].transform("nunique")
    common = usable[counts == n_methods]
    table = summarise(common)
    table.to_csv(OUT / "summary_common_windows.csv", index=False)
    print()
    print("windows every method has:", len(common) // n_methods)
    print(table.round(2).to_string(index=False))

    if agreement:
        agree = pd.DataFrame(agreement)
        agree.to_csv(OUT / "pos_agreement.csv", index=False)
        print("my POS vs toolbox POS, r per subject: median",
              round(agree["pos_r"].median(), 4), "min", round(agree["pos_r"].min(), 4))


if __name__ == "__main__":
    main()
