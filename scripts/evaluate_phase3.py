"""Phase 3: motion suppression and HR tracking on top of the phase 2 fusion.

    python scripts/evaluate_phase3.py

Runs on the UBFC-rPPG dev subjects and on whichever UBFC-Phys dev people
have been reduced (scripts/reduce_ubfc_phys.py). The settings are chosen by
person-wise cross-validation on UBFC-Phys, only among variants the live app
could run (no offline tracking).

The cut between still and motion windows is the 95th percentile of motion on
UBFC-rPPG dev, written to outputs/phase3/motion_threshold.json the first time
and reused after that, so it is fixed before any motion data is looked at.
Writes outputs/phase3/.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pulsecam import metrics
from pulsecam.config import load_config
from pulsecam.datasets import UBFC_PHYS_BVP_FS, UBFC_PHYS_TASKS, load_ubfc_subject
from pulsecam.fusion import fused_spectrum, snr_weights
from pulsecam.motion import motion_amount, motion_signals, motion_spectra, suppress_motion
from pulsecam.pipeline import all_snr, candidate_spectra, fill_gaps
from pulsecam.reference import reference_hr, spectral_reference_hr
from pulsecam.rppg import run_method
from pulsecam.sync import estimate_lag, hr_track
from pulsecam.tracking import viterbi_track

OUT = Path("outputs") / "phase3"
MIN_SYNC_R = 0.7
METHODS = ["CHROM", "POS"]
RULE = "snr2"
TAUS = [None, 1e-3, 1e-4, 1e-5, 1e-6]
SIGMAS = [None, 1.0, 2.0, 3.0, 5.0, 8.0]
# None = offline (whole recording), otherwise seconds of delay in the live app
LAGS = [None, 0, 2, 5]


def ubfc_rppg_subject(root, name, cfg, variant="raw"):
    win = cfg["window"]
    ref_cfg = cfg["reference"]
    band = cfg["hr_band_bpm"]
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

    if variant == "raw":
        patches = np.load(root / "derived" / (name + "_patches.npz"))
        faces = np.load(root / "derived" / (name + ".npz"))
    else:
        # a stress variant keeps landmarks and patch colour in one file
        patches = np.load(root / "derived" / (name + "_" + variant + ".npz"))
        faces = patches
    rgb = patches["rgb"].astype(float)
    patch_names = [str(p) for p in patches["names"]]
    n = min(len(rgb), len(times))
    starts = ref["start_s"]

    candidates, bpm, spectra = candidate_spectra(
        times[:n], rgb[:n], patch_names, METHODS, starts, win["length_s"], band)
    signals = motion_signals(faces["landmarks"][:n], faces["found"][:n])
    _, motion = motion_spectra(times[:n], signals, starts, win["length_s"], band)

    data = {
        "person": name,
        "task": "",
        "candidates": candidates,
        "bpm": bpm,
        "spectra": spectra,
        "motion": motion,
        "refs": {"strict": (ref["hr_bpm"], ref["ok"])},
        "start_s": starts,
    }
    return data


def person_lag(derived, person):
    """Video-after-BVP offset for one person, from whole-face POS on the rest task.

    T2 and T3 are left out because the wrist BVP is unreliable there and pooling
    them gave offsets up to the search limit. A weak match means the HR curve
    can't fix the offset, so the lag stays 0. The baseline is used, not the fused
    method, so the alignment can't favour the method being tested.
    Returns (lag used, lag found, r).
    """
    saved = np.load(derived / (person + "_T1_raw.npz"))
    fps = float(saved["fps"])
    face = list(saved["names"]).index("face")
    pulse = run_method("POS", fill_gaps(saved["rgb"][:, face].astype(float)), fps)
    bvp = np.load(derived / (person + "_T1_bvp.npy"))
    video_track = hr_track(np.arange(len(pulse)) / fps, pulse)
    contact_track = hr_track(np.arange(len(bvp)) / UBFC_PHYS_BVP_FS, bvp)
    found, r = estimate_lag([(video_track, contact_track)])
    used = found if r >= MIN_SYNC_R else 0.0
    return used, found, r


def ubfc_phys_task(derived, person, task, cfg, lag_s, variant="raw"):
    win = cfg["window"]
    ref_cfg = cfg["reference"]
    band = cfg["hr_band_bpm"]
    bvp = np.load(derived / (person + "_" + task + "_bvp.npy"))
    bvp_times = np.arange(len(bvp)) / UBFC_PHYS_BVP_FS
    ref = reference_hr(
        bvp_times,
        bvp,
        length_s=win["length_s"],
        step_s=win["step_s"],
        band_bpm=band,
        min_beats=ref_cfg["min_beats"],
        max_ibi_cv=ref_cfg["max_ibi_cv"],
        max_disagree_bpm=ref_cfg["max_disagree_bpm"],
        max_gap_s=ref_cfg["max_gap_s"],
    )
    loose = spectral_reference_hr(
        bvp_times,
        bvp,
        length_s=win["length_s"],
        step_s=win["step_s"],
        band_bpm=band,
        min_snr_db=ref_cfg["loose_min_snr_db"],
        max_gap_s=ref_cfg["max_gap_s"],
    )

    saved = np.load(derived / (person + "_" + task + "_" + variant + ".npz"))
    rgb = saved["rgb"].astype(float)
    patch_names = [str(p) for p in saved["names"]]
    times = np.arange(len(rgb)) / float(saved["fps"])
    # the BVP window starting at t is compared with the video window at t + lag
    starts = ref["start_s"] + lag_s
    inside = (starts >= 0) & (starts + win["length_s"] <= times[-1])
    starts = np.maximum(starts, 0)

    candidates, bpm, spectra = candidate_spectra(
        times, rgb, patch_names, METHODS, starts, win["length_s"], band)
    signals = motion_signals(saved["landmarks"], saved["found"])
    _, motion = motion_spectra(times, signals, starts, win["length_s"], band)

    data = {
        "person": person,
        "task": task,
        "candidates": candidates,
        "bpm": bpm,
        "spectra": spectra,
        "motion": motion,
        "refs": {
            "strict": (ref["hr_bpm"], ref["ok"] & inside),
            "loose": (loose["hr_bpm"], loose["ok"] & inside),
        },
        "start_s": starts,
    }
    return data


def predict(data, tau, sigma, lag, jump=0.0, sharpen=1, drop_patches=(), rule=RULE):
    keep = []
    for i, (patch, method) in enumerate(data["candidates"]):
        if patch not in drop_patches:
            keep.append(i)
    spectra = data["spectra"][keep]
    if tau is not None:
        spectra = suppress_motion(spectra, data["motion"], tau)
    weights = snr_weights(all_snr(data["bpm"], spectra), rule)
    combined = fused_spectrum(spectra, weights)
    if sigma is None:
        hr = data["bpm"][np.argmax(combined, axis=1)]
        no_data = weights.sum(axis=0) == 0
        return np.where(no_data, np.nan, hr)
    combined = combined ** sharpen
    return viterbi_track(data["bpm"], combined, sigma_bpm=sigma, lag=lag, jump=jump)


def predict_final(data, tracking):
    return predict(
        data,
        tracking["motion_tau"],
        tracking["sigma_bpm"],
        tracking["lag_s"],
        jump=tracking["jump"],
        sharpen=tracking["sharpen"],
        drop_patches=tracking["drop_patches"],
    )


def variant_label(tau, sigma, lag):
    parts = []
    if tau is None:
        parts.append("no suppression")
    else:
        parts.append("tau " + format(tau, ".0e"))
    if sigma is None:
        parts.append("no tracking")
    else:
        if lag is None:
            mode = "offline"
        else:
            mode = "lag " + str(lag)
        parts.append("track " + str(sigma) + " " + mode)
    return " / ".join(parts)


def baseline_face_pos(data):
    row = data["candidates"].index(("face", "POS"))
    spectra = np.nan_to_num(data["spectra"][row])
    hr = data["bpm"][np.argmax(spectra, axis=1)]
    return np.where(spectra.sum(axis=1) > 0, hr, np.nan)


def window_rows(dataset, data, label, pred):
    """One row per scored window and reference ("strict" and, on UBFC-Phys, "loose")."""
    rows = []
    amount = motion_amount(data["motion"])
    for reference, (ref_hr, ok) in data["refs"].items():
        for i in np.flatnonzero(ok):
            if np.isnan(pred[i]):
                continue
            rows.append({
                "dataset": dataset,
                "reference": reference,
                "subject": data["person"],
                "task": data["task"],
                "start_s": data["start_s"][i],
                "variant": label,
                "motion": amount[i],
                "ref_hr": ref_hr[i],
                "pred_hr": pred[i],
            })
    return rows


def summarise(windows, threshold):
    rows = []
    for (dataset, reference, label), used in windows.groupby(["dataset", "reference", "variant"]):
        moving = used["motion"] > threshold
        for part, chosen in [("all", used), ("still", used[~moving]), ("motion", used[moving])]:
            if len(chosen) == 0:
                continue
            row = {"dataset": dataset, "reference": reference, "variant": label, "windows": part}
            row.update(metrics.summary(chosen["pred_hr"].to_numpy(), chosen["ref_hr"].to_numpy()))
            rows.append(row)
    return pd.DataFrame(rows)


def per_person_table(windows):
    rows = []
    phys = windows[windows["dataset"] == "ubfc_phys"]
    for (reference, label, person, task), used in phys.groupby(["reference", "variant", "subject", "task"]):
        row = {"reference": reference, "variant": label, "person": person, "task": task}
        errors = (used["pred_hr"] - used["ref_hr"]).abs()
        row["n"] = len(used)
        row["mae"] = errors.mean()
        row["median_error"] = errors.median()
        rows.append(row)
    return pd.DataFrame(rows)


def cross_validate(windows, folds):
    """Pick the variant with the lowest MAE on the other folds' UBFC-Phys
    windows (strict reference), score it on the held-out fold under both
    references. Offline tracking is left out."""
    phys = windows[windows["dataset"] == "ubfc_phys"].copy()
    phys["fold"] = phys["subject"].map(folds)
    allowed = ~phys["variant"].str.contains("offline") & (phys["variant"] != "face POS")
    phys_allowed = phys[allowed]
    parts = []
    choices = []
    for fold in sorted(phys["fold"].unique()):
        train = phys_allowed[(phys_allowed["fold"] != fold) & (phys_allowed["reference"] == "strict")]
        if len(train) == 0:
            continue
        errors = (train["pred_hr"] - train["ref_hr"]).abs()
        train_mae = errors.groupby(train["variant"]).mean()
        best = train_mae.idxmin()
        choices.append({"fold": fold, "chosen": best, "train_mae": train_mae[best]})
        held_out = phys_allowed[(phys_allowed["fold"] == fold) & (phys_allowed["variant"] == best)]
        parts.append(held_out)
    if not parts:
        return None, pd.DataFrame(choices)
    cv = pd.concat(parts).drop(columns="fold")
    cv["variant"] = "CV-chosen"
    return cv, pd.DataFrame(choices)


def main():
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    root = Path(cfg["data_dir"]) / "ubfc_rppg"
    split = json.loads(Path("splits/ubfc_rppg.json").read_text())

    per_subject = {}
    for name in split["dev"]:
        per_subject[("ubfc_rppg", name)] = ubfc_rppg_subject(root, name, cfg)
        print(name, flush=True)

    derived = Path(cfg["data_dir"]) / "ubfc_phys" / "derived"
    phys_split = json.loads(Path("splits/ubfc_phys.json").read_text())
    lags = []
    for person in phys_split["dev"]:
        if not (derived / (person + "_T3_raw.npz")).exists():
            continue
        lag_s, found, r = person_lag(derived, person)
        lags.append({"person": person, "lag_s": lag_s, "found_s": found, "r": r})
        print(person, "lag used", lag_s, "found", found, "r =", round(r, 2), flush=True)
        for task in UBFC_PHYS_TASKS:
            data = ubfc_phys_task(derived, person, task, cfg, lag_s)
            per_subject[("ubfc_phys", person + "_" + task)] = data
    pd.DataFrame(lags).to_csv(OUT / "ubfc_phys_lags.csv", index=False)

    threshold_path = OUT / "motion_threshold.json"
    if threshold_path.exists():
        threshold = json.loads(threshold_path.read_text())["threshold"]
    else:
        amounts = []
        for (dataset, name), data in per_subject.items():
            if dataset == "ubfc_rppg":
                ok = data["refs"]["strict"][1]
                amounts.append(motion_amount(data["motion"])[ok])
        threshold = float(np.percentile(np.concatenate(amounts), 95))
        threshold_path.write_text(json.dumps({"threshold": threshold, "from": "UBFC-rPPG dev, 95th percentile"}))
    print("motion threshold", threshold)

    rows = []
    for (dataset, name), data in per_subject.items():
        rows += window_rows(dataset, data, "face POS", baseline_face_pos(data))
        for tau in TAUS:
            for sigma in SIGMAS:
                lags = [None] if sigma is None else LAGS
                for lag in lags:
                    pred = predict(data, tau, sigma, lag)
                    label = variant_label(tau, sigma, lag)
                    rows += window_rows(dataset, data, label, pred)
        rows += window_rows(dataset, data, "final", predict_final(data, cfg["tracking"]))
    windows = pd.DataFrame(rows)
    cv, choices = cross_validate(windows, phys_split["folds"])
    choices.to_csv(OUT / "cv_choices.csv", index=False)
    if cv is not None:
        windows = pd.concat([windows, cv])
        print(choices.to_string(index=False))
    windows.to_csv(OUT / "windows.csv", index=False)

    # the same windows again without the recordings rPPG-Toolbox excludes
    phys = windows[windows["dataset"] == "ubfc_phys"]
    recording = phys["subject"] + "_" + phys["task"]
    unlisted = phys[~recording.isin(cfg["ubfc_phys_toolbox_excluded"])].copy()
    unlisted["dataset"] = "ubfc_phys_unlisted"
    table = summarise(pd.concat([windows, unlisted]), threshold)
    table.to_csv(OUT / "summary.csv", index=False)
    per_person_table(windows).to_csv(OUT / "per_person.csv", index=False)
    pd.set_option("display.width", 220)
    columns = ["dataset", "reference", "windows", "variant", "n", "mae", "within_ec13", "harmonic"]
    for (dataset, reference), part in table.groupby(["dataset", "reference"]):
        shown = part[part["windows"] == "all"]
        print(shown.sort_values("mae")[columns].head(10).round(3).to_string(index=False))
    key = ["face POS", "no suppression / no tracking", "CV-chosen", "final"]
    picked = table[table["variant"].isin(key)]
    picked = picked.sort_values(["dataset", "reference", "windows", "variant"])
    print(picked[columns].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
