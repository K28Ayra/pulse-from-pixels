"""Phase 5: does the deep candidate earn its place, and does every part of the
method help when it is taken away? Writes to outputs/phase5/.
"""

import itertools
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from evaluate_phase3 import RULE, predict, predict_final
from evaluate_phase4 import load_recordings
from pulsecam.config import load_config
from pulsecam.fusion import window_spectra
from pulsecam.metrics import harmonic_errors
from pulsecam.motion import motion_amount

OUT = Path("outputs/phase5")
PHYS_FOLDER = Path("ubfc_phys") / "derived"
RPPG_FOLDER = Path("ubfc_rppg") / "derived"
FULL_NAME = "full"


def deep_spectra(cfg, dataset, person, task, data):
    if dataset == "ubfc_rppg":
        path = Path(cfg["data_dir"]) / RPPG_FOLDER / (person + "_deep.npy")
    else:
        path = Path(cfg["data_dir"]) / PHYS_FOLDER / (person + "_" + task + "_deep.npy")
    pulse = np.load(path).astype(float)
    times = np.arange(len(pulse)) / 30.0
    if dataset == "ubfc_rppg":
        faces = np.load(Path(cfg["data_dir"]) / RPPG_FOLDER / (person + ".npz"))
        times = faces["times"][:len(pulse)] - faces["times"][0]
    bpm, spectra = window_spectra(times, pulse, data["start_s"], cfg["window"]["length_s"], cfg["hr_band_bpm"])
    return spectra


def only_candidates(data, wanted):
    """A copy of data keeping the candidates for which wanted(patch, method) is true."""
    keep = []
    for i, (patch, method) in enumerate(data["candidates"]):
        if wanted(patch, method):
            keep.append(i)
    smaller = dict(data)
    smaller["candidates"] = [data["candidates"][i] for i in keep]
    smaller["spectra"] = data["spectra"][keep]
    return smaller


def with_deep(data, deep):
    bigger = dict(data)
    bigger["candidates"] = list(data["candidates"]) + [("deep", "FactorizePhys")]
    bigger["spectra"] = np.concatenate([data["spectra"], deep[None]])
    return bigger


def all_variants(data, deep, cfg):
    tracking = cfg["tracking"]
    best = cfg["deep_tracking"]
    tau = tracking["motion_tau"]
    sigma = tracking["sigma_bpm"]
    lag = tracking["lag_s"]
    jump = tracking["jump"]
    sharpen = tracking["sharpen"]
    drop = tracking["drop_patches"]
    deep_only = with_deep(data, deep)
    deep_only = only_candidates(deep_only, lambda patch, method: patch == "deep")
    face_only = only_candidates(data, lambda patch, method: patch == "face")
    pos_only = only_candidates(data, lambda patch, method: method == "POS")
    chrom_only = only_candidates(data, lambda patch, method: method == "CHROM")
    plus_deep = with_deep(data, deep)
    return {
        FULL_NAME: predict_final(data, tracking),
        "without patches (face only)": predict(face_only, tau, sigma, lag, jump, sharpen),
        "without POS": predict(chrom_only, tau, sigma, lag, jump, sharpen, drop),
        "without CHROM": predict(pos_only, tau, sigma, lag, jump, sharpen, drop),
        "without SNR weights": predict(data, tau, sigma, lag, jump, sharpen, drop, rule="equal"),
        "without motion suppression": predict(data, None, sigma, lag, jump, sharpen, drop),
        "without tracker": predict(data, tau, None, None, 0.0, 1, drop),
        "without jump": predict(data, tau, sigma, lag, 0.0, sharpen, drop),
        "without sharpening": predict(data, tau, sigma, lag, jump, 1, drop),
        "with jaw and chin": predict(data, tau, sigma, lag, jump, sharpen),
        "plus deep": predict(plus_deep, tau, sigma, lag, jump, sharpen, drop),
        "deep alone": predict(deep_only, None, None, None),
        "deep tracked": predict(deep_only, None, sigma, lag, jump, sharpen),
        "deep final": predict(deep_only, None, best["sigma_bpm"], best["lag_s"], best["jump"], best["sharpen"]),
        "deep final, no sharpening": predict(deep_only, None, best["sigma_bpm"], best["lag_s"], best["jump"], 1),
    }


def removal_combinations(data, tracking):
    """Patches, POS, SNR weights and motion suppression taken away in every combination."""
    tau = tracking["motion_tau"]
    sigma = tracking["sigma_bpm"]
    lag = tracking["lag_s"]
    jump = tracking["jump"]
    sharpen = tracking["sharpen"]
    drop = tracking["drop_patches"]
    results = {}
    for no_patches, no_pos, no_snr, no_motion in itertools.product([False, True], repeat=4):
        removed = []
        chosen = data
        if no_patches:
            removed.append("patches")
            chosen = only_candidates(chosen, lambda patch, method: patch == "face")
        if no_pos:
            removed.append("POS")
            chosen = only_candidates(chosen, lambda patch, method: method == "CHROM")
        rule = RULE
        if no_snr:
            removed.append("SNR weights")
            rule = "equal"
        use_tau = tau
        if no_motion:
            removed.append("motion")
            use_tau = None
        if not removed:
            continue
        label = "remove " + " + ".join(removed)
        results[label] = predict(chosen, use_tau, sigma, lag, jump, sharpen, drop, rule=rule)
    return results


def score_recording(item):
    dataset, person, task, data, cfg = item
    deep = deep_spectra(cfg, dataset, person, task, data)
    predictions = all_variants(data, deep, cfg)
    predictions.update(removal_combinations(data, cfg["tracking"]))
    amount = motion_amount(data["motion"])
    ref_hr, ok = data["refs"]["strict"]
    # windows past the last full deep chunk are left out for every variant
    usable = ok & ~np.any(np.isnan(deep), axis=1)
    rows = []
    name = person + "_" + task
    for t in np.flatnonzero(usable):
        row = {
            "dataset": dataset,
            "person": person,
            "task": task,
            "listed": name in cfg["ubfc_phys_toolbox_excluded"],
            "ref_hr": ref_hr[t],
            "motion": amount[t],
        }
        for label, pred in predictions.items():
            row[label] = pred[t]
        rows.append(row)
    return rows


def groups(table):
    rppg = table[table["dataset"] == "ubfc_rppg"]
    phys = table[table["dataset"] == "ubfc_phys"]
    return {
        "UBFC-rPPG": rppg,
        "UBFC-Phys all": phys,
        "UBFC-Phys unlisted": phys[~phys["listed"]],
        "both, unlisted": table[~table["listed"]],
    }


def report(table, labels, threshold):
    rows = []
    for group, part in groups(table).items():
        people = part["person"].to_numpy()
        ref = part["ref_hr"].to_numpy()
        full_errors = np.abs(part[FULL_NAME].to_numpy() - ref)
        full = pd.Series(full_errors).groupby(people).mean()
        for label in labels:
            pred = part[label].to_numpy()
            errors = np.abs(pred - ref)
            per_person = pd.Series(errors).groupby(people).mean()
            p_value = np.nan
            p_variant = np.nan
            if label != FULL_NAME:
                p_value = stats.wilcoxon(full, per_person, alternative="less").pvalue
                p_variant = stats.wilcoxon(per_person, full, alternative="less").pvalue
            rows.append({
                "group": group,
                "variant": label,
                "n": len(part),
                "mae": errors.mean(),
                "mean_person_mae": per_person.mean(),
                "moving_mae": errors[part["motion"].to_numpy() > threshold].mean(),
                "within_5": 100 * np.mean(errors <= 5),
                "harmonic": harmonic_errors(pred, ref),
                "full_better": int((full < per_person).sum()),
                "full_worse": int((full > per_person).sum()),
                "p_full_better": p_value,
                "p_variant_better": p_variant,
            })
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    loaded, folds = load_recordings(cfg)
    jobs = []
    for dataset, person, task, data in loaded:
        jobs.append((dataset, person, task, data, cfg))
    with Pool(14) as pool:
        results = pool.map(score_recording, jobs, chunksize=1)
    rows = []
    for part in results:
        rows += part
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "windows.csv", index=False)
    labels = [c for c in table.columns if c not in ["dataset", "person", "task", "listed", "ref_hr", "motion"]]
    threshold = json.loads(Path("outputs/phase3/motion_threshold.json").read_text())["threshold"]
    summary = report(table, labels, threshold)
    summary.to_csv(OUT / "ablation.csv", index=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 100)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
