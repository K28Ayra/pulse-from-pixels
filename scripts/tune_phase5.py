"""Tracking settings when FactorizePhys is the main signal, about 15 minutes.

blend is the deep spectrum's share against the fused classical spectrum. Same
rules as tune_phase3.py: every person counts the same, listed recordings don't
count. Writes outputs/phase5/tuning.csv, tuning_cv.csv and deep_flag.csv.
"""

import itertools
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from evaluate_phase3 import RULE
from evaluate_phase4 import load_recordings
from evaluate_phase5 import deep_spectra, only_candidates
from pulsecam.config import load_config
from pulsecam.fusion import fused_spectrum, snr_weights
from pulsecam.motion import suppress_motion
from pulsecam.pipeline import all_snr
from pulsecam.tracking import viterbi_track

OUT = Path("outputs/phase5")
NAMES = ["blend", "tau", "sigma", "lag", "jump", "sharpen"]
BLENDS = [1.0, 0.75, 0.5]
TAUS = [None, 1e-4]
SIGMAS = [2, 3, 5]
LAGS = [2, 5]
JUMPS = [0.0, 1e-3, 1e-2]
SHARPEN = [1, 2]
GRID = list(itertools.product(BLENDS, TAUS, SIGMAS, LAGS, JUMPS, SHARPEN))
BAD_BPM = 5.0


def unit_rows(spectra):
    totals = spectra.sum(axis=1, keepdims=True)
    totals = np.where(totals > 0, totals, 1.0)
    return spectra / totals


def classical_spectrum(data, tau, drop):
    chosen = only_candidates(data, lambda patch, method: patch not in drop)
    spectra = chosen["spectra"]
    if tau is not None:
        spectra = suppress_motion(spectra, data["motion"], tau)
    weights = snr_weights(all_snr(data["bpm"], spectra), RULE)
    return unit_rows(fused_spectrum(spectra, weights))


def score_recording(item):
    dataset, person, task, data, cfg = item
    deep = deep_spectra(cfg, dataset, person, task, data)
    ref_hr, ok = data["refs"]["strict"]
    usable = ok & ~np.any(np.isnan(deep), axis=1)
    drop = cfg["tracking"]["drop_patches"]
    plain = {}
    classical = {}
    for tau in TAUS:
        if tau is None:
            plain[tau] = unit_rows(np.nan_to_num(deep))
        else:
            plain[tau] = unit_rows(np.nan_to_num(suppress_motion(deep, data["motion"], tau)))
        classical[tau] = classical_spectrum(data, tau, drop)
    errors = np.zeros((len(GRID), usable.sum()), dtype=np.float32)
    for g, (blend, tau, sigma, lag, jump, sharpen) in enumerate(GRID):
        combined = blend * plain[tau] + (1 - blend) * classical[tau]
        hr = viterbi_track(data["bpm"], combined ** sharpen, sigma_bpm=sigma, lag=lag, jump=jump)
        errors[g] = np.abs(hr[usable] - ref_hr[usable])
    # the flag for the deep signal alone, one setting close to the phase 3 one
    hr = viterbi_track(data["bpm"], plain[None] ** 2, sigma_bpm=2, lag=5, jump=0.01)
    near = np.abs(data["bpm"][None, :] - hr[:, None]) <= BAD_BPM
    share = (plain[None] * near).sum(axis=1)
    return dataset, person, task, errors, hr[usable], ref_hr[usable], share[usable]


def main():
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    excluded = cfg["ubfc_phys_toolbox_excluded"]
    loaded, folds = load_recordings(cfg)
    jobs = []
    for dataset, person, task, data in loaded:
        jobs.append((dataset, person, task, data, cfg))
    with Pool(14) as pool:
        results = pool.map(score_recording, jobs, chunksize=1)

    by_person = {}
    for dataset, person, task, errors, hr, ref, share in results:
        if person + "_" + task not in excluded and errors.shape[1] > 0:
            by_person.setdefault(person, []).append(errors)
    people = sorted(by_person)
    mae = np.zeros((len(GRID), len(people)))
    for j, person in enumerate(people):
        mae[:, j] = np.concatenate(by_person[person], axis=1).mean(axis=1)
    table = pd.DataFrame(GRID, columns=NAMES)
    table["mean_person_mae"] = mae.mean(axis=1)
    table.sort_values("mean_person_mae").to_csv(OUT / "tuning.csv", index=False)
    print(table.sort_values("mean_person_mae").head(10).round(3).to_string(index=False))

    person_fold = np.array([folds[person] for person in people])
    picks = []
    held = np.zeros(len(people))
    for fold in sorted(set(person_fold)):
        train = person_fold != fold
        best = int(np.argmin(mae[:, train].mean(axis=1)))
        row = dict(zip(NAMES, GRID[best]))
        row["fold"] = fold
        row["train_mae"] = mae[best, train].mean()
        row["held_out_mae"] = mae[best, ~train].mean()
        held[~train] = mae[best, ~train]
        picks.append(row)
    picks = pd.DataFrame(picks)
    picks.to_csv(OUT / "tuning_cv.csv", index=False)
    print(picks.round(3).to_string(index=False))
    print("mean per person, out of fold:", round(held.mean(), 3))

    rows = []
    for dataset, person, task, errors, hr, ref, share in results:
        for i in range(len(hr)):
            rows.append({
                "dataset": dataset,
                "person": person,
                "listed": person + "_" + task in excluded,
                "error": abs(hr[i] - ref[i]),
                "share_at_hr": share[i],
            })
    flag = pd.DataFrame(rows)
    parts = {
        "UBFC-rPPG": flag[flag["dataset"] == "ubfc_rppg"],
        "UBFC-Phys all": flag[flag["dataset"] == "ubfc_phys"],
        "UBFC-Phys unlisted": flag[(flag["dataset"] == "ubfc_phys") & ~flag["listed"]],
        "pooled all": flag,
        "pooled unlisted": flag[~flag["listed"]],
    }
    summary = []
    for name, part in parts.items():
        cutoff = np.quantile(part["share_at_hr"], 0.2)
        kept = part["share_at_hr"] >= cutoff
        summary.append({
            "group": name,
            "auroc": roc_auc_score(part["error"] > BAD_BPM, -part["share_at_hr"]),
            "mae": part["error"].mean(),
            "mae_kept_80": part.loc[kept, "error"].mean(),
        })
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "deep_flag.csv", index=False)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
