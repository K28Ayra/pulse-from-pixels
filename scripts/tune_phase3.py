"""Choose the tracking settings on dev people, about 15 minutes.

Every person counts the same. UBFC-Phys recordings on rPPG-Toolbox's exclusion
list are left out, so people where no setting works can't decide the choice.
Writes outputs/phase3/tuning.csv and tuning_cv.csv.
"""

import itertools
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_phase3 import person_lag, predict, ubfc_phys_task, ubfc_rppg_subject
from pulsecam.config import load_config
from pulsecam.datasets import UBFC_PHYS_TASKS

OUT = Path("outputs/phase3")
TAUS = [None, 1e-4]
SIGMAS = [2, 3, 5]
LAGS = [0, 2, 5]
JUMPS = [0.0, 1e-4, 1e-3, 1e-2]
SHARPEN = [1, 2]
DROPS = [(), ("jaw_left", "jaw_right", "chin")]
GRID = list(itertools.product(TAUS, SIGMAS, LAGS, JUMPS, SHARPEN, DROPS))


def errors_for_all_settings(item):
    person, data = item
    ref_hr, ok = data["refs"]["strict"]
    rows = []
    for tau, sigma, lag, jump, sharpen, drop in GRID:
        pred = predict(data, tau, sigma, lag, jump=jump, sharpen=sharpen, drop_patches=drop)
        rows.append(np.abs(pred[ok] - ref_hr[ok]))
    return person, np.array(rows)


def load_recordings(cfg):
    recordings = []
    folds = {}
    root = Path(cfg["data_dir"]) / "ubfc_rppg"
    split = json.loads(Path("splits/ubfc_rppg.json").read_text())
    folds.update(split["folds"])
    for name in split["dev"]:
        recordings.append((name, ubfc_rppg_subject(root, name, cfg)))

    derived = Path(cfg["data_dir"]) / "ubfc_phys" / "derived"
    split = json.loads(Path("splits/ubfc_phys.json").read_text())
    folds.update(split["folds"])
    for person in split["dev"]:
        lag_s, found, r = person_lag(derived, person)
        for task in UBFC_PHYS_TASKS:
            if person + "_" + task in cfg["ubfc_phys_toolbox_excluded"]:
                continue
            recordings.append((person, ubfc_phys_task(derived, person, task, cfg, lag_s)))
    return recordings, folds


def main():
    cfg = load_config()
    recordings, folds = load_recordings(cfg)
    with Pool(14) as pool:
        results = pool.map(errors_for_all_settings, recordings, chunksize=1)

    by_person = {}
    for person, errors in results:
        by_person.setdefault(person, []).append(errors)
    people = sorted(by_person)
    mae = np.zeros((len(GRID), len(people)))
    for j, person in enumerate(people):
        mae[:, j] = np.concatenate(by_person[person], axis=1).mean(axis=1)

    table = pd.DataFrame(GRID, columns=["tau", "sigma", "lag", "jump", "sharpen", "drop"])
    table["mean_person_mae"] = mae.mean(axis=1)
    table.sort_values("mean_person_mae").to_csv(OUT / "tuning.csv", index=False)

    person_fold = np.array([folds[person] for person in people])
    picks = []
    for fold in sorted(set(person_fold)):
        train = person_fold != fold
        best = int(np.argmin(mae[:, train].mean(axis=1)))
        row = dict(zip(["tau", "sigma", "lag", "jump", "sharpen", "drop"], GRID[best]))
        row["fold"] = fold
        row["train_mae"] = mae[best, train].mean()
        row["held_out_mae"] = mae[best, ~train].mean()
        picks.append(row)
    picks = pd.DataFrame(picks)
    picks.to_csv(OUT / "tuning_cv.csv", index=False)
    print(picks.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
