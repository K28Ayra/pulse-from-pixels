"""Phase 4: a small model that scores candidates, for fusion and for the flag.

Scores are out-of-fold: a person is always scored by a model trained on other
people. Writes to outputs/phase4/.
"""

import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from evaluate_phase3 import person_lag, ubfc_phys_task, ubfc_rppg_subject
from pulsecam.config import load_config
from pulsecam.datasets import UBFC_PHYS_TASKS
from pulsecam.fusion import fused_spectrum, snr_weights
from pulsecam.motion import suppress_motion
from pulsecam.pipeline import all_snr
from pulsecam.quality import (
    WINDOW_FEATURE_NAMES,
    candidate_features,
    fit_quality_model,
    keep_top_share,
    quality_scores,
    window_features,
)
from pulsecam.tracking import viterbi_track

OUT = Path("outputs/phase4")
BAD_BPM = 5.0
POWERS = [1, 2, 4]


def load_recordings(cfg):
    recordings = []
    root = Path(cfg["data_dir"]) / "ubfc_rppg"
    split = json.loads(Path("splits/ubfc_rppg.json").read_text())
    folds = dict(split["folds"])
    for name in split["dev"]:
        data = ubfc_rppg_subject(root, name, cfg)
        recordings.append(("ubfc_rppg", name, "", data))

    derived = Path(cfg["data_dir"]) / "ubfc_phys" / "derived"
    split = json.loads(Path("splits/ubfc_phys.json").read_text())
    folds.update(split["folds"])
    for person in split["dev"]:
        lag_s, found, r = person_lag(derived, person)
        for task in UBFC_PHYS_TASKS:
            data = ubfc_phys_task(derived, person, task, cfg, lag_s)
            recordings.append(("ubfc_phys", person, task, data))
    return recordings, folds


def prepare(item):
    dataset, person, task, data, tracking = item
    keep = []
    for i, (patch, method) in enumerate(data["candidates"]):
        if patch not in tracking["drop_patches"]:
            keep.append(i)
    spectra = suppress_motion(data["spectra"][keep], data["motion"], tracking["motion_tau"])
    features, peaks = candidate_features(data["bpm"], spectra, data["motion"])
    ref_hr, ok = data["refs"]["strict"]
    return {
        "dataset": dataset,
        "person": person,
        "task": task,
        "bpm": data["bpm"],
        "motion": data["motion"],
        "spectra": spectra,
        "features": features,
        "peaks": peaks,
        "ref_hr": ref_hr,
        "ok": ok,
    }


def training_rows(recordings, excluded, use_excluded):
    """Labelled candidate rows, every recording counts the same."""
    all_x = []
    all_y = []
    all_w = []
    for rec in recordings:
        name = rec["person"] + "_" + rec["task"]
        if name in excluded and not use_excluded:
            continue
        x = rec["features"][:, rec["ok"]]
        peaks = rec["peaks"][:, rec["ok"]]
        y = np.abs(peaks - rec["ref_hr"][rec["ok"]]) <= BAD_BPM
        x = x.reshape(-1, x.shape[-1])
        y = y.reshape(-1)
        usable = ~np.any(np.isnan(x), axis=1)
        all_x.append(x[usable])
        all_y.append(y[usable])
        all_w.append(np.full(usable.sum(), 1.0 / max(usable.sum(), 1)))
    weights = np.concatenate(all_w)
    return np.concatenate(all_x), np.concatenate(all_y), weights * len(weights)


def out_of_fold_scores(recordings, folds, excluded, use_excluded, kind):
    scores = {}
    for fold in sorted(set(folds.values())):
        train = []
        for rec in recordings:
            if folds[rec["person"]] != fold:
                train.append(rec)
        x, y, w = training_rows(train, excluded, use_excluded)
        model = fit_quality_model(x, y, sample_weight=w, kind=kind)
        for i, rec in enumerate(recordings):
            if folds[rec["person"]] == fold:
                scores[i] = quality_scores(model, rec["features"])
    return scores


def track(rec, weights, tracking):
    combined = fused_spectrum(rec["spectra"], weights) ** tracking["sharpen"]
    return viterbi_track(
        rec["bpm"],
        combined,
        sigma_bpm=tracking["sigma_bpm"],
        lag=tracking["lag_s"],
        jump=tracking["jump"],
    )


def fusion_table(recordings, tracking, excluded, candidate_scores):
    """One row per scored window with the HR from every fusion rule."""
    rows = []
    for i, rec in enumerate(recordings):
        snr = all_snr(rec["bpm"], rec["spectra"])
        snr_w = snr_weights(snr, "snr2")
        variants = {"snr2": snr_w}
        for label, scores in candidate_scores.items():
            quality = scores[i]
            for power in POWERS:
                variants[label + "^" + str(power)] = quality ** power
            variants[label + "*snr2"] = quality * snr_w
        predictions = {}
        for label, weights in variants.items():
            predictions[label] = track(rec, weights, tracking)
        features = window_features(rec["bpm"], rec["spectra"], snr, snr_w, rec["motion"], predictions["snr2"])
        rec["window_features"] = features
        rec["hr"] = predictions["snr2"]
        name = rec["person"] + "_" + rec["task"]
        for t in np.flatnonzero(rec["ok"]):
            row = {
                "dataset": rec["dataset"],
                "person": rec["person"],
                "task": rec["task"],
                "t": t,
                "listed": name in excluded,
                "ref_hr": rec["ref_hr"][t],
            }
            for label in variants:
                row["hr_" + label] = predictions[label][t]
            for j, feature in enumerate(WINDOW_FEATURE_NAMES):
                row["f_" + feature] = features[t, j]
            rows.append(row)
    return pd.DataFrame(rows)


def window_scores(table, folds, kind, use_excluded):
    """Out-of-fold chance that the final HR of a window is within 5 bpm."""
    columns = ["f_" + name for name in WINDOW_FEATURE_NAMES]
    x = table[columns].to_numpy()
    good = (table["hr_snr2"] - table["ref_hr"]).abs().to_numpy() <= BAD_BPM
    fold_of = table["person"].map(folds).to_numpy()
    per_recording = table.groupby(["person", "task"])["t"].transform("count").to_numpy()
    weights = 1.0 / per_recording
    weights = weights * len(weights) / weights.sum()
    allowed = np.ones(len(table), dtype=bool)
    if not use_excluded:
        allowed = ~table["listed"].to_numpy()
    scores = np.zeros(len(table))
    importance = []
    for fold in sorted(set(fold_of)):
        train = (fold_of != fold) & allowed
        model = fit_quality_model(x[train], good[train], sample_weight=weights[train], kind=kind)
        held = fold_of == fold
        scores[held] = model.predict_proba(x[held])[:, 1]
        if kind == "logistic":
            importance.append(model[-1].coef_[0])
    return scores, importance


def groups(table):
    rppg = table[table["dataset"] == "ubfc_rppg"]
    phys = table[table["dataset"] == "ubfc_phys"]
    return {
        "ubfc_rppg": rppg,
        "ubfc_phys_all": phys,
        "ubfc_phys_unlisted": phys[~phys["listed"]],
        "pooled_all": table,
        "pooled_unlisted": table[~table["listed"]],
    }


def fusion_report(table):
    rows = []
    hr_columns = [c for c in table.columns if c.startswith("hr_")]
    for name, part in groups(table).items():
        ref = part["ref_hr"].to_numpy()
        base = pd.Series(np.abs(part["hr_snr2"].to_numpy() - ref)).groupby(part["person"].to_numpy()).mean()
        for column in hr_columns:
            errors = np.abs(part[column].to_numpy() - ref)
            per_person = pd.Series(errors).groupby(part["person"].to_numpy()).mean()
            p_value = np.nan
            if column != "hr_snr2":
                p_value = stats.wilcoxon(per_person, base, alternative="less").pvalue
            rows.append({
                "group": name,
                "fusion": column[3:],
                "n": len(part),
                "mae": errors.mean(),
                "mean_person_mae": per_person.mean(),
                "better": int((per_person < base).sum()),
                "worse": int((per_person > base).sum()),
                "p": p_value,
            })
    return pd.DataFrame(rows)


def flag_report(table, score_columns):
    rows = []
    for column in score_columns:
        cutoff = np.quantile(table[column], 0.2)
        for name, part in groups(table).items():
            pred = part["hr_snr2"].to_numpy()
            ref = part["ref_hr"].to_numpy()
            score = part[column].to_numpy()
            bad = np.abs(pred - ref) > BAD_BPM
            before, after, share = keep_top_share(pred, ref, score)
            kept = score >= cutoff
            rows.append({
                "score": column,
                "group": name,
                "share_bad": 100 * bad.mean(),
                "auroc": roc_auc_score(bad, -score),
                "mae": before,
                "cut_keep80": 100 * (1 - after / before),
                "fixed_kept": 100 * kept.mean(),
                "fixed_cut": 100 * (1 - np.abs(pred - ref)[kept].mean() / before),
            })
    return pd.DataFrame(rows)


def bootstrap_auroc(part, model_column, simple_column, n_boot=500, seed=0):
    """95% intervals over people: model, simple score, model minus simple."""
    rng = np.random.default_rng(seed)
    people = part["person"].unique()
    by_person = {}
    for person in people:
        by_person[person] = part[part["person"] == person]
    values = []
    for _ in range(n_boot):
        chosen = rng.choice(people, len(people))
        sample = pd.concat([by_person[person] for person in chosen])
        bad = (sample["hr_snr2"] - sample["ref_hr"]).abs() > BAD_BPM
        if 0 < bad.sum() < len(bad):
            model = roc_auc_score(bad, -sample[model_column])
            simple = roc_auc_score(bad, -sample[simple_column])
            values.append((model, simple, model - simple))
    return np.percentile(np.array(values), [2.5, 97.5], axis=0).T


def flag_per_person(table, column, cutoff):
    errors = (table["hr_snr2"] - table["ref_hr"]).abs()
    rows = []
    for person, part in table.groupby("person"):
        kept = part[column] >= cutoff
        part_errors = errors[part.index]
        rows.append({
            "dataset": part["dataset"].iloc[0],
            "person": person,
            "n": len(part),
            "mae": part_errors.mean(),
            "kept_pct": 100 * kept.mean(),
            "mae_kept": part_errors[kept].mean(),
            "share_bad": 100 * (part_errors > BAD_BPM).mean(),
        })
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    tracking = cfg["tracking"]
    excluded = cfg["ubfc_phys_toolbox_excluded"]
    loaded, folds = load_recordings(cfg)
    jobs = []
    for dataset, person, task, data in loaded:
        jobs.append((dataset, person, task, data, tracking))
    with Pool(14) as pool:
        recordings = pool.map(prepare, jobs, chunksize=1)

    candidate_scores = {}
    for kind in ["logistic", "boosted"]:
        for use_excluded, tag in [(False, "unlisted"), (True, "all")]:
            label = kind + "_" + tag
            candidate_scores[label] = out_of_fold_scores(recordings, folds, excluded, use_excluded, kind)
    table = fusion_table(recordings, tracking, excluded, candidate_scores)

    score_columns = []
    for kind in ["logistic", "boosted"]:
        for use_excluded, tag in [(False, "unlisted"), (True, "all")]:
            column = "q_" + kind + "_" + tag
            scores, importance = window_scores(table, folds, kind, use_excluded)
            table[column] = scores
            score_columns.append(column)
            if importance:
                mean_coef = np.mean(importance, axis=0)
                print(column, dict(zip(WINDOW_FEATURE_NAMES, mean_coef.round(2))))
    for name in ["fused_snr_db", "best_snr_db", "share_at_hr"]:
        table["base_" + name] = table["f_" + name]
        score_columns.append("base_" + name)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 300)
    fusion = fusion_report(table)
    flag = flag_report(table, score_columns)
    table.to_csv(OUT / "windows.csv", index=False)
    fusion.to_csv(OUT / "fusion.csv", index=False)
    flag.to_csv(OUT / "flag.csv", index=False)

    rows = []
    for name, part in groups(table).items():
        interval = bootstrap_auroc(part, "q_boosted_unlisted", "base_share_at_hr")
        rows.append({
            "group": name,
            "model_low": interval[0, 0],
            "model_high": interval[0, 1],
            "simple_low": interval[1, 0],
            "simple_high": interval[1, 1],
            "difference_low": interval[2, 0],
            "difference_high": interval[2, 1],
        })
    intervals = pd.DataFrame(rows)
    intervals.to_csv(OUT / "auroc_intervals.csv", index=False)
    print(intervals.round(3).to_string(index=False))
    cutoff = cfg["quality"]["cutoff"]
    per_person = flag_per_person(table, "base_share_at_hr", cutoff)
    per_person.to_csv(OUT / "flag_per_person.csv", index=False)
    print(per_person[per_person["dataset"] == "ubfc_phys"].round(2).to_string(index=False))

    best = fusion.sort_values("mean_person_mae")
    print(best[best["group"].isin(["pooled_all", "pooled_unlisted"])].head(14).round(4).to_string(index=False))
    print(fusion[fusion["fusion"] == "snr2"].round(3).to_string(index=False))
    print(flag.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
