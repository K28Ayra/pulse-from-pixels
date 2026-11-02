"""Phase 8: the frozen method on the test people, once.

    python scripts/evaluate_phase8.py dev    # same code on dev people, to check it
    python scripts/evaluate_phase8.py test   # the one run

"test" refuses to run unless configs/default.yaml says frozen: true, git has no
uncommitted changes to tracked files, and outputs/phase8/run.json doesn't exist
yet. It writes the commit hash there before reading any test reference.

Pipelines: "deep" is the final method (FactorizePhys + tracker, phase 5),
"classical" the phase 2-4 method, "POS" whole-face POS, and the rPPG-Toolbox
deep models (PURE checkpoints) as run by scripts/run_toolbox.py test.
Targets, fixed before the run (docs/plan.md):
  still people (UBFC-rPPG, UBFC-Phys T1): >= 90% within EC13 and MAE <= 3 bpm
  deep beats POS on every test set
  deep beats the best toolbox deep model on UBFC-Phys
UBFC-Phys is reported with all recordings and without rPPG-Toolbox's exclusion list.
"""

import glob
import json
import pickle
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from evaluate_phase3 import METHODS, baseline_face_pos, person_lag, ubfc_phys_task, ubfc_rppg_subject
from evaluate_phase6 import deep_path, deep_spectrum, track_with_share
from pulsecam import metrics
from pulsecam.config import load_config
from pulsecam.datasets import UBFC_PHYS_TASKS, list_mpu_recordings, load_mpu_recording
from pulsecam.motion import motion_amount, motion_signals, motion_spectra
from pulsecam.pipeline import candidate_spectra
from pulsecam.reference import reference_hr
from pulsecam.rppg import windowed_pulse_hr
from pulsecam.skin import ita_degrees, ita_group
from tune_phase5 import classical_spectrum, unit_rows

TOOLBOX_JOBS = ["PhysNet", "TS-CAN", "EfficientPhys", "FactorizePhys"]
TOOLBOX_FS = {"ubfc_phys": 35.0}
MPU = "mpu_rppg_sample"
SKIN_PATCHES = ["forehead", "cheek_left", "cheek_right"]
BAD_BPM = 5.0


def check_frozen(cfg, out):
    if not cfg.get("frozen", False):
        raise SystemExit("configs/default.yaml is not frozen: true")
    changed = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                             capture_output=True, text=True).stdout.strip()
    if changed:
        raise SystemExit("uncommitted changes:\n" + changed)
    if (out / "run.json").exists():
        raise SystemExit("the test run has already happened, see " + str(out / "run.json"))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps({"commit": commit, "started": time.ctime()}, indent=1))
    return commit


def load_toolbox(folder):
    """job -> recording name -> pulse, for the jobs that have run."""
    found = {}
    for job in TOOLBOX_JOBS:
        paths = glob.glob(str(folder / job / "**" / "*_outputs.pickle"), recursive=True)
        if not paths:
            continue
        with open(paths[0], "rb") as handle:
            saved = pickle.load(handle)
        signals = {}
        for name, chunks in saved["predictions"].items():
            pulse = np.concatenate([np.asarray(chunks[i].cpu()).ravel() for i in sorted(chunks)])
            if saved["label_type"] == "DiffNormalized":
                pulse = np.cumsum(pulse)
            signals[name] = pulse
        found[job] = signals
    return found


def mpu_recording(cfg, name, csv_path):
    """Same layout as evaluate_phase3.ubfc_rppg_subject, for one MPU-rPPG video."""
    win = cfg["window"]
    ref_cfg = cfg["reference"]
    band = cfg["hr_band_bpm"]
    recording = load_mpu_recording(Path(cfg["data_dir"]) / MPU / (name[3:] + "_Output.mp4"), csv_path)
    ref = reference_hr(
        recording["ppg_times"], recording["ppg"],
        length_s=win["length_s"], step_s=win["step_s"], band_bpm=band,
        min_beats=ref_cfg["min_beats"], max_ibi_cv=ref_cfg["max_ibi_cv"],
        max_disagree_bpm=ref_cfg["max_disagree_bpm"], max_gap_s=ref_cfg["max_gap_s"],
    )
    saved = np.load(Path(cfg["data_dir"]) / MPU / "derived" / (name + ".npz"))
    times = saved["times"]
    starts = ref["start_s"]
    candidates, bpm, spectra = candidate_spectra(
        times, saved["rgb"].astype(float), [str(p) for p in saved["names"]], METHODS, starts, win["length_s"], band)
    signals = motion_signals(saved["landmarks"], saved["found"])
    _, motion = motion_spectra(times, signals, starts, win["length_s"], band)
    return {
        "person": name, "task": "", "candidates": candidates, "bpm": bpm, "spectra": spectra,
        "motion": motion, "refs": {"strict": (ref["hr_bpm"], ref["ok"])}, "start_s": starts,
    }


def video_times(cfg, dataset, stem):
    """Frame times that the toolbox outputs line up with."""
    if dataset == "ubfc_rppg":
        saved = np.load(Path(cfg["data_dir"]) / dataset / "derived" / (stem + ".npz"))
        return saved["times"] - saved["times"][0]
    return None


def score_recording(item):
    dataset, person, task, lag_s, extra, toolbox, cfg = item
    tracking = cfg["tracking"]
    if dataset == "ubfc_rppg":
        data = ubfc_rppg_subject(Path(cfg["data_dir"]) / dataset, person, cfg)
        stem = person
    elif dataset == "ubfc_phys":
        data = ubfc_phys_task(Path(cfg["data_dir"]) / dataset / "derived", person, task, cfg, lag_s)
        stem = person + "_" + task
    else:
        data = mpu_recording(cfg, person, extra)
        stem = person
    data["deep_path"] = deep_path(cfg, dataset, stem, "raw")
    ref_hr, ok = data["refs"]["strict"]
    bpm = data["bpm"]

    classical = classical_spectrum(data, tracking["motion_tau"], tracking["drop_patches"])
    deep = deep_spectrum(cfg, dataset, stem, data)
    has_deep = ~np.any(np.isnan(deep), axis=1)
    hr_c, share_c = track_with_share(bpm, classical, tracking)
    hr_d, share_d = track_with_share(bpm, unit_rows(np.nan_to_num(deep)), cfg["deep_tracking"])
    predictions = {"deep": hr_d, "classical": hr_c, "POS": baseline_face_pos(data)}

    for job, signals in toolbox.items():
        if stem not in signals:
            continue
        pulse = signals[stem]
        times = video_times(cfg, dataset, stem)
        if times is None:
            times = np.arange(len(pulse)) / TOOLBOX_FS.get(dataset, 30.0)
        n = min(len(pulse), len(times))
        predictions[job] = windowed_pulse_hr(times[:n], pulse[:n], data["start_s"],
                                             cfg["window"]["length_s"], cfg["hr_band_bpm"])

    motion = motion_amount(data["motion"])
    rows = []
    for t in np.flatnonzero(ok & has_deep):
        row = {
            "dataset": dataset, "person": person, "task": task,
            "listed": person + "_" + task in cfg["ubfc_phys_toolbox_excluded"],
            "t": t, "motion": motion[t], "ref_hr": ref_hr[t],
            "share_deep": share_d[t], "share_classical": share_c[t],
        }
        for name, hr in predictions.items():
            row["hr_" + name] = hr[t]
        rows.append(row)
    return rows


def skin_ita(cfg, dataset, person):
    if dataset == "ubfc_rppg":
        path = Path(cfg["data_dir"]) / dataset / "derived" / (person + "_patches.npz")
    elif dataset == "ubfc_phys":
        path = Path(cfg["data_dir"]) / dataset / "derived" / (person + "_T1_raw.npz")
    else:
        path = Path(cfg["data_dir"]) / MPU / "derived" / (person + ".npz")
    saved = np.load(path)
    names = [str(n) for n in saved["names"]]
    columns = [names.index(p) for p in SKIN_PATCHES]
    return float(ita_degrees(np.nanmean(saved["rgb"][:, columns].reshape(-1, 3), axis=0)))


def test_sets(table):
    phys = table[table["dataset"] == "ubfc_phys"]
    unlisted = phys[~phys["listed"]]
    return {
        "UBFC-rPPG": table[table["dataset"] == "ubfc_rppg"],
        "UBFC-Phys all": phys,
        "UBFC-Phys without list": unlisted,
        "UBFC-Phys T1 all": phys[phys["task"] == "T1"],
        "UBFC-Phys T1 without list": unlisted[unlisted["task"] == "T1"],
        "MPU-rPPG": table[table["dataset"] == MPU],
    }


def summarise(table, pipelines):
    rows = []
    for name, part in test_sets(table).items():
        if len(part) == 0:
            continue
        for pipeline in pipelines:
            chosen = part[part["hr_" + pipeline].notna()]
            if len(chosen) == 0:
                continue
            pred = chosen["hr_" + pipeline].to_numpy()
            ref = chosen["ref_hr"].to_numpy()
            row = {"test_set": name, "pipeline": pipeline, "people": chosen["person"].nunique()}
            row.update(metrics.summary(pred, ref))
            row["mae_ci_low"], row["mae_ci_high"] = metrics.bootstrap_ci(pred, ref, chosen["person"].to_numpy())
            errors = np.abs(pred - ref)
            row["mean_person_mae"] = pd.Series(errors).groupby(chosen["person"].to_numpy()).mean().mean()
            # one number per recording: mean predicted HR against mean reference HR
            per_video = chosen.assign(error=pred - ref).groupby(["person", "task"])["error"].mean()
            row["video_mae"] = per_video.abs().mean()
            if pipeline in ("deep", "classical"):
                bad = errors > BAD_BPM
                share = chosen["share_" + pipeline].to_numpy()
                row["flag_auroc"] = roc_auc_score(bad, -share) if 0 < bad.sum() < len(bad) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def per_person(table, ita, pipelines):
    rows = []
    for (dataset, person), part in table.groupby(["dataset", "person"]):
        row = {"dataset": dataset, "person": person, "windows": len(part),
               "ita": ita[person], "skin": ita_group(ita[person])}
        for pipeline in pipelines:
            row["mae_" + pipeline] = (part["hr_" + pipeline] - part["ref_hr"]).abs().mean()
        rows.append(row)
    return pd.DataFrame(rows)


def deep_vs(table, other):
    """Per-person MAE of deep against another pipeline on the windows both have, one-sided Wilcoxon."""
    rows = []
    for name, part in test_sets(table).items():
        part = part[part["hr_" + other].notna()]
        if part["person"].nunique() < 2:
            continue
        mine = (part["hr_deep"] - part["ref_hr"]).abs().groupby(part["person"]).mean()
        theirs = (part["hr_" + other] - part["ref_hr"]).abs().groupby(part["person"]).mean()
        p = stats.wilcoxon(mine, theirs, alternative="less").pvalue if (mine != theirs).any() else np.nan
        rows.append({
            "test_set": name, "other": other, "windows": len(part),
            "mae_deep": (part["hr_deep"] - part["ref_hr"]).abs().mean(),
            "mae_other": (part["hr_" + other] - part["ref_hr"]).abs().mean(),
            "people_better": int((mine < theirs).sum()), "people": len(mine), "p": p,
        })
    return pd.DataFrame(rows)


def targets(summary, comparisons, toolbox_jobs):
    rows = []
    deep = summary[summary["pipeline"] == "deep"].set_index("test_set")
    for name in ["UBFC-rPPG", "UBFC-Phys T1 all", "UBFC-Phys T1 without list"]:
        if name in deep.index:
            mae = deep.loc[name, "mae"]
            ec13 = deep.loc[name, "within_ec13"]
            rows.append({"target": "still: MAE <= 3 and EC13 >= 90%", "test_set": name,
                         "result": "MAE %.2f, EC13 %.1f%%" % (mae, ec13), "pass": mae <= 3 and ec13 >= 90})
    pos = comparisons[comparisons["other"] == "POS"]
    for _, row in pos.iterrows():
        if row["test_set"].startswith("UBFC-Phys T1"):
            continue
        rows.append({"target": "deep beats POS", "test_set": row["test_set"],
                     "result": "%.2f vs %.2f" % (row["mae_deep"], row["mae_other"]),
                     "pass": row["mae_deep"] < row["mae_other"]})
    for name in ["UBFC-Phys all", "UBFC-Phys without list"]:
        part = comparisons[(comparisons["test_set"] == name) & comparisons["other"].isin(toolbox_jobs)]
        if len(part) == 0:
            continue
        best = part.loc[part["mae_other"].idxmin()]
        rows.append({"target": "deep beats best toolbox deep model", "test_set": name,
                     "result": "%.2f vs %.2f (%s)" % (best["mae_deep"], best["mae_other"], best["other"]),
                     "pass": best["mae_deep"] < best["mae_other"]})
    return pd.DataFrame(rows)


def main():
    mode = sys.argv[1]
    cfg = load_config()
    if mode == "test":
        out = Path("outputs/phase8")
        commit = check_frozen(cfg, out)
    elif mode == "dev":
        out = Path("outputs/phase8_dev_check")
        commit = None
    else:
        raise SystemExit("dev or test")
    out.mkdir(parents=True, exist_ok=True)

    toolbox_root = Path("outputs/phase8/toolbox") if mode == "test" else Path("outputs/none")
    jobs = []
    for name in json.loads(Path("splits/ubfc_rppg.json").read_text())[mode]:
        jobs.append(("ubfc_rppg", name, "", 0.0, None, load_toolbox(toolbox_root / "ubfc_rppg"), cfg))
    derived = Path(cfg["data_dir"]) / "ubfc_phys" / "derived"
    phys_toolbox = load_toolbox(toolbox_root / "ubfc_phys")
    lags = {}
    for person in json.loads(Path("splits/ubfc_phys.json").read_text())[mode]:
        lag_s, found, r = person_lag(derived, person)
        lags[person] = {"used_s": lag_s, "found_s": found, "r": r}
        for task in UBFC_PHYS_TASKS:
            jobs.append(("ubfc_phys", person, task, lag_s, None, phys_toolbox, cfg))
    if mode == "test":
        for name, _, csv_path in list_mpu_recordings(Path(cfg["data_dir"]) / MPU):
            jobs.append((MPU, name, "", 0.0, csv_path, {}, cfg))

    with Pool(14) as pool:
        results = pool.map(score_recording, jobs, chunksize=1)
    table = pd.DataFrame([row for part in results for row in part])
    table.to_csv(out / "windows.csv", index=False)

    toolbox_jobs = [job for job in TOOLBOX_JOBS if "hr_" + job in table]
    pipelines = ["deep", "classical", "POS"] + toolbox_jobs
    summary = summarise(table, pipelines)
    comparisons = pd.concat([deep_vs(table, other) for other in pipelines[1:]], ignore_index=True)
    ita = {person: skin_ita(cfg, dataset, person) for dataset, person, *_ in jobs}
    people = per_person(table, ita, pipelines)
    target_table = targets(summary, comparisons, toolbox_jobs)

    summary.to_csv(out / "summary.csv", index=False)
    comparisons.to_csv(out / "deep_vs_others.csv", index=False)
    people.to_csv(out / "per_person.csv", index=False)
    target_table.to_csv(out / "targets.csv", index=False)
    (out / "ubfc_phys_lags.json").write_text(json.dumps(lags, indent=1))
    if commit is not None:
        run = json.loads((out / "run.json").read_text())
        run["finished"] = time.ctime()
        (out / "run.json").write_text(json.dumps(run, indent=1))

    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    columns = ["test_set", "pipeline", "people", "n", "mae", "mae_ci_low", "mae_ci_high", "rmse", "pearson",
               "bias", "within_5bpm", "within_ec13", "video_mae", "flag_auroc"]
    print(summary[[c for c in columns if c in summary]].round(3).to_string(index=False))
    print(comparisons.round(3).to_string(index=False))
    print(people.round(2).to_string(index=False))
    print(target_table.to_string(index=False))


if __name__ == "__main__":
    main()
