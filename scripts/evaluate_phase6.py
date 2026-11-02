"""Phase 6: both pipelines on compressed and dimmed copies of the dev videos,
by skin tone and by amount of motion. Writes to outputs/phase6/.

"classical" is the phase 2-4 method, "deep" FactorizePhys with the tracker
(phase 5), "POS" whole-face POS. Every variant is scored on the same windows:
those with a strict reference and a deep output in every variant.
"""

import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from evaluate_phase3 import baseline_face_pos, person_lag, ubfc_phys_task, ubfc_rppg_subject
from pulsecam import metrics
from pulsecam.config import load_config
from pulsecam.datasets import UBFC_PHYS_TASKS
from pulsecam.fusion import window_spectra
from pulsecam.motion import motion_amount
from pulsecam.skin import ita_degrees, ita_group
from pulsecam.tracking import viterbi_track
from reduce_ubfc_phys import variant_names
from tune_phase5 import classical_spectrum, unit_rows

OUT = Path("outputs/phase6")
PIPELINES = ["classical", "deep", "POS"]
BAD_BPM = 5.0
WORKS_BPM = 5.0
SKIN_PATCHES = ["forehead", "cheek_left", "cheek_right"]


def deep_path(cfg, dataset, stem, variant):
    if variant == "raw":
        name = stem + "_deep.npy"
    else:
        name = stem + "_" + variant + "_deep.npy"
    return Path(cfg["data_dir"]) / dataset / "derived" / name


def deep_spectrum(cfg, dataset, stem, data):
    pulse = np.load(data["deep_path"]).astype(float)
    times = np.arange(len(pulse)) / 30.0
    if dataset == "ubfc_rppg":
        faces = np.load(Path(cfg["data_dir"]) / dataset / "derived" / (stem + ".npz"))
        times = faces["times"][:len(pulse)] - faces["times"][0]
    _, spectra = window_spectra(times, pulse, data["start_s"], cfg["window"]["length_s"], cfg["hr_band_bpm"])
    return spectra


def track_with_share(bpm, spectrum, settings):
    """Tracked HR and the share of the spectrum within 5 bpm of it (the flag)."""
    hr = viterbi_track(
        bpm,
        spectrum ** settings["sharpen"],
        sigma_bpm=settings["sigma_bpm"],
        lag=settings["lag_s"],
        jump=settings["jump"],
    )
    near = np.abs(bpm[None, :] - hr[:, None]) <= BAD_BPM
    share = (spectrum * near).sum(axis=1)
    return hr, share


def load_variant(cfg, dataset, person, task, variant, lag_s):
    if dataset == "ubfc_rppg":
        root = Path(cfg["data_dir"]) / "ubfc_rppg"
        data = ubfc_rppg_subject(root, person, cfg, variant)
        stem = person
    else:
        derived = Path(cfg["data_dir"]) / "ubfc_phys" / "derived"
        data = ubfc_phys_task(derived, person, task, cfg, lag_s, variant)
        stem = person + "_" + task
    data["deep_path"] = deep_path(cfg, dataset, stem, variant)
    return data, stem


def skin_ita(cfg, dataset, person):
    if dataset == "ubfc_rppg":
        path = Path(cfg["data_dir"]) / "ubfc_rppg" / "derived" / (person + "_patches.npz")
    else:
        path = Path(cfg["data_dir"]) / "ubfc_phys" / "derived" / (person + "_T1_raw.npz")
    saved = np.load(path)
    names = [str(n) for n in saved["names"]]
    columns = [names.index(p) for p in SKIN_PATCHES]
    colour = np.nanmean(saved["rgb"][:, columns].reshape(-1, 3), axis=0)
    return float(ita_degrees(colour))


def score_recording(item):
    dataset, person, task, lag_s, variants, cfg = item
    tracking = cfg["tracking"]
    deep_settings = cfg["deep_tracking"]
    results = {}
    usable = None
    raw_motion = None
    for variant in variants:
        data, stem = load_variant(cfg, dataset, person, task, variant, lag_s)
        ref_hr, ok = data["refs"]["strict"]
        bpm = data["bpm"]
        classical = classical_spectrum(data, tracking["motion_tau"], tracking["drop_patches"])
        deep = deep_spectrum(cfg, dataset, stem, data)
        has_deep = ~np.any(np.isnan(deep), axis=1)
        deep = unit_rows(np.nan_to_num(deep))
        hr_c, share_c = track_with_share(bpm, classical, tracking)
        hr_d, share_d = track_with_share(bpm, deep, deep_settings)
        results[variant] = {
            "classical": (hr_c, share_c),
            "deep": (hr_d, share_d),
            "POS": (baseline_face_pos(data), np.full(len(ref_hr), np.nan)),
        }
        mask = ok & has_deep
        usable = mask if usable is None else usable & mask
        if variant == "raw":
            raw_motion = motion_amount(data["motion"])
            raw_ref = ref_hr
    rows = []
    for variant, by_pipeline in results.items():
        for t in np.flatnonzero(usable):
            row = {
                "dataset": dataset,
                "person": person,
                "task": task,
                "listed": person + "_" + task in cfg["ubfc_phys_toolbox_excluded"],
                "variant": variant,
                "t": t,
                "motion": raw_motion[t],
                "ref_hr": raw_ref[t],
            }
            for pipeline, (hr, share) in by_pipeline.items():
                row["hr_" + pipeline] = hr[t]
                row["share_" + pipeline] = share[t]
            rows.append(row)
    return rows


def groups(table):
    phys = table[table["dataset"] == "ubfc_phys"]
    return {
        "UBFC-rPPG": table[table["dataset"] == "ubfc_rppg"],
        "UBFC-Phys unlisted": phys[~phys["listed"]],
        "UBFC-Phys all": phys,
    }


def error_summary(part, pipeline):
    pred = part["hr_" + pipeline].to_numpy()
    ref = part["ref_hr"].to_numpy()
    errors = np.abs(pred - ref)
    per_person = pd.Series(errors).groupby(part["person"].to_numpy()).mean()
    return {
        "n": len(part),
        "mae": errors.mean(),
        "mean_person_mae": per_person.mean(),
        "within_ec13": metrics.within_ec13(pred, ref),
    }


def by_variant(table, order):
    rows = []
    for group, part in groups(table).items():
        for variant in order:
            chosen = part[part["variant"] == variant]
            for pipeline in PIPELINES:
                row = {"group": group, "variant": variant, "pipeline": pipeline}
                row.update(error_summary(chosen, pipeline))
                rows.append(row)
    return pd.DataFrame(rows)


def lowest_working_bitrate(variant_table, bitrates):
    """Lowest bitrate with MAE <= WORKS_BPM there and at every higher bitrate."""
    rows = []
    for (group, pipeline), part in variant_table.groupby(["group", "pipeline"]):
        mae = part.set_index("variant")["mae"]
        lowest = None
        if mae["raw"] <= WORKS_BPM:
            for kbps in sorted(bitrates, reverse=True):
                if mae["x264_" + str(kbps) + "k"] > WORKS_BPM:
                    break
                lowest = kbps
        rows.append({"group": group, "pipeline": pipeline, "lowest_working_kbps": lowest})
    return pd.DataFrame(rows)


def by_skin(table, ita):
    raw = table[table["variant"] == "raw"].copy()
    raw["ita"] = raw["person"].map(ita)
    raw["skin"] = raw["ita"].map(ita_group)
    rows = []
    for view, part in [("all", raw), ("unlisted", raw[~raw["listed"]])]:
        for skin, chosen in part.groupby("skin"):
            for pipeline in PIPELINES:
                row = {"view": view, "skin": skin, "people": chosen["person"].nunique(), "pipeline": pipeline}
                row.update(error_summary(chosen, pipeline))
                rows.append(row)
    return pd.DataFrame(rows)


def by_motion(table, threshold):
    raw = table[(table["variant"] == "raw") & ~table["listed"]].copy()
    moving = raw[raw["motion"] > threshold]
    edges = [threshold] + list(np.quantile(moving["motion"], [1 / 3, 2 / 3])) + [np.inf]
    labels = ["still", "motion low", "motion mid", "motion high"]
    raw["bin"] = pd.cut(raw["motion"], [-np.inf] + edges, labels=labels)
    rows = []
    for name, chosen in raw.groupby("bin", observed=True):
        for pipeline in PIPELINES:
            row = {"bin": name, "pipeline": pipeline}
            row.update(error_summary(chosen, pipeline))
            rows.append(row)
    return pd.DataFrame(rows)


def flag_under_stress(table, cutoff):
    stressed = table[table["variant"] != "raw"]
    rows = []
    for group, part in groups(stressed).items():
        for pipeline in ["classical", "deep"]:
            errors = (part["hr_" + pipeline] - part["ref_hr"]).abs()
            bad = errors > BAD_BPM
            share = part["share_" + pipeline]
            kept = share >= cutoff
            rows.append({
                "group": group,
                "pipeline": pipeline,
                "share_bad": 100 * bad.mean(),
                "auroc": roc_auc_score(bad, -share),
                "kept_pct": 100 * kept.mean(),
                "mae": errors.mean(),
                "mae_kept": errors[kept].mean(),
            })
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    variants = variant_names(cfg)
    jobs = []
    for name in json.loads(Path("splits/ubfc_rppg.json").read_text())["dev"]:
        jobs.append(("ubfc_rppg", name, "", 0.0, variants, cfg))
    derived = Path(cfg["data_dir"]) / "ubfc_phys" / "derived"
    for person in json.loads(Path("splits/ubfc_phys.json").read_text())["dev"]:
        lag_s, found, r = person_lag(derived, person)
        for task in UBFC_PHYS_TASKS:
            jobs.append(("ubfc_phys", person, task, lag_s, variants, cfg))
    with Pool(14) as pool:
        results = pool.map(score_recording, jobs, chunksize=1)
    rows = []
    for part in results:
        rows += part
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "windows.csv", index=False)

    ita = {}
    for dataset, person, task, lag_s, _, _ in jobs:
        ita[person] = skin_ita(cfg, dataset, person)
    pd.Series(ita, name="ita").to_csv(OUT / "ita.csv")

    threshold = json.loads(Path("outputs/phase3/motion_threshold.json").read_text())["threshold"]
    variant_table = by_variant(table, variants)
    working = lowest_working_bitrate(variant_table, cfg["stress_variants"]["bitrates_kbps"])
    skin = by_skin(table, ita)
    motion = by_motion(table, threshold)
    flag = flag_under_stress(table, cfg["quality"]["cutoff"])
    variant_table.to_csv(OUT / "by_variant.csv", index=False)
    working.to_csv(OUT / "lowest_working_bitrate.csv", index=False)
    skin.to_csv(OUT / "by_skin.csv", index=False)
    motion.to_csv(OUT / "by_motion.csv", index=False)
    flag.to_csv(OUT / "flag_under_stress.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    wide = variant_table.pivot_table(index=["group", "variant"], columns="pipeline", values="mae", sort=False)
    print(wide.round(2).to_string())
    for frame in [working, skin, motion, flag]:
        print(frame.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
