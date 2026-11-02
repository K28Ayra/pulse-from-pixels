"""Phase 2: skin patches x methods, fused by SNR, on the UBFC-rPPG dev subjects.

    python scripts/evaluate_phase2.py

Needs data/ubfc_rppg/derived/<subject>_patches.npz (scripts/extract_patches.py).

Every combination of candidate set and weighting rule is scored on all dev
windows, but the headline number is cross-validated: for each of the 5
person-wise folds the combination is picked on the other 4 folds and then
scored on the held-out one. Writes outputs/phase2/.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from pulsecam import metrics
from pulsecam.config import load_config
from pulsecam.datasets import load_ubfc_subject
from pulsecam.fusion import fused_hr, snr_weights
from pulsecam.pipeline import all_snr, candidate_spectra
from pulsecam.reference import reference_hr

OUT = Path("outputs") / "phase2"
METHODS = ["GREEN", "CHROM", "POS"]
RULES = ["equal", "snr", "snr2", "best"]
BASELINE = ("face", "POS")


def candidate_sets(candidates):
    """Named subsets of the (patch, method) candidates to try."""
    sets = {}
    sets["all"] = list(candidates)
    sets["no GREEN"] = [c for c in candidates if c[1] != "GREEN"]
    sets["patches, no GREEN"] = [c for c in candidates if c[1] != "GREEN" and c[0] != "face"]
    sets["POS only"] = [c for c in candidates if c[1] == "POS"]
    sets["face, POS+CHROM"] = [c for c in candidates if c[0] == "face" and c[1] != "GREEN"]
    return sets


def subject_spectra(root, name, cfg):
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

    saved = np.load(root / "derived" / (name + "_patches.npz"))
    rgb = saved["rgb"].astype(float)
    patch_names = [str(p) for p in saved["names"]]
    n = min(len(rgb), len(times))
    candidates, bpm, spectra = candidate_spectra(
        times[:n], rgb[:n], patch_names, METHODS, ref["start_s"], win["length_s"], band)

    result = {
        "candidates": candidates,
        "bpm": bpm,
        "spectra": spectra,
        "snr": all_snr(bpm, spectra),
        "ref_hr": ref["hr_bpm"],
        "ok": ref["ok"],
        "start_s": ref["start_s"],
    }
    return result


def predict(data, chosen, rule):
    rows = [data["candidates"].index(c) for c in chosen]
    spectra = data["spectra"][rows]
    weights = snr_weights(data["snr"][rows], rule)
    return fused_hr(data["bpm"], spectra, weights)


def window_table(per_subject, variants):
    """One row per subject, window and variant, only windows with an ok reference."""
    rows = []
    for name, data in per_subject.items():
        ok = data["ok"]
        for label, (chosen, rule) in variants.items():
            pred = predict(data, chosen, rule)
            for i in np.flatnonzero(ok):
                if np.isnan(pred[i]):
                    continue
                rows.append({
                    "subject": name,
                    "start_s": data["start_s"][i],
                    "variant": label,
                    "ref_hr": data["ref_hr"][i],
                    "pred_hr": pred[i],
                })
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    root = Path(cfg["data_dir"]) / "ubfc_rppg"
    split = json.loads(Path("splits/ubfc_rppg.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)

    per_subject = {}
    for name in split["dev"]:
        per_subject[name] = subject_spectra(root, name, cfg)
        print(name, flush=True)
    candidates = per_subject[split["dev"][0]]["candidates"]

    # single candidates on their own (peak of their own spectrum)
    singles = {}
    for c in candidates:
        singles[c[0] + " " + c[1]] = ([c], "equal")
    single_windows = window_table(per_subject, singles)
    single_mae = single_windows.groupby("variant").apply(
        lambda g: metrics.mae(g["pred_hr"].to_numpy(), g["ref_hr"].to_numpy()), include_groups=False)
    single_mae = single_mae.sort_values()
    single_mae.to_csv(OUT / "single_candidates.csv", header=["mae"])

    variants = {}
    for set_name, chosen in candidate_sets(candidates).items():
        for rule in RULES:
            variants[set_name + " / " + rule] = (chosen, rule)
    windows = window_table(per_subject, variants)

    # cross-validated choice of variant
    folds = split["folds"]
    windows["fold"] = windows["subject"].map(folds)
    cv_parts = []
    choices = []
    for fold in sorted(windows["fold"].unique()):
        train = windows[windows["fold"] != fold]
        errors = (train["pred_hr"] - train["ref_hr"]).abs()
        train_mae = errors.groupby(train["variant"]).mean()
        best = train_mae.idxmin()
        choices.append({"fold": fold, "chosen": best, "train_mae": train_mae[best]})
        held_out = windows[(windows["fold"] == fold) & (windows["variant"] == best)]
        cv_parts.append(held_out)
    cv = pd.concat(cv_parts)
    cv["variant"] = "fused, CV-chosen"
    pd.DataFrame(choices).to_csv(OUT / "cv_choices.csv", index=False)

    baseline_label = BASELINE[0] + " " + BASELINE[1]
    baseline = single_windows[single_windows["variant"] == baseline_label]
    everything = pd.concat([windows, cv, baseline])
    everything.to_csv(OUT / "windows.csv", index=False)

    summaries = []
    for label, used in everything.groupby("variant"):
        pred = used["pred_hr"].to_numpy()
        ref_hr = used["ref_hr"].to_numpy()
        row = {"variant": label}
        row.update(metrics.summary(pred, ref_hr))
        low, high = metrics.bootstrap_ci(pred, ref_hr, used["subject"].to_numpy())
        row["mae_ci_low"] = low
        row["mae_ci_high"] = high
        summaries.append(row)
    table = pd.DataFrame(summaries).sort_values("mae")
    table.to_csv(OUT / "summary.csv", index=False)

    # per-person MAE, fused (CV) against whole-face POS, paired
    def per_person(frame):
        errors = (frame["pred_hr"] - frame["ref_hr"]).abs()
        return errors.groupby(frame["subject"]).mean()
    fused_people = per_person(cv)
    base_people = per_person(baseline).loc[fused_people.index]
    test = stats.wilcoxon(fused_people, base_people, alternative="less")
    gain = 1 - metrics.mae(cv["pred_hr"].to_numpy(), cv["ref_hr"].to_numpy()) / \
        metrics.mae(baseline["pred_hr"].to_numpy(), baseline["ref_hr"].to_numpy())

    pd.set_option("display.width", 220)
    print(table.round(2).to_string(index=False))
    print()
    print("best single candidates:")
    print(single_mae.head(8).round(2).to_string())
    print()
    print(pd.DataFrame(choices).to_string(index=False))
    print("CV fused vs whole-face POS: MAE", round(100 * gain, 1), "% lower,",
          "Wilcoxon one-sided p =", round(test.pvalue, 4),
          "; people better/worse:", int((fused_people < base_people).sum()), "/",
          int((fused_people > base_people).sum()))


if __name__ == "__main__":
    main()
