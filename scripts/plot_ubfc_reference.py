"""Reference HR from the PPG next to the HR the oximeter displayed.

    python scripts/plot_ubfc_reference.py subject24 subject37

Writes docs/figures/ubfc_reference.png. Signals only, no video frames.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pulsecam.config import load_config
from pulsecam.datasets import load_ubfc_subject
from pulsecam.reference import reference_hr


def main():
    names = sys.argv[1:]
    if not names:
        names = ["subject24", "subject37"]
    cfg = load_config()
    root = Path(cfg["data_dir"]) / "ubfc_rppg"
    ref_cfg = cfg["reference"]
    length = cfg["window"]["length_s"]

    fig, axes = plt.subplots(len(names), 1, figsize=(8, 2.6 * len(names)), squeeze=False)
    for ax, name in zip(axes[:, 0], names):
        subject = load_ubfc_subject(root, name)
        times = subject["ppg_times"] - subject["ppg_times"][0]
        ref = reference_hr(
            subject["ppg_times"],
            subject["ppg"],
            length_s=length,
            step_s=cfg["window"]["step_s"],
            band_bpm=cfg["hr_band_bpm"],
            min_beats=ref_cfg["min_beats"],
            max_ibi_cv=ref_cfg["max_ibi_cv"],
            max_disagree_bpm=ref_cfg["max_disagree_bpm"],
            max_gap_s=ref_cfg["max_gap_s"],
        )
        middle = ref["start_s"] + length / 2
        ok = ref["ok"]
        valid = subject["device_hr"] >= 30

        ax.plot(times[valid], subject["device_hr"][valid], lw=1, label="oximeter display")
        ax.plot(middle[ok], ref["hr_bpm"][ok], ".", ms=4, label="from PPG beats")
        ax.plot(middle[~ok], ref["hr_bpm"][~ok], "x", ms=4, color="grey", label="window not used")
        ax.set_title(name, fontsize=10)
        ax.set_ylabel("bpm")
        ax.legend(fontsize=7, loc="upper right")
    axes[-1, 0].set_xlabel("time (s)")
    fig.tight_layout()

    out = Path("docs") / "figures" / "ubfc_reference.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    main()
