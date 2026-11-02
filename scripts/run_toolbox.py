"""Run rPPG-Toolbox baselines on the UBFC-rPPG dev subjects.

    python scripts/run_toolbox.py               # all jobs
    python scripts/run_toolbox.py POS_CHROM     # one job by name

Each job takes the toolbox's own config, changes only paths (and the device),
and runs the toolbox's main.py in envs/rppg-toolbox. The toolbox only sees
dev subjects: it reads a folder of links to them. Logs and saved outputs go
to outputs/phase1/toolbox/<job>/, the toolbox's printed video-level metrics
to outputs/phase1/toolbox_video_metrics.csv.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

TOOLBOX = Path("tools") / "rPPG-Toolbox"
TOOLBOX_PYTHON = Path("envs") / "rppg-toolbox" / "bin" / "python"
STUBS = Path("toolbox_stubs")
DATA = Path("data") / "ubfc_rppg"
DEV_LINKS = DATA / "toolbox_dev"
CACHE = Path("data") / "toolbox_cache"
OUT = Path("outputs") / "phase1" / "toolbox"

# job name -> toolbox config, checkpoint
JOBS = {
    "unsupervised": ("UBFC-rPPG_UNSUPERVISED.yaml", None),
    "PhysNet": ("PURE_UBFC-rPPG_PHYSNET_BASIC.yaml", "PURE_PhysNet_DiffNormalized.pth"),
    "TS-CAN": ("PURE_UBFC-rPPG_TSCAN_BASIC.yaml", "PURE_TSCAN.pth"),
    "EfficientPhys": ("PURE_UBFC-rPPG_EFFICIENTPHYS.yaml", "PURE_EfficientPhys.pth"),
    "FactorizePhys": ("PURE_UBFC-rPPG_FactorizePhys_FSAM_Res.yaml", "PURE_FactorizePhys_FSAM_Res.pth"),
}


def make_dev_links():
    split = json.loads(Path("splits/ubfc_rppg.json").read_text())
    DEV_LINKS.mkdir(exist_ok=True)
    for old in DEV_LINKS.iterdir():
        old.unlink()
    for name in split["dev"]:
        target = Path("..") / "DATASET_2" / name
        (DEV_LINKS / name).symlink_to(target)
    return len(split["dev"])


def write_config(job, config_name, checkpoint):
    source = TOOLBOX / "configs" / "infer_configs" / config_name
    cfg = yaml.safe_load(source.read_text())
    if "UNSUPERVISED" in cfg:
        data = cfg["UNSUPERVISED"]["DATA"]
    else:
        data = cfg["TEST"]["DATA"]

    # the toolbox runs from its own folder, so every path is made absolute
    data["DATA_PATH"] = str(DEV_LINKS.resolve())
    data["CACHED_PATH"] = str(CACHE.resolve())
    data["DO_PREPROCESS"] = True
    data["BEGIN"] = 0.0
    data["END"] = 1.0
    data.pop("FILE_LIST_PATH", None)
    cfg["LOG"] = {"PATH": str((OUT / job).resolve())}
    if checkpoint is not None:
        cfg["INFERENCE"]["MODEL_PATH"] = str((TOOLBOX / "final_model_release" / checkpoint).resolve())
        cfg["DEVICE"] = "cuda:0"

    out_path = OUT / job / "config.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out_path


def parse_metrics(log_text, job):
    """Video-level FFT metrics the toolbox prints, one row per method."""
    rows = []
    method = job
    row = None
    for line in log_text.splitlines():
        started = re.search(r"Unsupervised Method \( (\w+) \) Predicting", line)
        if started:
            method = started.group(1)
        found = re.match(r"FFT (MAE|RMSE|MAPE|Pearson) \(FFT Label\): ([-\d.e]+)", line)
        if found is None:
            continue
        if row is None or row["method"] != method:
            row = {"method": method}
            rows.append(row)
        row[found.group(1).lower()] = float(found.group(2))
    return rows


def run_job(job):
    config_name, checkpoint = JOBS[job]
    config_path = write_config(job, config_name, checkpoint)
    log_path = OUT / job / "log.txt"
    command = [str(TOOLBOX_PYTHON.resolve()), "main.py", "--config_file", str(config_path.resolve())]
    print("running", job, "log in", log_path, flush=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(STUBS.resolve())
    with open(log_path, "w") as log:
        result = subprocess.run(command, cwd=TOOLBOX, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        print(job, "failed, see", log_path)
        return []
    return parse_metrics(log_path.read_text(), job)


def main():
    n_dev = make_dev_links()
    print(n_dev, "dev subjects linked in", DEV_LINKS)

    jobs = sys.argv[1:]
    if not jobs:
        jobs = list(JOBS)

    table_path = Path("outputs") / "phase1" / "toolbox_video_metrics.csv"
    if table_path.exists():
        table = pd.read_csv(table_path)
    else:
        table = pd.DataFrame(columns=["method"])

    for job in jobs:
        rows = run_job(job)
        for row in rows:
            print(row)
            table = table[table["method"] != row["method"]]
            table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
        table.to_csv(table_path, index=False)


if __name__ == "__main__":
    main()
