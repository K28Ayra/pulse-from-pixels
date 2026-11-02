"""Run rPPG-Toolbox baselines on the UBFC-rPPG dev subjects.

    python scripts/run_toolbox.py               # all jobs
    python scripts/run_toolbox.py POS_CHROM     # one job by name
    python scripts/run_toolbox.py test          # phase 8: test people, both UBFC sets

Each job takes the toolbox's own config, changes only paths (and the device),
and runs the toolbox's main.py in envs/rppg-toolbox. The toolbox only sees
dev subjects: it reads a folder of links to them. Logs and saved outputs go
to outputs/phase1/toolbox/<job>/, the toolbox's printed video-level metrics
to outputs/phase1/toolbox_video_metrics.csv.

"test" runs the deep jobs on the test people only, after the config is frozen,
and writes to outputs/phase8/toolbox/<dataset>_<job>/. For UBFC-Phys it uses the
toolbox's UBFC-PHYS settings (35 fps, dynamic face box) with its exclusion list
switched off; phase 8 reports the recordings both ways.
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

TEST_LINKS = {"ubfc_rppg": DATA / "toolbox_test", "ubfc_phys": Path("data") / "ubfc_phys" / "toolbox_test"}
TEST_CACHE = Path("data") / "toolbox_cache_test"
TEST_OUT = Path("outputs") / "phase8" / "toolbox"
DEEP_JOBS = ["PhysNet", "TS-CAN", "EfficientPhys", "FactorizePhys"]
# the toolbox has no FactorizePhys config for UBFC-Phys, so it gets PhysNet's data settings
PHYS_CONFIGS = {
    "PhysNet": "PURE_UBFC-PHYS_PHYSNET_BASIC.yaml",
    "TS-CAN": "PURE_UBFC-PHYS_TSCAN_BASIC.yaml",
    "EfficientPhys": "PURE_UBFC-PHYS_EFFICIENTPHYS.yaml",
}

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


def make_test_links():
    for dataset, folder in TEST_LINKS.items():
        split = json.loads(Path("splits/" + dataset + ".json").read_text())
        folder.mkdir(exist_ok=True)
        for old in folder.iterdir():
            old.unlink()
        for name in split["test"]:
            if dataset == "ubfc_rppg":
                target = Path("..") / "DATASET_2" / name
            else:
                target = Path("..") / "raw" / name
            (folder / name).symlink_to(target)


def phys_test_data(job):
    """TEST.DATA for UBFC-Phys: the toolbox's own UBFC-PHYS block, no exclusion list."""
    name = PHYS_CONFIGS.get(job, PHYS_CONFIGS["PhysNet"])
    data = yaml.safe_load((TOOLBOX / "configs" / "infer_configs" / name).read_text())["TEST"]["DATA"]
    data["FILTERING"]["USE_EXCLUSION_LIST"] = False
    return data


def write_config(job, config_name, checkpoint, data_path=DEV_LINKS, cache=CACHE, out=OUT, phys=False):
    source = TOOLBOX / "configs" / "infer_configs" / config_name
    cfg = yaml.safe_load(source.read_text())
    if phys:
        own = cfg["TEST"]["DATA"]
        cfg["TEST"]["DATA"] = phys_test_data(job)
        # keep the model's own clip length and input format
        cfg["TEST"]["DATA"]["PREPROCESS"]["CHUNK_LENGTH"] = own["PREPROCESS"]["CHUNK_LENGTH"]
        cfg["TEST"]["DATA"]["PREPROCESS"]["DATA_TYPE"] = own["PREPROCESS"]["DATA_TYPE"]
        cfg["TEST"]["DATA"]["PREPROCESS"]["LABEL_TYPE"] = own["PREPROCESS"]["LABEL_TYPE"]
    if "UNSUPERVISED" in cfg:
        data = cfg["UNSUPERVISED"]["DATA"]
    else:
        data = cfg["TEST"]["DATA"]

    # the toolbox runs from its own folder, so every path is made absolute
    data["DATA_PATH"] = str(Path(data_path).resolve())
    data["CACHED_PATH"] = str(Path(cache).resolve())
    data["DO_PREPROCESS"] = True
    data["BEGIN"] = 0.0
    data["END"] = 1.0
    data.pop("FILE_LIST_PATH", None)
    cfg["LOG"] = {"PATH": str((out / job).resolve())}
    if checkpoint is not None:
        cfg["INFERENCE"]["MODEL_PATH"] = str((TOOLBOX / "final_model_release" / checkpoint).resolve())
        cfg["DEVICE"] = "cuda:0"

    out_path = out / job / "config.yaml"
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


def run_job(job, config_path=None, out=OUT):
    config_name, checkpoint = JOBS[job]
    if config_path is None:
        config_path = write_config(job, config_name, checkpoint)
    log_path = out / job / "log.txt"
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


def run_test():
    make_test_links()
    for dataset in ["ubfc_rppg", "ubfc_phys"]:
        out = TEST_OUT / dataset
        for job in DEEP_JOBS:
            config_name, checkpoint = JOBS[job]
            config_path = write_config(
                job, config_name, checkpoint, TEST_LINKS[dataset], TEST_CACHE / dataset, out,
                phys=dataset == "ubfc_phys",
            )
            for row in run_job(job, config_path, out):
                print(dataset, row, flush=True)


def main():
    if sys.argv[1:] == ["test"]:
        run_test()
        return
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
