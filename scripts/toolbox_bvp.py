"""Save the toolbox's classical pulse signals, which its own runner only scores.

Runs in the toolbox env, after scripts/run_toolbox.py has made the cache:

    envs/rppg-toolbox/bin/python scripts/toolbox_bvp.py

Reads the toolbox's cached face crops (Haar box x1.5, 72x72) for each dev
subject and writes outputs/phase1/toolbox_bvp/<subject>.npz with the mean
RGB of the crop and one pulse signal per method.
"""

import glob
import os
import sys

import numpy as np

TOOLBOX = os.path.join("tools", "rPPG-Toolbox")
sys.path.insert(0, TOOLBOX)

from unsupervised_methods.methods.CHROME_DEHAAN import CHROME_DEHAAN  # noqa: E402
from unsupervised_methods.methods.GREEN import GREEN  # noqa: E402
from unsupervised_methods.methods.ICA_POH import ICA_POH  # noqa: E402
from unsupervised_methods.methods.LGI import LGI  # noqa: E402
from unsupervised_methods.methods.OMIT import OMIT  # noqa: E402
from unsupervised_methods.methods.PBV import PBV  # noqa: E402
from unsupervised_methods.methods.POS_WANG import POS_WANG  # noqa: E402

FS = 30


def main():
    pattern = os.path.join("data", "toolbox_cache", "UBFC-rPPG_*unsupervised")
    cache_dirs = glob.glob(pattern)
    if len(cache_dirs) != 1:
        raise SystemExit("expected one unsupervised cache folder, found " + str(cache_dirs))
    cache = cache_dirs[0]
    out_dir = os.path.join("outputs", "phase1", "toolbox_bvp")
    os.makedirs(out_dir, exist_ok=True)

    inputs = sorted(glob.glob(os.path.join(cache, "*_input0.npy")))
    for path in inputs:
        name = os.path.basename(path).replace("_input0.npy", "")
        frames = np.load(path)[..., :3]
        signals = {
            "mean_rgb": frames.mean(axis=(1, 2)),
            "POS": POS_WANG(frames, FS),
            "CHROM": CHROME_DEHAAN(frames, FS),
            "ICA": ICA_POH(frames, FS),
            "GREEN": GREEN(frames),
            "LGI": LGI(frames),
            "PBV": PBV(frames),
            "OMIT": OMIT(frames),
        }
        np.savez(os.path.join(out_dir, name + ".npz"), **signals)
        print(name, len(frames), "frames", flush=True)


if __name__ == "__main__":
    main()
