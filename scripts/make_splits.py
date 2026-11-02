"""Write the dev/test splits. Run once; the json files are committed.

The subject lists are typed in by hand (from the dataset file listings), so
the split is fixed before any video is opened.
"""

from pathlib import Path

from pulsecam.splits import dev_folds, make_split, save_split

SEED = 2026
N_FOLDS = 5

UBFC_RPPG_NUMBERS = [
    1, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 23, 24,
    25, 26, 27, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    45, 46, 47, 48, 49,
]

# only s1-s12 are downloaded (about 14 GB each)
UBFC_PHYS_NUMBERS = list(range(1, 13))


def main():
    out_dir = Path("splits")
    out_dir.mkdir(exist_ok=True)

    ubfc_ids = []
    for n in UBFC_RPPG_NUMBERS:
        ubfc_ids.append("subject" + str(n))
    ubfc = make_split(ubfc_ids, n_test=12, seed=SEED)
    ubfc["folds"] = dev_folds(ubfc["dev"], N_FOLDS, SEED)
    save_split(ubfc, out_dir / "ubfc_rppg.json")

    phys_ids = []
    for n in UBFC_PHYS_NUMBERS:
        phys_ids.append("s" + str(n))
    phys = make_split(phys_ids, n_test=4, seed=SEED)
    phys["folds"] = dev_folds(phys["dev"], N_FOLDS, SEED)
    save_split(phys, out_dir / "ubfc_phys.json")

    print("UBFC-rPPG dev", len(ubfc["dev"]), "test", len(ubfc["test"]))
    print("  test:", ubfc["test"])
    print("UBFC-Phys dev", len(phys["dev"]), "test", len(phys["test"]))
    print("  test:", phys["test"])


if __name__ == "__main__":
    main()
