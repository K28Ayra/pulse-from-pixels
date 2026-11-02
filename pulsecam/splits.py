import json

import numpy as np


def make_split(ids, n_test, seed):
    """Pick n_test ids at random for the held-out test set; the rest are dev."""
    ids = sorted(ids)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    test_idx = order[:n_test]

    test = []
    dev = []
    for i, name in enumerate(ids):
        if i in test_idx:
            test.append(name)
        else:
            dev.append(name)

    split = {"seed": seed, "dev": dev, "test": test}
    return split


def dev_folds(dev_ids, n_folds, seed):
    """Assign each dev id to one of n_folds cross-validation folds."""
    ids = sorted(dev_ids)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    folds = {}
    for rank, idx in enumerate(order):
        folds[ids[idx]] = int(rank % n_folds)
    return folds


def save_split(split, path):
    with open(path, "w") as f:
        json.dump(split, f, indent=2)


def load_split(path):
    with open(path, "r") as f:
        return json.load(f)
