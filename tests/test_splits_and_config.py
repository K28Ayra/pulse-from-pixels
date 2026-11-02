import json
from pathlib import Path

from pulsecam.config import load_config
from pulsecam.splits import dev_folds, make_split

REPO = Path(__file__).resolve().parents[1]


def test_split_is_deterministic_and_disjoint():
    ids = ["a", "b", "c", "d", "e", "f", "g"]
    first = make_split(ids, n_test=3, seed=5)
    second = make_split(ids, n_test=3, seed=5)
    assert first == second
    assert len(first["test"]) == 3
    assert set(first["dev"]).isdisjoint(first["test"])
    assert sorted(first["dev"] + first["test"]) == sorted(ids)


def test_folds_cover_every_dev_id():
    dev = ["s" + str(i) for i in range(10)]
    folds = dev_folds(dev, n_folds=5, seed=0)
    assert sorted(folds) == sorted(dev)
    counts = [list(folds.values()).count(k) for k in range(5)]
    assert counts == [2, 2, 2, 2, 2]


def test_committed_splits_have_no_overlap():
    for name in ["ubfc_rppg.json", "ubfc_phys.json"]:
        with open(REPO / "splits" / name) as f:
            split = json.load(f)
        assert set(split["dev"]).isdisjoint(split["test"])
        assert sorted(split["folds"]) == sorted(split["dev"])


def test_live_mode_off_by_default_and_overridable():
    path = REPO / "configs" / "default.yaml"
    cfg = load_config(path)
    assert cfg["live_mode"] is False
    cfg = load_config(path, live_mode=True)
    assert cfg["live_mode"] is True
