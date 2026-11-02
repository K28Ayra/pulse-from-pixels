import numpy as np

from pulsecam import metrics


def test_basic_errors():
    ref = np.array([60.0, 80.0, 100.0, 120.0])
    pred = np.array([62.0, 77.0, 100.0, 124.0])
    assert metrics.mae(pred, ref) == 2.25
    assert np.isclose(metrics.rmse(pred, ref), np.sqrt(29 / 4))
    assert metrics.within_bpm(pred, ref, 2.0) == 50.0


def test_ec13_uses_the_larger_allowance():
    ref = np.array([40.0, 40.0, 120.0, 120.0])
    # 40 bpm: 5 bpm allowed (10% would be 4); 120 bpm: 12 bpm allowed
    pred = np.array([45.0, 46.0, 132.0, 133.0])
    assert metrics.within_ec13(pred, ref) == 50.0


def test_bland_altman_on_a_constant_offset():
    ref = np.linspace(60, 100, 20)
    bias, low, high = metrics.bland_altman(ref + 3, ref)
    assert np.isclose(bias, 3)
    assert np.isclose(low, 3) and np.isclose(high, 3)


def test_summary_keys_and_perfect_prediction():
    ref = np.linspace(60, 100, 20)
    result = metrics.summary(ref, ref)
    assert result["mae"] == 0
    assert result["within_ec13"] == 100
    assert np.isclose(result["pearson"], 1)


def test_bootstrap_resamples_people():
    rng = np.random.default_rng(0)
    groups = np.repeat(["a", "b", "c", "d", "e"], 50)
    ref = rng.uniform(60, 100, len(groups))
    # person "e" is 10 bpm off, the rest are close
    pred = ref + rng.normal(0, 1, len(groups))
    pred[groups == "e"] += 10
    low, high = metrics.bootstrap_ci(pred, ref, groups, n_boot=500)
    point = metrics.mae(pred, ref)
    assert low < point < high
    # with one bad person out of five, the interval has to be wide
    assert high - low > 2
