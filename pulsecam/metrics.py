"""HR error metrics. pred and ref are HR in bpm, one value per window (or per video)."""

import numpy as np


def mae(pred, ref):
    return float(np.mean(np.abs(pred - ref)))


def rmse(pred, ref):
    return float(np.sqrt(np.mean((pred - ref) ** 2)))


def pearson(pred, ref):
    if len(pred) < 3 or np.std(pred) == 0 or np.std(ref) == 0:
        return float("nan")
    return float(np.corrcoef(pred, ref)[0, 1])


def bland_altman(pred, ref):
    """Bias and 95% limits of agreement of pred - ref."""
    diff = pred - ref
    bias = np.mean(diff)
    spread = 1.96 * np.std(diff, ddof=1)
    return float(bias), float(bias - spread), float(bias + spread)


def within_bpm(pred, ref, bpm=5.0):
    """Percentage of values with |error| <= bpm."""
    close = np.abs(pred - ref) <= bpm
    return float(100 * np.mean(close))


def within_ec13(pred, ref):
    """Percentage within the ANSI/AAMI EC13 bar: +-10% of reference or +-5 bpm, whichever is larger."""
    allowed = np.maximum(5.0, 0.1 * ref)
    close = np.abs(pred - ref) <= allowed
    return float(100 * np.mean(close))


def harmonic_errors(pred, ref):
    """Percentage of windows that landed on double or half the true rate.

    An error counts when it is above the EC13 bar and the prediction is
    within 10% of 2 * ref or 0.5 * ref.
    """
    ratio = pred / ref
    near_double = np.abs(ratio - 2.0) <= 0.2
    near_half = np.abs(ratio - 0.5) <= 0.05
    allowed = np.maximum(5.0, 0.1 * ref)
    wrong = np.abs(pred - ref) > allowed
    harmonic = wrong & (near_double | near_half)
    return float(100 * np.mean(harmonic))


def summary(pred, ref):
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    bias, low, high = bland_altman(pred, ref)
    result = {
        "n": len(pred),
        "mae": mae(pred, ref),
        "rmse": rmse(pred, ref),
        "pearson": pearson(pred, ref),
        "bias": bias,
        "loa_low": low,
        "loa_high": high,
        "within_5bpm": within_bpm(pred, ref, 5.0),
        "within_ec13": within_ec13(pred, ref),
        "harmonic": harmonic_errors(pred, ref),
    }
    return result


def bootstrap_ci(pred, ref, groups, metric=mae, n_boot=2000, seed=0):
    """95% interval of a metric when whole people (groups) are resampled, not windows."""
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    groups = np.asarray(groups)
    names = np.unique(groups)
    rows_of = {}
    for name in names:
        rows_of[name] = np.flatnonzero(groups == name)

    rng = np.random.default_rng(seed)
    values = np.empty(n_boot)
    for i in range(n_boot):
        picked = rng.choice(names, size=len(names), replace=True)
        rows = np.concatenate([rows_of[name] for name in picked])
        values[i] = metric(pred[rows], ref[rows])
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)
