import numpy as np

from pulsecam.quality import (
    FEATURE_NAMES,
    WINDOW_FEATURE_NAMES,
    candidate_features,
    fit_quality_model,
    keep_top_share,
    quality_scores,
    window_features,
)

BPM = np.arange(42.0, 211.0, 1.0)


def peak_at(hr, noise, rng):
    row = np.exp(-0.5 * ((BPM - hr) / 1.5) ** 2) + noise * rng.random(len(BPM))
    return row / row.sum()


def test_features_separate_a_clean_peak_from_noise():
    rng = np.random.default_rng(0)
    spectra = np.zeros((3, 1, len(BPM)))
    spectra[0, 0] = peak_at(75, 0.01, rng)
    spectra[1, 0] = peak_at(76, 0.01, rng)
    spectra[2, 0] = peak_at(150, 2.0, rng)
    motion = np.zeros((1, len(BPM)))
    features, peaks = candidate_features(BPM, spectra, motion)
    assert features.shape == (3, 1, len(FEATURE_NAMES))
    snr = FEATURE_NAMES.index("snr_db")
    agreement = FEATURE_NAMES.index("agreement")
    assert features[0, 0, snr] > features[2, 0, snr]
    assert features[0, 0, agreement] == 0.5
    assert features[2, 0, agreement] == 0.0
    assert abs(peaks[0, 0] - 75) <= 1


def test_empty_spectrum_gives_nan_and_scores_zero():
    rng = np.random.default_rng(1)
    spectra = np.zeros((2, 2, len(BPM)))
    spectra[0, 0] = peak_at(70, 0.01, rng)
    spectra[1, 0] = peak_at(70, 0.01, rng)
    spectra[:, 1] = np.nan
    features, peaks = candidate_features(BPM, spectra, np.zeros((2, len(BPM))))
    assert np.all(np.isnan(features[:, 1]))
    x = np.vstack([rng.normal(0, 1, (50, 9)), rng.normal(3, 1, (50, 9))])
    y = np.array([0] * 50 + [1] * 50)
    model = fit_quality_model(x, y)
    scores = quality_scores(model, features)
    assert scores.shape == (2, 2)
    assert np.all(scores[:, 1] == 0)


def test_keep_top_share_drops_the_low_scores():
    ref = np.full(10, 70.0)
    pred = ref.copy()
    pred[:2] += 20
    score = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float)
    before, after, share = keep_top_share(pred, ref, score, 0.8)
    assert before == 4.0
    assert after == 0.0
    assert share == 0.8


def test_window_features_see_a_tracker_far_from_the_evidence():
    rng = np.random.default_rng(2)
    spectra = np.zeros((2, 3, len(BPM)))
    for c in range(2):
        for t in range(3):
            spectra[c, t] = peak_at(80, 0.01, rng)
    weights = np.ones((2, 3))
    snr = np.zeros((2, 3))
    motion = np.zeros((3, len(BPM)))
    on_peak = window_features(BPM, spectra, snr, weights, motion, np.full(3, 80.0))
    off_peak = window_features(BPM, spectra, snr, weights, motion, np.full(3, 120.0))
    assert on_peak.shape == (3, len(WINDOW_FEATURE_NAMES))
    gap = WINDOW_FEATURE_NAMES.index("tracker_gap")
    share = WINDOW_FEATURE_NAMES.index("share_at_hr")
    agree = WINDOW_FEATURE_NAMES.index("agreement_at_hr")
    assert on_peak[0, gap] == 0
    assert off_peak[0, gap] == 40
    assert on_peak[0, share] > off_peak[0, share]
    assert on_peak[0, agree] == 1
    assert off_peak[0, agree] == 0
