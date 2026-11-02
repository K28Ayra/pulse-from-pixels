import numpy as np

from pulsecam.fusion import fused_hr, snr_db, snr_weights, window_spectra
from pulsecam.reference import window_starts


def two_candidates(true_hr=72, seconds=30, fs=30):
    rng = np.random.default_rng(0)
    t = np.arange(seconds * fs) / fs
    clean = np.sin(2 * np.pi * true_hr / 60 * t) + 0.3 * rng.standard_normal(len(t))
    # the bad candidate has a strong wrong peak at 100 bpm and lots of noise
    bad = 0.6 * np.sin(2 * np.pi * 100 / 60 * t) + 1.5 * rng.standard_normal(len(t))
    return t, clean, bad


def test_spectra_rows_sum_to_one_and_end_is_nan():
    t, clean, _ = two_candidates()
    starts = np.array([0.0, 10.0, 25.0])
    bpm, spectra = window_spectra(t, clean, starts, 10, (42, 210))
    assert np.isclose(spectra[0].sum(), 1)
    assert np.isnan(spectra[2]).all()
    assert bpm[np.argmax(spectra[0])] == 72


def test_clean_candidate_has_higher_snr():
    t, clean, bad = two_candidates()
    starts = window_starts(t[-1], 10, 1)
    bpm, good_spec = window_spectra(t, clean, starts, 10, (42, 210))
    _, bad_spec = window_spectra(t, bad, starts, 10, (42, 210))
    assert np.all(snr_db(bpm, good_spec) > snr_db(bpm, bad_spec))


def test_snr_fusion_follows_the_clean_candidate():
    t, clean, bad = two_candidates()
    starts = window_starts(t[-1], 10, 1)
    bpm, good_spec = window_spectra(t, clean, starts, 10, (42, 210))
    _, bad_spec = window_spectra(t, bad, starts, 10, (42, 210))
    spectra = np.stack([good_spec, bad_spec, bad_spec, bad_spec])
    snr = np.stack([snr_db(bpm, s) for s in spectra])
    for rule in ["snr", "snr2", "best"]:
        hr = fused_hr(bpm, spectra, snr_weights(snr, rule))
        assert np.all(np.abs(hr - 72) <= 1), rule


def test_best_rule_is_one_hot():
    snr = np.array([[1.0, 5.0], [3.0, np.nan]])
    weights = snr_weights(snr, "best")
    assert weights.tolist() == [[0.0, 1.0], [1.0, 0.0]]
