import numpy as np

from pulsecam.tracking import OnlineTracker, viterbi_track


def peaked_spectra(bpm, path, width=2.0):
    rows = []
    for hr in path:
        rows.append(np.exp(-0.5 * ((bpm - hr) / width) ** 2))
    return np.array(rows)


def test_one_harmonic_window_is_ignored():
    bpm = np.arange(42, 211, 1.0)
    true = np.linspace(70, 76, 20)
    spectra = peaked_spectra(bpm, true)
    # window 10 has a much taller peak at double the rate
    spectra[10] = 0.2 * spectra[10] + peaked_spectra(bpm, [2 * true[10]])[0]
    assert bpm[np.argmax(spectra[10])] > 140

    for lag in [None, 0, 3]:
        hr = viterbi_track(bpm, spectra, sigma_bpm=3.0, lag=lag)
        assert np.max(np.abs(hr - true)) <= 1.0


def test_follows_a_real_change():
    bpm = np.arange(42, 211, 1.0)
    true = np.concatenate([np.full(10, 70.0), np.linspace(70, 100, 15), np.full(10, 100.0)])
    spectra = peaked_spectra(bpm, true)
    hr = viterbi_track(bpm, spectra, sigma_bpm=3.0)
    assert np.max(np.abs(hr - true)) <= 1.0


def test_empty_windows_do_not_break_it():
    bpm = np.arange(42, 211, 1.0)
    spectra = peaked_spectra(bpm, np.full(8, 80.0))
    spectra[3] = np.nan
    spectra[4] = 0
    hr = viterbi_track(bpm, spectra)
    assert np.all(hr == 80)


def test_lag_zero_only_uses_the_past():
    bpm = np.arange(42, 211, 1.0)
    spectra = peaked_spectra(bpm, np.linspace(60, 90, 30))
    full = viterbi_track(bpm, spectra, lag=0)
    cut = viterbi_track(bpm, spectra[:15], lag=0)
    assert np.array_equal(full[:15], cut)


def test_jump_lets_the_track_follow_a_sudden_change():
    bpm = np.arange(40.0, 200.0, 1.0)
    spectra = np.full((60, len(bpm)), 1e-3)
    spectra[:30, 30] = 1.0
    spectra[30:, 60] = 1.0
    stuck = viterbi_track(bpm, spectra, sigma_bpm=0.5)
    moved = viterbi_track(bpm, spectra, sigma_bpm=0.5, jump=1e-3)
    assert abs(moved[-1] - bpm[60]) < 1
    assert abs(moved[0] - bpm[30]) < 1
    assert abs(np.abs(np.diff(moved)).max() - 30) < 1
    assert np.abs(np.diff(stuck)).max() < 30


def test_online_tracker_matches_offline():
    bpm = np.arange(42, 211, 1.0)
    rng = np.random.default_rng(0)
    spectra = rng.random((40, len(bpm))) ** 8
    for lag in [0, 1, 5]:
        offline = viterbi_track(bpm, spectra, sigma_bpm=2.0, lag=lag, jump=0.01)
        tracker = OnlineTracker(bpm, sigma_bpm=2.0, lag=lag, jump=0.01)
        for t in range(len(spectra)):
            index, hr = tracker.step(spectra[t])
            if t >= lag:
                assert index == t - lag
                assert hr == offline[t - lag]
            else:
                assert index == 0
