import numpy as np
import pytest

from pulsecam.reference import find_beats, reference_hr, to_uniform, window_starts
from pulsecam.synthetic import fake_ppg


@pytest.mark.parametrize("true_hr", [50, 72, 110, 160])
def test_reference_hr_matches_known_rate(true_hr):
    times, ppg = fake_ppg(true_hr, fs=30, seconds=40)
    ref = reference_hr(times, ppg, length_s=10, step_s=1)
    good = ref["hr_bpm"][ref["ok"]]
    assert len(good) > 0
    assert np.all(np.abs(good - true_hr) < 1.0)


def test_beat_count_is_right():
    times, ppg = fake_ppg(60, fs=60, seconds=30, noise=0.0)
    beats, _ = find_beats(ppg, fs=60)
    # filtfilt edges can cost a beat at either end
    assert 28 <= len(beats) <= 30


def test_irregular_timestamps_are_handled():
    times, ppg = fake_ppg(80, fs=30, seconds=30)
    rng = np.random.default_rng(1)
    jitter = rng.uniform(-0.005, 0.005, len(times))
    ref = reference_hr(times + jitter, ppg)
    good = ref["hr_bpm"][ref["ok"]]
    assert np.all(np.abs(good - 80) < 1.5)


def test_windows_with_a_recording_gap_are_flagged():
    times, ppg = fake_ppg(80, fs=30, seconds=30)
    keep = (times < 14) | (times > 15)
    ref = reference_hr(times[keep], ppg[keep])
    starts = ref["start_s"]
    touches_gap = (starts < 15) & (starts + 10 > 14)
    clear_of_gap = (starts > 15.1) | (starts + 10 < 13.9)
    assert not ref["ok"][touches_gap].any()
    assert ref["ok"][clear_of_gap].all()


def test_noise_only_windows_are_flagged():
    rng = np.random.default_rng(2)
    times = np.arange(30 * 30) / 30
    noise = rng.standard_normal(len(times))
    ref = reference_hr(times, noise)
    assert ref["ok"].mean() < 0.2


def test_window_starts():
    starts = window_starts(duration_s=15, length_s=10, step_s=1)
    assert list(starts) == [0, 1, 2, 3, 4, 5]
    assert len(window_starts(5, 10, 1)) == 0


def test_to_uniform_keeps_values_on_grid():
    times = np.array([0.0, 0.5, 1.0])
    values = np.array([0.0, 1.0, 2.0])
    grid, out = to_uniform(times, values, fs=4)
    assert np.allclose(grid, [0, 0.25, 0.5, 0.75, 1.0])
    assert np.allclose(out, [0, 0.5, 1.0, 1.5, 2.0])


def test_spectral_reference_keeps_clean_and_drops_noise():
    from pulsecam.reference import spectral_reference_hr
    fs = 64
    t, clean = fake_ppg(75, fs, 40, noise=0.1)
    rng = np.random.default_rng(3)
    noise = rng.standard_normal(len(t))
    good = spectral_reference_hr(t, clean)
    bad = spectral_reference_hr(t, noise)
    assert good["ok"].all()
    assert np.all(np.abs(good["hr_bpm"] - 75) <= 1)
    assert bad["ok"].mean() < 0.2
