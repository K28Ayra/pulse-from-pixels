import numpy as np
import pytest

from pulsecam.reference import window_starts
from pulsecam.rppg import METHODS, run_method, windowed_pulse_hr


def fake_skin_rgb(hr_bpm, fs=30, seconds=30, seed=0):
    """Mean skin colour with a small pulse, a slow brightness drift and noise."""
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    t = np.arange(n) / fs
    pulse = np.sin(2 * np.pi * hr_bpm / 60 * t)
    # blood absorbs green most; these are roughly the PBV proportions
    strength = np.array([0.33, 0.77, 0.53])
    skin = np.array([180.0, 130.0, 110.0])
    drift = 1 + 0.05 * np.sin(2 * np.pi * 0.05 * t)
    rgb = skin * drift[:, None] * (1 + 0.003 * pulse[:, None] * strength)
    rgb = rgb + 0.05 * rng.standard_normal((n, 3))
    return t, rgb


@pytest.mark.parametrize("name", sorted(METHODS))
@pytest.mark.parametrize("true_hr", [55, 80, 120])
def test_methods_find_the_pulse(name, true_hr):
    t, rgb = fake_skin_rgb(true_hr)
    pulse = run_method(name, rgb, fs=30)
    assert len(pulse) == len(rgb)
    starts = window_starts(t[-1], 10, 1)
    hr = windowed_pulse_hr(t, pulse, starts, 10, (42, 210))
    assert np.all(np.abs(hr - true_hr) < 1.5)


def test_pos_ignores_brightness_changes():
    t, rgb = fake_skin_rgb(70)
    # a strong flicker at 100 bpm that changes all channels by the same factor
    flicker = 1 + 0.02 * np.sin(2 * np.pi * 100 / 60 * t)
    pulse = run_method("POS", rgb * flicker[:, None], fs=30)
    starts = window_starts(t[-1], 10, 1)
    hr = windowed_pulse_hr(t, pulse, starts, 10, (42, 210))
    assert np.all(np.abs(hr - 70) < 1.5)


def test_windows_past_the_end_are_nan():
    t, rgb = fake_skin_rgb(70, seconds=12)
    pulse = run_method("POS", rgb, fs=30)
    hr = windowed_pulse_hr(t, pulse, np.array([0.0, 5.0]), 10, (42, 210))
    assert not np.isnan(hr[0])
    assert np.isnan(hr[1])
