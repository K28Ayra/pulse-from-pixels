import numpy as np

from pulsecam.sync import estimate_lag, hr_track


def pulse_with_changing_hr(seconds, fs, start_s, seed):
    """A pulse whose HR wanders between 60 and 100 bpm, starting start_s into the HR curve."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * fs)) / fs
    knots = np.arange(0, seconds + start_s + 20, 8.0)
    curve = 80 + 20 * np.sin(np.cumsum(rng.normal(0, 1, len(knots))))
    hr = np.interp(t + start_s, knots, curve)
    phase = 2 * np.pi * np.cumsum(hr / 60) / fs
    return t, np.sin(phase) + 0.2 * rng.standard_normal(len(t))


def test_finds_a_known_offset():
    pairs = []
    for seed in [1, 2]:
        # the contact recording starts 4 s into the story, the video at 0 s,
        # so the same HR change is 4 s later in the video
        t_contact, contact = pulse_with_changing_hr(120, 64, start_s=4.0, seed=seed)
        t_video, video = pulse_with_changing_hr(120, 35, start_s=0.0, seed=seed)
        pairs.append((hr_track(t_video, video), hr_track(t_contact, contact)))
    lag, r = estimate_lag(pairs)
    assert abs(lag - 4.0) <= 0.5
    assert r > 0.9
