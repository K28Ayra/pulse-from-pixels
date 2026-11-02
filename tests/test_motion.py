import numpy as np

from pulsecam.fusion import window_spectra
from pulsecam.motion import (CHEEK_LEFT, CHEEK_RIGHT, motion_amount, motion_signals,
                             motion_spectra, suppress_motion)
from pulsecam.reference import window_starts


def moving_face(n_frames=300, fs=30, shake_bpm=100, shake_px=5.0):
    """Landmarks of a face 200 px wide that shakes sideways."""
    rng = np.random.default_rng(0)
    base = rng.uniform(0, 200, size=(478, 2))
    base[CHEEK_LEFT] = [0, 100]
    base[CHEEK_RIGHT] = [200, 100]
    t = np.arange(n_frames) / fs
    shift = shake_px * np.sin(2 * np.pi * shake_bpm / 60 * t)
    landmarks = np.repeat(base[None], n_frames, axis=0)
    landmarks[:, :, 0] += shift[:, None]
    return t, landmarks, shift


def test_x_is_in_face_widths():
    t, landmarks, shift = moving_face()
    found = np.ones(len(t), dtype=bool)
    signals = motion_signals(landmarks, found)
    x = signals[:, 0] - signals[:, 0].mean()
    expected = (shift - shift.mean()) / 200
    assert np.allclose(x, expected, atol=1e-9)
    # nothing else moves
    assert np.allclose(signals[:, 2], np.log(200))


def test_missing_frames_are_filled():
    t, landmarks, _ = moving_face()
    found = np.ones(len(t), dtype=bool)
    found[50:60] = False
    landmarks[50:60] = 0
    signals = motion_signals(landmarks, found)
    assert np.isfinite(signals).all()
    assert np.abs(signals[55, 2] - np.log(200)) < 1e-9


def test_suppression_removes_the_shake_peak():
    fs = 30
    t, landmarks, shift = moving_face(n_frames=600, fs=fs)
    signals = motion_signals(landmarks, np.ones(len(t), dtype=bool))
    # the colour trace has a weak pulse at 72 and a stronger artefact from the shake
    pulse = 0.5 * np.sin(2 * np.pi * 72 / 60 * t) + shift
    starts = window_starts(t[-1], 10, 1)
    bpm, spectra = window_spectra(t, pulse, starts, 10, (42, 210))
    _, motion = motion_spectra(t, signals, starts, 10, (42, 210))
    assert bpm[np.argmax(spectra[0])] == 100

    # the artefact has 100 times the pulse power, so it needs a strong setting
    tau = 1e-4 * np.nanmax(motion)
    cleaned = suppress_motion(spectra, motion, tau)
    for i in range(len(starts) - 1):
        assert abs(bpm[np.argmax(cleaned[i])] - 72) <= 3
    assert np.isclose(cleaned[0].sum(), 1)


def test_no_motion_changes_nothing():
    rng = np.random.default_rng(1)
    spectra = rng.random((5, 20))
    spectra = spectra / spectra.sum(axis=1, keepdims=True)
    motion = np.zeros((5, 20))
    assert np.allclose(suppress_motion(spectra, motion, tau=1.0), spectra)
    assert np.all(motion_amount(motion) == 0)
