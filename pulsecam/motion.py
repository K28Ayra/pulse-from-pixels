"""Head and mouth motion from face landmarks, and removing it from pulse spectra."""

import numpy as np

from pulsecam.face import FACE_OVAL
from pulsecam.reference import band_spectrum, bandpass, to_uniform

# MediaPipe face mesh numbers
CHEEK_LEFT = 234
CHEEK_RIGHT = 454
EYE_LEFT = 33
EYE_RIGHT = 263
LIP_TOP = 13
LIP_BOTTOM = 14

MOTION_NAMES = ["x", "y", "scale", "roll", "mouth"]


def fill_missing(values, found):
    """Linear interpolation over frames without a face, per column."""
    filled = values.copy()
    if found.all() or not found.any():
        return filled
    index = np.arange(len(values))
    for k in range(values.shape[1]):
        filled[~found, k] = np.interp(index[~found], index[found], values[found, k])
    return filled


def motion_signals(landmarks, found):
    """(frames, 5) array: x, y, scale, roll, mouth.

    x and y are the face-oval centre and mouth is the lip gap, all divided by
    the face width so that one unit is one face width at any resolution.
    scale is the log of the face width and roll is the eye-line angle in radians.
    """
    landmarks = np.asarray(landmarks, dtype=float)
    found = np.asarray(found, dtype=bool)

    width = np.linalg.norm(landmarks[:, CHEEK_RIGHT] - landmarks[:, CHEEK_LEFT], axis=1)
    width = np.where(width > 0, width, np.nan)
    centre = landmarks[:, FACE_OVAL].mean(axis=1)

    eye_line = landmarks[:, EYE_RIGHT] - landmarks[:, EYE_LEFT]
    roll = np.arctan2(eye_line[:, 1], eye_line[:, 0])
    lip_gap = np.linalg.norm(landmarks[:, LIP_BOTTOM] - landmarks[:, LIP_TOP], axis=1)

    signals = np.column_stack([
        centre[:, 0] / width,
        centre[:, 1] / width,
        np.log(width),
        roll,
        lip_gap / width,
    ])
    usable = found & np.isfinite(signals).all(axis=1)
    signals = fill_missing(signals, usable)
    return signals


def motion_spectra(times, signals, starts, length_s, band_bpm, fs=30.0):
    """Summed periodogram of all motion signals per window, not normalised,
    on the same bpm axis as fusion.window_spectra. Returns bpm, (n_windows, n_bins).

    A Hann taper keeps a strong motion peak from leaking into far-away bins,
    which would otherwise be suppressed too.
    """
    n_win = int(round(length_s * fs))
    bpm, _ = band_spectrum(np.zeros(n_win), fs, band_bpm)
    total = np.full((len(starts), len(bpm)), np.nan)

    filtered = []
    for k in range(signals.shape[1]):
        grid, values = to_uniform(times, signals[:, k], fs)
        values = bandpass(values, fs, band_bpm[0] / 60, band_bpm[1] / 60)
        filtered.append(values)

    for i, start in enumerate(starts):
        first = int(round(start * fs))
        if first + n_win > len(filtered[0]):
            continue
        power = np.zeros(len(bpm))
        for values in filtered:
            _, p = band_spectrum(values[first:first + n_win], fs, band_bpm, taper="hann")
            power += p
        total[i] = power
    return bpm, total


def motion_amount(motion):
    """One number per window: total in-band motion power (sum over bins)."""
    return np.nansum(motion, axis=1)


def suppress_motion(spectra, motion, tau):
    """Down-weight the bins where motion has power: each bin is multiplied by
    1 / (1 + motion / tau), then rows are scaled back to sum 1.

    spectra can be (n_windows, n_bins) or (n_candidates, n_windows, n_bins).
    A small tau removes more; when motion is far below tau nothing changes.
    """
    weight = 1.0 / (1.0 + np.nan_to_num(motion) / tau)
    out = spectra * weight
    sums = out.sum(axis=-1, keepdims=True)
    sums = np.where(sums > 0, sums, np.nan)
    out = out / sums
    return out
