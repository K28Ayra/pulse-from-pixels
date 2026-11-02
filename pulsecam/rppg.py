"""Classical rPPG methods. Input is an (n_frames, 3) array of mean R, G, B."""

import math

import numpy as np

from pulsecam.reference import bandpass, spectral_hr, to_uniform


def green(rgb):
    pulse = rgb[:, 1] - np.mean(rgb[:, 1])
    return pulse


def pos(rgb, fs, window_s=1.6):
    """Plane-orthogonal-to-skin, Wang et al. 2017, Algorithm 1 (overlap-add)."""
    n_frames = len(rgb)
    length = math.ceil(window_s * fs)
    projection = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
    pulse = np.zeros(n_frames)

    for start in range(0, n_frames - length + 1):
        end = start + length
        chunk = rgb[start:end]
        normalised = chunk / np.mean(chunk, axis=0)
        s = normalised @ projection.T
        s1 = s[:, 0]
        s2 = s[:, 1]
        alpha = np.std(s1) / np.std(s2)
        h = s1 + alpha * s2
        pulse[start:end] += h - np.mean(h)
    return pulse


def chrom(rgb, fs, window_s=1.6, band_hz=(0.7, 2.5)):
    """Chrominance method, de Haan and Jeanne 2013, Hann-windowed overlap-add."""
    n_frames = len(rgb)
    length = math.ceil(window_s * fs)
    if length % 2 == 1:
        length += 1
    half = length // 2
    hann = np.hanning(length)
    pulse = np.zeros(n_frames)

    for start in range(0, n_frames - length + 1, half):
        end = start + length
        chunk = rgb[start:end]
        normalised = chunk / np.mean(chunk, axis=0)
        r = normalised[:, 0]
        g = normalised[:, 1]
        b = normalised[:, 2]
        x = 3 * r - 2 * g
        y = 1.5 * r + g - 1.5 * b
        x = bandpass(x, fs, band_hz[0], band_hz[1])
        y = bandpass(y, fs, band_hz[0], band_hz[1])
        alpha = np.std(x) / np.std(y)
        s = x - alpha * y
        pulse[start:end] += s * hann
    return pulse


METHODS = {"GREEN": green, "POS": pos, "CHROM": chrom}


def run_method(name, rgb, fs):
    function = METHODS[name]
    if name == "GREEN":
        return function(rgb)
    return function(rgb, fs)


def windowed_pulse_hr(times, pulse, starts, length_s, band_bpm, fs=30.0):
    """Spectral-peak HR of a pulse signal in each window; times in seconds from 0."""
    grid, values = to_uniform(times, pulse, fs)
    values = bandpass(values, fs, band_bpm[0] / 60, band_bpm[1] / 60)
    n_win = int(round(length_s * fs))
    hr = np.full(len(starts), np.nan)
    for i, start in enumerate(starts):
        first = int(round(start * fs))
        chunk = values[first:first + n_win]
        if len(chunk) < n_win:
            continue
        hr[i] = spectral_hr(chunk, fs, band_bpm)
    return hr
