"""Combining many candidate pulse signals into one HR per window."""

import numpy as np

from pulsecam.reference import band_spectrum, bandpass, peak_snr_db, to_uniform


def window_spectra(times, pulse, starts, length_s, band_bpm, fs=30.0):
    """Power spectrum of each window, scaled to sum 1. Rows are NaN where the
    window runs past the end of the signal. Returns bpm, (n_windows, n_bins)."""
    grid, values = to_uniform(times, pulse, fs)
    values = bandpass(values, fs, band_bpm[0] / 60, band_bpm[1] / 60)
    n_win = int(round(length_s * fs))

    # one dummy call to get the bpm axis, it is the same for every window
    bpm, _ = band_spectrum(np.zeros(n_win), fs, band_bpm)
    spectra = np.full((len(starts), len(bpm)), np.nan)
    for i, start in enumerate(starts):
        first = int(round(start * fs))
        chunk = values[first:first + n_win]
        if len(chunk) < n_win:
            continue
        _, power = band_spectrum(chunk, fs, band_bpm)
        total = power.sum()
        if total > 0:
            spectra[i] = power / total
    return bpm, spectra


def snr_db(bpm, spectra, half_width_bpm=6.0):
    """SNR of each spectrum row (reference.peak_snr_db), NaN for empty rows."""
    snr = np.full(len(spectra), np.nan)
    for i, row in enumerate(spectra):
        if np.isnan(row[0]):
            continue
        snr[i] = peak_snr_db(bpm, row, half_width_bpm)
    return snr


def snr_weights(snr, rule):
    """Fusion weights from SNR in dB, shape (n_candidates, n_windows).

    equal: every candidate the same; snr: linear SNR; snr2: linear SNR squared;
    best: only the candidate with the highest SNR in that window.
    """
    if rule == "equal":
        weights = np.ones_like(snr)
    elif rule == "snr":
        weights = 10 ** (snr / 10)
    elif rule == "snr2":
        weights = 10 ** (snr / 5)
    elif rule == "best":
        filled = np.where(np.isnan(snr), -np.inf, snr)
        weights = np.zeros_like(snr)
        winners = np.argmax(filled, axis=0)
        weights[winners, np.arange(snr.shape[1])] = 1.0
    else:
        raise ValueError("unknown rule: " + rule)
    weights = np.where(np.isnan(snr), 0.0, weights)
    return weights


def fused_spectrum(spectra, weights):
    """Weighted sum of candidate spectra, (n_candidates, n_windows, n_bins) to (n_windows, n_bins)."""
    filled = np.nan_to_num(spectra)
    combined = np.zeros(filled.shape[1:])
    for c in range(len(filled)):
        combined += weights[c][:, None] * filled[c]
    return combined


def fused_hr(bpm, spectra, weights):
    """Peak of the weighted mean spectrum. spectra: (n_candidates, n_windows, n_bins)."""
    combined = fused_spectrum(spectra, weights)
    hr = bpm[np.argmax(combined, axis=1)]
    no_data = weights.sum(axis=0) == 0
    hr = np.where(no_data, np.nan, hr)
    return hr
