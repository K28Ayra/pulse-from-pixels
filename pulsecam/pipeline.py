"""From per-patch colour traces to candidate spectra, shared by the evaluation scripts."""

import numpy as np

from pulsecam.fusion import snr_db, window_spectra
from pulsecam.rppg import run_method


def fill_gaps(rgb):
    """Interpolate frames where a patch had no pixels (NaN)."""
    filled = rgb.copy()
    index = np.arange(len(rgb))
    missing = np.isnan(rgb[:, 0])
    if missing.all() or not missing.any():
        return filled
    known = ~missing
    for channel in range(3):
        filled[missing, channel] = np.interp(index[missing], index[known], rgb[known, channel])
    return filled


def candidate_spectra(times, patch_rgb, patch_names, methods, starts, length_s, band_bpm):
    """Every patch times every method.

    patch_rgb is (frames, patches, 3). Returns the (patch, method) list, the bpm
    axis and spectra of shape (n_candidates, n_windows, n_bins).
    """
    fs = (len(times) - 1) / (times[-1] - times[0])
    candidates = []
    all_spectra = []
    bpm = None
    for j, patch in enumerate(patch_names):
        rgb = fill_gaps(patch_rgb[:, j])
        for method in methods:
            pulse = run_method(method, rgb, fs)
            bpm, spectra = window_spectra(times, pulse, starts, length_s, band_bpm)
            candidates.append((patch, method))
            all_spectra.append(spectra)
    return candidates, bpm, np.stack(all_spectra)


def all_snr(bpm, spectra):
    """SNR in dB for (n_candidates, n_windows, n_bins) spectra."""
    rows = []
    for s in spectra:
        rows.append(snr_db(bpm, s))
    return np.stack(rows)
