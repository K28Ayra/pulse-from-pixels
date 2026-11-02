"""Tracking HR across windows so a single bad window can't jump to a harmonic."""

import numpy as np


def viterbi_track(bpm, spectra, sigma_bpm=3.0, floor=1e-4, lag=None, jump=0.0):
    """Most likely HR path through the windows' spectra.

    spectra is (n_windows, n_bins), each row a fused spectrum. A row of NaN or
    zeros (no data) counts as flat. Moving d bpm between neighbouring windows
    costs (d / sigma_bpm)^2 / 2 in log terms. jump is the probability of leaving
    that Gaussian for any other bin, so a large real HR change stays possible.

    lag=None uses the whole recording (backtracking from the end). With a
    number, the HR of window t is decided once window t + lag has been seen,
    by backtracking lag steps from the best path end at that point, as the
    live app would with lag seconds of delay. lag=0 only looks at the past.
    """
    n_windows, n_bins = spectra.shape
    probs = np.full((n_windows, n_bins), 1.0 / n_bins)
    for t in range(n_windows):
        row = np.nan_to_num(spectra[t])
        total = row.sum()
        if total > 0:
            probs[t] = row / total
    emission = np.log(probs + floor)

    step = bpm[:, None] - bpm[None, :]
    transition = -0.5 * (step / sigma_bpm) ** 2
    if jump > 0:
        transition = np.logaddexp(transition + np.log(1 - jump), np.log(jump / n_bins))

    score = emission[0].copy()
    came_from = np.zeros((n_windows, n_bins), dtype=int)
    hr = np.full(n_windows, np.nan)
    if lag == 0:
        hr[0] = bpm[np.argmax(score)]
    for t in range(1, n_windows):
        # candidates[i, j]: be at bin i before and move to bin j now
        candidates = score[:, None] + transition
        came_from[t] = np.argmax(candidates, axis=0)
        best_before = candidates[came_from[t], np.arange(n_bins)]
        score = best_before + emission[t]
        score = score - score.max()
        if lag is not None and t >= lag:
            state = np.argmax(score)
            for back in range(t, t - lag, -1):
                state = came_from[back, state]
            hr[t - lag] = bpm[state]

    # what is still undecided at the end comes from the full backtrack
    path = np.zeros(n_windows, dtype=int)
    path[-1] = np.argmax(score)
    for t in range(n_windows - 1, 0, -1):
        path[t - 1] = came_from[t, path[t]]
    full = bpm[path]
    undecided = np.isnan(hr)
    hr[undecided] = full[undecided]
    return hr
