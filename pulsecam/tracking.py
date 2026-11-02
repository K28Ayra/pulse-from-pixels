"""Tracking HR across windows so a single bad window can't jump to a harmonic."""

import numpy as np


def log_emission(spectra, floor=1e-4):
    """Each row scaled to sum 1, then log. Empty rows become flat."""
    n_windows, n_bins = spectra.shape
    probs = np.full((n_windows, n_bins), 1.0 / n_bins)
    for t in range(n_windows):
        row = np.nan_to_num(spectra[t])
        total = row.sum()
        if total > 0:
            probs[t] = row / total
    return np.log(probs + floor)


def log_transition(bpm, sigma_bpm, jump):
    """transition[i, j]: log chance of moving from bin i to bin j in one window."""
    step = bpm[:, None] - bpm[None, :]
    transition = -0.5 * (step / sigma_bpm) ** 2
    if jump > 0:
        transition = np.logaddexp(transition + np.log(1 - jump), np.log(jump / len(bpm)))
    return transition


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
    emission = log_emission(spectra, floor)
    transition = log_transition(bpm, sigma_bpm, jump)

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


class OnlineTracker:
    """viterbi_track(lag=lag) one window at a time, for the live app.

    step() returns the index of the window it just decided and its HR. While
    fewer than lag windows exist, the lag is shorter, down to 0 for the first.
    From window lag on, the answers are the same as viterbi_track's.
    """

    def __init__(self, bpm, sigma_bpm=3.0, lag=0, jump=0.0, floor=1e-4):
        self.bpm = bpm
        self.lag = lag
        self.floor = floor
        self.transition = log_transition(bpm, sigma_bpm, jump)
        self.score = None
        self.came_from = []
        self.n_windows = 0

    def step(self, spectrum):
        emission = log_emission(spectrum[None], self.floor)[0]
        if self.score is None:
            self.score = emission
        else:
            candidates = self.score[:, None] + self.transition
            came_from = np.argmax(candidates, axis=0)
            best_before = candidates[came_from, np.arange(len(self.bpm))]
            self.score = best_before + emission
            self.score = self.score - self.score.max()
            self.came_from.append(came_from)
            # only the last lag steps are ever walked back
            self.came_from = self.came_from[-max(self.lag, 1):]
        self.n_windows += 1

        lag = min(self.lag, self.n_windows - 1)
        state = np.argmax(self.score)
        if lag > 0:
            for came_from in reversed(self.came_from[-lag:]):
                state = came_from[state]
        return self.n_windows - 1 - lag, self.bpm[state]
