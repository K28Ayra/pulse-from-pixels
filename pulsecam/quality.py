"""Quality features of a candidate spectrum and the model that scores them."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pulsecam.reference import peak_snr_db

FEATURE_NAMES = [
    "snr_db",
    "peak_share",
    "entropy",
    "second_peak",
    "harmonic",
    "agreement",
    "distance_to_median",
    "motion_total",
    "motion_at_peak",
]
PEAK_WIDTH_BPM = 6.0
AGREE_BPM = 5.0
WINDOW_FEATURE_NAMES = [
    "peak_share",
    "share_at_hr",
    "fused_snr_db",
    "tracker_gap",
    "agreement_at_hr",
    "best_snr_db",
    "recent_peak_spread",
    "recent_snr_db",
    "motion_total",
]
RECENT_WINDOWS = 10


def spectrum_features(bpm, row, motion_row):
    """Features of one spectrum that don't depend on the other candidates."""
    power = row / row.sum()
    peak = np.argmax(power)
    near = np.abs(bpm - bpm[peak]) <= PEAK_WIDTH_BPM
    second = power[~near].max() if np.any(~near) else 0.0
    double = np.abs(bpm - 2 * bpm[peak]) <= PEAK_WIDTH_BPM
    harmonic = power[double].max() if np.any(double) else 0.0
    positive = power[power > 0]
    entropy = -np.sum(positive * np.log(positive)) / np.log(len(power))
    motion_total = np.log10(np.sum(motion_row) + 1e-9)
    motion_at_peak = np.log10(motion_row[peak] + 1e-9)
    return [
        peak_snr_db(bpm, power, PEAK_WIDTH_BPM),
        power[peak],
        entropy,
        second / power[peak],
        harmonic / power[peak],
        motion_total,
        motion_at_peak,
    ]


def candidate_features(bpm, spectra, motion):
    """Features for every candidate in every window.

    spectra is (n_candidates, n_windows, n_bins), motion (n_windows, n_bins).
    Returns features (n_candidates, n_windows, n_features) and the peak HR of
    each candidate (n_candidates, n_windows). Empty spectra give NaN rows.
    """
    n_candidates, n_windows, _ = spectra.shape
    features = np.full((n_candidates, n_windows, len(FEATURE_NAMES)), np.nan)
    peaks = np.full((n_candidates, n_windows), np.nan)
    clean_motion = np.nan_to_num(motion)
    for t in range(n_windows):
        for c in range(n_candidates):
            row = spectra[c, t]
            if np.any(np.isnan(row)) or row.sum() <= 0:
                continue
            peaks[c, t] = bpm[np.argmax(row)]
            single = spectrum_features(bpm, row, clean_motion[t])
            features[c, t, :5] = single[:5]
            features[c, t, 7:] = single[5:]
        found = peaks[:, t]
        if np.all(np.isnan(found)):
            continue
        middle = np.nanmedian(found)
        for c in range(n_candidates):
            if np.isnan(found[c]):
                continue
            others = np.delete(found, c)
            others = others[~np.isnan(others)]
            if len(others) > 0:
                features[c, t, 5] = np.mean(np.abs(others - found[c]) <= AGREE_BPM)
            else:
                features[c, t, 5] = 0.0
            features[c, t, 6] = abs(found[c] - middle)
    return features, peaks


def window_features(bpm, spectra, snr, weights, motion, hr):
    """Features of the final HR of each window, using nothing from the future.

    spectra (n_candidates, n_windows, n_bins), snr and weights (n_candidates,
    n_windows), motion (n_windows, n_bins), hr (n_windows) from the tracker.
    """
    n_windows = spectra.shape[1]
    features = np.full((n_windows, len(WINDOW_FEATURE_NAMES)), np.nan)
    filled = np.nan_to_num(spectra)
    candidate_peaks = bpm[np.argmax(filled, axis=2)]
    best_snr = np.nanmax(np.where(np.isnan(snr), -50.0, snr), axis=0)
    fused_peaks = np.full(n_windows, np.nan)
    fused_snr = np.full(n_windows, np.nan)
    clean_motion = np.nan_to_num(motion)
    for t in range(n_windows):
        row = (weights[:, t, None] * filled[:, t]).sum(axis=0)
        total = row.sum()
        if total <= 0:
            continue
        row = row / total
        peak = np.argmax(row)
        fused_peaks[t] = bpm[peak]
        fused_snr[t] = peak_snr_db(bpm, row, PEAK_WIDTH_BPM)
        first = max(0, t - RECENT_WINDOWS + 1)
        near_hr = np.abs(bpm - hr[t]) <= AGREE_BPM
        features[t, 0] = row[peak]
        features[t, 1] = row[near_hr].sum()
        features[t, 2] = fused_snr[t]
        features[t, 3] = abs(fused_peaks[t] - hr[t])
        features[t, 4] = np.mean(np.abs(candidate_peaks[:, t] - hr[t]) <= AGREE_BPM)
        features[t, 5] = best_snr[t]
        features[t, 6] = np.nanstd(fused_peaks[first:t + 1])
        features[t, 7] = np.nanmean(fused_snr[first:t + 1])
        features[t, 8] = np.log10(np.sum(clean_motion[t]) + 1e-9)
    return features


def fit_quality_model(features, is_good, sample_weight=None, kind="logistic"):
    """Chance of being within 5 bpm: logistic regression or small boosted trees."""
    if kind == "logistic":
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(features, is_good, logisticregression__sample_weight=sample_weight)
        return model
    model = HistGradientBoostingClassifier(max_depth=3, max_iter=100, learning_rate=0.1)
    model.fit(features, is_good, sample_weight=sample_weight)
    return model


def quality_scores(model, features):
    """Chance of being good, same leading shape as features. NaN rows score 0."""
    n_features = features.shape[-1]
    flat = features.reshape(-1, n_features)
    usable = ~np.any(np.isnan(flat), axis=1)
    scores = np.zeros(len(flat))
    if np.any(usable):
        scores[usable] = model.predict_proba(flat[usable])[:, 1]
    return scores.reshape(features.shape[:-1])


def keep_top_share(pred, ref, score, share=0.8):
    """MAE of all windows and of the share with the highest score."""
    errors = np.abs(pred - ref)
    cutoff = np.quantile(score, 1 - share)
    kept = score >= cutoff
    return errors.mean(), errors[kept].mean(), kept.mean()
