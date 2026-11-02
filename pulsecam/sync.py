"""Finding the time offset between a video and its contact pulse recording.

UBFC-Phys video and wrist BVP are not started together: in s1 the same HR
changes show up in the video about 5 s after the BVP.
"""

import numpy as np
from scipy import signal

from pulsecam.reference import bandpass, to_uniform


def hr_track(times, pulse, band_bpm=(42, 210), fs=30.0, length_s=10.0, hop_s=0.5):
    """Spectral-peak HR every hop_s seconds, from length_s long windows."""
    grid, values = to_uniform(times, pulse, fs)
    values = bandpass(values, fs, band_bpm[0] / 60, band_bpm[1] / 60)
    n_seg = int(round(length_s * fs))
    n_hop = int(round(hop_s * fs))
    freqs, _, power = signal.spectrogram(values, fs, nperseg=n_seg, noverlap=n_seg - n_hop,
                                         nfft=int(60 * fs))
    bpm = freqs * 60
    keep = (bpm >= band_bpm[0]) & (bpm <= band_bpm[1])
    track = bpm[keep][np.argmax(power[keep], axis=0)]
    return track


def estimate_lag(pairs, max_lag_s=10.0, hop_s=0.5):
    """How many seconds later things happen in the video than in the contact signal.

    pairs is a list of (video_track, contact_track) HR tracks from hr_track,
    one per recording of the same person; they share one lag. The lag with the
    highest Pearson correlation of the overlapping parts wins.
    Returns (lag_s, correlation).
    """
    max_steps = int(round(max_lag_s / hop_s))
    best_lag = 0.0
    best_r = -np.inf
    for k in range(-max_steps, max_steps + 1):
        video_parts = []
        contact_parts = []
        for video, contact in pairs:
            n = min(len(video), len(contact))
            if k >= 0:
                video_parts.append(video[k:n])
                contact_parts.append(contact[:n - k])
            else:
                video_parts.append(video[:n + k])
                contact_parts.append(contact[-k:n])
        a = np.concatenate(video_parts)
        b = np.concatenate(contact_parts)
        if len(a) < 10 or np.std(a) == 0 or np.std(b) == 0:
            continue
        r = np.corrcoef(a, b)[0, 1]
        if r > best_r:
            best_r = r
            best_lag = k * hop_s
    return best_lag, best_r
