"""Reference heart rate from a contact PPG signal (pulse oximeter or wrist sensor)."""

import numpy as np
from scipy import signal


def to_uniform(times, values, fs):
    """Resample an irregularly sampled signal onto a uniform grid at fs Hz."""
    start = times[0]
    end = times[-1]
    n_samples = int(np.floor((end - start) * fs)) + 1
    new_times = start + np.arange(n_samples) / fs
    new_values = np.interp(new_times, times, values)
    return new_times, new_values


def bandpass(x, fs, low_hz, high_hz, order=3):
    nyquist = fs / 2.0
    low = low_hz / nyquist
    high = high_hz / nyquist
    sos = signal.butter(order, [low, high], btype="bandpass", output="sos")
    filtered = signal.sosfiltfilt(sos, x)
    return filtered


def find_beats(ppg, fs, band_bpm=(42, 210)):
    """Return the sample indices of systolic peaks in a PPG signal."""
    low_hz = band_bpm[0] / 60.0
    high_hz = band_bpm[1] / 60.0
    clean = bandpass(ppg, fs, low_hz, high_hz)

    # two peaks can't be closer than one beat at the highest allowed rate
    min_gap = int(fs / high_hz)

    # prominence relative to the signal spread, so the threshold works for
    # any amplitude scale (UBFC and UBFC-Phys use very different units)
    spread = np.percentile(clean, 95) - np.percentile(clean, 5)
    min_prominence = 0.2 * spread

    peaks, _ = signal.find_peaks(clean, distance=min_gap, prominence=min_prominence)
    return peaks, clean


def refine_peaks(x, peaks):
    """Sub-sample peak positions from a parabola through each peak and its neighbours.

    Without this, beat intervals snap to whole samples and HR at 160 bpm
    comes out 3-4 bpm off.
    """
    refined = peaks.astype(float)
    for i, p in enumerate(peaks):
        if p == 0 or p == len(x) - 1:
            continue
        left = x[p - 1]
        mid = x[p]
        right = x[p + 1]
        denom = left - 2 * mid + right
        if denom == 0:
            continue
        shift = 0.5 * (left - right) / denom
        refined[i] = p + shift
    return refined


def band_spectrum(x, fs, band_bpm=(42, 210), taper="boxcar"):
    """Periodogram inside the band, zero-padded to 1 bpm steps or finer. Returns bpm, power."""
    n_fft = max(len(x), int(60 * fs))
    freqs, power = signal.periodogram(x, fs=fs, window=taper, nfft=n_fft, detrend="linear")
    bpm = freqs * 60.0
    in_band = (bpm >= band_bpm[0]) & (bpm <= band_bpm[1])
    return bpm[in_band], power[in_band]


def peak_snr_db(bpm, power, half_width_bpm=6.0):
    """SNR as in de Haan and Jeanne 2013: power within half_width of the
    highest peak and of twice its rate, against the rest of the band."""
    peak = bpm[np.argmax(power)]
    near_peak = np.abs(bpm - peak) <= half_width_bpm
    near_harmonic = np.abs(bpm - 2 * peak) <= half_width_bpm
    inside = near_peak | near_harmonic
    signal_power = power[inside].sum()
    noise_power = power[~inside].sum()
    if noise_power <= 0:
        noise_power = 1e-12
    return 10 * np.log10(signal_power / noise_power)


def spectral_hr(x, fs, band_bpm=(42, 210)):
    """HR from the highest periodogram peak inside the band."""
    bpm, power = band_spectrum(x, fs, band_bpm)
    best = np.argmax(power)
    return bpm[best]


def window_starts(duration_s, length_s, step_s):
    last_start = duration_s - length_s
    if last_start < 0:
        return np.array([])
    n = int(np.floor(last_start / step_s)) + 1
    starts = np.arange(n) * step_s
    return starts


def windowed_hr(beat_times, starts, length_s, min_beats=5, max_ibi_cv=0.3):
    """Heart rate per window from beat times.

    Returns (hr_bpm, ok). hr is 60 / median inter-beat interval. ok is False
    when the window has too few beats or the intervals jump around too much,
    which usually means the contact signal itself is bad there.
    """
    hr = np.full(len(starts), np.nan)
    ok = np.zeros(len(starts), dtype=bool)

    for i, start in enumerate(starts):
        end = start + length_s
        inside = (beat_times >= start) & (beat_times < end)
        times = beat_times[inside]
        if len(times) < min_beats:
            continue

        ibis = np.diff(times)
        median_ibi = np.median(ibis)
        hr[i] = 60.0 / median_ibi

        cv = np.std(ibis) / np.mean(ibis)
        ok[i] = cv <= max_ibi_cv

    return hr, ok


def reference_hr(times, ppg, length_s=10, step_s=1, band_bpm=(42, 210),
                 min_beats=5, max_ibi_cv=0.15, max_disagree_bpm=5.0,
                 max_gap_s=0.3, fs=60.0):
    """Windowed reference HR from a raw PPG trace with timestamps in seconds.

    Window start times are relative to times[0]. A window is ok when its beat
    intervals are regular, the peak-based HR agrees with the spectral HR (on
    band-passed noise the two rarely agree) and the recording has no hole
    longer than max_gap_s in it (a longer hole can hide a beat).
    """
    times = np.asarray(times, dtype=float)
    ppg = np.asarray(ppg, dtype=float)
    times = times - times[0]

    grid, values = to_uniform(times, ppg, fs)
    peaks, clean = find_beats(values, fs, band_bpm)
    exact_peaks = refine_peaks(clean, peaks)
    beat_times = exact_peaks / fs

    starts = window_starts(grid[-1], length_s, step_s)
    hr, ok = windowed_hr(beat_times, starts, length_s, min_beats, max_ibi_cv)

    spec_hr = np.full(len(starts), np.nan)
    n_win = int(round(length_s * fs))
    for i, start in enumerate(starts):
        first = int(round(start * fs))
        chunk = clean[first:first + n_win]
        spec_hr[i] = spectral_hr(chunk, fs, band_bpm)

    gap_ends = times[1:][np.diff(times) > max_gap_s]
    no_gap = np.ones(len(starts), dtype=bool)
    for end in gap_ends:
        inside = (starts < end) & (starts + length_s > end - max_gap_s)
        no_gap[inside] = False

    agree = np.abs(hr - spec_hr) <= max_disagree_bpm
    in_band = (hr >= band_bpm[0]) & (hr <= band_bpm[1])
    ok = ok & in_band & agree & no_gap

    result = {
        "start_s": starts,
        "hr_bpm": hr,
        "spectral_hr_bpm": spec_hr,
        "ok": ok,
        "beat_times": beat_times,
    }
    return result


def spectral_reference_hr(times, ppg, length_s=10, step_s=1, band_bpm=(42, 210),
                          min_snr_db=0.0, max_gap_s=0.3, fs=60.0):
    """A looser reference: the spectral peak of the contact signal, trusted
    when its SNR is at least min_snr_db and there is no recording gap.

    For wrist BVP during movement, where single beats can't be found reliably
    but the pulse is still the strongest rhythm. Window starts are relative
    to times[0], as in reference_hr.
    """
    times = np.asarray(times, dtype=float)
    ppg = np.asarray(ppg, dtype=float)
    times = times - times[0]
    grid, values = to_uniform(times, ppg, fs)
    clean = bandpass(values, fs, band_bpm[0] / 60.0, band_bpm[1] / 60.0)

    starts = window_starts(grid[-1], length_s, step_s)
    hr = np.full(len(starts), np.nan)
    snr = np.full(len(starts), np.nan)
    n_win = int(round(length_s * fs))
    for i, start in enumerate(starts):
        first = int(round(start * fs))
        chunk = clean[first:first + n_win]
        bpm, power = band_spectrum(chunk, fs, band_bpm)
        hr[i] = bpm[np.argmax(power)]
        snr[i] = peak_snr_db(bpm, power)

    gap_ends = times[1:][np.diff(times) > max_gap_s]
    no_gap = np.ones(len(starts), dtype=bool)
    for end in gap_ends:
        inside = (starts < end) & (starts + length_s > end - max_gap_s)
        no_gap[inside] = False

    result = {
        "start_s": starts,
        "hr_bpm": hr,
        "snr_db": snr,
        "ok": (snr >= min_snr_db) & no_gap,
    }
    return result
