"""Heart rate from a stream of frames, one frame at a time, for the webcam app.

Same method as the phase 5 evaluation: a 72x72 face crop whose box moves every
BOX_EVERY frames, FactorizePhys (exported to ONNX) on 160-frame chunks, 10 s
windows every second and the tracker from configs/default.yaml (deep_tracking).
What differs: the model runs once per window on the latest frames and the
overlapping outputs are averaged, and the first windows are shorter, so the
first HR comes after about 6 s instead of 15 s.

With background=True the model runs in a thread (about 0.15 s per call on two
cores), so the camera loop doesn't stall. The replay evaluation runs it inline
so its results don't depend on timing.
"""

import threading

import numpy as np
import onnxruntime

from pulsecam.face import BOX_EVERY, crop_box, face_crop
from pulsecam.reference import band_spectrum, bandpass, to_uniform
from pulsecam.tracking import OnlineTracker

CHUNK = 160
FS = 30.0
NEAR_BPM = 5.0


def load_model(path, threads):
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return onnxruntime.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


class LiveHeartRate:
    """Call push() with every frame. find_face(rgb, time_s) gives (478, 2) landmarks or None."""

    def __init__(self, cfg, find_face, model=None, background=False):
        self.live = cfg["live"]
        self.tracking = cfg["deep_tracking"]
        self.band = cfg["hr_band_bpm"]
        self.length_s = cfg["window"]["length_s"]
        self.step_s = cfg["window"]["step_s"]
        self.cutoff = cfg["quality"]["cutoff"]
        self.crop_size = cfg["crop_size"]
        self.find_face = find_face
        if model is None:
            model = load_model(self.live["model"], self.live["threads"])
        self.model = model
        self.background = background
        self.lock = threading.Lock()
        self.worker = None
        # the chunk outputs are averaged with this weight, so chunk edges count less
        self.chunk_weight = np.hanning(CHUNK) + 0.1
        self.reset()

    def reset(self):
        with self.lock:
            self.box = None
            self.frames_since_check = 0
            self.lost_since = None
            self.start_time = None
            self.times = []
            self.sums = []
            self.weights = []
            # frames numbered from the start; times, sums and weights begin at frame n_dropped
            self.n_frames = 0
            self.n_dropped = 0
            self.crops = []
            self.finished = None
            # a thread still running from before a reset sees a new number and gives up
            self.generation = getattr(self, "generation", 0) + 1
            self.next_end = self.live["first_window_s"]
            self.tracker = None
            self.recent_spectra = []
            self.status = "looking for a face"

    def push(self, rgb, time_s):
        """Returns a reading (dict) when a window has just finished, else None."""
        if self.box is None or self.lost_since is not None or self.frames_since_check >= BOX_EVERY:
            self.check_face(rgb, time_s)
        if self.box is None:
            return None
        self.frames_since_check += 1

        if self.start_time is None:
            self.start_time = time_s
        with self.lock:
            self.times.append(time_s - self.start_time)
            self.sums.append(0.0)
            self.weights.append(0.0)
            self.n_frames += 1
            self.forget_old()
        self.crops.append(face_crop(rgb, self.box, self.crop_size))
        self.crops = self.crops[-(CHUNK + 1):]

        if self.times[-1] >= self.next_end:
            end = self.next_end
            self.next_end += self.step_s
            if len(self.crops) < CHUNK + 1:
                self.status = "starting"
            else:
                job = (np.array(self.crops, dtype=np.float32), self.n_frames - (CHUNK + 1), end, self.generation)
                if not self.background:
                    self.finish_window(*job)
                elif self.worker is None or not self.worker.is_alive():
                    self.worker = threading.Thread(target=self.finish_window, args=job, daemon=True)
                    self.worker.start()
                # else: the last window is still running, this one is skipped

        reading = self.finished
        self.finished = None
        return reading

    def check_face(self, rgb, time_s):
        points = self.find_face(rgb, time_s)
        self.frames_since_check = 0
        if points is not None:
            self.box = crop_box(points)
            self.lost_since = None
            return
        if self.box is None:
            return
        if self.lost_since is None:
            self.lost_since = time_s
        # a short miss keeps the old box, a long one starts again
        if time_s - self.lost_since > self.live["face_lost_reset_s"]:
            self.reset()

    def forget_old(self):
        keep = int(self.live["buffer_s"] * FS * 1.5)
        if len(self.times) > keep:
            cut = len(self.times) - keep
            del self.times[:cut]
            del self.sums[:cut]
            del self.weights[:cut]
            self.n_dropped += cut

    def finish_window(self, chunk, first_frame, end, generation):
        x = chunk.transpose(3, 0, 1, 2)[None]
        output = self.model.run(None, {"frames": x})[0][0]
        output = (output - output.mean()) / (output.std() + 1e-9)
        with self.lock:
            if generation != self.generation:
                return
            # output sample i belongs to frame i of the chunk, as in scripts/run_deep.py
            first = first_frame - self.n_dropped
            for i in range(CHUNK):
                if first + i >= 0:
                    self.sums[first + i] += self.chunk_weight[i] * output[i]
                    self.weights[first + i] += self.chunk_weight[i]
            self.finished = self.read_window(end)

    def pulse(self):
        """Times and pulse of the frames the model has covered so far."""
        with self.lock:
            weights = np.array(self.weights)
            times = np.array(self.times)
            sums = np.array(self.sums)
        covered = weights > 0
        return times[covered], sums[covered] / weights[covered]

    def read_window(self, end):
        """Called with the lock held."""
        weights = np.array(self.weights)
        covered = weights > 0
        times = np.array(self.times)[covered]
        values = np.array(self.sums)[covered] / weights[covered]
        recent = times >= end - self.live["buffer_s"]
        times = times[recent]
        values = values[recent]
        length = min(self.length_s, end - times[0])

        grid, uniform = to_uniform(times, values, FS)
        uniform = bandpass(uniform, FS, self.band[0] / 60, self.band[1] / 60)
        bpm, power = band_spectrum(uniform[grid >= end - length], FS, self.band)
        spectrum = power / power.sum()

        if self.tracker is None:
            self.tracker = OnlineTracker(
                bpm,
                sigma_bpm=self.tracking["sigma_bpm"],
                lag=self.live["lag_s"],
                jump=self.tracking["jump"],
            )
        self.recent_spectra.append((end, length, spectrum))
        self.recent_spectra = self.recent_spectra[-(self.live["lag_s"] + 1):]
        index, hr = self.tracker.step(spectrum ** self.tracking["sharpen"])

        lag = self.tracker.n_windows - 1 - index
        window_end, window_length, window_spectrum = self.recent_spectra[-1 - lag]
        share = float(window_spectrum[np.abs(bpm - hr) <= NEAR_BPM].sum())
        self.status = "ok" if share >= self.cutoff else "not trusted"
        return {
            "time_s": end,
            "hr": float(hr),
            "share": share,
            "trusted": share >= self.cutoff,
            "window_end_s": window_end,
            "window_s": window_length,
        }
