# Phase 7: the webcam app

```bash
conda env create -f environment-live.yml
conda activate pulsecam-live
python scripts/live.py
```

You get your face with a box around it, a number in bpm, and the last few seconds of the pulse drawn along the bottom. The number turns grey with "not trusted, keep still" when the phase 4 flag fires. `--video file.avi` plays a file instead of the camera, and `--no-window` only prints the readings.

It runs the phase 5 winner, FactorizePhys with the tracker. The app env has no torch, so the model goes through onnxruntime. `scripts/export_onnx.py` makes `models/factorizephys.onnx` (226 KB) from the toolbox checkpoint. The one snag was `torch.diff`, which ONNX can't export. It's used once, on the input frames, so during the export I swap it for slicing and subtracting. The ONNX output matches the phase 5 output to 0.0002, and the HR is identical in every window on four dev videos.

| Target | Result | |
|---|---|---|
| >= 30 fps at 640x480 on a laptop CPU | 163 fps flat out on 4 cores of a Xeon Gold 5317, 30 fps paced with nothing dropped. 30.0 fps on my laptop's webcam | pass |
| first HR within 10 s | 6.0 to 7.0 s on all 30 dev people | pass |
| privacy test (no file writes, no sockets) | `tests/test_privacy.py` passes | pass |

## Getting the first number out sooner

Offline, the first HR needs a full 10 s window plus 5 s of tracker delay, so 15 s. Too slow for the target. In the app the first windows are shorter: the first one is 6 s (the model needs 161 frames, about 5.4 s), and they grow by a second until they reach 10. The tracker's delay also starts at 0 and grows to 5 windows. So the number shows up at 6 s. From 11 s on, windows and delay are the same as offline.

The first reading is less precise. Against the reference for the first 10 s it's 1.7 bpm off (median), and 25 of 28 people are within 5 bpm. The worst was subject 41, 91 against 103.

## Does live give the same answers as offline?

Offline, the model runs on back-to-back 160-frame chunks of the whole recording. Live can't do that, so it runs the model once per window on the latest 161 frames and averages the overlapping outputs, with a Hann weight so the chunk edges count less. The band-pass filter also only sees the last 20 s instead of the whole recording.

To check, `scripts/evaluate_phase7.py replay` feeds every dev UBFC-rPPG video through `pulsecam/live.py` frame by frame, with the real timestamps, as a camera would. Same 1,426 windows for every row:

| | MAE | per person | within 5 bpm |
|---|---|---|---|
| phase 5 offline, lag 5 | 0.771 | 0.856 | 98.9% |
| live, lag 5 (what the app shows) | 0.798 | 0.888 | 99.0% |
| live, lag 2 | 0.830 | 0.911 | 98.9% |
| live, lag 0 | 0.964 | 1.038 | 97.9% |

Live is 3.5% worse than offline. I also checked whether the tracker delay is worth it, since a number that's 5 s old feels odd on a live screen. Lag 0 is 21% worse, so I kept 5. The number on screen describes the 10 s that ended 5 s ago.

The replay is UBFC-rPPG only. UBFC-Phys runs at 35 fps, and the phase 5 numbers there were made by resampling to 30 Hz, which the app doesn't do.

## Speed

`taskset -c 0-3 python scripts/evaluate_phase7.py speed`, one 640x480 video, ms per frame:

| stage | ms |
|---|---|
| read frame | 0.51 |
| BGR to RGB | 0.12 |
| face landmarks (once every 30 frames, 17 ms a call) | 0.58 |
| model (once a second, 129 ms a call) | 3.84 |
| rest of push (crop, spectrum, tracker) | 0.78 |
| drawing | 0.30 |
| all | 6.13 |

Face landmarks only run once a second because the crop box only moves once a second (phase 5 did the same). Every frame would be about 23 ms instead of 6, still above 30 fps.

The model was the real problem. At 130-200 ms a call, running it in the camera loop would freeze the picture for a fifth of a second every second, and the camera keeps capturing, so frame times would bunch up. Now it runs in a thread. onnxruntime lets go of Python's lock while it works, so the loop keeps going. With frames fed at 30 fps the median frame takes 2.6 ms and none are dropped. 47 frames of 1,547 take 34-43 ms, and every one of them is a face-landmark frame. Landmarks take 17 ms a call when the loop runs flat out and about 40 ms when it's paced. I think the CPU clocks down between frames, but I haven't checked.

onnxruntime is slower than torch here: 152 ms against 114 ms on 2 threads. Full graph optimisation didn't help (152 vs 153).

## Privacy

Live mode doesn't write anything or open a socket. Instead of trusting myself on that, `pulsecam/privacy.py` adds a Python audit hook (PEP 578) once the models are loaded. From then on, any `open` for writing, any `os` file change and any socket call raises `PermissionError`. An audit hook can't be removed. The app also sets `sys.dont_write_bytecode`, otherwise Python would try to write `.pyc` files.

The hook only sees calls made from Python. OpenCV, MediaPipe and onnxruntime are C++ and could write without it noticing, so the tests check the filesystem too:

- one runs the live pipeline under the hook on made-up frames with a stand-in model, in an empty folder. Writes, `mkdir`, `socket` and `urlopen` must all be refused, the folder must still be empty at the end, and it must read 72 bpm off frames that pulse at 72 bpm, within 10 s. It runs in CI.
- the other runs the real app on subject 1 and checks that no file appeared in the repo or the temp folder. It needs the models and data, so it's skipped in CI.

Frames never leave memory. The app keeps the last 161 face crops (72x72) and 30 s of pulse values, and nothing else.

## Not done

- Two tries on my laptop (i5-1135G7, 8 GB, Windows, built-in webcam). The first time it showed about 94 bpm while I counted 99 at my wrist over a minute, and went green only once, at 92. The second time I let it print the fps: 2671 frames in 89 s, 30.0 fps on average, which is all the webcam gives, so the app keeps up. That run gave 83 readings with a median of 88 bpm, and only 8 were trusted. I counted 94 at my wrist, so it read about 6 low, like the first time. A hand count is a rough check anyway, not a reference. What the flat-out speed is on the laptop I don't know, since the camera caps it at 30.
- The flag cutoff (0.44) was set on clean dataset video. A webcam stream is compressed, and phase 6 already showed the flag dropping more than half the windows on compressed video, so mostly grey on a webcam fits that. I haven't retuned it.
- Webcams often drop to 15 fps in dim light. Mine stayed at 30 in both tries. The model was trained on 30 fps video, and at 15 fps the first number would take 11 s. I haven't tried resampling.
- The classical pipeline isn't in the app. It would need its own windowed POS and CHROM, and phase 6 says it's worse on compressed video anyway.
- `models/factorizephys.onnx` isn't in git. It's made from the rPPG-Toolbox checkpoint, which comes under the toolbox's own licence (a Responsible AI License, v1.1). `./setup_tools.sh --toolbox-env` builds it. To run the app on a laptop without that env, you have to copy `models/` over.
