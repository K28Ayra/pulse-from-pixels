# Phase 3: motion and tracking

This phase moved to UBFC-Phys, where people talk and do mental arithmetic in front of the camera. I added two things.

- Motion suppression. Head motion comes from the landmarks (x, y, scale, roll, mouth). Wherever motion has power in the spectrum, that frequency gets down-weighted: weight = 1 / (1 + motion / tau). `pulsecam/motion.py`
- A Viterbi tracker. It follows HR from window to window, so one window jumping to double the rate gets ignored. It waits 5 s before committing, which a live app can afford. `pulsecam/tracking.py`

Final setting: tau = 1e-4, sigma = 2 bpm, 5 s delay, jump = 0.01, fused spectrum squared before tracking, jaw and chin patches dropped. All of it is in `configs/default.yaml` under `tracking`.

## Results

MAE in bpm, dev people only.

| | Phase 2 | Phase 3 |
|---|---|---|
| UBFC-rPPG, all windows | 0.887 | 0.820 |
| UBFC-Phys without exclusion list | 3.00 | 2.37 |
| UBFC-Phys moving windows, without list (124) | 5.18 | 2.42 |
| UBFC-Phys moving windows, all 8 people (311) | 16.25 | 14.38 |
| harmonic errors, UBFC-Phys | 1.47% | 0% |

UBFC-rPPG improves for 25 of 30 people (p = 0.0004). UBFC-Phys alone isn't convincing: 2 of 6 people better, and most of the gain comes from s3. 107 of the 124 moving windows are s3 too.

My target for moving windows was 20% below phase 2. Without the list it's 53%. With all 8 people it's 12%, a miss.

## Problem 1: the video and wrist sensor aren't in sync

On the first person, s1, the camera HR curve matched the wrist BVP only after shifting it 5 s, so I assumed a 5 s offset. With all 8 people, the offsets I got were all over the place, from -10 s to +10 s, and two hit the edge of the search. They were noise, and they pushed good data out of line (s12 got -9 s).

The fix was to estimate the offset only on the rest task (T1), where the wrist signal is clean, and only use it if the match is strong (r >= 0.7). Otherwise it's 0. The people who match well sit near zero: s2 -0.5 s, s3 0 s, s12 -1 s. s1 keeps 5 s. The offset comes from whole-face POS, never from my own method, so it can't favour it.

## Problem 2: s8 and s9

These two are around 23 bpm off on every task. On s8 at rest, the wrist says 77 and the camera says 108, steadily, and 108/77 isn't a harmonic. I checked the obvious things. The face is found in over 99% of frames, crops look fine, and the wrong peak isn't in the background or in the head motion.

rPPG-Toolbox has an exclusion list for UBFC-Phys in its own configs (67 recordings, no reasons given), and s8 and s9 are on it for every task. I only found it after seeing my errors. So from here on I report UBFC-Phys both ways, all windows and without the list. The list is in `configs/default.yaml`. It isn't a perfect match with my errors either. s3 T1 is listed and I get 1.69 bpm on it.

## Problem 3: the tracker hurt fast HR changes

My first setting (sigma 2, no jump) took s1 from 3.38 to 6.91. Their HR swings between 65 and 100 within seconds, and the tracker refused to move that far. Three changes fixed it:

- `jump`, a small chance of moving to any HR, so strong evidence can pull the track over
- squaring the spectrum, so a clear peak beats the tracker's preference for staying put
- dropping jaw and chin patches, which move when people talk

I swept 288 combinations with `scripts/tune_phase3.py`, with every person counting the same. s1 went back to 3.60. Subject 3 on UBFC-rPPG is still worse with tracking, 4.35 to 6.65, and I haven't fixed that one.

```bash
python scripts/tune_phase3.py      # about 15 min
python scripts/evaluate_phase3.py  # about 6 min
```
