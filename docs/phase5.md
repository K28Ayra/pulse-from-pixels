# Phase 5: deep model and ablation

Two questions here. Does a deep model help, and does every part of my pipeline actually pull its weight?

I took FactorizePhys with the PURE checkpoint from rPPG-Toolbox (51,840 parameters) and ran it on 72x72 face crops of every dev recording (`scripts/make_crops.py`, `scripts/run_deep.py`). Then I tried it two ways, as one more signal in the fusion and as the main signal with my tracker on top.

## Results

MAE in bpm, dev people, the same windows for every row.

| | UBFC-rPPG | UBFC-Phys without list | UBFC-Phys all | per person, both |
|---|---|---|---|---|
| classical pipeline (phases 2-4) | 0.812 | 2.374 | 8.751 | 1.158 |
| classical + FactorizePhys as a signal | 0.813 | 2.286 | 8.855 | 1.132 |
| FactorizePhys alone | 0.858 | 2.250 | 8.638 | 1.107 |
| FactorizePhys + tracker | 0.782 | 1.946 | 8.515 | 1.020 |

As one more signal it barely helps (2%). As the main signal with the tracker it's 12% better per person. That's 3.7% on UBFC-rPPG and 18% on UBFC-Phys without the list (5 of 6 people better, p = 0.031). Over all 36 people p = 0.093, so not significant. The 1.020 was tuned on these people, and out of fold it's 1.099.

The tracker matters for the deep model too. It takes it from 1.107 to 1.020, with 30 of 36 people better (p = 0.0004). Motion suppression and mixing in the classical signals both made it worse.

Speed is fine. 0.114 s per 160 frames on 2 CPU threads, 0.058 s on 4, for the model alone.

## Ablation

Take one part out of the classical pipeline at a time. Mean per-person MAE without the list, full pipeline 1.158:

| Removed | MAE | change | p |
|---|---|---|---|
| tracker | 1.275 | +10.1% | 0.002 |
| CHROM | 1.243 | +7.3% | 0.107 |
| sharpening | 1.217 | +5.1% | 0.74 |
| jump | 1.184 | +2.3% | 0.30 |
| keeping jaw and chin | 1.180 | +1.9% | 0.48 |
| motion suppression | 1.163 | +0.4% | 0.12 |
| SNR weights | 1.162 | +0.3% | 0.31 |
| patches | 1.162 | +0.3% | 0.13 |
| POS | 1.156 | -0.2% | 0.13 |

Only the tracker is clearly needed on its own. Patches and POS cover for each other. Take both out and it's 1.359 (+17%, p = 0.028). Motion suppression barely changes the mean, but without it UBFC-Phys gets harmonic errors back (1.08% against 0%).

## The bug that almost killed it

My first run put FactorizePhys at 1.32 on UBFC-rPPG, against 0.88 in phase 1, and I nearly dropped it. The problem was timing. I'd resampled every video to an even 30 Hz from its average frame rate, but UBFC-rPPG frames aren't evenly spaced, and two subjects run at about 23 fps. Feeding the real frames and scoring on the real timestamps gave 0.858.

## s8 and s9 again

FactorizePhys reads 107 bpm on s8 at rest. POS reads 108 and the wrist 77. The two methods have nothing in common except the video, so I now suspect the wrist more than the camera. It isn't proof, since something in the video itself could fool both.

```bash
python scripts/make_crops.py
envs/rppg-toolbox/bin/python scripts/run_deep.py run
python scripts/evaluate_phase5.py   # about 5 min
python scripts/tune_phase5.py       # about 15 min
```
