# Phase 6: compression, dim light, skin tone, motion

A phone app won't get clean lossless video. So I broke the dev videos on purpose and ran all three pipelines on every broken copy: classical (phases 2-4), FactorizePhys with the tracker (phase 5), and plain whole-face POS as the baseline.

The copies (`scripts/make_stress.py`):

- x264 at 4000, 2000, 1000, 500 and 250 kbps
- `light_50` and `light_25`: pixels times 0.5 or 0.25, plus Gaussian noise (sd 2 in 8-bit units) so it isn't just a clean rescale

Landmarks are found again on each copy, as the app would have to. Every pipeline is scored on the same windows: 1,496 on UBFC-rPPG, and 1,393 on UBFC-Phys (809 without the exclusion list).

I set the targets before running anything, including the rule for the breaking bitrate: the lowest bitrate where MAE is at most 5 bpm, at that bitrate and at every higher one.

| Target | Result | |
|---|---|---|
| breaks at a lower bitrate than POS | UBFC-rPPG: classical 1000k, deep 250k, POS 2000k. UBFC-Phys without list: classical and deep 1000k, POS 4000k | pass |
| MAE <= 5 at half light | 0.84 UBFC-rPPG, 2.40 UBFC-Phys without list | pass |
| worst skin group better than POS | deep yes, classical no (one person, see below) | half |
| flag AUROC >= 0.75 on stressed video | 0.82 to 0.91 | pass |

Every UBFC-Phys "all" number breaks at 4000k already, because s8 and s9 are ~23 bpm off on clean video.

## Compression

MAE in bpm.

| kbps | classical | deep | POS |
|---|---|---|---|
| UBFC-rPPG raw | 0.81 | 0.78 | 1.21 |
| 4000 | 2.40 | 1.01 | 3.26 |
| 2000 | 1.89 | 1.07 | 3.32 |
| 1000 | 2.61 | 1.11 | 5.01 |
| 500 | 6.60 | 1.43 | 7.36 |
| 250 | 6.58 | 1.51 | 7.56 |
| UBFC-Phys without list, raw | 2.37 | 1.95 | 4.06 |
| 4000 | 2.70 | 2.31 | 4.92 |
| 2000 | 2.79 | 2.47 | 5.93 |
| 1000 | 4.67 | 3.24 | 10.82 |
| 500 | 11.89 | 10.19 | 18.31 |
| 250 | 16.92 | 12.22 | 24.67 |

This is the clearest result of the phase. The classical pipeline loses most of its advantage the moment the video is compressed: 0.81 becomes 2.40 even at 4000k. FactorizePhys barely notices on UBFC-rPPG and still reads 1.5 bpm at 250k. My guess is that x264 smooths away exactly the small, slow colour changes that averaging over patches depends on, while the model was trained on a view of the face that survives it better. I haven't tested that guess.

UBFC-Phys falls apart for everything below 1000k. Those videos have more motion, so the encoder spends its bits on the movement.

The curve isn't monotone, and that bothered me. Classical is worse at 4000k than at 2000k on UBFC-rPPG. Nearly all of it is subject 9: 0.8 bpm raw, 30 at 4000k, 3.4 at 2000k, 34 at 1000k. At 4000k and 1000k the tracker locks onto about 72 bpm, while the true rate is 108. POS and deep stay near 100-108 on the same copies. It isn't one fixed frequency from the encoder, because it moves between bitrates. I haven't found the cause. With 30 people, one person like that is enough to bend the averages.

## Dim light

Hardly matters. At a quarter of the light, classical goes 0.81 to 0.99 on UBFC-rPPG and deep 0.78 to 0.84. UBFC-Phys without the list moves by 0.1-0.15 bpm. The pulse is a relative colour change, so scaling the pixels doesn't remove it, and sd 2 of noise gets mostly averaged away over a patch. A real dark room adds compression on top, and that's the part that hurts.

## Skin tone

ITA angle from the median skin colour of the face (`pulsecam/skin.py`), grouped with the Chardon et al. (1991) cut-offs. The data is very light-skinned:

| group | people | classical | deep | POS |
|---|---|---|---|---|
| very light (> 55°) | 20 | 0.72 | 0.73 | 1.25 |
| light (41-55°) | 9 | 1.17 | 0.97 | 1.42 |
| intermediate (28-41°) | 5 | 2.29 | 2.03 | 4.54 |
| tan (10-28°) | 1 | 3.62 | 2.24 | 3.14 |
| brown (-30 to 10°) | 1 | 0.98 | 0.92 | 1.33 |

That's without the exclusion list, raw video. With the list, s8 and s9 land in "intermediate" and it goes to ~10 bpm for every method.

The tan group is one person, s1, and the brown group is one person, UBFC-rPPG subject 24. On s1, classical is worse than POS (3.62 against 3.14), so the classical pipeline misses that target. s1 is also the person whose HR swings 65-100 within seconds, which is the phase 3 problem and probably has nothing to do with skin. Subject 24, the darkest face here, is one of the easier people.

Error does go up as skin gets darker, from very light to intermediate. But skin tone and dataset are mixed together here: almost all the UBFC-Phys people are intermediate, and UBFC-Phys is the harder dataset anyway. I can't separate the two with this data, so I don't claim anything about darker skin. Two people below 28° is nothing.

## Motion

Raw video, both datasets, without the list. Windows above the phase 3 motion threshold are split in three equal parts, 65 windows each. Everything below it is "still", and that's mostly UBFC-rPPG.

| | windows | classical | deep | POS |
|---|---|---|---|---|
| still | 2110 | 1.33 | 1.15 | 1.95 |
| low | 65 | 2.06 | 1.85 | 3.95 |
| mid | 65 | 1.56 | 1.57 | 5.41 |
| high | 65 | 1.55 | 1.42 | 5.80 |

POS gets worse steadily as motion goes up. Both of my pipelines stay flat. It's only 195 moving windows and 174 of them are s3, so I wouldn't read much into the low bin being the worst.

## The flag on broken video

All stressed copies pooled, flag at 0.44 as in phase 4.

| | AUROC | kept | MAE | MAE of kept |
|---|---|---|---|---|
| UBFC-rPPG classical | 0.86 | 39% | 3.13 | 0.73 |
| UBFC-rPPG deep | 0.91 | 75% | 1.11 | 0.76 |
| UBFC-Phys without list, classical | 0.87 | 46% | 6.27 | 1.60 |
| UBFC-Phys without list, deep | 0.89 | 58% | 4.91 | 1.66 |

It still ranks windows well. The cutoff was picked on clean video, though, and on compressed video it throws away more than half of the classical windows. For the app that's probably the right call, since a missing number is better than a wrong one. I didn't retune it.

## Which one goes into the app

Deep. It's as good or better everywhere in this phase, and much better under compression, which is what a phone gives you. The cost is torch (or an ONNX export) in the live env, and a 160-frame chunk, about 5 s, before the first number. The classical pipeline stays in the repo for comparison.

```bash
python scripts/make_stress.py
envs/rppg-toolbox/bin/python scripts/run_deep.py run
python scripts/evaluate_phase6.py   # about 2 min
```
