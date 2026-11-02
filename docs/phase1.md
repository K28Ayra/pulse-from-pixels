# Phase 1: baselines

Everything here is the 30 UBFC-rPPG dev people. The toolbox only ever sees a folder of links to dev subjects (`data/ubfc_rppg/toolbox_dev`), so the 12 test people stay untouched.

I used rPPG-Toolbox at commit `b7500b8` in its own env (Python 3.8, torch 2.1.2). `scripts/run_toolbox.py` takes the toolbox's own configs and only changes paths and the GPU. The deep models are the PURE-trained checkpoints, so none of them has seen a UBFC person. The toolbox imports PhysMamba at start even if you don't use it, which needs `mamba_ssm`. Instead of compiling that, `toolbox_stubs/` has a stand-in that raises if anyone actually calls it.

## Does it match the paper?

One HR per video, as in Table 2 of the toolbox paper (Liu et al., arXiv 2210.00716). The paper uses all 42 people and I use 30, so some gap is expected.

| Method | dev 30, 0.6-3.3 Hz | dev 30, 0.75-2.5 Hz | paper |
|---|---|---|---|
| POS | 3.49 | | 4.00 |
| CHROM | 3.60 | | 3.98 |
| PhysNet | 3.08 | 1.61 | 0.98 |
| TS-CAN | 0.97 | 0.97 | 1.29 |
| EfficientPhys | 3.13 | 2.34 | 2.07 |
| FactorizePhys | 1.14 | 0.79 | 1.04 |

The toolbox commit searches HR in 0.6-3.3 Hz, and a comment in its own code says the paper used 0.75-2.5 Hz. With the commit band, PhysNet lands at 59 bpm on subject 26 where the truth is 101, and that one video adds 1.4 bpm to the mean. With the paper's band it gets 1.61, close enough.

The toolbox treats every video as 30 fps. Subjects 25 to 27 are about 23 fps, so its "bpm" for them is about 28% too high. The label goes through the same mistake, so its error stays fair, but I score everything on the real timestamps.

## My POS against theirs

First try: r = 0.98 between my POS and the toolbox's, and mine was worse per window, 3.23 against 1.96 bpm. The algorithm wasn't the difference. The toolbox's POS detrends (smoothness priors, lambda 100) and filters 0.75-3 Hz inside the method. Put my output through the same steps and r = 1.000 on every subject.

The face region matters a lot more. Same POS code, 3.23 bpm on the toolbox's face box (1.5x bigger, with hair and background in it) and 1.21 on the MediaPipe face outline.

## Per window

10 s windows, 1 s step, HR from the spectral peak in 42-210 bpm, against the phase 0 reference. Only windows where every method has a value (1490). The CI is a bootstrap over people.

| Method | MAE | 95% CI | within 5 bpm | within EC13 |
|---|---|---|---|---|
| FactorizePhys | 0.88 | 0.75-1.04 | 98.5% | 99.6% |
| TS-CAN | 0.96 | 0.79-1.17 | 98.5% | 99.3% |
| CHROM, mine, face outline | 1.12 | 0.83-1.53 | 97.4% | 98.4% |
| POS, mine, face outline | 1.23 | 0.83-1.77 | 97.5% | 98.8% |
| POS, toolbox | 2.02 | 1.01-3.53 | 94.0% | 96.0% |
| EfficientPhys | 2.54 | 0.94-5.54 | 94.6% | 96.1% |
| CHROM, toolbox | 2.81 | 1.53-4.38 | 92.2% | 94.2% |
| GREEN, mine | 4.54 | 1.59-8.40 | 90.5% | 91.8% |
| PhysNet | 5.22 | 1.62-10.08 | 89.2% | 90.6% |
| LGI / OMIT, toolbox | 8.5 | 4.2-13.8 | 81.2% | 83.0% |
| PBV, toolbox | 10.55 | 6.52-14.88 | 73.0% | 75.0% |
| ICA, toolbox | 14.71 | 8.93-20.96 | 65.0% | 66.1% |
| GREEN, toolbox | 19.47 | 12.84-26.57 | 57.1% | 59.1% |

PhysNet is 1.61 per video but 5.22 per window. Most of its bad windows (104 of 161) are just wrong peaks rather than double or half the rate. A whole video averages them out and a 10 s window can't.

LGI and OMIT really do give identical numbers. LGI removes the main colour direction of the whole video and OMIT the one of the first frame. On a face that barely moves, those are almost the same vector.

FactorizePhys and TS-CAN are already under 1 bpm on this dataset. UBFC-rPPG people sit still, so it's the easy case, and the harder tests come later.

Full tables are in `outputs/phase1/summary.csv` and `summary_common_windows.csv`.
