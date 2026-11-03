# Phase 8: the test set

This is the only time the test people were used. Before the run I set `frozen: true` in `configs/default.yaml` and committed it. `scripts/evaluate_phase8.py test` won't start unless the config is frozen, git has no uncommitted changes, and `outputs/phase8/run.json` doesn't exist yet. It writes that file with the commit hash before it reads any reference, so a second run is refused. The run was at commit 471264a.

The test people:

- UBFC-rPPG, the 12 subjects in `splits/`
- UBFC-Phys s4, s6, s7 and s10, all three tasks
- MPU-rPPG sample, 4 recordings. The plan said 5, but one of the five CSVs (55551383) has no video.

What runs is the Phase 7 setup: FactorizePhys (PURE checkpoint) with my tracker. Next to it are the classical pipeline from Phase 3, plain POS on the face outline, and the toolbox's four PURE-trained deep models (PhysNet, TS-CAN, EfficientPhys, FactorizePhys) run through rPPG-Toolbox itself. Scoring is the same as in every phase: 10 s windows every 1 s, and a window only counts if the contact reference is clean.

## Against the targets

| Target | Test set | Result | |
|---|---|---|---|
| still people: MAE <= 3 bpm and >= 90% within EC13 | UBFC-rPPG | 0.96 bpm, 99.7% | pass |
| | UBFC-Phys T1 | 1.26 bpm, 100% | pass |
| beat POS on every test set | UBFC-rPPG | 0.96 vs 1.14 | pass |
| | UBFC-Phys all | 2.41 vs 3.15 | pass |
| | UBFC-Phys without list | 2.12 vs 2.91 | pass |
| | MPU-rPPG | 13.4 vs 27.1 | pass |
| beat the best toolbox deep model on UBFC-Phys | all | 2.41 vs 2.68 (FactorizePhys) | pass |
| | without list | 2.12 vs 2.32 (FactorizePhys) | pass |

None of the test T1 recordings is on the toolbox exclusion list, so T1 gives the same numbers both ways.

On UBFC-rPPG the gap to POS is clear: 11 of 12 people are better and p < 0.001 (one-sided Wilcoxon over people). On UBFC-Phys there are only 4 people, so the smallest p-value the test can give is 0.0625. There, "beats" just means a lower average. Deep was better than the toolbox's FactorizePhys for all 4 people, but that's still only 4.

## All the numbers

MAE in bpm, per window, with a 95% bootstrap interval over people.

| | UBFC-rPPG | UBFC-Phys all | UBFC-Phys without list | UBFC-Phys T1 | MPU-rPPG |
|---|---|---|---|---|---|
| deep (mine) | 0.96 (0.70-1.28) | 2.41 (1.62-2.99) | 2.12 (1.42-2.46) | 1.26 | 13.4 (7.7-23.4) |
| classical (mine) | 0.85 (0.68-1.03) | 2.33 | 2.03 | 1.26 | 15.2 |
| POS | 1.14 | 3.15 | 2.91 | 1.38 | 27.1 |
| toolbox FactorizePhys | 1.10 | 2.68 | 2.32 | 1.35 | |
| toolbox EfficientPhys | 1.32 | 4.28 | 3.59 | 1.77 | |
| toolbox TS-CAN | 1.36 | 7.27 | 7.22 | 7.51 | |
| toolbox PhysNet | 3.38 | 7.66 | 7.65 | 9.50 | |

I didn't run the toolbox models on MPU-rPPG. The toolbox has no loader for it.

Deep isn't better than classical on clean test video. On UBFC-rPPG classical is ahead, 0.85 against 0.96 (deep better for only 5 of 12 people). On UBFC-Phys classical is ahead for all 4 people, by about 0.1 bpm. On dev it was the other way round on UBFC-Phys without the list (deep 1.95, classical 2.37), and on UBFC-rPPG they were close (0.78 and 0.81). I picked deep in Phase 6 because it still works on compressed video (1.5 bpm at 250 kbps, where classical stops working). These test videos aren't compressed, so that advantage doesn't show here.

The quality flag (share < 0.44) gets an AUROC of 0.96 on UBFC-rPPG, 0.77 on UBFC-Phys all and 0.80 without the list. On dev those were 0.98, 0.86 and 0.85, so it lost a bit on UBFC-Phys.

## MPU-rPPG doesn't work

13.4 bpm, and only 48% of windows within EC13. Per recording:

| recording | windows | reference median | deep median | deep MAE | classical MAE |
|---|---|---|---|---|---|
| mpu55553417 | 585 | 87 | 84 | 13.0 | 17.5 |
| mpu55553423 | 693 | 81 | 80 | 6.2 | 11.7 |
| mpu55553429 | 399 | 119 | 88 | 30.1 | 21.7 |
| mpu55553438 | 661 | 88 | 92 | 11.2 | 13.0 |

These videos are different from anything on dev:

- **Motion.** The median head motion is about 13 times what it is in UBFC-rPPG.
- **The heart rate itself.** mpu55553429 goes from 71 to 155 bpm, with a median of 119. The model sits near 88 most of the time, so it's following something other than the pulse.
- **Skin tone.** All four people are tan or brown by ITA. On dev I had only two people darker than "intermediate".
- **Format.** The videos are 60 fps mp4, so each one is compressed and is halved to 30 fps for the model.

The flag sees that it's wrong. It trusts only 1% of MPU windows, and the AUROC there is 0.56, which is close to chance, so it can't tell the good windows from the bad ones. I can't work out which of the four differences above matters most from four videos. I also can't tune anything on them now. That's the main open problem from this phase.

## Skin tone

The plan said this would be a listing, not a statistic, and with these numbers there's nothing more to say. The darker test people:

- s7 (UBFC-Phys, ITA -7, brown): deep 2.36, classical 2.27, POS 3.15. PhysNet and TS-CAN from the toolbox fail on them, at 18.8 and 15.8 bpm.
- The four MPU people (ITA 4 to 21): bad for every method, as above, for reasons I can't separate from motion and compression.

All 12 UBFC-rPPG test subjects are "very light". So UBFC-rPPG says nothing about skin tone.

## Two things that went wrong on the way

The first toolbox run on the UBFC-Phys test people quietly left out 2 or 3 of the 12 recordings, a different set for each model. The kernel log had 8 out-of-memory kills. The toolbox preprocesses up to 8 videos at once, and each worker holds a whole 1024x1024 video, about 35 GB, on a machine with 251 GB. A killed worker just leaves its recording out of the cache, with no error. `scripts/toolbox_main.py` now starts the toolbox with at most 4 workers, and the rerun had all 12 recordings for every model. Before this phase I had only run the toolbox on UBFC-rPPG, whose videos are much smaller, so it hadn't come up.

The other one is still open. For each UBFC-Phys person the script checks the offset between the video and the contact signal, for reporting only, since the offset used stays at 0 s. For s7 the best offset came out at -5.5 s and for s6 at -2 s, with weak correlations (0.57 and 0.23). If s7's recording really is shifted, its 2.4 bpm is a bit pessimistic. I didn't change anything after seeing it.

Subject 27 is in the test set. They appear as a row of numbers in `outputs/phase8/per_person.csv` and nowhere else, with no face crop or figure.
