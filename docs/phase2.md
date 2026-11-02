# Phase 2: skin patches and SNR fusion

Result first. On the 30 dev people, mixing 16 patch-and-method signals by how clean each one is gives 0.89 bpm per window. Plain POS on the whole face gives 1.20. That's 26% lower, with 21 people better and 9 worse, and a one-sided Wilcoxon test on per-person MAE gives p = 0.013.

| My target (dev, still people) | Result | |
|---|---|---|
| MAE <= 3 bpm | 0.89 (CI 0.74-1.09) | pass |
| >= 90% within EC13 | 99.4% | pass |
| >= 15% below whole-face POS, p < 0.05 | 26.3%, p = 0.013 | pass |
| face found in >= 99% of frames | 100% | pass |

## What I built

Seven patches from MediaPipe landmark numbers (`PATCHES` in `pulsecam/face.py`): forehead, nose bridge, both cheeks, both lower cheeks next to the mouth ("jaw") and chin. The whole face outline is an eighth region. I drew all 478 landmarks on one dev face to pick them and checked the shapes on three more, one with a beard. The forehead starts at landmarks 108/151/337 instead of the top of the mesh, because hair covered the top on the first face I looked at.

Every region goes through POS and CHROM. For each 10 s window, each signal gets a power spectrum and an SNR: power near its peak and near twice the peak, against everything else in 42-210 bpm (de Haan and Jeanne 2013). The final spectrum is a weighted mean, and HR is its peak (`pulsecam/fusion.py`).

## Picking the setting

I tried 5 sets of signals with 4 weighting rules, picked the best on four folds and scored it on the fifth. All five folds picked the same one: POS and CHROM on every region, weighted by SNR squared.

| Setting | MAE |
|---|---|
| POS+CHROM everywhere, SNR² | 0.89 |
| same, SNR | 0.91 |
| same, equal weights | 0.94 |
| same, only the best-SNR signal per window | 1.19 |
| with GREEN added, SNR² | 1.30 |
| face outline only, POS+CHROM | 1.03 |
| face outline POS (baseline) | 1.20 |

Picking only the best signal per window is worse than averaging. A noisy patch sometimes gets one tall, narrow wrong peak and wins the window. GREEN hurts for the same reason. Sharper weights (SNR to the power 3, 4, 6) and other peak widths didn't beat 0.887 either.

Most weight goes to the face outline (37%), forehead (22%) and cheeks (25%). Jaw, nose and chin together get 16%.

## Who got worse

Subject 3 went from 3.47 to 4.35 and subject 4 from 2.31 to 2.84. Subject 3 also had the worst reference in phase 0. The biggest win was subject 32, 5.75 to 1.15.

FactorizePhys from phase 1 sits at 0.88 and the confidence intervals overlap almost completely, so on this dataset it's a tie with no training. UBFC-rPPG people don't move, though. UBFC-Phys is the real test.
