# Phase 4: when to not trust a reading

The app needs to say "ignore this number" sometimes. I planned a small learned model whose score would do two jobs: weight the signals in fusion and raise the flag. It didn't earn its place, so I went with something simpler.

The flag is `share_at_hr`, the share of the fused spectrum that sits within 5 bpm of the HR being reported. Below 0.44 the reading gets flagged. No training needed.

"Bad" below means more than 5 bpm off. All numbers are dev people, with person-wise 5-fold CV wherever a model is trained.

| Target | Result | |
|---|---|---|
| AUROC >= 0.80 | 0.885 all dev, 0.924 without the exclusion list | pass |
| keep 80% of windows, MAE 30% lower | 47% all dev, 33% without list | pass pooled |
| same, inside each dataset | 25-28% | miss |
| learned weights beat SNR weights | 16 variants tried, none did | miss |

## What I tried

Version one was logistic regression on nine features per signal: SNR, peak share, entropy, second peak, harmonic peak, agreement with other signals, distance to the median, and two motion features. As a flag it did worse than plain SNR (0.841 against 0.890). It never looked at the HR the tracker actually reported.

So I added nine features about the final HR, `share_at_hr` among them, and tried logistic regression and small boosted trees (depth 3). The trees got 0.899. `share_at_hr` alone got 0.885. With a bootstrap over people, the difference between them includes zero everywhere, and on UBFC-rPPG the simple score is actually better (0.969 against 0.959).

For fusion, none of the 16 quality-weight variants beat SNR squared. The best was 0.001 bpm worse per person.

## What the flag does at 0.44

| | kept | MAE | MAE of kept |
|---|---|---|---|
| UBFC-rPPG | 92.9% | 0.82 | 0.68 |
| UBFC-Phys without list | 77.6% | 2.37 | 1.65 |
| UBFC-Phys all | 65.5% | 8.71 | 5.09 |
| s8 | 37.7% | 23.35 | 26.54 |
| s9 | 27.3% | 23.31 | 19.89 |

It throws away about two thirds of s8 and s9, but what gets through is just as wrong. Those windows have a clean peak at the wrong rate, and nothing built from the spectrum alone can tell that apart from a real pulse.

It also flags some fine windows. UBFC-rPPG subject 4 keeps only 29%, with no bad windows among them.

```bash
python scripts/evaluate_phase4.py   # about 10 min
```
