# pulsecam

You point a camera at a face and get heart rate out of it. No sensor on the finger. It reads tiny colour changes in the skin, turns them into a pulse signal and tracks HR over time, and it flags readings it doesn't trust.

I tested every step against a contact pulse reference, on two public datasets, and kept the test people locked away until the end.

## Setup

You need conda (or mamba). A GPU only matters for the deep model.

```bash
git clone <this repo>
cd pulsecam
./setup_env.sh                 # env goes to ./envs/pulsecam
conda activate ./envs/pulsecam
./setup_tools.sh               # rPPG-Toolbox + MediaPipe face model
pytest -q
```

`conda env create -f environment.yml` also works if you don't care where the env lives.

The deep model (FactorizePhys) runs from rPPG-Toolbox, which wants its own older env:

```bash
./setup_tools.sh --toolbox-env
```

For the webcam demo there's a smaller env without torch. It works on Windows too:

```bash
conda env create -f environment-live.yml
conda activate pulsecam-live
```

## Data

Nothing in `data/` goes into git. Put things here:

```
data/
  ubfc_rppg/DATASET_2/subject1/vid.avi
                              /ground_truth.txt
  ubfc_phys/raw/s1/vid_s1_T1.avi ...
  mpu_rppg_sample/
```

- UBFC-rPPG (DATASET_2, 42 people, about 75 GB) is a Google Drive folder linked from https://sites.google.com/view/ybenezeth/ubfcrppg. Normal downloads hit "quota exceeded". rclone with your own Google login works:

  ```bash
  rclone authorize "drive"     # on a machine with a browser, gives you a token
  RCLONE_CONFIG=.cache/rclone.conf rclone config create gdrive drive scope=drive token='<token>'
  scripts/download_ubfc_drive.sh
  ```

  Subjects 21 and 27 didn't agree to their faces being shown, so don't put them in figures.
- UBFC-Phys is at https://doi.org/10.21227/5da0-7344 (free IEEE DataPort account). I use s1 to s12. Each person is about 15 GB, so I shrink them after unzipping (see below).
- MPU-rPPG sample: `scripts/download_mpu.sh`. Test only.

The dev/test splits in `splits/` were made before I opened a single video. Don't regenerate them.

## Usage

Every phase has an evaluation script, and they write CSVs to `outputs/phaseN/`. Run them from the repo root with the env active.

```bash
# landmarks and skin-patch colour for UBFC-rPPG
python scripts/extract_faces.py
python scripts/extract_patches.py

# shrink UBFC-Phys to landmarks, patch colour and 72x72 face crops
python scripts/reduce_ubfc_phys.py s1 s2 s3 s5 s8 s9 s11 s12

# then any phase, for example
python scripts/evaluate_phase3.py
```

For the deep model, make crops and run it in the toolbox env first:

```bash
python scripts/make_crops.py
envs/rppg-toolbox/bin/python scripts/run_deep.py run
envs/rppg-toolbox/bin/python scripts/run_deep.py speed   # CPU speed
python scripts/evaluate_phase5.py
```

The baselines from phase 1 come from the toolbox itself:

```bash
python scripts/run_toolbox.py
envs/rppg-toolbox/bin/python scripts/toolbox_bvp.py
PYTHONPATH=toolbox_stubs envs/rppg-toolbox/bin/python scripts/toolbox_video_bands.py
python scripts/evaluate_phase1.py
```

That run takes about 1.5 h on one GPU. Most of it is the toolbox cropping every video again for each model.

## How it's put together

`pulsecam/` is the package and `scripts/` just calls it. All settings live in `configs/default.yaml`. The tracker settings, the quality cutoff and the list of UBFC-Phys recordings that rPPG-Toolbox excludes are all there.

`live_mode` is false for datasets. The webcam app sets it to true.

Everything gets scored the same way. HR comes from 10 s windows every 1 s and is compared with HR from the contact pulse in the same window. A window only counts if that reference looks clean. Settings are picked by person-wise cross-validation on dev people.

UBFC-Phys gets reported twice, all windows and without the recordings on rPPG-Toolbox's exclusion list. You'll see why in phase 3.

## Phases

### Phase 0: reference and splits

[docs/phase0.md](docs/phase0.md)

The contact PPG becomes one HR per window. A window only counts if the beats are regular and the peak-based HR agrees with the spectral one. 2223 of 2390 windows pass. The oximeter's own HR column lags and jumps around, so I don't use it as truth.

### Phase 1: baselines

[docs/phase1.md](docs/phase1.md)

I ran rPPG-Toolbox's classical methods and four PURE-trained deep models on the 30 dev people. They land close to the toolbox paper. My own POS matches the toolbox's exactly once the same filtering is applied. Per window, FactorizePhys gets 0.88 bpm and my POS on a MediaPipe face outline gets 1.23. The same POS on the toolbox's looser face box gets 3.23, so where you measure matters more than which method you use.

### Phase 2: skin patches and SNR fusion

[docs/phase2.md](docs/phase2.md)

Seven skin patches plus the face outline, each through POS and CHROM. They get mixed per window by how clean their spectrum is. 0.89 bpm against 1.20 for plain POS, 26% lower (p = 0.013). That ties FactorizePhys with no training.

### Phase 3: motion and tracking

[docs/phase3.md](docs/phase3.md)

Head motion from the landmarks is used to down-weight frequencies where the head moves, and a Viterbi tracker follows HR with 5 s of delay. UBFC-rPPG goes from 0.887 to 0.820 (25 of 30 people better). UBFC-Phys without the exclusion list goes from 3.00 to 2.37. Two people, s8 and s9, stay around 23 bpm whatever I do.

### Phase 4: quality flag

[docs/phase4.md](docs/phase4.md)

The flag is the share of the spectrum within 5 bpm of the reported HR. Below 0.44 it says "don't trust this". AUROC is 0.885 over all dev windows. I tried a learned model too, and it wasn't better.

### Phase 5: deep model and ablation

[docs/phase5.md](docs/phase5.md)

FactorizePhys with my tracker beats the classical pipeline, 1.02 bpm per person against 1.16. Taking parts away one at a time, only the tracker is clearly needed. Both pipelines go into phase 6.
