#!/usr/bin/env bash
# External tools that are not pip packages:
#   tools/rPPG-Toolbox            baselines, pinned commit, only the PURE checkpoints
#   models/face_landmarker.task   MediaPipe face landmark model
#   envs/rppg-toolbox             the toolbox's own env (python 3.8), only with --toolbox-env
# Safe to re-run; finished steps are skipped.
set -e
cd "$(dirname "$0")"

TOOLBOX_COMMIT=b7500b848f84ad7f86e277b4612563b69f4f88f9
TOOLBOX_URL=https://github.com/ubicomplab/rPPG-Toolbox
LANDMARKER_URL=https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

# Checkpoints trained on PURE, so they have never seen UBFC-rPPG or UBFC-Phys people
CHECKPOINTS="PURE_PhysNet_DiffNormalized.pth PURE_TSCAN.pth PURE_EfficientPhys.pth PURE_FactorizePhys_FSAM_Res.pth"

mkdir -p tools models

retry() {
    for i in 1 2 3 4 5 6; do
        if "$@"; then
            return 0
        fi
        echo "failed, retrying in 30 s"
        sleep 30
    done
    return 1
}

if [ ! -f tools/rPPG-Toolbox/main.py ]; then
    # Full clones kept dying on our slow link (the repo carries ~200 MB of
    # checkpoints), so: blobless clone, then check out code and only the
    # checkpoints listed above.
    rm -rf tools/rPPG-Toolbox
    retry git -c http.version=HTTP/1.1 clone --filter=blob:none --no-checkout "$TOOLBOX_URL" tools/rPPG-Toolbox
    cd tools/rPPG-Toolbox
    patterns="/* !/final_model_release/*"
    for name in $CHECKPOINTS; do
        patterns="$patterns /final_model_release/$name"
    done
    git sparse-checkout set --no-cone $patterns
    retry git -c http.version=HTTP/1.1 checkout -q "$TOOLBOX_COMMIT"
    cd ../..
fi

if [ ! -f models/face_landmarker.task ]; then
    retry curl -L -o models/face_landmarker.task "$LANDMARKER_URL"
fi

if [ "$1" = "--toolbox-env" ]; then
    export CONDA_PKGS_DIRS="$PWD/.cache/conda_pkgs"
    export PIP_CACHE_DIR="$PWD/.cache/pip"
    # pip unpacks wheels in TMPDIR; torch and its CUDA libraries are ~5 GB,
    # more than our root disk had free
    export TMPDIR="$PWD/.cache/tmp"
    mkdir -p "$TMPDIR"
    if [ ! -x envs/rppg-toolbox/bin/python ]; then
        # absolute prefix, a relative one ends up in the zoneinfo path
        mamba create -y -p "$PWD/envs/rppg-toolbox" python=3.8
    fi
    py=envs/rppg-toolbox/bin/python
    retry $py -m pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 \
        --index-url https://download.pytorch.org/whl/cu121
    # mamba-ssm and causal-conv1d are only for PhysMamba, which I don't use,
    # and they need a CUDA compiler to build
    grep -v -E "mamba-ssm|causal-conv1d" tools/rPPG-Toolbox/requirements.txt > .cache/toolbox_requirements.txt
    retry $py -m pip install -r .cache/toolbox_requirements.txt
fi

echo "tools ready"
