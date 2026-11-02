#!/usr/bin/env bash
# Creates the conda env inside this folder (./envs/pulsecam).
# The conda and pip caches
# are kept here too, so a full root disk does not break the install.
set -e
cd "$(dirname "$0")"

export CONDA_PKGS_DIRS="$PWD/.cache/conda_pkgs"
export PIP_CACHE_DIR="$PWD/.cache/pip"
# pip unpacks wheels here, the torch wheel alone is 2 GB
export TMPDIR="$PWD/.cache/tmp"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR"

# the classic conda solver is very slow on this file
if command -v mamba > /dev/null; then
    tool=mamba
else
    tool="conda --solver=libmamba"
fi

# Absolute prefix on purpose: with a relative one, python's zoneinfo
# search path gets baked in as a relative path and warns on every start.
prefix="$PWD/envs/pulsecam"

if [ -d "$prefix" ]; then
    $tool env update -p "$prefix" -f environment.yml --prune
else
    $tool env create -y -p "$prefix" -f environment.yml
fi

echo "done. activate with:  conda activate ./envs/pulsecam"
