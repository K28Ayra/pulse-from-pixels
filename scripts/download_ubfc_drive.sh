#!/usr/bin/env bash
# Downloads UBFC-rPPG DATASET_2 (about 75 GB) from the authors' Google Drive folder.
#
# Anonymous downloads of these videos fail with "quota exceeded" because so
# many people fetch them. Downloading through the Drive API with your own
# Google login (an rclone remote called "gdrive", see README) was not blocked
# when I tried it, at 2-5 MB/s.
#
# Re-running skips files that are already complete, so it is safe to restart.
# Config is kept in .cache/rclone.conf, outside git.
cd "$(dirname "$0")/.."

export RCLONE_CONFIG="$PWD/.cache/rclone.conf"
UBFC_FOLDER_ID=1o0XU4gTIo46YfwaWjIgbtCncc-oF44Xk
src="gdrive,root_folder_id=$UBFC_FOLDER_ID:DATASET_2"
dest=data/ubfc_rppg/DATASET_2
mkdir -p "$dest"

# --tpslimit: rclone's shared client id runs into Google's per-minute
# request cap quickly, so keep the request rate low
for attempt in 1 2 3 4 5 6 7 8 9 10; do
    if rclone copy "$src" "$dest" --transfers 2 --tpslimit 4 --retries 5 \
        --stats 5m --stats-one-line -v; then
        echo "finished $(date)"
        exit 0
    fi
    echo "attempt $attempt failed, waiting 5 minutes"
    sleep 300
done
echo "gave up after 10 attempts"
exit 1
