#!/usr/bin/env bash
# Downloads the MPU-rPPG sample (Figshare article 29377835, CC BY 4.0) into
# data/mpu_rppg_sample. Files are fetched one by one with resume, because
# Figshare's "download all" zip can't resume and kept failing.
# Every file on Figshare is called Output.csv / Output.mp4, so the file id
# is kept in the name; a csv and the video right after it belong together.
#
# The fifth recording is a 7.25 GB mkv and is only fetched with --with-mkv.
cd "$(dirname "$0")/.."
dest=data/mpu_rppg_sample
mkdir -p "$dest"

# small recordings first, the 7.25 GB mkv last
files="
55553414 55553414_Output.csv
55553417 55553417_Output.mp4
55553420 55553420_Output.csv
55553423 55553423_Output.mp4
55553426 55553426_Output.csv
55553429 55553429_Output.mp4
55553435 55553435_Output.csv
55553438 55553438_Output.mp4
"
if [ "$1" = "--with-mkv" ]; then
    files="$files
55551383 55551383_Output.csv
55551389 55551389_Output.mkv
"
fi

echo "$files" | while read -r id name; do
    [ -z "$id" ] && continue
    out="$dest/$name"
    if [ -f "$out.done" ]; then
        continue
    fi
    for attempt in $(seq 1 50); do
        if curl -sS -L -C - --retry 5 -o "$out" "https://ndownloader.figshare.com/files/$id"; then
            touch "$out.done"
            echo "ok $name $(du -h "$out" | cut -f1)"
            break
        fi
        echo "retry $attempt $name"
        sleep 20
    done
done
