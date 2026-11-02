#!/bin/zsh
# Wait for data/ubfc_phys/<person>.zip to finish copying, then unzip it,
# delete the zip and reduce the person.   scripts/process_phys_zip.sh s5
cd "$(dirname "$0")/.."
person=$1
zip_path=data/ubfc_phys/$person.zip
prev=-1
while true; do
  size=$(stat -c %s $zip_path 2>/dev/null || echo 0)
  # same size as 3 minutes ago and a readable central directory: copy is done
  if [ "$size" = "$prev" ] && unzip -l $zip_path > /dev/null 2>&1; then
    break
  fi
  prev=$size
  sleep 180
done
echo "$person zip complete, $size bytes, $(date)"
unzip -q -o $zip_path -d data/ubfc_phys/raw/ || { echo "$person UNZIP FAILED"; exit 1; }
rm -f $zip_path
echo "$person unzipped, zip deleted, $(date)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ./envs/pulsecam
python scripts/reduce_ubfc_phys.py $person 2>&1 | grep -E "frames, face|Error|Traceback"
echo "$person reduced, $(date)"
