#!/bin/zsh
# Fetch UBFC-Phys zips from signed IEEE DataPort links, one link per line in
# data/ubfc_phys/links.txt, then process each one. Partial zips are resumed.
#   setsid nohup scripts/download_phys.sh > outputs/logs/phys_download.log 2>&1 &
# The links last 1 hour but a running transfer outlives that, so all downloads
# start together and the slow reduction waits until they are finished.
cd "$(dirname "$0")/.."
links=data/ubfc_phys/links.txt
people=()

while read -r url; do
  case "$url" in
    http*) ;;
    *) continue ;;
  esac
  person=$(echo "$url" | grep -oE '/s[0-9]+\.zip' | head -1 | tr -d '/' | sed 's/\.zip//')
  if [ -z "$person" ]; then
    echo "no sN.zip in a link, skipped"
    continue
  fi
  if [ -e "data/ubfc_phys/derived/${person}_T3_light_25.npz" ]; then
    echo "$person already reduced"
    continue
  fi
  people+=($person)
  zip_path=data/ubfc_phys/$person.zip
  if unzip -l $zip_path > /dev/null 2>&1; then
    echo "$person zip already complete"
    continue
  fi
  echo "$person download start, $(date)"
  # --fail keeps an S3 error page out of the zip; the url is never echoed
  (
    curl -sS --fail -L -C - --retry 20 --retry-delay 30 --retry-all-errors -o $zip_path "$url" \
      && echo "$person download done, $(stat -c %s $zip_path) bytes, $(date)" \
      || echo "$person DOWNLOAD FAILED at $(stat -c %s $zip_path) bytes, $(date)"
  ) &
done < $links
wait

for person in $people; do
  if unzip -l data/ubfc_phys/$person.zip > /dev/null 2>&1; then
    scripts/process_phys_zip.sh $person
  else
    echo "$person zip incomplete, not processed"
  fi
done
echo "all done, $(date)"
