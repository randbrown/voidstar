#!/usr/bin/env bash

# Usage:
#   ./run_parallel.sh "pattern"
#
# Example:
#   ./run_parallel.sh "/mnt/c/users/brown/Videos/wild_mountain_thyme_3[a-c].mp4"

MAX_JOBS=3

if [ $# -lt 1 ]; then
  echo "Usage: $0 \"file_pattern\""
  exit 1
fi

PATTERN="$1"

function run_job() {
  local f="$1"
  echo "▶ Starting: $f"

  python reels_cv_overlay.py "$f" \
    --min-det-conf 0.5 \
    --min-trk-conf 0.5 \
    --draw-ids false \
    --smear true \
    --smear-frames 17 \
    --smear-decay 0.99 \
    --trail true \
    --trail-alpha .999 \
    --beat-sync true \
    --overlay-color 164,132,172 \
    > "${f}.log" 2>&1

  echo "✓ Finished: $f"
}

export -f run_job

FILES=( $PATTERN )

if [ ${#FILES[@]} -eq 0 ]; then
  echo "No files matched pattern: $PATTERN"
  exit 1
fi

echo "Matched files:"
for f in "${FILES[@]}"; do
  echo "  $f"
done
echo ""

for f in "${FILES[@]}"; do
  while [ "$(jobs -r | wc -l)" -ge "$MAX_JOBS" ]; do
    sleep 1
  done

  run_job "$f" &
done

wait
echo "All jobs completed!"
