#!/usr/bin/env bash

# A simple FFmpeg script to copy a clip from a video file without re-encoding.
# Usage:
#   ff_clip_copy.sh <input.mp4> <start> <duration>
# E.g. to copy a 1-minute clip starting at 23:00:
#   ff_clip_copy.sh input.mp4 00:23:00 60


set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ff_clip_copy.sh <input.mp4> <start> <duration>

Where:
  <start>    = HH:MM:SS[.ms]  (e.g. 00:23:00 or 00:23:00.500)
  <duration> = seconds OR HH:MM:SS[.ms] (e.g. 60 or 00:01:00)

Examples:
  ./ff_clip_copy.sh "in.mp4" 00:23:00 60
  ./ff_clip_copy.sh "in.mp4" 00:23:00 00:01:00
EOF
}

if [[ $# -lt 3 ]]; then
  usage
  exit 2
fi

IN="$1"
START="$2"
DUR="$3"

if [[ ! -f "$IN" ]]; then
  echo "ERROR: input file not found: $IN" >&2
  exit 1
fi

# Sanitize timestamp strings for filenames (":" -> "h/m/s" style)
sanitize_ts() {
  local t="$1"
  # Replace ':' with nothing and keep digits + dot, then format like 002300 or 000100.500 -> 000100p500
  # Simpler readable: 00h23m00s, 00h01m00s
  if [[ "$t" =~ ^([0-9]{2}):([0-9]{2}):([0-9]{2})(\.[0-9]+)?$ ]]; then
    local hh="${BASH_REMATCH[1]}"
    local mm="${BASH_REMATCH[2]}"
    local ss="${BASH_REMATCH[3]}"
    local frac="${BASH_REMATCH[4]:-}"
    if [[ -n "$frac" ]]; then
      # ".500" -> "p500"
      frac="${frac#.}"
      echo "${hh}h${mm}m${ss}s"_"p${frac}"
    else
      echo "${hh}h${mm}m${ss}s"
    fi
  else
    # For plain seconds like "60" or "60.5"
    local s="$t"
    s="${s//./p}"
    echo "${s}s"
  fi
}

START_TAG="$(sanitize_ts "$START")"
DUR_TAG="$(sanitize_ts "$DUR")"

DIR="$(dirname "$IN")"
BASE="$(basename "$IN")"
NAME="${BASE%.*}"
EXT="${BASE##*.}"

OUT="${DIR}/${NAME}_s${START_TAG}_d${DUR_TAG}_copy.${EXT}"

echo "Input : $IN"
echo "Start : $START"
echo "Dur   : $DUR"
echo "Output: $OUT"
echo

ffmpeg -hide_banner -y \
  -ss "$START" -i "$IN" \
  -t "$DUR" \
  -map 0 -c copy \
  -fflags +genpts -avoid_negative_ts make_zero \
  -movflags +faststart \
  "$OUT"

echo
echo "Done: $OUT"
