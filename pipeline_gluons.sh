#!/usr/bin/env bash
# pipeline_gluons.sh
# Pipeline for generating target videos for an input mp4
#
# Usage:
#   ./pipeline_gluons.sh                        # uses default INPUT_VIDEO env var or fallback
#   ./pipeline_gluons.sh /path/to/video.mp4     # build all targets for that input
#   ./pipeline_gluons.sh preview /path/to.mp4   # build only 60s START target
#   ./pipeline_gluons.sh --end-only --input /path/to.mp4
#   ./pipeline_gluons.sh --force --input /path/to.mp4 --outdir /some/dir
#
# Key knobs:
#   --start-seconds <n>         Default 0 (used for 90/180 targets; 60 targets may override to middle 10m)
#   --youtube-full-seconds <n>  Default = full input duration (used for 90/180 targets; 60 targets may override to 600)
#   --cps <float>               Default 0.5
#   --glitch-seconds <n>        Default 2
#   --loop-seam-seconds <n>     Default = glitch-seconds
#   --logo <filename_pattern>   Repeatable. If provided, use these logo file(s) (globs allowed).
#                               If multiple logo files, rotate them across dvdlogo runs and
#                               add a filename suffix so outputs don't overwrite.

set -euo pipefail

# ----------------------------
# Helpers
# ----------------------------

die() { echo "Error: $*" >&2; exit 1; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

require_file() {
  local label="$1" path="$2"
  [[ -f "$path" ]] || die "Missing $label file: $path"
}

# Helper to check if a target needs to be rebuilt
should_rebuild() {
    local target="$1"
    local script="$0"

    if [[ "${FORCE:-0}" -eq 1 ]]; then
        return 0
    fi
    if [[ ! -f "$target" ]]; then
        return 0
    fi
    if [[ "$script" -nt "$target" ]]; then
        return 0
    fi
    echo "Target $target is up-to-date. Skipping."
    return 1
}

# Helper for renaming output files
rename_output() {
    local src="$1"
    local dst="$2"
    if [[ -f "$src" ]]; then
        mv -v "$src" "$dst"
    else
        echo "Warning: $src not found for renaming."
    fi
}

# Determine input video duration (seconds, integer)
get_video_duration_seconds() {
    local video="$1"
    # ffprobe returns float seconds; we floor to int for stable filenames.
    local dur
    dur="$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$video" 2>/dev/null || true)"
    [[ -n "$dur" ]] || die "Could not determine video duration via ffprobe: $video"
    python3 - <<PY
import math
d=float("$dur")
print(int(math.floor(d)))
PY
}

# Expand one or more glob patterns into concrete files
expand_logo_patterns() {
    local -a out=()
    local pat
    shopt -s nullglob
    for pat in "$@"; do
        local -a matches=( $pat )
        if (( ${#matches[@]} == 0 )); then
            die "Logo pattern did not match any files: $pat"
        fi
        out+=( "${matches[@]}" )
    done
    shopt -u nullglob
    printf '%s\n' "${out[@]}"
}

# Global logo rotation index
DVDLOGO_RUN_INDEX=0

# Pick logo (rotating) and optionally produce a suffix tag
pick_logo_for_run() {
    local -n _logos_ref=$1
    local count="${#_logos_ref[@]}"
    (( count > 0 )) || die "Internal error: no logos available"
    local idx=$(( DVDLOGO_RUN_INDEX % count ))
    local logo="${_logos_ref[$idx]}"
    local tag=""
    if (( count > 1 )); then
        tag="$(basename "${logo%.*}")"
    fi
    DVDLOGO_RUN_INDEX=$(( DVDLOGO_RUN_INDEX + 1 ))
    echo "$logo|$tag"
}

# For multi-logo, add a suffix so files don't overwrite
with_logo_suffix() {
    local base_path="$1"   # full path to .mp4
    local tag="$2"         # may be empty
    if [[ -z "$tag" ]]; then
        echo "$base_path"
        return 0
    fi
    echo "${base_path%.mp4}_logo-${tag}.mp4"
}

# Special logic for 60/60t: if input > 10 minutes, use the middle 10 minutes only.
compute_60_window() {
    local ss="$START_SECONDS"
    local full="$YOUTUBE_FULL_SECONDS"
    if (( INPUT_DURATION_SECONDS > 600 )); then
        ss=$(( (INPUT_DURATION_SECONDS - 600) / 2 ))
        full=600
    fi
    echo "$ss|$full"
}

# ----------------------------
# Targets
# ----------------------------

# 1. 60s highlight (START anchor)
run_60s_start() {
    echo "--- 60s highlight (START) ---"

    local w ss full
    w="$(compute_60_window)"
    ss="${w%%|*}"
    full="${w##*|}"
    if (( INPUT_DURATION_SECONDS > 600 )); then
        echo "60s START: input > 10m, using middle 10m segment: start-seconds=$ss youtube-full-seconds=$full"
    fi

    python3 "$DIVVY" highlights "$INPUT_VIDEO" \
        --start-seconds "$ss" --youtube-full-seconds "$full" --target-length-seconds 60 \
        --video-encoder libx264 --preset medium --out-dir "$OUTDIR" \
        --glitch-seconds "$GLITCH_SECONDS" --glitch-style vuwind --cps "$CPS" --loop-seam-seconds "$LOOP_SEAM_SECONDS" \
        --sampling-mode uniform-spread --n-segments 8 --sample-seconds 8 --truncate-to-full-clips

    python3 "$REELS_OVERLAY" "$OUTDIR/${STEM}__highlights__mode-uniform-spread__start-${ss}s__full-${full}s__target-60s.mp4" \
        --min-det-conf 0.05 --min-trk-conf 0.05 --draw-ids true \
        --smear true --smear-frames 17 --smear-decay 0.99 \
        --trail true --trail-alpha .999 --beat-sync true \
        --velocity-color true --velocity-color-mult 10

    rename_output "$OUTDIR/${STEM}__highlights__mode-uniform-spread__start-${ss}s__full-${full}s__target-60s_fps30_mc2_det0p05_trk0p05_trail1_ta1p00_tlotrue_scan1_velcolor_ids_beatsync_smear.mp4" \
        "$OUTDIR/${STEM}_highlights_60s_overlay.mp4"

    local picked logo tag target
    picked="$(pick_logo_for_run LOGOS)"
    logo="${picked%%|*}"
    tag="${picked##*|}"
    target="$(with_logo_suffix "$OUTDIR/${STEM}_highlights_60s_overlay_logo.mp4" "$tag")"

    should_rebuild "$target" || return 0

    python3 "$DVDLOGO" "$OUTDIR/${STEM}_highlights_60s_overlay.mp4" "$logo" \
        --speed 0 --logo-scale .4 --logo-rotate-speed 0 --trails 0.85 --opacity .5 \
        --audio-reactive-glow 1.0 --audio-reactive-scale 0.5 --audio-reactive-gain 2.0 \
        --edge-margin-px 0 --reels-local-overlay false --voidstar-preset wild \
        --local-point-track true --local-point-track-scale 1.2 --local-point-track-pad-px 0 \
        --local-point-track-max-points 256 --local-point-track-radius 256 --local-point-track-min-distance 128 \
        --local-point-track-refresh 8 --local-point-track-opacity 0.77 --local-point-track-decay 0.77 \
        --local-point-track-link-neighbors 8 --local-point-track-link-thickness 1 \
        --local-point-track-link-opacity .77 --voidstar-colorize true --start-x .8 --start-y .83 \
        --content-bbox-for-local false --voidstar-debug-bounds true --voidstar-debug-bounds-mode hit-glitch \
        --voidstar-debug-bounds-hit-threshold 0.8 --voidstar-debug-bounds-hit-prob 0.1 \
        --output "$target"
}

# 2. 90s highlight (START anchor)
run_90s_start() {
    echo "--- 90s highlight (START) ---"
    local ss="$START_SECONDS"
    local full="$YOUTUBE_FULL_SECONDS"

    python3 "$DIVVY" highlights "$INPUT_VIDEO" \
        --start-seconds "$ss" --youtube-full-seconds "$full" --target-length-seconds 90 \
        --video-encoder libx264 --preset medium --out-dir "$OUTDIR" \
        --glitch-seconds "$GLITCH_SECONDS" --glitch-style vuwind --cps "$CPS" --loop-seam-seconds "$LOOP_SEAM_SECONDS" \
        --sampling-mode uniform-spread --sample-seconds 16 --truncate-to-full-clips

    python3 "$REELS_OVERLAY" "$OUTDIR/${STEM}__highlights__mode-uniform-spread__start-${ss}s__full-${full}s__target-90s.mp4" \
        --min-det-conf 0.05 --min-trk-conf 0.05 --draw-ids true \
        --smear true --smear-frames 17 --smear-decay 0.99 \
        --trail true --trail-alpha .999 --beat-sync true \
        --velocity-color true --velocity-color-mult 10

    rename_output "$OUTDIR/${STEM}__highlights__mode-uniform-spread__start-${ss}s__full-${full}s__target-90s_fps30_mc2_det0p05_trk0p05_trail1_ta1p00_tlotrue_scan1_velcolor_ids_beatsync_smear.mp4" \
        "$OUTDIR/${STEM}_highlights_90s_overlay.mp4"

    local picked logo tag target
    picked="$(pick_logo_for_run LOGOS)"
    logo="${picked%%|*}"
    tag="${picked##*|}"
    target="$(with_logo_suffix "$OUTDIR/${STEM}_highlights_90s_overlay_logo.mp4" "$tag")"

    should_rebuild "$target" || return 0

    python3 "$DVDLOGO" "$OUTDIR/${STEM}_highlights_90s_overlay.mp4" "$logo" \
        --speed 0 --logo-scale .4 --logo-rotate-speed 0 --trails 0.85 --opacity .5 \
        --audio-reactive-glow 1.0 --audio-reactive-scale 0.5 --audio-reactive-gain 2.0 \
        --edge-margin-px 0 --reels-local-overlay false --voidstar-preset wild \
        --local-point-track true --local-point-track-scale 1.2 --local-point-track-pad-px 0 \
        --local-point-track-max-points 256 --local-point-track-radius 256 --local-point-track-min-distance 128 \
        --local-point-track-refresh 8 --local-point-track-opacity 0.77 --local-point-track-decay 0.77 \
        --local-point-track-link-neighbors 8 --local-point-track-link-thickness 1 \
        --local-point-track-link-opacity .77 --voidstar-colorize true --start-x .8 --start-y .83 \
        --content-bbox-for-local false --voidstar-debug-bounds true --voidstar-debug-bounds-mode hit-glitch \
        --voidstar-debug-bounds-hit-threshold 0.8 --voidstar-debug-bounds-hit-prob 0.1 \
        --output "$target"
}

# 3. 180s highlight (START anchor)
run_180s_start() {
    echo "--- 180s highlight (START) ---"
    local ss="$START_SECONDS"
    local full="$YOUTUBE_FULL_SECONDS"

    python3 "$DIVVY" highlights "$INPUT_VIDEO" \
        --start-seconds "$ss" --youtube-full-seconds "$full" --target-length-seconds 180 \
        --video-encoder libx264 --preset medium --out-dir "$OUTDIR" \
        --glitch-seconds "$GLITCH_SECONDS" --glitch-style vuwind --cps "$CPS" --loop-seam-seconds "$LOOP_SEAM_SECONDS" \
        --sampling-mode uniform-spread --n-segments 6 --sample-seconds 32 --truncate-to-full-clips

    python3 "$REELS_OVERLAY" "$OUTDIR/${STEM}__highlights__mode-uniform-spread__start-${ss}s__full-${full}s__target-180s.mp4" \
        --min-det-conf 0.05 --min-trk-conf 0.05 --draw-ids true \
        --smear true --smear-frames 17 --smear-decay 0.99 \
        --trail true --trail-alpha .999 --beat-sync true \
        --velocity-color true --velocity-color-mult 10

    rename_output "$OUTDIR/${STEM}__highlights__mode-uniform-spread__start-${ss}s__full-${full}s__target-180s_fps30_mc2_det0p05_trk0p05_trail1_ta1p00_tlotrue_scan1_velcolor_ids_beatsync_smear.mp4" \
        "$OUTDIR/${STEM}_highlights_180s_overlay.mp4"

    local picked logo tag target
    picked="$(pick_logo_for_run LOGOS)"
    logo="${picked%%|*}"
    tag="${picked##*|}"
    target="$(with_logo_suffix "$OUTDIR/${STEM}_highlights_180s_overlay_logo.mp4" "$tag")"

    should_rebuild "$target" || return 0

    python3 "$DVDLOGO" "$OUTDIR/${STEM}_highlights_180s_overlay.mp4" "$logo" \
        --speed 0 --logo-scale .4 --logo-rotate-speed 0 --trails 0.85 --opacity .5 \
        --audio-reactive-glow 1.0 --audio-reactive-scale 0.5 --audio-reactive-gain 2.0 \
        --edge-margin-px 0 --reels-local-overlay false --voidstar-preset wild \
        --local-point-track true --local-point-track-scale 1.2 --local-point-track-pad-px 0 \
        --local-point-track-max-points 256 --local-point-track-radius 256 --local-point-track-min-distance 128 \
        --local-point-track-refresh 8 --local-point-track-opacity 0.77 --local-point-track-decay 0.77 \
        --local-point-track-link-neighbors 8 --local-point-track-link-thickness 1 \
        --local-point-track-link-opacity .77 --voidstar-colorize true --start-x .8 --start-y .83 \
        --content-bbox-for-local false --voidstar-debug-bounds true --voidstar-debug-bounds-mode hit-glitch \
        --voidstar-debug-bounds-hit-threshold 0.8 --voidstar-debug-bounds-hit-prob 0.1 \
        --output "$target"
}

# 4. FULL (no divvy, just overlay and logo)
run_full() {
    echo "--- FULL (YouTube) ---"
    local base_overlay="$OUTDIR/${STEM}_full_overlay.mp4"

    should_rebuild "$base_overlay" || return 0

    python3 "$REELS_OVERLAY" "$INPUT_VIDEO" \
        --min-det-conf 0.05 --min-trk-conf 0.05 --draw-ids true \
        --smear true --smear-frames 17 --smear-decay 0.99 \
        --trail true --trail-alpha .999 --beat-sync true \
        --velocity-color true --velocity-color-mult 10

    rename_output "$OUTDIR/${STEM}_fps30_mc2_det0p05_trk0p05_trail1_ta1p00_tlotrue_scan1_velcolor_ids_beatsync_smear.mp4" \
        "$base_overlay"

    local picked logo tag target
    picked="$(pick_logo_for_run LOGOS)"
    logo="${picked%%|*}"
    tag="${picked##*|}"
    target="$(with_logo_suffix "$OUTDIR/${STEM}_full_overlay_logo.mp4" "$tag")"

    should_rebuild "$target" || return 0

    python3 "$DVDLOGO" "$base_overlay" "$logo" \
        --speed 0 --logo-scale .4 --logo-rotate-speed 0 --trails 0.85 --opacity .5 \
        --audio-reactive-glow 1.0 --audio-reactive-scale 0.5 --audio-reactive-gain 2.0 \
        --edge-margin-px 0 --reels-local-overlay false --voidstar-preset wild \
        --local-point-track true --local-point-track-scale 1.2 --local-point-track-pad-px 0 \
        --local-point-track-max-points 256 --local-point-track-radius 256 --local-point-track-min-distance 128 \
        --local-point-track-refresh 8 --local-point-track-opacity 0.77 --local-point-track-decay 0.77 \
        --local-point-track-link-neighbors 8 --local-point-track-link-thickness 1 \
        --local-point-track-link-opacity .77 --voidstar-colorize true --start-x .8 --start-y .83 \
        --content-bbox-for-local false --voidstar-debug-bounds true --voidstar-debug-bounds-mode hit-glitch \
        --voidstar-debug-bounds-hit-threshold 0.8 --voidstar-debug-bounds-hit-prob 0.1 \
        --output "$target"
}

# 5. 60s highlight (END anchor)
run_60s_end() {
    echo "--- 60s highlight (END) ---"

    local w ss full
    w="$(compute_60_window)"
    ss="${w%%|*}"
    full="${w##*|}"
    if (( INPUT_DURATION_SECONDS > 600 )); then
        echo "60t END: input > 10m, using middle 10m segment: start-seconds=$ss youtube-full-seconds=$full"
    fi

    python3 "$DIVVY" highlights "$INPUT_VIDEO" \
        --start-seconds "$ss" --youtube-full-seconds "$full" --target-length-seconds 60 \
        --video-encoder libx264 --preset medium --out-dir "$OUTDIR" \
        --glitch-seconds "$GLITCH_SECONDS" --glitch-style vuwind --cps "$CPS" --loop-seam-seconds "$LOOP_SEAM_SECONDS" \
        --sampling-mode uniform-spread --n-segments 8 --sample-seconds 8 --sample-anchor end \
        --truncate-to-full-clips

    rename_output "$OUTDIR/${STEM}__highlights__mode-uniform-spread__start-${ss}s__full-${full}s__target-60s.mp4" \
        "$OUTDIR/${STEM}_highlights_60t.mp4"

    python3 "$REELS_OVERLAY" "$OUTDIR/${STEM}_highlights_60t.mp4" \
        --min-det-conf 0.05 --min-trk-conf 0.05 --draw-ids true \
        --smear true --smear-frames 17 --smear-decay 0.99 \
        --trail true --trail-alpha .999 --beat-sync true \
        --velocity-color true --velocity-color-mult 10 \
        --output "$OUTDIR/${STEM}_highlights_60t_overlay.mp4"

    rename_output "$OUTDIR/${STEM}_highlights_60t_fps30_mc2_det0p05_trk0p05_trail1_ta1p00_tlotrue_scan1_velcolor_ids_beatsync_smear.mp4" \
        "$OUTDIR/${STEM}_highlights_60t_overlay.mp4"

    local picked logo tag target
    picked="$(pick_logo_for_run LOGOS)"
    logo="${picked%%|*}"
    tag="${picked##*|}"
    target="$(with_logo_suffix "$OUTDIR/${STEM}_highlights_60t_overlay_logo.mp4" "$tag")"

    should_rebuild "$target" || return 0

    python3 "$DVDLOGO" "$OUTDIR/${STEM}_highlights_60t_overlay.mp4" "$logo" \
        --speed 0 --logo-scale .4 --logo-rotate-speed 0 --trails 0.85 --opacity .5 \
        --audio-reactive-glow 1.0 --audio-reactive-scale 0.5 --audio-reactive-gain 2.0 \
        --edge-margin-px 0 --reels-local-overlay false --voidstar-preset wild \
        --local-point-track true --local-point-track-scale 1.2 --local-point-track-pad-px 0 \
        --local-point-track-max-points 256 --local-point-track-radius 256 --local-point-track-min-distance 128 \
        --local-point-track-refresh 8 --local-point-track-opacity 0.77 --local-point-track-decay 0.77 \
        --local-point-track-link-neighbors 8 --local-point-track-link-thickness 1 \
        --local-point-track-link-opacity .77 --voidstar-colorize true --start-x .8 --start-y .83 \
        --content-bbox-for-local false --voidstar-debug-bounds true --voidstar-debug-bounds-mode hit-glitch \
        --voidstar-debug-bounds-hit-threshold 0.8 --voidstar-debug-bounds-hit-prob 0.1 \
        --output "$target"
}

# 6. 90s highlight (END anchor)
run_90s_end() {
    echo "--- 90s highlight (END) ---"
    local ss="$START_SECONDS"
    local full="$YOUTUBE_FULL_SECONDS"

    python3 "$DIVVY" highlights "$INPUT_VIDEO" \
        --start-seconds "$ss" --youtube-full-seconds "$full" --target-length-seconds 90 \
        --video-encoder libx264 --preset medium --out-dir "$OUTDIR" \
        --glitch-seconds "$GLITCH_SECONDS" --glitch-style vuwind --cps "$CPS" --loop-seam-seconds "$LOOP_SEAM_SECONDS" \
        --sampling-mode uniform-spread --sample-seconds 16 --sample-anchor end --truncate-to-full-clips

    rename_output "$OUTDIR/${STEM}__highlights__mode-uniform-spread__start-${ss}s__full-${full}s__target-90s.mp4" \
        "$OUTDIR/${STEM}_highlights_90t.mp4"

    python3 "$REELS_OVERLAY" "$OUTDIR/${STEM}_highlights_90t.mp4" \
        --min-det-conf 0.05 --min-trk-conf 0.05 --draw-ids true \
        --smear true --smear-frames 17 --smear-decay 0.99 \
        --trail true --trail-alpha .999 --beat-sync true \
        --velocity-color true --velocity-color-mult 10 \
        --output "$OUTDIR/${STEM}_highlights_90t_overlay.mp4"

    rename_output "$OUTDIR/${STEM}_highlights_90t_fps30_mc2_det0p05_trk0p05_trail1_ta1p00_tlotrue_scan1_velcolor_ids_beatsync_smear.mp4" \
        "$OUTDIR/${STEM}_highlights_90t_overlay.mp4"

    local picked logo tag target
    picked="$(pick_logo_for_run LOGOS)"
    logo="${picked%%|*}"
    tag="${picked##*|}"
    target="$(with_logo_suffix "$OUTDIR/${STEM}_highlights_90t_overlay_logo.mp4" "$tag")"

    should_rebuild "$target" || return 0

    python3 "$DVDLOGO" "$OUTDIR/${STEM}_highlights_90t_overlay.mp4" "$logo" \
        --speed 0 --logo-scale .4 --logo-rotate-speed 0 --trails 0.85 --opacity .5 \
        --audio-reactive-glow 1.0 --audio-reactive-scale 0.5 --audio-reactive-gain 2.0 \
        --edge-margin-px 0 --reels-local-overlay false --voidstar-preset wild \
        --local-point-track true --local-point-track-scale 1.2 --local-point-track-pad-px 0 \
        --local-point-track-max-points 256 --local-point-track-radius 256 --local-point-track-min-distance 128 \
        --local-point-track-refresh 8 --local-point-track-opacity 0.77 --local-point-track-decay 0.77 \
        --local-point-track-link-neighbors 8 --local-point-track-link-thickness 1 \
        --local-point-track-link-opacity .77 --voidstar-colorize true --start-x .8 --start-y .83 \
        --content-bbox-for-local false --voidstar-debug-bounds true --voidstar-debug-bounds-mode hit-glitch \
        --voidstar-debug-bounds-hit-threshold 0.8 --voidstar-debug-bounds-hit-prob 0.1 \
        --output "$target"
}

# 7. 180s highlight (END anchor)
run_180s_end() {
    echo "--- 180s highlight (END) ---"
    local ss="$START_SECONDS"
    local full="$YOUTUBE_FULL_SECONDS"

    python3 "$DIVVY" highlights "$INPUT_VIDEO" \
        --start-seconds "$ss" --youtube-full-seconds "$full" --target-length-seconds 180 \
        --video-encoder libx264 --preset medium --out-dir "$OUTDIR" \
        --glitch-seconds "$GLITCH_SECONDS" --glitch-style vuwind --cps "$CPS" --loop-seam-seconds "$LOOP_SEAM_SECONDS" \
        --sampling-mode uniform-spread --n-segments 6 --sample-seconds 32 --sample-anchor end --truncate-to-full-clips

    rename_output "$OUTDIR/${STEM}__highlights__mode-uniform-spread__start-${ss}s__full-${full}s__target-180s.mp4" \
        "$OUTDIR/${STEM}_highlights_180t.mp4"

    python3 "$REELS_OVERLAY" "$OUTDIR/${STEM}_highlights_180t.mp4" \
        --min-det-conf 0.05 --min-trk-conf 0.05 --draw-ids true \
        --smear true --smear-frames 17 --smear-decay 0.99 \
        --trail true --trail-alpha .999 --beat-sync true \
        --velocity-color true --velocity-color-mult 10 \
        --output "$OUTDIR/${STEM}_highlights_180t_overlay.mp4"

    rename_output "$OUTDIR/${STEM}_highlights_180t_fps30_mc2_det0p05_trk0p05_trail1_ta1p00_tlotrue_scan1_velcolor_ids_beatsync_smear.mp4" \
        "$OUTDIR/${STEM}_highlights_180t_overlay.mp4"

    local picked logo tag target
    picked="$(pick_logo_for_run LOGOS)"
    logo="${picked%%|*}"
    tag="${picked##*|}"
    target="$(with_logo_suffix "$OUTDIR/${STEM}_highlights_180t_overlay_logo.mp4" "$tag")"

    should_rebuild "$target" || return 0

    python3 "$DVDLOGO" "$OUTDIR/${STEM}_highlights_180t_overlay.mp4" "$logo" \
        --speed 0 --logo-scale .4 --logo-rotate-speed 0 --trails 0.85 --opacity .5 \
        --audio-reactive-glow 1.0 --audio-reactive-scale 0.5 --audio-reactive-gain 2.0 \
        --edge-margin-px 0 --reels-local-overlay false --voidstar-preset wild \
        --local-point-track true --local-point-track-scale 1.2 --local-point-track-pad-px 0 \
        --local-point-track-max-points 256 --local-point-track-radius 256 --local-point-track-min-distance 128 \
        --local-point-track-refresh 8 --local-point-track-opacity 0.77 --local-point-track-decay 0.77 \
        --local-point-track-link-neighbors 8 --local-point-track-link-thickness 1 \
        --local-point-track-link-opacity .77 --voidstar-colorize true --start-x .8 --start-y .83 \
        --content-bbox-for-local false --voidstar-debug-bounds true --voidstar-debug-bounds-mode hit-glitch \
        --voidstar-debug-bounds-hit-threshold 0.8 --voidstar-debug-bounds-hit-prob 0.1 \
        --output "$target"
}

# ----------------------------
# Main
# ----------------------------

main() {
    FORCE=0
    END_ONLY=0
    PREVIEW=0

    # Video / output
    INPUT_VIDEO=""
    OUTDIR=""

    # Parameterized knobs
    START_SECONDS=0
    YOUTUBE_FULL_SECONDS=""   # default later to full duration
    CPS=0.5
    GLITCH_SECONDS=2
    LOOP_SEAM_SECONDS=""      # default later to glitch-seconds

    # Logo patterns (repeatable)
    LOGO_PATTERNS=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --force|-f) FORCE=1; shift ;;
            --end-only) END_ONLY=1; shift ;;
            preview)    PREVIEW=1; shift ;;

            --input)    INPUT_VIDEO="$2"; shift 2 ;;
            --outdir)   OUTDIR="$2"; shift 2 ;;

            --start-seconds) START_SECONDS="$2"; shift 2 ;;
            --youtube-full-seconds) YOUTUBE_FULL_SECONDS="$2"; shift 2 ;;
            --cps) CPS="$2"; shift 2 ;;
            --glitch-seconds) GLITCH_SECONDS="$2"; shift 2 ;;
            --loop-seam-seconds) LOOP_SEAM_SECONDS="$2"; shift 2 ;;

            --logo)
                LOGO_PATTERNS+=( "$2" )
                shift 2
                ;;

            -h|--help)
                sed -n '1,120p' "$0"
                return 0
                ;;

            # Positional mp4 (only if it doesn't look like a flag)
            *)
                if [[ "$1" != -* ]]; then
                    INPUT_VIDEO="$1"
                    shift
                else
                    die "Unknown arg: $1 (try --help)"
                fi
                ;;
        esac
    done

    # Defaults (support env vars, keep prior behavior as fallback)
    if [[ -z "${INPUT_VIDEO}" ]]; then
        INPUT_VIDEO=$(eval echo "${INPUT_VIDEO:-${VOIDSTAR_INPUT_VIDEO:-~/WinVideos/gluons_voidstar_0.mp4}}")
    else
        INPUT_VIDEO=$(eval echo "$INPUT_VIDEO")
    fi

    STEM="$(basename "${INPUT_VIDEO%.*}")"

    if [[ -z "${OUTDIR}" ]]; then
        OUTDIR=$(eval echo "${OUTDIR:-${VOIDSTAR_OUTDIR:-~/WinVideos/${STEM}/}}")
    else
        OUTDIR=$(eval echo "$OUTDIR")
    fi

    # Tooling dependencies
    require_cmd python3
    require_cmd ffprobe

    mkdir -p "$OUTDIR"

    require_file "INPUT_VIDEO" "$INPUT_VIDEO"

    # Compute duration and defaults that depend on duration
    INPUT_DURATION_SECONDS="$(get_video_duration_seconds "$INPUT_VIDEO")"
    if [[ -z "${YOUTUBE_FULL_SECONDS}" ]]; then
        YOUTUBE_FULL_SECONDS="$INPUT_DURATION_SECONDS"
    fi
    if [[ -z "${LOOP_SEAM_SECONDS}" ]]; then
        LOOP_SEAM_SECONDS="$GLITCH_SECONDS"
    fi

    echo "Input: $INPUT_VIDEO"
    echo "Stem:  $STEM"
    echo "Out:   $OUTDIR"
    echo "Dur:   ${INPUT_DURATION_SECONDS}s"
    echo "Args:  start=${START_SECONDS}s full=${YOUTUBE_FULL_SECONDS}s cps=${CPS} glitch=${GLITCH_SECONDS}s loop_seam=${LOOP_SEAM_SECONDS}s"

    # Auto-detect required scripts and logo image(s)
    PROJECT_ROOT="/home/$USER/code/voidstar"
    find_script() {
        local script_name="$1"
        local found
        found=$(find "$PROJECT_ROOT" -type f -name "$script_name" | head -n 1)
        if [[ -z "$found" ]]; then
            die "Could not find required script: $script_name in $PROJECT_ROOT"
        fi
        echo "$found"
    }
    find_void_logos_default() {
        # Default: all void*.png under project root (sorted for stable rotation)
        # This enables variety without passing --logo.
        find "$PROJECT_ROOT" -type f -iname "void*.png" | sort
    }

    DIVVY=$(find_script "divvy.py")
    REELS_OVERLAY=$(find_script "reels_cv_overlay.py")
    DVDLOGO=$(find_script "voidstar_dvd_logo.py")

    require_file "DIVVY" "$DIVVY"
    require_file "REELS_OVERLAY" "$REELS_OVERLAY"
    require_file "DVDLOGO" "$DVDLOGO"

    # Build LOGOS array
    LOGOS=()
    if (( ${#LOGO_PATTERNS[@]} > 0 )); then
        mapfile -t LOGOS < <(expand_logo_patterns "${LOGO_PATTERNS[@]}")
        echo "Using user-specified logos (${#LOGOS[@]}):"
        printf '  %s\n' "${LOGOS[@]}"
    else
        mapfile -t LOGOS < <(find_void_logos_default)
        if (( ${#LOGOS[@]} == 0 )); then
            die "Could not find any default logos matching void*.png in $PROJECT_ROOT (pass --logo to specify)"
        fi
        # Validate each found logo
        for f in "${LOGOS[@]}"; do
            require_file "LOGO_IMG" "$f"
        done
        echo "Using default auto-detected void*.png logos (${#LOGOS[@]}):"
        printf '  %s
' "${LOGOS[@]}"
    fi

    if [[ "$PREVIEW" -eq 1 ]]; then
        run_60s_start
        return 0
    fi

    if [[ "$END_ONLY" -eq 1 ]]; then
        run_60s_end
        run_90s_end
        run_180s_end
        return 0
    fi

    run_60s_start
    run_90s_start
    run_180s_start
    run_full
    run_60s_end
    run_90s_end
    run_180s_end
}

main "$@"
