#!/usr/bin/env bash
# pipeline_counterfactual_dimensions_voidstar_0.sh
# Based on pipeline_gluons.sh (keeps same flow and defaults, plus optional glitchfield layer).
#
# Performance optimizations:
# - Reels CV overlay is expensive: run it ONCE on the full input video to create a cached
#   "base overlay" video, then run divvy on that base overlay to make each highlight.
#   This avoids running reels overlay 6+ times.
# - Optional parallelism for independent targets (divvy + dvdlogo) via --jobs N.
#   Logo assignment remains deterministic and matches the sequential target order.

set -euo pipefail

die() { echo "Error: $*" >&2; exit 1; }
require_cmd() { command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"; }
require_file() { local label="$1" path="$2"; [[ -f "$path" ]] || die "Missing $label file: $path"; }

should_rebuild() {
    local target="$1"
    local script="$0"
    if [[ "${FORCE:-0}" -eq 1 ]]; then return 0; fi
    if [[ ! -f "$target" ]]; then return 0; fi
    if [[ "$script" -nt "$target" ]]; then return 0; fi
    echo "Target $target is up-to-date. Skipping."
    return 1
}

rename_output() {
    local src="$1" dst="$2"
    if [[ -f "$src" ]]; then mv -v "$src" "$dst"; else echo "Warning: $src not found for renaming."; fi
}

get_video_duration_seconds() {
    local video="$1"
    local dur
    dur="$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$video" 2>/dev/null || true)"
    [[ -n "$dur" ]] || die "Could not determine video duration via ffprobe: $video"
    python3 - <<PY
import math
d=float("$dur")
print(int(math.floor(d)))
PY
}

expand_logo_patterns() {
    local -a out=()
    local pat
    shopt -s nullglob
    for pat in "$@"; do
        local -a matches=( $pat )
        (( ${#matches[@]} > 0 )) || die "Logo pattern did not match any files: $pat"
        out+=( "${matches[@]}" )
    done
    shopt -u nullglob
    printf '%s\n' "${out[@]}"
}

find_void_logos_default() {
    # Default: all void*.png under project root (sorted for stable rotation)
    find "$PROJECT_ROOT" -type f -iname "void*.png" | sort
}

with_logo_suffix() {
    local base_path="$1" tag="$2"
    if [[ -z "$tag" ]]; then echo "$base_path"; else echo "${base_path%.mp4}_logo-${tag}.mp4"; fi
}

# Deterministic logo assignment that matches sequential target order:
# logo_index = target_ordinal % num_logos
logo_for_ordinal() {
    local ordinal="$1"
    local count="${#LOGOS[@]}"
    (( count > 0 )) || die "Internal error: no logos available"
    local idx=$(( ordinal % count ))
    local logo="${LOGOS[$idx]}"
    local tag=""
    if (( count > 1 )); then tag="$(basename "${logo%.*}")"; fi
    echo "$logo|$tag"
}

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
# Parallel job control (optional)
# ----------------------------
JOBS=1

_sem_init() {
    local n="$1"
    mkfifo /tmp/voidstar_sem.$$ || true
    exec 9<>/tmp/voidstar_sem.$$
    rm -f /tmp/voidstar_sem.$$
    for _ in $(seq 1 "$n"); do printf '.' >&9; done
}
_sem_acquire() { read -r -n 1 <&9; }
_sem_release() { printf '.' >&9; }

# ----------------------------
# Core expensive cache: base reels overlay for full input
# ----------------------------
build_base_reels_overlay() {
    echo "--- Base reels overlay (cache) ---"
    local target="$BASE_REELS_OVERLAY"
    should_rebuild "$target" || return 0

    python3 "$REELS_OVERLAY" "$INPUT_VIDEO" \
        --min-det-conf 0.05 --min-trk-conf 0.05 --draw-ids true \
        --smear true --smear-frames 17 --smear-decay 0.99 \
        --trail true --trail-alpha .999 --beat-sync true \
        --velocity-color true --velocity-color-mult 10 \
        --output "$target"

    # rename_output "$OUTDIR/${STEM}_fps30_mc2_det0p05_trk0p05_trail1_ta1p00_tlotrue_scan1_velcolor_ids_beatsync_smear.mp4" \
    #     "$target"
}

run_optional_glitchfield_on_clip() {
    local input_clip="$1"
    local target="$2"
    local stage_label="${3:-clip}"

    GLITCHFIELD_STAGE_OUT="$input_clip"
    [[ "$ENABLE_GLITCHFIELD_STAGE" -eq 1 ]] || return 0

    echo "--- Optional glitchfield stage (${GLITCHFIELD_PRESET}) on ${stage_label} ---"
    require_file "GLITCHFIELD" "$GLITCHFIELD"

    local -a gf_args
    case "$GLITCHFIELD_PRESET" in
        clean)
            gf_args=(
                --effect combo --audio-reactive
                --beat-threshold 0.58 --beat-prob 1.0 --glitch-hold 12
                --combo-glyph-prob 0.6 --combo-stutter-prob 0.4 --combo-alternate-bias 0.55
                --color-mode input --input-palette-k 6 --palette-refresh 6
                --stutter-slice-prob 0.75 --stutter-slice-shift-max 120
                --fx-rgb-split --fx-rgb-shift-max 2 --fx-rgb-prob 0.45
                --fx-trail-frames 1 --fx-trail-strength 0.18
                --fx-scanline-jitter 0.08 --fx-scanline-step 2 --fx-scanline-shift-max 1 --fx-scanline-brightness 0.95
                --fx-vhold-prob 0.03 --fx-vhold-max 4
                --min-gate-period 6
            )
            ;;
        gritty)
            gf_args=(
                --effect combo --audio-reactive
                --beat-threshold 0.56 --beat-prob 1.0 --glitch-hold 13
                --combo-glyph-prob 0.45 --combo-stutter-prob 0.55 --combo-alternate-bias 0.6
                --color-mode audio --palette fire --audio-color-reactive
                --stutter-slice-prob 0.9 --stutter-slice-shift-max 170
                --stutter-frame-jitter 3 --stutter-jitter-prob 0.65
                --fx-rgb-split --fx-rgb-shift-max 3 --fx-rgb-prob 0.7
                --fx-trail-frames 2 --fx-trail-strength 0.3
                --fx-scanline-jitter 0.22 --fx-scanline-step 2 --fx-scanline-shift-max 2 --fx-scanline-brightness 0.9
                --fx-line-dropout-prob 0.06 --fx-line-dropout-thickness 2 --fx-line-dropout-bright
                --fx-posterize-levels 6
                --fx-vhold-prob 0.08 --fx-vhold-max 10
                --min-gate-period 8
            )
            ;;
        chaos)
            gf_args=(
                --effect combo --audio-reactive
                --beat-threshold 0.52 --beat-prob 1.0 --glitch-hold 14
                --combo-glyph-prob 0.3 --combo-stutter-prob 0.7 --combo-alternate-bias 0.45
                --color-mode fixed --palette insane --glitch-prob 0.45
                --stutter-slice-prob 0.95 --stutter-slice-shift-max 220
                --stutter-slice-min-bands 4 --stutter-slice-max-bands 12
                --stutter-frame-jitter 4 --stutter-jitter-prob 0.75
                --fx-rgb-split --fx-rgb-shift-max 6 --fx-rgb-prob 0.9
                --fx-trail-frames 3 --fx-trail-strength 0.38
                --fx-scanline-jitter 0.3 --fx-scanline-step 2 --fx-scanline-shift-max 3 --fx-scanline-brightness 0.85
                --fx-line-dropout-prob 0.12 --fx-line-dropout-thickness 3 --fx-line-dropout-bright
                --fx-block-jitter-prob 0.05 --fx-block-size 12 --fx-block-shift-max 4
                --fx-posterize-levels 5
                --fx-bitcrush-bits 6
                --fx-vhold-prob 0.12 --fx-vhold-max 14
            )
            ;;
        custom)
            if [[ -z "$GLITCHFIELD_CUSTOM_ARGS" ]]; then
                die "GLITCHFIELD_PRESET=custom requires GLITCHFIELD_CUSTOM_ARGS"
            fi
            read -r -a gf_args <<< "$GLITCHFIELD_CUSTOM_ARGS"
            ;;
        *)
            die "Unknown GLITCHFIELD_PRESET: $GLITCHFIELD_PRESET (use clean|gritty|chaos|custom)"
            ;;
    esac

    if [[ -n "${GLITCHFIELD_MIN_GATE_PERIOD:-}" ]]; then
        gf_args+=(--min-gate-period "$GLITCHFIELD_MIN_GATE_PERIOD")
    fi

    if [[ "$USE_GLITCHFIELD_CACHE" -eq 1 ]]; then
        if ! should_rebuild "$target"; then
            GLITCHFIELD_STAGE_OUT="$target"
            echo "[glitchfield] using cached: $target"
            return 0
        fi
    fi

    python3 "$GLITCHFIELD" "$input_clip" \
        "${gf_args[@]}" \
        --seed "$GLITCHFIELD_SEED"

    local generated_src in_dir in_stem
    in_dir="$(dirname "$input_clip")"
    in_stem="$(basename "${input_clip%.*}")"
    generated_src="$(ls -t "$in_dir/${in_stem}"_*.mp4 2>/dev/null | head -n 1 || true)"
    [[ -n "$generated_src" && -f "$generated_src" ]] || die "Could not locate glitchfield output for stem: $in_stem"

    rename_output "$generated_src" "$target"
    require_file "GLITCHFIELD_STAGE" "$target"
    GLITCHFIELD_STAGE_OUT="$target"
    echo "[glitchfield] staged clip: $target"
}

# ----------------------------
# Targets
# ----------------------------
run_60s_start() {
    local ordinal=0
    echo "--- 60s highlight (START) ---"
    local w ss full; w="$(compute_60_window)"; ss="${w%%|*}"; full="${w##*|}"
    if (( INPUT_DURATION_SECONDS > 600 )); then
        echo "60s START: input > 10m, using middle 10m segment: start-seconds=$ss youtube-full-seconds=$full"
    fi

    python3 "$DIVVY" highlights "$BASE_REELS_OVERLAY" \
        --start-seconds "$ss" --youtube-full-seconds "$full" --target-length-seconds 60 \
        --video-encoder libx264 --preset medium --out-dir "$OUTDIR" \
        --glitch-seconds "$GLITCH_SECONDS" --glitch-style vuwind --cps "$CPS" --loop-seam-seconds "$LOOP_SEAM_SECONDS" \
        --sampling-mode uniform-spread --n-segments 3 --sample-seconds 20 --truncate-to-full-clips

    local base_stem; base_stem="$(basename "${BASE_REELS_OVERLAY%.*}")"
    local divvy_src="$OUTDIR/${base_stem}__highlights__mode-uniform-spread__start-${ss}s__full-${full}s__target-60s.mp4"
    local divvy_dst="$OUTDIR/${STEM}_highlights_60s_overlay.mp4"
    rename_output "$divvy_src" "$divvy_dst"

    local source_for_logo="$divvy_dst"
    local glitch_dst="$OUTDIR/${STEM}_highlights_60s_overlay_glitchfield_${GLITCHFIELD_PRESET}.mp4"
    run_optional_glitchfield_on_clip "$divvy_dst" "$glitch_dst" "60s-start"
    source_for_logo="$GLITCHFIELD_STAGE_OUT"

    local picked logo tag target
    picked="$(logo_for_ordinal "$ordinal")"; logo="${picked%%|*}"; tag="${picked##*|}"
    target="$(with_logo_suffix "$OUTDIR/${STEM}_highlights_60s_overlay_logo.mp4" "$tag")"
    should_rebuild "$target" || return 0

    python3 "$DVDLOGO" "$source_for_logo" "$logo" \
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

run_90s_start() {
    local ordinal=1
    echo "--- 90s highlight (START) ---"
    local ss="$START_SECONDS" full="$YOUTUBE_FULL_SECONDS"

    python3 "$DIVVY" highlights "$BASE_REELS_OVERLAY" \
        --start-seconds "$ss" --youtube-full-seconds "$full" --target-length-seconds 90 \
        --video-encoder libx264 --preset medium --out-dir "$OUTDIR" \
        --glitch-seconds "$GLITCH_SECONDS" --glitch-style vuwind --cps "$CPS" --loop-seam-seconds "$LOOP_SEAM_SECONDS" \
        --sampling-mode uniform-spread --sample-seconds 16 --truncate-to-full-clips

    local base_stem; base_stem="$(basename "${BASE_REELS_OVERLAY%.*}")"
    local divvy_src="$OUTDIR/${base_stem}__highlights__mode-uniform-spread__start-${ss}s__full-${full}s__target-90s.mp4"
    local divvy_dst="$OUTDIR/${STEM}_highlights_90s_overlay.mp4"
    rename_output "$divvy_src" "$divvy_dst"

    local source_for_logo="$divvy_dst"
    local glitch_dst="$OUTDIR/${STEM}_highlights_90s_overlay_glitchfield_${GLITCHFIELD_PRESET}.mp4"
    run_optional_glitchfield_on_clip "$divvy_dst" "$glitch_dst" "90s-start"
    source_for_logo="$GLITCHFIELD_STAGE_OUT"

    local picked logo tag target
    picked="$(logo_for_ordinal "$ordinal")"; logo="${picked%%|*}"; tag="${picked##*|}"
    target="$(with_logo_suffix "$OUTDIR/${STEM}_highlights_90s_overlay_logo.mp4" "$tag")"
    should_rebuild "$target" || return 0

    python3 "$DVDLOGO" "$source_for_logo" "$logo" \
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

run_180s_start() {
    local ordinal=2
    echo "--- 180s highlight (START) ---"
    local ss="$START_SECONDS" full="$YOUTUBE_FULL_SECONDS"

    python3 "$DIVVY" highlights "$BASE_REELS_OVERLAY" \
        --start-seconds "$ss" --youtube-full-seconds "$full" --target-length-seconds 180 \
        --video-encoder libx264 --preset medium --out-dir "$OUTDIR" \
        --glitch-seconds "$GLITCH_SECONDS" --glitch-style vuwind --cps "$CPS" --loop-seam-seconds "$LOOP_SEAM_SECONDS" \
        --sampling-mode uniform-spread --n-segments 6 --sample-seconds 32 --truncate-to-full-clips

    local base_stem; base_stem="$(basename "${BASE_REELS_OVERLAY%.*}")"
    local divvy_src="$OUTDIR/${base_stem}__highlights__mode-uniform-spread__start-${ss}s__full-${full}s__target-180s.mp4"
    local divvy_dst="$OUTDIR/${STEM}_highlights_180s_overlay.mp4"
    rename_output "$divvy_src" "$divvy_dst"

    local source_for_logo="$divvy_dst"
    local glitch_dst="$OUTDIR/${STEM}_highlights_180s_overlay_glitchfield_${GLITCHFIELD_PRESET}.mp4"
    run_optional_glitchfield_on_clip "$divvy_dst" "$glitch_dst" "180s-start"
    source_for_logo="$GLITCHFIELD_STAGE_OUT"

    local picked logo tag target
    picked="$(logo_for_ordinal "$ordinal")"; logo="${picked%%|*}"; tag="${picked##*|}"
    target="$(with_logo_suffix "$OUTDIR/${STEM}_highlights_180s_overlay_logo.mp4" "$tag")"
    should_rebuild "$target" || return 0

    python3 "$DVDLOGO" "$source_for_logo" "$logo" \
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

run_full() {
    local ordinal=3
    echo "--- FULL (YouTube) ---"
    local base_overlay="$OUTDIR/${STEM}_reels_base_overlay.mp4"
    # The actual generated file is <STEM>_fps30_mc2_det0p05_trk0p05_trail1_ta1p00_tlotrue_scan1_velcolor_ids_beatsync_smear.mp4
    local reels_base="$OUTDIR/${STEM}_fps30_mc2_det0p05_trk0p05_trail1_ta1p00_tlotrue_scan1_velcolor_ids_beatsync_smear.mp4"
    echo "[run_full] Checking if reels overlay needs rerun or cached overlay can be used..."
    if should_rebuild "$base_overlay"; then
        echo "[run_full] Decided to rerun reels overlay (base_overlay: $base_overlay)."
        if [[ -f "$reels_base" ]]; then
            echo "[run_full] Found reels overlay output: $reels_base. Moving to $base_overlay."
            mv -f "$reels_base" "$base_overlay"
        else
            echo "[run_full] Warning: $reels_base not found for renaming."
        fi
    else
        echo "[run_full] Using cached reels overlay: $base_overlay (up-to-date)."
    fi

    local picked logo tag target
    picked="$(logo_for_ordinal "$ordinal")"; logo="${picked%%|*}"; tag="${picked##*|}"
    local source_for_logo="$base_overlay"
    local glitch_dst="$OUTDIR/${STEM}_full_overlay_glitchfield_${GLITCHFIELD_PRESET}.mp4"
    run_optional_glitchfield_on_clip "$base_overlay" "$glitch_dst" "full"
    source_for_logo="$GLITCHFIELD_STAGE_OUT"

    target="$(with_logo_suffix "$OUTDIR/${STEM}_full_overlay_logo.mp4" "$tag")"
    should_rebuild "$target" || return 0

    python3 "$DVDLOGO" "$source_for_logo" "$logo" \
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

run_60s_end() {
    local ordinal=4
    echo "--- 60s highlight (END) ---"
    local w ss full; w="$(compute_60_window)"; ss="${w%%|*}"; full="${w##*|}"
    if (( INPUT_DURATION_SECONDS > 600 )); then
        echo "60t END: input > 10m, using middle 10m segment: start-seconds=$ss youtube-full-seconds=$full"
    fi

    python3 "$DIVVY" highlights "$BASE_REELS_OVERLAY" \
        --start-seconds "$ss" --youtube-full-seconds "$full" --target-length-seconds 60 \
        --video-encoder libx264 --preset medium --out-dir "$OUTDIR" \
        --glitch-seconds "$GLITCH_SECONDS" --glitch-style vuwind --cps "$CPS" --loop-seam-seconds "$LOOP_SEAM_SECONDS" \
        --sampling-mode uniform-spread --n-segments 3 --sample-seconds 20 --sample-anchor end \
        --truncate-to-full-clips

    local base_stem; base_stem="$(basename "${BASE_REELS_OVERLAY%.*}")"
    local divvy_src="$OUTDIR/${base_stem}__highlights__mode-uniform-spread__start-${ss}s__full-${full}s__target-60s.mp4"
    local divvy_dst="$OUTDIR/${STEM}_highlights_60t_overlay.mp4"
    rename_output "$divvy_src" "$divvy_dst"

    local source_for_logo="$divvy_dst"
    local glitch_dst="$OUTDIR/${STEM}_highlights_60t_overlay_glitchfield_${GLITCHFIELD_PRESET}.mp4"
    run_optional_glitchfield_on_clip "$divvy_dst" "$glitch_dst" "60s-end"
    source_for_logo="$GLITCHFIELD_STAGE_OUT"

    local picked logo tag target
    picked="$(logo_for_ordinal "$ordinal")"; logo="${picked%%|*}"; tag="${picked##*|}"
    target="$(with_logo_suffix "$OUTDIR/${STEM}_highlights_60t_overlay_logo.mp4" "$tag")"
    should_rebuild "$target" || return 0

    python3 "$DVDLOGO" "$source_for_logo" "$logo" \
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

run_90s_end() {
    local ordinal=5
    echo "--- 90s highlight (END) ---"
    local ss="$START_SECONDS" full="$YOUTUBE_FULL_SECONDS"

    python3 "$DIVVY" highlights "$BASE_REELS_OVERLAY" \
        --start-seconds "$ss" --youtube-full-seconds "$full" --target-length-seconds 90 \
        --video-encoder libx264 --preset medium --out-dir "$OUTDIR" \
        --glitch-seconds "$GLITCH_SECONDS" --glitch-style vuwind --cps "$CPS" --loop-seam-seconds "$LOOP_SEAM_SECONDS" \
        --sampling-mode uniform-spread --sample-seconds 16 --sample-anchor end --truncate-to-full-clips

    local base_stem; base_stem="$(basename "${BASE_REELS_OVERLAY%.*}")"
    local divvy_src="$OUTDIR/${base_stem}__highlights__mode-uniform-spread__start-${ss}s__full-${full}s__target-90s.mp4"
    local divvy_dst="$OUTDIR/${STEM}_highlights_90t_overlay.mp4"
    rename_output "$divvy_src" "$divvy_dst"

    local source_for_logo="$divvy_dst"
    local glitch_dst="$OUTDIR/${STEM}_highlights_90t_overlay_glitchfield_${GLITCHFIELD_PRESET}.mp4"
    run_optional_glitchfield_on_clip "$divvy_dst" "$glitch_dst" "90s-end"
    source_for_logo="$GLITCHFIELD_STAGE_OUT"

    local picked logo tag target
    picked="$(logo_for_ordinal "$ordinal")"; logo="${picked%%|*}"; tag="${picked##*|}"
    target="$(with_logo_suffix "$OUTDIR/${STEM}_highlights_90t_overlay_logo.mp4" "$tag")"
    should_rebuild "$target" || return 0

    python3 "$DVDLOGO" "$source_for_logo" "$logo" \
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

run_180s_end() {
    local ordinal=6
    echo "--- 180s highlight (END) ---"
    local ss="$START_SECONDS" full="$YOUTUBE_FULL_SECONDS"

    python3 "$DIVVY" highlights "$BASE_REELS_OVERLAY" \
        --start-seconds "$ss" --youtube-full-seconds "$full" --target-length-seconds 180 \
        --video-encoder libx264 --preset medium --out-dir "$OUTDIR" \
        --glitch-seconds "$GLITCH_SECONDS" --glitch-style vuwind --cps "$CPS" --loop-seam-seconds "$LOOP_SEAM_SECONDS" \
        --sampling-mode uniform-spread --n-segments 6 --sample-seconds 32 --sample-anchor end --truncate-to-full-clips

    local base_stem; base_stem="$(basename "${BASE_REELS_OVERLAY%.*}")"
    local divvy_src="$OUTDIR/${base_stem}__highlights__mode-uniform-spread__start-${ss}s__full-${full}s__target-180s.mp4"
    local divvy_dst="$OUTDIR/${STEM}_highlights_180t_overlay.mp4"
    rename_output "$divvy_src" "$divvy_dst"

    local source_for_logo="$divvy_dst"
    local glitch_dst="$OUTDIR/${STEM}_highlights_180t_overlay_glitchfield_${GLITCHFIELD_PRESET}.mp4"
    run_optional_glitchfield_on_clip "$divvy_dst" "$glitch_dst" "180s-end"
    source_for_logo="$GLITCHFIELD_STAGE_OUT"

    local picked logo tag target
    picked="$(logo_for_ordinal "$ordinal")"; logo="${picked%%|*}"; tag="${picked##*|}"
    target="$(with_logo_suffix "$OUTDIR/${STEM}_highlights_180t_overlay_logo.mp4" "$tag")"
    should_rebuild "$target" || return 0

    python3 "$DVDLOGO" "$source_for_logo" "$logo" \
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

main() {
    FORCE=0; END_ONLY=0; PREVIEW=1
    PIPELINE_MODE="preview"
    INPUT_VIDEO=""; OUTDIR=""
    START_SECONDS=108; YOUTUBE_FULL_SECONDS="784"; CPS=0.5; GLITCH_SECONDS=0.5; LOOP_SEAM_SECONDS="0.5"
    LOGO_PATTERNS=()
    USE_REELS_CACHE=1
    ENABLE_GLITCHFIELD_STAGE=1
    USE_GLITCHFIELD_CACHE=1
    GLITCHFIELD_PRESET="clean"
    GLITCHFIELD_SEED=1337
    GLITCHFIELD_MIN_GATE_PERIOD="7"
    GLITCHFIELD_CUSTOM_ARGS=""
    JOBS=1

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --force|-f) FORCE=1; shift ;;
            --end-only) END_ONLY=1; shift ;;
            preview) PREVIEW=1; shift ;;
            --mode)
                case "$2" in
                    preview)
                        PREVIEW=1
                        END_ONLY=0
                        PIPELINE_MODE="preview"
                        ;;
                    end-only)
                        END_ONLY=1
                        PREVIEW=0
                        PIPELINE_MODE="end-only"
                        ;;
                    all)
                        PREVIEW=0
                        END_ONLY=0
                        PIPELINE_MODE="all"
                        ;;
                    *)
                        die "Unknown --mode value: $2 (use preview|end-only|all)"
                        ;;
                esac
                shift 2
                ;;
            --input) INPUT_VIDEO="$2"; shift 2 ;;
            --outdir) OUTDIR="$2"; shift 2 ;;
            --start-seconds) START_SECONDS="$2"; shift 2 ;;
            --youtube-full-seconds) YOUTUBE_FULL_SECONDS="$2"; shift 2 ;;
            --cps) CPS="$2"; shift 2 ;;
            --glitch-seconds) GLITCH_SECONDS="$2"; shift 2 ;;
            --loop-seam-seconds) LOOP_SEAM_SECONDS="$2"; shift 2 ;;
            --logo) LOGO_PATTERNS+=( "$2" ); shift 2 ;;
            --jobs|-j) JOBS="$2"; shift 2 ;;
            --no-reels-cache) USE_REELS_CACHE=0; shift ;;
            --skip-reels-overlay) USE_REELS_CACHE=0; shift ;;
            --no-glitchfield) ENABLE_GLITCHFIELD_STAGE=0; shift ;;
            --no-glitchfield-cache) USE_GLITCHFIELD_CACHE=0; shift ;;
            --glitchfield-preset) GLITCHFIELD_PRESET="$2"; shift 2 ;;
            --glitchfield-seed) GLITCHFIELD_SEED="$2"; shift 2 ;;
            --glitchfield-min-gate-period) GLITCHFIELD_MIN_GATE_PERIOD="$2"; shift 2 ;;
            --glitchfield-custom-args) GLITCHFIELD_CUSTOM_ARGS="$2"; shift 2 ;;
            -h|--help) sed -n '1,130p' "$0"; return 0 ;;
            *)
                if [[ "$1" != -* ]]; then INPUT_VIDEO="$1"; shift
                else die "Unknown arg: $1 (try --help)"; fi
                ;;
        esac
    done

    if [[ -z "${INPUT_VIDEO}" ]]; then
        INPUT_VIDEO=$(eval echo "${INPUT_VIDEO:-${VOIDSTAR_INPUT_VIDEO:-/mnt/c/Users/brown/Videos/counterfactual_dimensions_voidstar_0/counterfactual_dimensions_voidstar_0.mp4}}")
    else
        INPUT_VIDEO=$(eval echo "$INPUT_VIDEO")
    fi
    STEM="$(basename "${INPUT_VIDEO%.*}")"

    if [[ -z "${OUTDIR}" ]]; then
        OUTDIR=$(eval echo "${OUTDIR:-${VOIDSTAR_OUTDIR:-/mnt/c/Users/brown/Videos/counterfactual_dimensions_voidstar_0}}")
    else
        OUTDIR=$(eval echo "$OUTDIR")
    fi

    require_cmd python3
    require_cmd ffprobe
    mkdir -p "$OUTDIR"
    require_file "INPUT_VIDEO" "$INPUT_VIDEO"

    INPUT_DURATION_SECONDS="$(get_video_duration_seconds "$INPUT_VIDEO")"
    if [[ -z "${YOUTUBE_FULL_SECONDS}" ]]; then YOUTUBE_FULL_SECONDS="$INPUT_DURATION_SECONDS"; fi
    if [[ -z "${LOOP_SEAM_SECONDS}" ]]; then LOOP_SEAM_SECONDS="$GLITCH_SECONDS"; fi

    echo "Input: $INPUT_VIDEO"
    echo "Stem:  $STEM"
    echo "Out:   $OUTDIR"
    echo "Dur:   ${INPUT_DURATION_SECONDS}s"
    echo "Args:  start=${START_SECONDS}s full=${YOUTUBE_FULL_SECONDS}s cps=${CPS} glitch=${GLITCH_SECONDS}s loop_seam=${LOOP_SEAM_SECONDS}s"
    echo "Perf:  reels_cache=${USE_REELS_CACHE} jobs=${JOBS}"

    PROJECT_ROOT="/home/$USER/code/voidstar"
    find_script() {
        local script_name="$1"
        local found
        found=$(find "$PROJECT_ROOT" -type f -name "$script_name" | head -n 1)
        [[ -n "$found" ]] || die "Could not find required script: $script_name in $PROJECT_ROOT"
        echo "$found"
    }

    DIVVY=$(find_script "divvy.py")
    REELS_OVERLAY=$(find_script "reels_cv_overlay.py")
    DVDLOGO=$(find_script "voidstar_dvd_logo.py")
    GLITCHFIELD=$(find_script "glitchfield.py")
    require_file "DIVVY" "$DIVVY"
    require_file "REELS_OVERLAY" "$REELS_OVERLAY"
    require_file "DVDLOGO" "$DVDLOGO"
    require_file "GLITCHFIELD" "$GLITCHFIELD"

    LOGOS=()
    if (( ${#LOGO_PATTERNS[@]} > 0 )); then
        mapfile -t LOGOS < <(expand_logo_patterns "${LOGO_PATTERNS[@]}")
        echo "Using user-specified logos (${#LOGOS[@]}):"
        printf '  %s\n' "${LOGOS[@]}"
    else
        mapfile -t LOGOS < <(find_void_logos_default)
        (( ${#LOGOS[@]} > 0 )) || die "Could not find any default logos matching void*.png in $PROJECT_ROOT (pass --logo to specify)"
        for f in "${LOGOS[@]}"; do require_file "LOGO_IMG" "$f"; done
        echo "Using default auto-detected void*.png logos (${#LOGOS[@]}):"
        printf '  %s\n' "${LOGOS[@]}"
    fi

    BASE_REELS_OVERLAY="$OUTDIR/${STEM}_reels_base_overlay.mp4"
    if [[ "$USE_REELS_CACHE" -eq 1 ]]; then
        build_base_reels_overlay
    else
        BASE_REELS_OVERLAY="$INPUT_VIDEO"
    fi

    declare -a TARGETS=()
    if [[ "$PREVIEW" -eq 1 ]]; then
        TARGETS=(run_60s_start)
    elif [[ "$END_ONLY" -eq 1 ]]; then
        TARGETS=(run_60s_end run_90s_end run_180s_end)
    else
        TARGETS=(run_60s_start run_90s_start run_180s_start run_60s_end run_90s_end run_180s_end run_full)
    fi

    if (( JOBS <= 1 )); then
        for fn in "${TARGETS[@]}"; do "$fn"; done
        return 0
    fi

    echo "--- Running targets in parallel (jobs=$JOBS) ---"
    _sem_init "$JOBS"

    pids=()
    for fn in "${TARGETS[@]}"; do
        _sem_acquire
        (
            set -euo pipefail
            "$fn"
        ) &
        pids+=( "$!" )
        _sem_release
    done

    failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then failed=1; fi
    done
    (( failed == 0 )) || die "One or more parallel jobs failed."
}

main "$@"
