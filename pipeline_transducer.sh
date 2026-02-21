#!/usr/bin/env bash
# pipeline_transducer.sh
#
# Performance optimizations:
# - Reels CV overlay is expensive: run it ONCE on the full input video to create a cached
#   "base overlay" video, then run divvy on that base overlay to make each highlight.
#   This avoids running reels overlay 6+ times.
# - Optional parallelism for independent targets (divvy + dvdlogo) via --jobs N.
#   Logo assignment remains deterministic and matches the sequential target order.

set -euo pipefail

# ============================================================
# USER TUNING (edit here, then run script with no args)
# ============================================================
# Optional CLI args still work as overrides, but this block is the primary workflow.

# QUICK START PRESETS (copy values into variables below)
# 1) FAST ITERATE (skip expensive steps)
#    ENABLE_REELS_OVERLAY_STEP=0
#    ENABLE_GLITCHFIELD_STAGE=0
#    PIPELINE_MODE_DEFAULT="preview"
#    JOBS_DEFAULT=1
#
# 2) FULL QUALITY (cached reels + all targets)
#    ENABLE_REELS_OVERLAY_STEP=1
#    USE_REELS_CACHE_DEFAULT=1
#    ENABLE_GLITCHFIELD_STAGE=0
#    PIPELINE_MODE_DEFAULT="all"
#
# 3) GLITCHFIELD EXPERIMENT (with presets from notes)
#    ENABLE_REELS_OVERLAY_STEP=0        # optional: bypass reels for speed
#    ENABLE_GLITCHFIELD_STAGE=1
#    GLITCHFIELD_PRESET="clean"        # clean | gritty | chaos | custom
#    GLITCHFIELD_START_SECONDS=690
#    GLITCHFIELD_DURATION_SECONDS=15
#    PIPELINE_MODE_DEFAULT="preview"

# Pipeline mode: all | end-only | preview | custom
PIPELINE_MODE_DEFAULT="custom"

# For custom mode, choose exactly which targets run.
RUN_60S_START=1
RUN_90S_START=0
RUN_180S_START=0
RUN_60S_END=1
RUN_90S_END=0
RUN_180S_END=0
RUN_FULL=0

# Input/output defaults.
INPUT_VIDEO_DEFAULT="~/WinVideos/transducer/transducer_voidstar_0.mp4"
OUTDIR_DEFAULT="~/WinVideos/transducer"

# Highlight sampling defaults (leave start/full empty for divvy auto defaults).
START_SECONDS_DEFAULT="501"
YOUTUBE_FULL_SECONDS_DEFAULT=""
DETECT_AUDIO_START_END_DEFAULT=1

# Timing/style defaults.
CPS_DEFAULT=0.5
GLITCH_SECONDS_DEFAULT=2
LOOP_SEAM_SECONDS_DEFAULT=""

# Reels overlay stage controls.
ENABLE_REELS_OVERLAY_STEP=0      # set 0 to bypass reels overlay completely
USE_REELS_CACHE_DEFAULT=1        # if 1, reuse cached base overlay when up-to-date

# Optional glitchfield stage (runs after reels/input base, before divvy highlights).
ENABLE_GLITCHFIELD_STAGE=1       # set 1 to enable
GLITCHFIELD_PRESET="chaos"      # clean | gritty | chaos | custom
GLITCHFIELD_START_SECONDS=""      # empty => glitchfield default
GLITCHFIELD_DURATION_SECONDS=""   # empty => glitchfield default
GLITCHFIELD_SEED=1337
GLITCHFIELD_MIN_GATE_PERIOD=
GLITCHFIELD_CUSTOM_ARGS=""      # used when preset=custom

# Optional parallelism and force rebuild.
JOBS_DEFAULT=1
FORCE_DEFAULT=0

# Optional copy of final rendered outputs to Google Drive (WSL path style).
ENABLE_GDRIVE_COPY_DEFAULT=1
GDRIVE_OUTDIR_DEFAULT="/mnt/g/My Drive/Music/voidstar/transducer"   # e.g. /mnt/c/Users/<you>/Google Drive/My Drive/Videos

# Glitchfield preset examples (manual reference):
# clean:
#   python glitchfield.py <input>.mp4 --effect combo --audio-reactive --beat-threshold 0.58 --beat-prob 1.0 --glitch-hold 12 --combo-glyph-prob 0.6 --combo-stutter-prob 0.4 --combo-alternate-bias 0.55 --color-mode input --input-palette-k 6 --palette-refresh 6 --stutter-slice-prob 0.75 --stutter-slice-shift-max 120 --fx-rgb-split --fx-rgb-shift-max 2 --fx-rgb-prob 0.45 --fx-trail-frames 1 --fx-trail-strength 0.18 --fx-scanline-jitter 0.08 --fx-scanline-step 2 --fx-scanline-shift-max 1 --fx-scanline-brightness 0.95 --fx-vhold-prob 0.03 --fx-vhold-max 4 --min-gate-period 6 --start 690 --duration 15 --seed 1337
# gritty:
#   python glitchfield.py <input>.mp4 --effect combo --audio-reactive --beat-threshold 0.56 --beat-prob 1.0 --glitch-hold 13 --combo-glyph-prob 0.45 --combo-stutter-prob 0.55 --combo-alternate-bias 0.6 --color-mode audio --palette fire --audio-color-reactive --stutter-slice-prob 0.9 --stutter-slice-shift-max 170 --stutter-frame-jitter 3 --stutter-jitter-prob 0.65 --fx-rgb-split --fx-rgb-shift-max 3 --fx-rgb-prob 0.7 --fx-trail-frames 2 --fx-trail-strength 0.3 --fx-scanline-jitter 0.22 --fx-scanline-step 2 --fx-scanline-shift-max 2 --fx-scanline-brightness 0.9 --fx-line-dropout-prob 0.06 --fx-line-dropout-thickness 2 --fx-line-dropout-bright --fx-posterize-levels 6 --fx-vhold-prob 0.08 --fx-vhold-max 10 --min-gate-period 8 --start 690 --duration 15 --seed 1337
# chaos:
#   python glitchfield.py <input>.mp4 --effect combo --audio-reactive --beat-threshold 0.52 --beat-prob 1.0 --glitch-hold 14 --combo-glyph-prob 0.3 --combo-stutter-prob 0.7 --combo-alternate-bias 0.45 --color-mode fixed --palette insane --glitch-prob 0.45 --stutter-slice-prob 0.95 --stutter-slice-shift-max 220 --stutter-slice-min-bands 4 --stutter-slice-max-bands 12 --stutter-frame-jitter 4 --stutter-jitter-prob 0.75 --fx-rgb-split --fx-rgb-shift-max 6 --fx-rgb-prob 0.9 --fx-trail-frames 3 --fx-trail-strength 0.38 --fx-scanline-jitter 0.3 --fx-scanline-step 2 --fx-scanline-shift-max 3 --fx-scanline-brightness 0.85 --fx-line-dropout-prob 0.12 --fx-line-dropout-thickness 3 --fx-line-dropout-bright --fx-block-jitter-prob 0.05 --fx-block-size 12 --fx-block-shift-max 4 --fx-posterize-levels 5 --fx-bitcrush-bits 6 --fx-vhold-prob 0.12 --fx-vhold-max 14 --min-gate-period 10 --start 690 --duration 15 --seed 1337

die() { echo "Error: $*" >&2; exit 1; }
require_cmd() { command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"; }
require_file() { local label="$1" path="$2"; [[ -f "$path" ]] || die "Missing $label file: $path"; }

copy_to_gdrive_if_enabled() {
    local src="$1"
    [[ "${ENABLE_GDRIVE_COPY:-0}" -eq 1 ]] || return 0
    [[ -n "${GDRIVE_OUTDIR:-}" ]] || die "ENABLE_GDRIVE_COPY is on but GDRIVE_OUTDIR is empty"
    [[ -f "$src" ]] || { echo "[gdrive] warning: source file not found: $src"; return 0; }

    mkdir -p "$GDRIVE_OUTDIR"
    local dst="$GDRIVE_OUTDIR/$(basename "$src")"
    cp -f "$src" "$dst"
    echo "[gdrive] copied: $dst"
}

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
    find "$PROJECT_ROOT/art/logos_alpha" -type f -iname "void*.png" | sort
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

build_highlights_time_args() {
    HIGHLIGHTS_TIME_ARGS=()
    if [[ "${DETECT_AUDIO_START_END:-1}" -eq 1 ]]; then
        HIGHLIGHTS_TIME_ARGS+=(--detect-audio-start-end)
    fi
    if [[ -n "${START_SECONDS:-}" ]]; then
        HIGHLIGHTS_TIME_ARGS+=(--start-seconds "$START_SECONDS")
    fi
    if [[ -n "${YOUTUBE_FULL_SECONDS:-}" ]]; then
        HIGHLIGHTS_TIME_ARGS+=(--youtube-full-seconds "$YOUTUBE_FULL_SECONDS")
    fi
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
    if [[ "$USE_REELS_CACHE" -eq 1 ]]; then
        should_rebuild "$target" || return 0
    else
        echo "[reels] cache disabled: rebuilding base overlay"
    fi

    python3 "$REELS_OVERLAY" "$INPUT_VIDEO" \
        --min-det-conf 0.05 --min-trk-conf 0.05 --draw-ids true \
        --smear true --smear-frames 17 --smear-decay 0.99 \
        --trail true --trail-alpha .999 --beat-sync true \
        --velocity-color true --velocity-color-mult 10 \
        --output "$target"

    # rename_output "$OUTDIR/${STEM}_fps30_mc2_det0p05_trk0p05_trail1_ta1p00_tlotrue_scan1_velcolor_ids_beatsync_smear.mp4" \
    #     "$target"
}

run_optional_glitchfield_stage() {
    [[ "$ENABLE_GLITCHFIELD_STAGE" -eq 1 ]] || return 0

    echo "--- Optional glitchfield stage (${GLITCHFIELD_PRESET}) ---"
    local glitchfield_script="${PROJECT_ROOT}/glitchfield/glitchfield.py"
    require_file "GLITCHFIELD" "$glitchfield_script"

    local preset_tag="${GLITCHFIELD_PRESET}"
    local target="$OUTDIR/${STEM}_base_glitchfield_${preset_tag}.mp4"
    should_rebuild "$target" || {
        BASE_REELS_OVERLAY="$target"
        echo "[glitchfield] using cached: $BASE_REELS_OVERLAY"
        return 0
    }

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

    local -a gf_time_args
    gf_time_args=()
    if [[ -n "${GLITCHFIELD_START_SECONDS:-}" ]]; then
        gf_time_args+=(--start "$GLITCHFIELD_START_SECONDS")
    fi
    if [[ -n "${GLITCHFIELD_DURATION_SECONDS:-}" ]]; then
        gf_time_args+=(--duration "$GLITCHFIELD_DURATION_SECONDS")
    fi

    python3 "$glitchfield_script" "$BASE_REELS_OVERLAY" \
        "${gf_args[@]}" \
        "${gf_time_args[@]}" \
        --seed "$GLITCHFIELD_SEED"

    local generated_src in_dir in_stem
    in_dir="$(dirname "$BASE_REELS_OVERLAY")"
    in_stem="$(basename "${BASE_REELS_OVERLAY%.*}")"
    generated_src="$(ls -t "$in_dir/${in_stem}"_*.mp4 2>/dev/null | head -n 1 || true)"
    if [[ -n "$generated_src" && -f "$generated_src" ]]; then
        rename_output "$generated_src" "$target"
    else
        die "Could not locate glitchfield output for stem: $in_stem"
    fi

    BASE_REELS_OVERLAY="$target"
    echo "[glitchfield] staged base: $BASE_REELS_OVERLAY"
}

run_divvy_uniform_highlights() {
    local output_path="$1"
    local target_seconds="$2"
    local sample_seconds="$3"
    local n_segments="$4"
    local sample_anchor="$5"

    local glitch_try="$GLITCH_SECONDS"
    local attempts=0
    local max_attempts=5

    while true; do
        local -a cmd
        cmd=(
            python3 "$DIVVY" highlights "$BASE_REELS_OVERLAY"
            "${HIGHLIGHTS_TIME_ARGS[@]}" --target-length-seconds "$target_seconds"
            --video-encoder libx264 --preset medium --out-dir "$OUTDIR"
            --output "$output_path"
            --glitch-seconds "$glitch_try" --glitch-style vuwind --cps "$CPS" --loop-seam-seconds "$LOOP_SEAM_SECONDS"
            --sampling-mode uniform-spread --sample-seconds "$sample_seconds" --truncate-to-full-clips
        )

        if [[ -n "$n_segments" ]]; then
            cmd+=(--n-segments "$n_segments")
        fi
        if [[ -n "$sample_anchor" ]]; then
            cmd+=(--sample-anchor "$sample_anchor")
        fi

        local err_file
        err_file="$(mktemp /tmp/voidstar_divvy_highlights_err.XXXXXX)"

        if "${cmd[@]}" 2> >(tee "$err_file" >&2); then
            rm -f "$err_file"
            return 0
        fi

        local rc=$?
        if grep -q -- "--glitch-seconds too large for shortest selected clip" "$err_file"; then
            rm -f "$err_file"
            if (( attempts >= max_attempts )); then
                echo "Error: highlights failed after glitch fallback attempts (last glitch-seconds=${glitch_try})" >&2
                return $rc
            fi

            local next_try
            next_try="$(python3 - <<PY
g=float("${glitch_try}")
g=max(0.0, g*0.75)
if g < 0.05:
    g = 0.0
print(f"{g:.6f}")
PY
)"

            if [[ "$next_try" == "$glitch_try" ]]; then
                next_try="0.0"
            fi
            echo "[pipeline] divvy glitch-seconds too large for selected clip; retrying with --glitch-seconds ${next_try}" >&2
            glitch_try="$next_try"
            attempts=$((attempts + 1))
            continue
        fi

        rm -f "$err_file"
        return $rc
    done
}

# ----------------------------
# Targets
# ----------------------------
run_60s_start() {
    local ordinal=0
    echo "--- 60s highlight (START) ---"
    local divvy_dst="$OUTDIR/${STEM}_highlights_60s_overlay.mp4"

    run_divvy_uniform_highlights "$divvy_dst" 60 4 15 ""

    local picked logo tag target
    picked="$(logo_for_ordinal "$ordinal")"; logo="${picked%%|*}"; tag="${picked##*|}"
    target="$(with_logo_suffix "$OUTDIR/${STEM}_highlights_60s_overlay_logo.mp4" "$tag")"
    if ! should_rebuild "$target"; then
        copy_to_gdrive_if_enabled "$target"
        return 0
    fi

    python3 "$DVDLOGO" "$divvy_dst" "$logo" \
        --speed 0 --logo-scale .4 --logo-rotate-speed 0 --trails 0.85 --opacity .5 \
        --audio-reactive-glow 1.0 --audio-reactive-scale 0.5 --audio-reactive-gain 2.0 \
        --edge-margin-px 0 --reels-local-overlay false --voidstar-preset wild \
        --local-point-track true --local-point-track-scale 1.2 --local-point-track-pad-px 0 \
        --local-point-track-max-points 256 --local-point-track-radius 256 --local-point-track-min-distance 128 \
        --local-point-track-refresh 8 --local-point-track-opacity 0.77 --local-point-track-decay 0.77 \
        --local-point-track-link-neighbors 8 --local-point-track-link-thickness 1 \
        --local-point-track-link-opacity .77 --voidstar-colorize true --start-x .5 --start-y .83 \
        --content-bbox-for-local false --voidstar-debug-bounds true --voidstar-debug-bounds-mode hit-glitch \
        --voidstar-debug-bounds-hit-threshold 0.8 --voidstar-debug-bounds-hit-prob 0.1 \
        --output "$target"
    copy_to_gdrive_if_enabled "$target"
}

run_90s_start() {
    local ordinal=1
    echo "--- 90s highlight (START) ---"
    local divvy_dst="$OUTDIR/${STEM}_highlights_90s_overlay.mp4"

    run_divvy_uniform_highlights "$divvy_dst" 90 16 "" ""

    local picked logo tag target
    picked="$(logo_for_ordinal "$ordinal")"; logo="${picked%%|*}"; tag="${picked##*|}"
    target="$(with_logo_suffix "$OUTDIR/${STEM}_highlights_90s_overlay_logo.mp4" "$tag")"
    if ! should_rebuild "$target"; then
        copy_to_gdrive_if_enabled "$target"
        return 0
    fi

    python3 "$DVDLOGO" "$divvy_dst" "$logo" \
        --speed 0 --logo-scale .4 --logo-rotate-speed 0 --trails 0.85 --opacity .5 \
        --audio-reactive-glow 1.0 --audio-reactive-scale 0.5 --audio-reactive-gain 2.0 \
        --edge-margin-px 0 --reels-local-overlay false --voidstar-preset wild \
        --local-point-track true --local-point-track-scale 1.2 --local-point-track-pad-px 0 \
        --local-point-track-max-points 256 --local-point-track-radius 256 --local-point-track-min-distance 128 \
        --local-point-track-refresh 8 --local-point-track-opacity 0.77 --local-point-track-decay 0.77 \
        --local-point-track-link-neighbors 8 --local-point-track-link-thickness 1 \
        --local-point-track-link-opacity .77 --voidstar-colorize true --start-x .5 --start-y .83 \
        --content-bbox-for-local false --voidstar-debug-bounds true --voidstar-debug-bounds-mode hit-glitch \
        --voidstar-debug-bounds-hit-threshold 0.8 --voidstar-debug-bounds-hit-prob 0.1 \
        --output "$target"
    copy_to_gdrive_if_enabled "$target"
}

run_180s_start() {
    local ordinal=2
    echo "--- 180s highlight (START) ---"
    local divvy_dst="$OUTDIR/${STEM}_highlights_180s_overlay.mp4"

    run_divvy_uniform_highlights "$divvy_dst" 180 32 6 ""

    local picked logo tag target
    picked="$(logo_for_ordinal "$ordinal")"; logo="${picked%%|*}"; tag="${picked##*|}"
    target="$(with_logo_suffix "$OUTDIR/${STEM}_highlights_180s_overlay_logo.mp4" "$tag")"
    if ! should_rebuild "$target"; then
        copy_to_gdrive_if_enabled "$target"
        return 0
    fi

    python3 "$DVDLOGO" "$divvy_dst" "$logo" \
        --speed 0 --logo-scale .4 --logo-rotate-speed 0 --trails 0.85 --opacity .5 \
        --audio-reactive-glow 1.0 --audio-reactive-scale 0.5 --audio-reactive-gain 2.0 \
        --edge-margin-px 0 --reels-local-overlay false --voidstar-preset wild \
        --local-point-track true --local-point-track-scale 1.2 --local-point-track-pad-px 0 \
        --local-point-track-max-points 256 --local-point-track-radius 256 --local-point-track-min-distance 128 \
        --local-point-track-refresh 8 --local-point-track-opacity 0.77 --local-point-track-decay 0.77 \
        --local-point-track-link-neighbors 8 --local-point-track-link-thickness 1 \
        --local-point-track-link-opacity .77 --voidstar-colorize true --start-x .5 --start-y .83 \
        --content-bbox-for-local false --voidstar-debug-bounds true --voidstar-debug-bounds-mode hit-glitch \
        --voidstar-debug-bounds-hit-threshold 0.8 --voidstar-debug-bounds-hit-prob 0.1 \
        --output "$target"
    copy_to_gdrive_if_enabled "$target"
}

run_full() {
    local ordinal=3
    echo "--- FULL (YouTube) ---"
    local base_overlay="$BASE_REELS_OVERLAY"
    require_file "BASE_REELS_OVERLAY" "$base_overlay"

    local picked logo tag target
    picked="$(logo_for_ordinal "$ordinal")"; logo="${picked%%|*}"; tag="${picked##*|}"
    target="$(with_logo_suffix "$OUTDIR/${STEM}_full_overlay_logo.mp4" "$tag")"
    if ! should_rebuild "$target"; then
        copy_to_gdrive_if_enabled "$target"
        return 0
    fi

    python3 "$DVDLOGO" "$base_overlay" "$logo" \
        --speed 0 --logo-scale .4 --logo-rotate-speed 0 --trails 0.85 --opacity .5 \
        --audio-reactive-glow 1.0 --audio-reactive-scale 0.5 --audio-reactive-gain 2.0 \
        --edge-margin-px 0 --reels-local-overlay false --voidstar-preset wild \
        --local-point-track true --local-point-track-scale 1.2 --local-point-track-pad-px 0 \
        --local-point-track-max-points 256 --local-point-track-radius 256 --local-point-track-min-distance 128 \
        --local-point-track-refresh 8 --local-point-track-opacity 0.77 --local-point-track-decay 0.77 \
        --local-point-track-link-neighbors 8 --local-point-track-link-thickness 1 \
        --local-point-track-link-opacity .77 --voidstar-colorize true --start-x .5 --start-y .83 \
        --content-bbox-for-local false --voidstar-debug-bounds true --voidstar-debug-bounds-mode hit-glitch \
        --voidstar-debug-bounds-hit-threshold 0.8 --voidstar-debug-bounds-hit-prob 0.1 \
        --output "$target"
    copy_to_gdrive_if_enabled "$target"
}

run_60s_end() {
    local ordinal=4
    echo "--- 60s highlight (END) ---"
    local divvy_dst="$OUTDIR/${STEM}_highlights_60t_overlay.mp4"

    run_divvy_uniform_highlights "$divvy_dst" 60 4 15 "end"

    local picked logo tag target
    picked="$(logo_for_ordinal "$ordinal")"; logo="${picked%%|*}"; tag="${picked##*|}"
    target="$(with_logo_suffix "$OUTDIR/${STEM}_highlights_60t_overlay_logo.mp4" "$tag")"
    if ! should_rebuild "$target"; then
        copy_to_gdrive_if_enabled "$target"
        return 0
    fi

    python3 "$DVDLOGO" "$divvy_dst" "$logo" \
        --speed 0 --logo-scale .4 --logo-rotate-speed 0 --trails 0.85 --opacity .5 \
        --audio-reactive-glow 1.0 --audio-reactive-scale 0.5 --audio-reactive-gain 2.0 \
        --edge-margin-px 0 --reels-local-overlay false --voidstar-preset wild \
        --local-point-track true --local-point-track-scale 1.2 --local-point-track-pad-px 0 \
        --local-point-track-max-points 256 --local-point-track-radius 256 --local-point-track-min-distance 128 \
        --local-point-track-refresh 8 --local-point-track-opacity 0.77 --local-point-track-decay 0.77 \
        --local-point-track-link-neighbors 8 --local-point-track-link-thickness 1 \
        --local-point-track-link-opacity .77 --voidstar-colorize true --start-x .5 --start-y .83 \
        --content-bbox-for-local false --voidstar-debug-bounds true --voidstar-debug-bounds-mode hit-glitch \
        --voidstar-debug-bounds-hit-threshold 0.8 --voidstar-debug-bounds-hit-prob 0.1 \
        --output "$target"
    copy_to_gdrive_if_enabled "$target"
}

run_90s_end() {
    local ordinal=5
    echo "--- 90s highlight (END) ---"
    local divvy_dst="$OUTDIR/${STEM}_highlights_90t_overlay.mp4"

    run_divvy_uniform_highlights "$divvy_dst" 90 16 "" "end"

    local picked logo tag target
    picked="$(logo_for_ordinal "$ordinal")"; logo="${picked%%|*}"; tag="${picked##*|}"
    target="$(with_logo_suffix "$OUTDIR/${STEM}_highlights_90t_overlay_logo.mp4" "$tag")"
    if ! should_rebuild "$target"; then
        copy_to_gdrive_if_enabled "$target"
        return 0
    fi

    python3 "$DVDLOGO" "$divvy_dst" "$logo" \
        --speed 0 --logo-scale .4 --logo-rotate-speed 0 --trails 0.85 --opacity .5 \
        --audio-reactive-glow 1.0 --audio-reactive-scale 0.5 --audio-reactive-gain 2.0 \
        --edge-margin-px 0 --reels-local-overlay false --voidstar-preset wild \
        --local-point-track true --local-point-track-scale 1.2 --local-point-track-pad-px 0 \
        --local-point-track-max-points 256 --local-point-track-radius 256 --local-point-track-min-distance 128 \
        --local-point-track-refresh 8 --local-point-track-opacity 0.77 --local-point-track-decay 0.77 \
        --local-point-track-link-neighbors 8 --local-point-track-link-thickness 1 \
        --local-point-track-link-opacity .77 --voidstar-colorize true --start-x .5 --start-y .83 \
        --content-bbox-for-local false --voidstar-debug-bounds true --voidstar-debug-bounds-mode hit-glitch \
        --voidstar-debug-bounds-hit-threshold 0.8 --voidstar-debug-bounds-hit-prob 0.1 \
        --output "$target"
    copy_to_gdrive_if_enabled "$target"
}

run_180s_end() {
    local ordinal=6
    echo "--- 180s highlight (END) ---"
    local divvy_dst="$OUTDIR/${STEM}_highlights_180t_overlay.mp4"

    run_divvy_uniform_highlights "$divvy_dst" 180 32 6 "end"

    local picked logo tag target
    picked="$(logo_for_ordinal "$ordinal")"; logo="${picked%%|*}"; tag="${picked##*|}"
    target="$(with_logo_suffix "$OUTDIR/${STEM}_highlights_180t_overlay_logo.mp4" "$tag")"
    if ! should_rebuild "$target"; then
        copy_to_gdrive_if_enabled "$target"
        return 0
    fi

    python3 "$DVDLOGO" "$divvy_dst" "$logo" \
        --speed 0 --logo-scale .4 --logo-rotate-speed 0 --trails 0.85 --opacity .5 \
        --audio-reactive-glow 1.0 --audio-reactive-scale 0.5 --audio-reactive-gain 2.0 \
        --edge-margin-px 0 --reels-local-overlay false --voidstar-preset wild \
        --local-point-track true --local-point-track-scale 1.2 --local-point-track-pad-px 0 \
        --local-point-track-max-points 256 --local-point-track-radius 256 --local-point-track-min-distance 128 \
        --local-point-track-refresh 8 --local-point-track-opacity 0.77 --local-point-track-decay 0.77 \
        --local-point-track-link-neighbors 8 --local-point-track-link-thickness 1 \
        --local-point-track-link-opacity .77 --voidstar-colorize true --start-x .5 --start-y .83 \
        --content-bbox-for-local false --voidstar-debug-bounds true --voidstar-debug-bounds-mode hit-glitch \
        --voidstar-debug-bounds-hit-threshold 0.8 --voidstar-debug-bounds-hit-prob 0.1 \
        --output "$target"
    copy_to_gdrive_if_enabled "$target"
}

main() {
    FORCE="$FORCE_DEFAULT"; END_ONLY=0; PREVIEW=0
    INPUT_VIDEO=""; OUTDIR=""
    START_SECONDS="$START_SECONDS_DEFAULT"; YOUTUBE_FULL_SECONDS="$YOUTUBE_FULL_SECONDS_DEFAULT"; DETECT_AUDIO_START_END="$DETECT_AUDIO_START_END_DEFAULT"; CPS="$CPS_DEFAULT"; GLITCH_SECONDS="$GLITCH_SECONDS_DEFAULT"; LOOP_SEAM_SECONDS="$LOOP_SEAM_SECONDS_DEFAULT"
    LOGO_PATTERNS=("voidstar_emblem_text_0.png")
    USE_REELS_CACHE="$USE_REELS_CACHE_DEFAULT"
    JOBS="$JOBS_DEFAULT"
    PIPELINE_MODE="$PIPELINE_MODE_DEFAULT"
    ENABLE_GDRIVE_COPY="$ENABLE_GDRIVE_COPY_DEFAULT"
    GDRIVE_OUTDIR="$GDRIVE_OUTDIR_DEFAULT"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --force|-f) FORCE=1; shift ;;
            --end-only) END_ONLY=1; shift ;;
            preview) PREVIEW=1; shift ;;
            --mode) PIPELINE_MODE="$2"; shift 2 ;;
            --input) INPUT_VIDEO="$2"; shift 2 ;;
            --outdir) OUTDIR="$2"; shift 2 ;;
            --start-seconds) START_SECONDS="$2"; shift 2 ;;
            --youtube-full-seconds) YOUTUBE_FULL_SECONDS="$2"; shift 2 ;;
            --detect-audio-start-end) DETECT_AUDIO_START_END=1; shift ;;
            --no-detect-audio-start-end) DETECT_AUDIO_START_END=0; shift ;;
            --skip-reels-overlay) ENABLE_REELS_OVERLAY_STEP=0; shift ;;
            --cps) CPS="$2"; shift 2 ;;
            --glitch-seconds) GLITCH_SECONDS="$2"; shift 2 ;;
            --loop-seam-seconds) LOOP_SEAM_SECONDS="$2"; shift 2 ;;
            --logo) LOGO_PATTERNS+=( "$2" ); shift 2 ;;
            --jobs|-j) JOBS="$2"; shift 2 ;;
            --no-reels-cache) USE_REELS_CACHE=0; shift ;;
            --copy-to-gdrive) ENABLE_GDRIVE_COPY=1; shift ;;
            --gdrive-outdir) GDRIVE_OUTDIR="$2"; shift 2 ;;
            -h|--help) sed -n '1,130p' "$0"; exit 0 ;;
            *)
                if [[ "$1" != -* ]]; then INPUT_VIDEO="$1"; shift
                else die "Unknown arg: $1 (try --help)"; fi
                ;;
        esac
    done

    if [[ -z "${INPUT_VIDEO}" ]]; then
        INPUT_VIDEO=$(eval echo "${INPUT_VIDEO:-${VOIDSTAR_INPUT_VIDEO:-$INPUT_VIDEO_DEFAULT}}")
    else
        INPUT_VIDEO=$(eval echo "$INPUT_VIDEO")
    fi
    STEM="$(basename "${INPUT_VIDEO%.*}")"

    if [[ -z "${OUTDIR}" ]]; then
        if [[ -n "$OUTDIR_DEFAULT" ]]; then
            OUTDIR=$(eval echo "${OUTDIR:-${VOIDSTAR_OUTDIR:-$OUTDIR_DEFAULT}}")
        else
            OUTDIR=$(eval echo "${OUTDIR:-${VOIDSTAR_OUTDIR:-~/WinVideos/${STEM}/}}")
        fi
    else
        OUTDIR=$(eval echo "$OUTDIR")
    fi

    if [[ "$ENABLE_GDRIVE_COPY" -eq 1 && -n "${GDRIVE_OUTDIR:-}" ]]; then
        GDRIVE_OUTDIR=$(eval echo "$GDRIVE_OUTDIR")
    fi

    require_cmd python3
    require_cmd ffprobe
    mkdir -p "$OUTDIR"
    require_file "INPUT_VIDEO" "$INPUT_VIDEO"

    INPUT_DURATION_SECONDS="$(get_video_duration_seconds "$INPUT_VIDEO")"
    if [[ -z "${LOOP_SEAM_SECONDS}" ]]; then LOOP_SEAM_SECONDS="$GLITCH_SECONDS"; fi
    build_highlights_time_args

    local start_dbg full_dbg detect_dbg
    start_dbg="${START_SECONDS:-auto}"
    full_dbg="${YOUTUBE_FULL_SECONDS:-auto}"
    detect_dbg="off"; if [[ "$DETECT_AUDIO_START_END" -eq 1 ]]; then detect_dbg="on"; fi

    echo "Input: $INPUT_VIDEO"
    echo "Stem:  $STEM"
    echo "Out:   $OUTDIR"
    echo "Dur:   ${INPUT_DURATION_SECONDS}s"
    echo "Args:  start=${start_dbg}s full=${full_dbg}s detect_audio=${detect_dbg} cps=${CPS} glitch=${GLITCH_SECONDS}s loop_seam=${LOOP_SEAM_SECONDS}s"
    echo "Perf:  reels_overlay=${ENABLE_REELS_OVERLAY_STEP} reels_cache=${USE_REELS_CACHE} glitchfield=${ENABLE_GLITCHFIELD_STAGE} jobs=${JOBS}"
    echo "Sync:  gdrive_copy=${ENABLE_GDRIVE_COPY} gdrive_outdir=${GDRIVE_OUTDIR:-unset}"

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
    require_file "DIVVY" "$DIVVY"
    require_file "REELS_OVERLAY" "$REELS_OVERLAY"
    require_file "DVDLOGO" "$DVDLOGO"

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
    if [[ "$ENABLE_REELS_OVERLAY_STEP" -eq 1 ]]; then
        build_base_reels_overlay
    else
        echo "[pipeline] reels overlay step bypassed (using input video directly)"
        BASE_REELS_OVERLAY="$INPUT_VIDEO"
    fi

    run_optional_glitchfield_stage

    declare -a TARGETS=()
    if [[ "$PREVIEW" -eq 1 || "$PIPELINE_MODE" == "preview" ]]; then
        TARGETS=(run_60s_start)
    elif [[ "$END_ONLY" -eq 1 || "$PIPELINE_MODE" == "end-only" ]]; then
        TARGETS=(run_60s_end run_90s_end run_180s_end)
    elif [[ "$PIPELINE_MODE" == "custom" ]]; then
        (( RUN_60S_START == 1 )) && TARGETS+=(run_60s_start)
        (( RUN_90S_START == 1 )) && TARGETS+=(run_90s_start)
        (( RUN_180S_START == 1 )) && TARGETS+=(run_180s_start)
        (( RUN_60S_END == 1 )) && TARGETS+=(run_60s_end)
        (( RUN_90S_END == 1 )) && TARGETS+=(run_90s_end)
        (( RUN_180S_END == 1 )) && TARGETS+=(run_180s_end)
        (( RUN_FULL == 1 )) && TARGETS+=(run_full)
        (( ${#TARGETS[@]} > 0 )) || die "PIPELINE_MODE=custom but no RUN_* targets enabled"
    else
        TARGETS=(run_60s_start run_90s_start run_180s_start run_60s_end run_90s_end run_180s_end run_full)
    fi

    if (( JOBS <= 1 )); then
        for fn in "${TARGETS[@]}"; do "$fn"; done
        exit 0
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
