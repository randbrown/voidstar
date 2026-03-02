#!/usr/bin/env bash
# pipeline_atomism_voidstar_2.sh
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
#    USE_GLITCHFIELD_CACHE_DEFAULT=1
#    GLITCHFIELD_PRESET="clean"        # clean | gritty | chaos | custom
#    PIPELINE_MODE_DEFAULT="preview"

# Pipeline mode: all | end-only | preview | custom | preview,custom
PIPELINE_MODE_DEFAULT="preview,custom"
PIPELINE_LOG_TAG_DEFAULT="atomism_2"

# For custom mode, choose exactly which targets run.
RUN_60S_START=1
RUN_180S_START=1
RUN_60S_END=0
RUN_180S_END=0
RUN_FULL=1

##NOTE: sensorium uses tiny electrical sparks (including flood in/out)

# Input/output defaults.
INPUT_VIDEO_DEFAULT="~/WinVideos/atomism/atomism_voidstar_2.mp4"
OUTDIR_DEFAULT="~/WinVideos/atomism"

# Highlight sampling defaults (leave start/full empty for divvy auto defaults).
START_SECONDS_DEFAULT="7.5"
YOUTUBE_FULL_SECONDS_DEFAULT=""
DETECT_AUDIO_START_END_DEFAULT=0

# Timing/style defaults.
CPS_DEFAULT=0.5
DIVVY_SAMPLING_MODE_DEFAULT="groove"   # uniform-spread | groove | minute-averages | n-averages | recursive-halves
DIVVY_GROOVE_BPM_DEFAULT="30"
DIVVY_60S_START_SAMPLE_SECONDS_DEFAULT=15
DIVVY_60S_START_SEGMENTS_DEFAULT=4
DIVVY_180S_START_SAMPLE_SECONDS_DEFAULT=15
DIVVY_180S_START_SEGMENTS_DEFAULT=12
DIVVY_60S_END_SAMPLE_SECONDS_DEFAULT=4
DIVVY_60S_END_SEGMENTS_DEFAULT=6
DIVVY_180S_END_SAMPLE_SECONDS_DEFAULT=15
DIVVY_180S_END_SEGMENTS_DEFAULT=12
GLITCH_SECONDS_DEFAULT=1.7
LOOP_SEAM_SECONDS_DEFAULT="2"
PREVIEW_REELS_OVERLAY_DEFAULT=0

##NOTE: sensorium reels overlay is white only with ids/boxes disabled
# Reels overlay stage controls.
ENABLE_REELS_OVERLAY_STEP=1      # set 0 to bypass reels overlay completely
USE_REELS_CACHE_DEFAULT=1        # if 1, reuse cached base overlay when up-to-date
REELS_CACHE_MODE_DEFAULT="base"  # base | per-target
BASE_REELS_OVERLAY_PREBUILT_DEFAULT="~/WinVideos/atomism/atomism_voidstar_2_reels_base_overlay.mp4"
REELS_BOOTSTRAP_EXISTING_CACHE_DEFAULT=1   # if prebuilt base overlay exists but cache key is missing, trust/seed cache key

# Optional glitchfield stage that runs BEFORE reels overlay.
ENABLE_PRE_REELS_GLITCHFIELD_STAGE=1
USE_PRE_REELS_GLITCHFIELD_CACHE_DEFAULT=1
PRE_REELS_GLITCHFIELD_PRESET="minimalist"   # minimalist | clean | gritty | chaos | custom
PRE_REELS_GLITCHFIELD_SEED=1337
PRE_REELS_GLITCHFIELD_MIN_GATE_PERIOD=11
PRE_REELS_GLITCHFIELD_CUSTOM_ARGS=""

# Optional glitchfield stage (runs after divvy highlights, before dvdlogo).
ENABLE_GLITCHFIELD_STAGE=0       # set 1 to enable
USE_GLITCHFIELD_CACHE_DEFAULT=1  # if 1, reuse cached glitchfield base when up-to-date
GLITCHFIELD_PRESET="minimalist"      # minimalist | clean | gritty | chaos | custom
GLITCHFIELD_SEED=1337
GLITCHFIELD_MIN_GATE_PERIOD=11
GLITCHFIELD_CUSTOM_ARGS=""      # used when preset=custom

# Optional particle sparks stage (runs after divvy highlights, before dvdlogo).
ENABLE_PARTICLE_SPARKS_STAGE=1
USE_PARTICLE_SPARKS_CACHE_DEFAULT=1
PARTICLE_SPARKS_MAX_POINTS_DEFAULT=16
PARTICLE_SPARKS_POINT_MIN_DISTANCE_DEFAULT=444
PARTICLE_SPARKS_MAX_LIVE_SPARKS_DEFAULT=16
PARTICLE_SPARKS_MOTION_THRESHOLD_DEFAULT=0.7
PARTICLE_SPARKS_RATE_DEFAULT=0.5
PARTICLE_SPARKS_SIZE_DEFAULT=14
PARTICLE_SPARKS_LIFE_FRAMES_DEFAULT=45
PARTICLE_SPARKS_SPEED_DEFAULT=1.9
PARTICLE_SPARKS_JITTER_DEFAULT=1.3
PARTICLE_SPARKS_OPACITY_DEFAULT=0.7
PARTICLE_SPARKS_AUDIO_GAIN_DEFAULT=1.3
PARTICLE_SPARKS_AUDIO_SMOOTH_DEFAULT=0.7
PARTICLE_SPARKS_COLOR_MODE_DEFAULT="emmons"   # white | rgb | random | audio-intensity | antiparticles | abstract-forms
PARTICLE_SPARKS_EMMONS_SPIN_SPEED_DEFAULT=0.02
PARTICLE_SPARKS_REPEL_STRENGTH_DEFAULT=1
PARTICLE_SPARKS_REPEL_RADIUS_DEFAULT=72
PARTICLE_SPARKS_COLOR_RGB_DEFAULT="255,255,255"
PARTICLE_SPARKS_FLOOD_IN_OUT_DEFAULT=1
PARTICLE_SPARKS_FLOOD_SECONDS_DEFAULT=2.0
PARTICLE_SPARKS_FLOOD_SPAWN_MULT_DEFAULT=4
PARTICLE_SPARKS_FLOOD_EXTRA_SOURCES_DEFAULT=16
PARTICLE_SPARKS_FLOOD_VELOCITY_MULT_DEFAULT=1.15

# Optional title hook stage (runs last, after logo/particle stages).
ENABLE_TITLE_HOOK_STAGE=1
USE_TITLE_HOOK_CACHE_DEFAULT=1
TITLE_HOOK_DURATION_DEFAULT=4.0
TITLE_HOOK_DURATION_600_DEFAULT=8.0
TITLE_HOOK_FADE_OUT_DURATION_DEFAULT=1.3
TITLE_HOOK_TITLE_DEFAULT='// atomism_voidstar_2 (hi60t)\n// voidstar'
TITLE_HOOK_SECONDARY_TEXT_DEFAULT='#livecoding\n#pedalsteel\n#improvisedmusic\n#opencvpython\n#vibecoding'
TITLE_HOOK_LOGO_DEFAULT='~/code/voidstar/art/logos_alpha/voidstar_logo_0.png'
TITLE_HOOK_LOGO_ALPHA_THRESHOLD_DEFAULT=0.99
TITLE_HOOK_LOGO_INTENSITY_DEFAULT=1.0
TITLE_HOOK_LOGO_IDLE_WIGGLE_DEFAULT=0.0003
TITLE_HOOK_LOGO_X_RATIO_DEFAULT=0.5
TITLE_HOOK_LOGO_Y_RATIO_DEFAULT=0.2
TITLE_HOOK_LOGO_MOTION_TRACK_SCALE_DEFAULT=0.67
TITLE_HOOK_LOGO_MOTION_TRACK_RADIUS_DEFAULT=512
TITLE_HOOK_LOGO_MOTION_TRACK_LINK_NEIGHBORS_DEFAULT=4
TITLE_HOOK_LOGO_MOTION_TRACK_MIN_DISTANCE_DEFAULT=64
TITLE_HOOK_LOGO_MOTION_TRACK_PAD_PX_DEFAULT=128
TITLE_HOOK_LOGO_MOTION_TRACK_LINK_OPACITY_DEFAULT=0.7
TITLE_HOOK_LOGO_MOTION_TRACK_REFRESH_DEFAULT=3
TITLE_HOOK_LOGO_MOTION_TRACK_DECAY_DEFAULT=0.5
TITLE_HOOK_LOGO_OPACITY_DEFAULT=0.7
TITLE_HOOK_BACKGROUND_DIM_DEFAULT=0.33
TITLE_HOOK_TITLE_LAYER_DIM_DEFAULT=0.0
TITLE_HOOK_TEXT_ALIGN_DEFAULT="left"
TITLE_HOOK_TITLE_JITTER_AUDIO_MULTIPLIER_DEFAULT=0.000001
TITLE_HOOK_SPARKS_DEFAULT=1
TITLE_HOOK_SPARKS_RATE_DEFAULT=1
TITLE_HOOK_SPARKS_MOTION_THRESHOLD_DEFAULT=0.5
TITLE_HOOK_SPARKS_OPACITY_DEFAULT=0.7

# Optional parallelism and force rebuild.
JOBS_DEFAULT=1
FORCE_DEFAULT=0

# Optional copy of final rendered outputs to Google Drive (WSL path style).
ENABLE_GDRIVE_COPY_DEFAULT=1
GDRIVE_OUTDIR_DEFAULT="~/GoogleDrive/Music/voidstar/atomism"   # e.g. /mnt/c/Users/<you>/Google Drive/My Drive/Videos mapped via symlink

# Logo assignment by target direction.
LOGO_START_DEFAULT="~/code/voidstar/art/logos_alpha/voidstar_logo_cosmos_atom_0.png"
LOGO_END_DEFAULT="~/code/voidstar/art/logos_alpha/voidstar_emblem_text_0.png"

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

GDRIVE_READY=0

ensure_gdrive_mount() {
  local mnt=/mnt/g

    # Validation only (no sudo): startup init handles privileged mount work.
    if ! mountpoint -q "$mnt"; then
        echo "[voidstar] ERROR: /mnt/g is not mounted"
        return 1
    fi

    ls "$mnt" >/dev/null 2>&1 || {
        echo "[voidstar] ERROR: /mnt/g not accessible (Google Drive not mounted/ready)"
        return 1
    }
}

initialize_gdrive_mount_once() {
    [[ "${ENABLE_GDRIVE_COPY:-0}" -eq 1 ]] || return 0
    [[ -n "${GDRIVE_OUTDIR:-}" ]] || die "ENABLE_GDRIVE_COPY is on but GDRIVE_OUTDIR is empty"

    local mnt=/mnt/g
    # Fast path: already mounted and readable; no sudo required.
    if ensure_gdrive_mount >/dev/null 2>&1; then
        echo "[gdrive] startup init: mount already ready (no sudo needed)"
        GDRIVE_READY=1
        return 0
    fi

    echo "[gdrive] startup init: mount not ready; attempting repair/mount"

    # Create mountpoint only if missing (may require sudo).
    if [[ ! -d "$mnt" ]]; then
        if ! sudo mkdir -p "$mnt"; then
            echo "[gdrive] warning: could not create $mnt; disabling Google Drive copy"
            ENABLE_GDRIVE_COPY=0
            return 0
        fi
    fi

    # If mounted but stale/bad, lazy-unmount it (requires sudo).
    if mountpoint -q "$mnt" && ! ls "$mnt" >/dev/null 2>&1; then
        sudo umount -l "$mnt" || true
    fi

    # Mount if not currently mounted.
    if ! mountpoint -q "$mnt"; then
        if ! sudo mount -t drvfs G: "$mnt"; then
            echo "[gdrive] warning: mount command failed; disabling Google Drive copy"
            ENABLE_GDRIVE_COPY=0
            return 0
        fi
    fi

    ensure_gdrive_mount || {
        echo "[gdrive] warning: mount validation failed; disabling Google Drive copy"
        ENABLE_GDRIVE_COPY=0
        return 0
    }

    GDRIVE_READY=1
}

copy_to_gdrive_if_enabled() {

    local src="$1"
    [[ "${ENABLE_GDRIVE_COPY:-0}" -eq 1 ]] || return 0
    [[ -n "${GDRIVE_OUTDIR:-}" ]] || die "ENABLE_GDRIVE_COPY is on but GDRIVE_OUTDIR is empty"
        [[ "${GDRIVE_READY:-0}" -eq 1 ]] || { echo "[gdrive] warning: Google Drive init not ready; skipping copy"; return 0; }
    [[ -f "$src" ]] || { echo "[gdrive] warning: source file not found: $src"; return 0; }

    mkdir -p "$GDRIVE_OUTDIR"
    local dst="$GDRIVE_OUTDIR/$(basename "$src")"
    cp -f "$src" "$dst"
    refresh_output_timestamp "$dst"
    echo "[gdrive] copied: $dst"
}

should_rebuild() {
    local target="$1"
    shift || true

    local -a deps=()
    local cache_sig=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dep)
                [[ $# -ge 2 ]] || die "should_rebuild: --dep requires a value"
                deps+=("$2")
                shift 2
                ;;
            --sig)
                [[ $# -ge 2 ]] || die "should_rebuild: --sig requires a value"
                cache_sig="$2"
                shift 2
                ;;
            *)
                deps+=("$1")
                shift
                ;;
        esac
    done

    if [[ "${FORCE:-0}" -eq 1 ]]; then return 0; fi
    if [[ ! -f "$target" ]]; then return 0; fi

    local dep
    for dep in "${deps[@]}"; do
        [[ -n "$dep" ]] || continue
        [[ -e "$dep" ]] || continue
        if [[ "$dep" -nt "$target" ]]; then
            echo "Target $target is older than dependency $dep. Rebuilding." >&2
            return 0
        fi
    done

    if [[ -n "$cache_sig" ]]; then
        local sig_file="${target}.cachekey"
        if [[ ! -f "$sig_file" ]]; then
            echo "Target $target missing cache key. Rebuilding." >&2
            return 0
        fi
        local existing_sig
        existing_sig="$(cat "$sig_file")"
        if [[ "$existing_sig" != "$cache_sig" ]]; then
            echo "Target $target cache key changed. Rebuilding." >&2
            return 0
        fi
    fi

    echo "Target $target is up-to-date. Skipping." >&2
    return 1
}

write_cache_signature() {
    local target="$1"
    local cache_sig="$2"
    [[ -n "$cache_sig" ]] || return 0
    printf '%s' "$cache_sig" > "${target}.cachekey"
}

file_fingerprint() {
    local path="$1"
    if [[ ! -e "$path" ]]; then
        echo "missing"
        return 0
    fi
    local mtime size
    mtime="$(stat -c '%Y' "$path" 2>/dev/null || echo 0)"
    size="$(stat -c '%s' "$path" 2>/dev/null || echo 0)"
    echo "${mtime}:${size}"
}

refresh_output_timestamp() {
    local path="$1"
    [[ -f "$path" ]] || return 0

    touch "$path" 2>/dev/null || true

    if command -v powershell.exe >/dev/null 2>&1 && command -v wslpath >/dev/null 2>&1; then
        local win_path
        win_path="$(wslpath -w "$path" 2>/dev/null || true)"
        if [[ -n "$win_path" ]]; then
            WIN_PATH="$win_path" powershell.exe -NoProfile -Command '$p=$env:WIN_PATH; if (Test-Path $p) { $t=Get-Date; $i=Get-Item $p; $i.CreationTime=$t; $i.LastWriteTime=$t; $i.LastAccessTime=$t }' >/dev/null 2>&1 || true
        fi
    fi
}

dvdlogo_cache_signature() {
    local profile="$1"
    local source_clip="$2"
    local logo_path="$3"
    echo "dvdlogo|profile=${profile}|input=${source_clip}|input_fp=$(file_fingerprint "$source_clip")|logo=${logo_path}|logo_fp=$(file_fingerprint "$logo_path")|script=${DVDLOGO}|script_fp=$(file_fingerprint "$DVDLOGO")"
}

titlehook_cache_signature() {
    local source_clip="$1"
    local logo_path="$2"
    local args_sig="$3"
    echo "titlehook|input=${source_clip}|input_fp=$(file_fingerprint "$source_clip")|logo=${logo_path}|logo_fp=$(file_fingerprint "$logo_path")|script=${TITLE_HOOK_SCRIPT}|script_fp=$(file_fingerprint "$TITLE_HOOK_SCRIPT")|args=${args_sig}"
}

title_hook_duration_for_target_seconds() {
    local target_seconds="$1"
    awk -v d60="$TITLE_HOOK_DURATION" -v d600="$TITLE_HOOK_DURATION_600" -v t="$target_seconds" '
        BEGIN {
            if (t <= 60) {
                printf "%.3f", d60
                exit
            }
            if (t >= 600) {
                printf "%.3f", d600
                exit
            }
            w = (t - 60.0) / (600.0 - 60.0)
            out = d60 + ((d600 - d60) * w)
            printf "%.3f", out
        }
    '
}

rename_output() {
    local src="$1" dst="$2"
    if [[ ! -f "$src" ]]; then
        echo "Warning: $src not found for renaming." >&2
        return 1
    fi

    if mv -vf "$src" "$dst" >&2; then
        refresh_output_timestamp "$dst"
        return 0
    fi

    echo "Warning: mv failed for $src -> $dst; trying copy fallback" >&2
    if cp -f "$src" "$dst" && rm -f "$src"; then
        refresh_output_timestamp "$dst"
        echo "copied '$src' -> '$dst' (fallback)" >&2
        return 0
    fi

    echo "Error: could not stage output to target: $dst" >&2
    return 1
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
        local m
        for m in "${matches[@]}"; do
            out+=( "$(readlink -f "$m")" )
        done
    done
    shopt -u nullglob
    printf '%s\n' "${out[@]}"
}

resolve_logo_patterns() {
    local -a out=()
    local pat
    for pat in "$@"; do
        if [[ "$pat" == */* || "$pat" == *"*"* || "$pat" == *"?"* || "$pat" == *"["* ]]; then
            out+=("$pat")
            continue
        fi

        if [[ -e "$pat" ]]; then
            out+=("$pat")
            continue
        fi

        if [[ -e "$PROJECT_ROOT/art/logos_alpha/$pat" ]]; then
            out+=("$PROJECT_ROOT/art/logos_alpha/$pat")
            continue
        fi

        if [[ -e "$PROJECT_ROOT/dvd_logo/$pat" ]]; then
            out+=("$PROJECT_ROOT/dvd_logo/$pat")
            continue
        fi

        out+=("$pat")
    done
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

canonical_target_output_path() {
    local target_abbrev="$1"
    echo "$OUTDIR/${STEM}_${target_abbrev}.mp4"
}

finalize_target_output_name() {
    local produced_path="$1"
    local target_abbrev="$2"
    local canonical_path
    canonical_path="$(canonical_target_output_path "$target_abbrev")"

    if [[ "$produced_path" != "$canonical_path" ]]; then
        rename_output "$produced_path" "$canonical_path" || die "Could not stage final output: $canonical_path"
    fi

    refresh_output_timestamp "$canonical_path"

    echo "$canonical_path"
}

compute_60_window() {
    local ss="$START_SECONDS"
    local full=""
    echo "$ss|$full"
}

build_highlights_time_args() {
    HIGHLIGHTS_TIME_ARGS=()
    if [[ "${DETECT_AUDIO_START_END:-1}" -eq 1 && ( -z "${START_SECONDS:-}" || -z "${YOUTUBE_FULL_SECONDS:-}" ) ]]; then
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
    local reels_cache_sig
    local source_video="${REELS_INPUT_VIDEO:-$INPUT_VIDEO}"
    reels_cache_sig="reels|input=${source_video}|input_fp=$(file_fingerprint "$source_video")|script=${REELS_OVERLAY}|script_fp=$(file_fingerprint "$REELS_OVERLAY")|min_det=0.05|min_trk=0.05|draw_ids=false|smear=true|smear_frames=17|smear_decay=0.99|trail=true|trail_alpha=.999|beat_sync=true|velocity_color=false|overlay_color=255,255,255"

    if [[ "$USE_REELS_CACHE" -eq 1 && "${REELS_BOOTSTRAP_EXISTING_CACHE:-0}" -eq 1 && -f "$target" && ! -f "${target}.cachekey" ]]; then
        if [[ "$source_video" -nt "$target" || "$REELS_OVERLAY" -nt "$target" ]]; then
            echo "[reels] existing base overlay is stale vs dependencies; rebuilding"
        else
            echo "[reels] bootstrapping cache key for existing base overlay: $target"
            write_cache_signature "$target" "$reels_cache_sig"
        fi
    fi

    if [[ "$USE_REELS_CACHE" -eq 1 ]]; then
        should_rebuild "$target" --dep "$source_video" --dep "$REELS_OVERLAY" --sig "$reels_cache_sig" || return 0
    else
        echo "[reels] cache disabled: rebuilding base overlay"
    fi

    local reels_tmp
    reels_tmp="$(mktemp -d "/tmp/reels_cv_overlay_${PIPELINE_LOG_TAG:-voidstar}_${STEM:-clip}_base_XXXXXX")"
    if ! TMPDIR="$reels_tmp" python3 "$REELS_OVERLAY" "$source_video" \
        --min-det-conf 0.05 --min-trk-conf 0.05 --draw-ids false \
        --smear true --smear-frames 17 --smear-decay 0.99 \
        --trail true --trail-alpha .999 --beat-sync true \
        --velocity-color false --overlay-color 255,255,255 \
        --output "$target"; then
        rm -rf "$reels_tmp"
        return 1
    fi
    rm -rf "$reels_tmp"

    write_cache_signature "$target" "$reels_cache_sig"

    # rename_output "$OUTDIR/${STEM}_fps30_mc2_det0p05_trk0p05_trail1_ta1p00_tlotrue_scan1_velcolor_ids_beatsync_smear.mp4" \
    #     "$target"
}

run_optional_reels_overlay_on_clip() {
    local input_clip="$1"
    local target="$2"
    local stage_label="${3:-clip}"

    [[ "$ENABLE_REELS_OVERLAY_STEP" -eq 1 ]] || { echo "$input_clip"; return 0; }
    [[ "$REELS_CACHE_MODE" == "per-target" ]] || { echo "$input_clip"; return 0; }

    echo "--- Optional per-target reels overlay cache on ${stage_label} ---" >&2

    local reels_cache_sig
    reels_cache_sig="reels_per_target|input=${input_clip}|input_fp=$(file_fingerprint "$input_clip")|script=${REELS_OVERLAY}|script_fp=$(file_fingerprint "$REELS_OVERLAY")|min_det=0.05|min_trk=0.05|draw_ids=false|smear=true|smear_frames=17|smear_decay=0.99|trail=true|trail_alpha=.999|beat_sync=true|velocity_color=false|overlay_color=255,255,255"

    if [[ "$USE_REELS_CACHE" -eq 1 ]]; then
        should_rebuild "$target" --dep "$input_clip" --dep "$REELS_OVERLAY" --sig "$reels_cache_sig" || {
            echo "[reels per-target] using cached: $target" >&2
            echo "$target"
            return 0
        }
    else
        echo "[reels per-target] cache disabled: rebuilding stage clip" >&2
    fi

    local reels_tmp
    reels_tmp="$(mktemp -d "/tmp/reels_cv_overlay_${PIPELINE_LOG_TAG:-voidstar}_${STEM:-clip}_pertarget_XXXXXX")"
    if ! TMPDIR="$reels_tmp" python3 "$REELS_OVERLAY" "$input_clip" \
        --min-det-conf 0.05 --min-trk-conf 0.05 --draw-ids false \
        --smear true --smear-frames 17 --smear-decay 0.99 \
        --trail true --trail-alpha .999 --beat-sync true \
        --velocity-color false --overlay-color 255,255,255 \
        --output "$target" \
        1>&2; then
        rm -rf "$reels_tmp"
        return 1
    fi
    rm -rf "$reels_tmp"

    [[ -f "$target" ]] || die "Per-target reels overlay did not produce output: $target"

    write_cache_signature "$target" "$reels_cache_sig"
    echo "[reels per-target] staged clip: $target" >&2
    echo "$target"
}

run_optional_pre_reels_glitchfield() {
    local input_clip="$1"
    local target="$2"

    [[ "$ENABLE_PRE_REELS_GLITCHFIELD_STAGE" -eq 1 ]] || { echo "$input_clip"; return 0; }

    echo "--- Optional pre-reels glitchfield stage (${PRE_REELS_GLITCHFIELD_PRESET}) ---" >&2
    local glitchfield_script="${PROJECT_ROOT}/glitchfield/glitchfield.py"
    require_file "GLITCHFIELD" "$glitchfield_script"

    local -a gf_args
    case "$PRE_REELS_GLITCHFIELD_PRESET" in
        minimalist)
            gf_args=(
                --effect combo --audio-reactive
                --beat-threshold 0.60 --beat-prob 1.0 --glitch-hold 9
                --combo-glyph-prob 0.75 --combo-stutter-prob 0.25 --combo-alternate-bias 0.60
                --color-mode input --input-palette-k 5 --palette-refresh 8
                --stutter-slice-prob 0.50 --stutter-slice-shift-max 64
                --fx-rgb-split --fx-rgb-shift-max 1 --fx-rgb-prob 0.18
                --fx-trail-frames 1 --fx-trail-strength 0.12
                --fx-scanline-jitter 0.04 --fx-scanline-step 2 --fx-scanline-shift-max 1 --fx-scanline-brightness 0.98
                --fx-vhold-prob 0.015 --fx-vhold-max 2
                --min-gate-period 10
            )
            ;;
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
            if [[ -z "$PRE_REELS_GLITCHFIELD_CUSTOM_ARGS" ]]; then
                die "PRE_REELS_GLITCHFIELD_PRESET=custom requires PRE_REELS_GLITCHFIELD_CUSTOM_ARGS"
            fi
            read -r -a gf_args <<< "$PRE_REELS_GLITCHFIELD_CUSTOM_ARGS"
            ;;
        *)
            die "Unknown PRE_REELS_GLITCHFIELD_PRESET: $PRE_REELS_GLITCHFIELD_PRESET (use minimalist|clean|gritty|chaos|custom)"
            ;;
    esac

    if [[ -n "${PRE_REELS_GLITCHFIELD_MIN_GATE_PERIOD:-}" ]]; then
        gf_args+=(--min-gate-period "$PRE_REELS_GLITCHFIELD_MIN_GATE_PERIOD")
    fi

    local gf_sig_args glitchfield_cache_sig
    gf_sig_args="${gf_args[*]}"
    glitchfield_cache_sig="pre_reels_glitchfield|input=${input_clip}|input_fp=$(file_fingerprint "$input_clip")|script=${glitchfield_script}|script_fp=$(file_fingerprint "$glitchfield_script")|preset=${PRE_REELS_GLITCHFIELD_PRESET}|seed=${PRE_REELS_GLITCHFIELD_SEED}|min_gate_period=${PRE_REELS_GLITCHFIELD_MIN_GATE_PERIOD:-}|args=${gf_sig_args}"

    if [[ "$USE_PRE_REELS_GLITCHFIELD_CACHE" -eq 1 ]]; then
        should_rebuild "$target" --dep "$input_clip" --dep "$glitchfield_script" --sig "$glitchfield_cache_sig" || {
            echo "[pre-reels glitchfield] using cached: $target" >&2
            echo "$target"
            return 0
        }
    else
        echo "[pre-reels glitchfield] cache disabled: rebuilding stage clip" >&2
    fi

    python3 "$glitchfield_script" "$input_clip" \
        "${gf_args[@]}" \
        --seed "$PRE_REELS_GLITCHFIELD_SEED" \
        1>&2

    local generated_src in_dir in_stem
    in_dir="$(dirname "$input_clip")"
    in_stem="$(basename "${input_clip%.*}")"
    generated_src="$(ls -t "$in_dir/${in_stem}"_*.mp4 2>/dev/null | head -n 1 || true)"
    if [[ -n "$generated_src" && -f "$generated_src" ]]; then
        rename_output "$generated_src" "$target" || die "Could not stage pre-reels glitchfield output to target: $target"
    else
        die "Could not locate pre-reels glitchfield output for stem: $in_stem"
    fi

    write_cache_signature "$target" "$glitchfield_cache_sig"
    echo "[pre-reels glitchfield] staged clip: $target" >&2
    echo "$target"
}

run_optional_glitchfield_on_clip() {
    local input_clip="$1"
    local target="$2"
    local stage_label="${3:-clip}"

    [[ "$ENABLE_GLITCHFIELD_STAGE" -eq 1 ]] || { echo "$input_clip"; return 0; }

    echo "--- Optional glitchfield stage (${GLITCHFIELD_PRESET}) on ${stage_label} ---" >&2
    local glitchfield_script="${PROJECT_ROOT}/glitchfield/glitchfield.py"
    require_file "GLITCHFIELD" "$glitchfield_script"

    local -a gf_args
    case "$GLITCHFIELD_PRESET" in
        minimalist)
            gf_args=(
                --effect combo --audio-reactive
                --beat-threshold 0.60 --beat-prob 1.0 --glitch-hold 9
                --combo-glyph-prob 0.75 --combo-stutter-prob 0.25 --combo-alternate-bias 0.60
                --color-mode input --input-palette-k 5 --palette-refresh 8
                --stutter-slice-prob 0.50 --stutter-slice-shift-max 64
                --fx-rgb-split --fx-rgb-shift-max 1 --fx-rgb-prob 0.18
                --fx-trail-frames 1 --fx-trail-strength 0.12
                --fx-scanline-jitter 0.04 --fx-scanline-step 2 --fx-scanline-shift-max 1 --fx-scanline-brightness 0.98
                --fx-vhold-prob 0.015 --fx-vhold-max 2
                --min-gate-period 10
            )
            ;;
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
                die "Unknown GLITCHFIELD_PRESET: $GLITCHFIELD_PRESET (use minimalist|clean|gritty|chaos|custom)"
            ;;
    esac

    if [[ -n "${GLITCHFIELD_MIN_GATE_PERIOD:-}" ]]; then
        gf_args+=(--min-gate-period "$GLITCHFIELD_MIN_GATE_PERIOD")
    fi

    local gf_sig_args glitchfield_cache_sig
    gf_sig_args="${gf_args[*]}"
    glitchfield_cache_sig="glitchfield|input=${input_clip}|input_fp=$(file_fingerprint "$input_clip")|script=${glitchfield_script}|script_fp=$(file_fingerprint "$glitchfield_script")|preset=${GLITCHFIELD_PRESET}|seed=${GLITCHFIELD_SEED}|min_gate_period=${GLITCHFIELD_MIN_GATE_PERIOD:-}|args=${gf_sig_args}"

    if [[ "$USE_GLITCHFIELD_CACHE" -eq 1 ]]; then
        should_rebuild "$target" --dep "$input_clip" --dep "$glitchfield_script" --sig "$glitchfield_cache_sig" || {
            echo "[glitchfield] using cached: $target" >&2
            echo "$target"
            return 0
        }
    else
        echo "[glitchfield] cache disabled: rebuilding glitchfield clip" >&2
    fi

    python3 "$glitchfield_script" "$input_clip" \
        "${gf_args[@]}" \
        --seed "$GLITCHFIELD_SEED" \
        1>&2

    local generated_src in_dir in_stem
    in_dir="$(dirname "$input_clip")"
    in_stem="$(basename "${input_clip%.*}")"
    generated_src="$(ls -t "$in_dir/${in_stem}"_*.mp4 2>/dev/null | head -n 1 || true)"
    if [[ -n "$generated_src" && -f "$generated_src" ]]; then
        rename_output "$generated_src" "$target" || die "Could not stage glitchfield output to target: $target"
    else
        die "Could not locate glitchfield output for stem: $in_stem"
    fi

    write_cache_signature "$target" "$glitchfield_cache_sig"
    echo "[glitchfield] staged clip: $target" >&2
    echo "$target"
}

run_optional_particle_sparks_on_clip() {
    local input_clip="$1"
    local target="$2"
    local stage_label="${3:-clip}"

    [[ "$ENABLE_PARTICLE_SPARKS_STAGE" -eq 1 ]] || { echo "$input_clip"; return 0; }

    echo "--- Optional particle sparks stage on ${stage_label} ---" >&2
    require_file "PARTICLE_SPARKS" "$PARTICLE_SPARKS"

    local sparks_cache_sig
    sparks_cache_sig="particle_sparks|input=${input_clip}|input_fp=$(file_fingerprint "$input_clip")|script=${PARTICLE_SPARKS}|script_fp=$(file_fingerprint "$PARTICLE_SPARKS")|max_points=${PARTICLE_SPARKS_MAX_POINTS}|point_min_distance=${PARTICLE_SPARKS_POINT_MIN_DISTANCE}|max_live_sparks=${PARTICLE_SPARKS_MAX_LIVE_SPARKS}|motion_threshold=${PARTICLE_SPARKS_MOTION_THRESHOLD}|spark_rate=${PARTICLE_SPARKS_RATE}|spark_size=${PARTICLE_SPARKS_SIZE}|spark_life=${PARTICLE_SPARKS_LIFE_FRAMES}|spark_speed=${PARTICLE_SPARKS_SPEED}|spark_jitter=${PARTICLE_SPARKS_JITTER}|spark_opacity=${PARTICLE_SPARKS_OPACITY}|emmons_spin_speed=${PARTICLE_SPARKS_EMMONS_SPIN_SPEED}|spark_repel_strength=${PARTICLE_SPARKS_REPEL_STRENGTH}|spark_repel_radius=${PARTICLE_SPARKS_REPEL_RADIUS}|audio_gain=${PARTICLE_SPARKS_AUDIO_GAIN}|audio_smooth=${PARTICLE_SPARKS_AUDIO_SMOOTH}|color_mode=${PARTICLE_SPARKS_COLOR_MODE}|color_rgb=${PARTICLE_SPARKS_COLOR_RGB}|flood_in_out=${PARTICLE_SPARKS_FLOOD_IN_OUT}|flood_seconds=${PARTICLE_SPARKS_FLOOD_SECONDS}|flood_spawn_mult=${PARTICLE_SPARKS_FLOOD_SPAWN_MULT}|flood_extra_sources=${PARTICLE_SPARKS_FLOOD_EXTRA_SOURCES}|flood_velocity_mult=${PARTICLE_SPARKS_FLOOD_VELOCITY_MULT}"

    if [[ "$USE_PARTICLE_SPARKS_CACHE" -eq 1 ]]; then
        should_rebuild "$target" --dep "$input_clip" --dep "$PARTICLE_SPARKS" --sig "$sparks_cache_sig" || {
            echo "[particle-sparks] using cached: $target" >&2
            echo "$target"
            return 0
        }
    else
        echo "[particle-sparks] cache disabled: rebuilding stage clip" >&2
    fi

    python3 "$PARTICLE_SPARKS" "$input_clip" \
        --output "$target" \
        --start 0 --duration 0 \
        --color-mode "$PARTICLE_SPARKS_COLOR_MODE" \
        --spark-size "$PARTICLE_SPARKS_SIZE" \
        --spark-life-frames "$PARTICLE_SPARKS_LIFE_FRAMES" \
        --max-points "$PARTICLE_SPARKS_MAX_POINTS" \
        --point-min-distance "$PARTICLE_SPARKS_POINT_MIN_DISTANCE" \
        --max-live-sparks "$PARTICLE_SPARKS_MAX_LIVE_SPARKS" \
        --spark-rate "$PARTICLE_SPARKS_RATE" \
        --motion-threshold "$PARTICLE_SPARKS_MOTION_THRESHOLD" \
        --spark-speed "$PARTICLE_SPARKS_SPEED" \
        --spark-jitter "$PARTICLE_SPARKS_JITTER" \
        --spark-opacity "$PARTICLE_SPARKS_OPACITY" \
        --emmons-spin-speed "$PARTICLE_SPARKS_EMMONS_SPIN_SPEED" \
        --spark-repel-strength "$PARTICLE_SPARKS_REPEL_STRENGTH" \
        --spark-repel-radius "$PARTICLE_SPARKS_REPEL_RADIUS" \
        --audio-reactive true \
        --audio-reactive-gain "$PARTICLE_SPARKS_AUDIO_GAIN" \
        --audio-reactive-smooth "$PARTICLE_SPARKS_AUDIO_SMOOTH" \
        --color-rgb "$PARTICLE_SPARKS_COLOR_RGB" \
        --flood-in-out "$( [[ "$PARTICLE_SPARKS_FLOOD_IN_OUT" -eq 1 ]] && echo true || echo false )" \
        --flood-seconds "$PARTICLE_SPARKS_FLOOD_SECONDS" \
        --flood-spawn-mult "$PARTICLE_SPARKS_FLOOD_SPAWN_MULT" \
        --flood-extra-sources "$PARTICLE_SPARKS_FLOOD_EXTRA_SOURCES" \
        --flood-velocity-mult "$PARTICLE_SPARKS_FLOOD_VELOCITY_MULT" \
        1>&2

    [[ -f "$target" ]] || die "Particle sparks stage did not produce output: $target"
    write_cache_signature "$target" "$sparks_cache_sig"
    echo "[particle-sparks] staged clip: $target" >&2
    echo "$target"
}

run_optional_title_hook_on_clip() {
    local input_clip="$1"
    local target="$2"
    local stage_label="${3:-clip}"
    local title_token="${4:-hi60t}"
    local hook_duration="${5:-$TITLE_HOOK_DURATION}"

    [[ "$ENABLE_TITLE_HOOK_STAGE" -eq 1 ]] || { echo "$input_clip"; return 0; }

    echo "--- Optional title hook stage on ${stage_label} ---" >&2
    require_file "TITLE_HOOK_SCRIPT" "$TITLE_HOOK_SCRIPT"
    require_file "TITLE_HOOK_LOGO" "$TITLE_HOOK_LOGO"

    local sparks_flag=""
    if [[ "$TITLE_HOOK_SPARKS" -eq 1 ]]; then
        sparks_flag="--title-hook-sparks"
    fi

    local args_sig
    local resolved_title
    resolved_title="${TITLE_HOOK_TITLE//hi60t/${title_token}}"
    args_sig="duration=${hook_duration}|fade=${TITLE_HOOK_FADE_OUT_DURATION}|title=${resolved_title}|secondary=${TITLE_HOOK_SECONDARY_TEXT}|logo_alpha_threshold=${TITLE_HOOK_LOGO_ALPHA_THRESHOLD}|logo_intensity=${TITLE_HOOK_LOGO_INTENSITY}|logo_idle_wiggle=${TITLE_HOOK_LOGO_IDLE_WIGGLE}|logo_x=${TITLE_HOOK_LOGO_X_RATIO}|logo_y=${TITLE_HOOK_LOGO_Y_RATIO}|track_scale=${TITLE_HOOK_LOGO_MOTION_TRACK_SCALE}|track_radius=${TITLE_HOOK_LOGO_MOTION_TRACK_RADIUS}|track_neighbors=${TITLE_HOOK_LOGO_MOTION_TRACK_LINK_NEIGHBORS}|track_min_distance=${TITLE_HOOK_LOGO_MOTION_TRACK_MIN_DISTANCE}|track_pad_px=${TITLE_HOOK_LOGO_MOTION_TRACK_PAD_PX}|track_link_opacity=${TITLE_HOOK_LOGO_MOTION_TRACK_LINK_OPACITY}|track_refresh=${TITLE_HOOK_LOGO_MOTION_TRACK_REFRESH}|track_decay=${TITLE_HOOK_LOGO_MOTION_TRACK_DECAY}|logo_opacity=${TITLE_HOOK_LOGO_OPACITY}|background_dim=${TITLE_HOOK_BACKGROUND_DIM}|title_layer_dim=${TITLE_HOOK_TITLE_LAYER_DIM}|text_align=${TITLE_HOOK_TEXT_ALIGN}|title_jitter_audio_multiplier=${TITLE_HOOK_TITLE_JITTER_AUDIO_MULTIPLIER}|sparks=${TITLE_HOOK_SPARKS}|sparks_rate=${TITLE_HOOK_SPARKS_RATE}|sparks_motion_threshold=${TITLE_HOOK_SPARKS_MOTION_THRESHOLD}|sparks_opacity=${TITLE_HOOK_SPARKS_OPACITY}|token=${title_token}"

    local titlehook_sig
    titlehook_sig="$(titlehook_cache_signature "$input_clip" "$TITLE_HOOK_LOGO" "$args_sig")"

    if [[ "$USE_TITLE_HOOK_CACHE" -eq 1 ]]; then
        should_rebuild "$target" --dep "$input_clip" --dep "$TITLE_HOOK_SCRIPT" --dep "$TITLE_HOOK_LOGO" --sig "$titlehook_sig" || {
            echo "[title-hook] using cached: $target" >&2
            echo "$target"
            return 0
        }
    else
        echo "[title-hook] cache disabled: rebuilding stage clip" >&2
    fi

    python3 "$TITLE_HOOK_SCRIPT" "$input_clip" \
        --output "$target" \
        --title "$resolved_title" \
        --secondary-text "$TITLE_HOOK_SECONDARY_TEXT" \
        --duration "$hook_duration" \
        --fade-out-duration "$TITLE_HOOK_FADE_OUT_DURATION" \
        --logo "$TITLE_HOOK_LOGO" \
        --logo-alpha-threshold "$TITLE_HOOK_LOGO_ALPHA_THRESHOLD" \
        --logo-intensity "$TITLE_HOOK_LOGO_INTENSITY" \
        --logo-idle-wiggle "$TITLE_HOOK_LOGO_IDLE_WIGGLE" \
        --logo-x-ratio "$TITLE_HOOK_LOGO_X_RATIO" \
        --logo-y-ratio "$TITLE_HOOK_LOGO_Y_RATIO" \
        --logo-motion-track-scale "$TITLE_HOOK_LOGO_MOTION_TRACK_SCALE" \
        --logo-motion-track-radius "$TITLE_HOOK_LOGO_MOTION_TRACK_RADIUS" \
        --logo-motion-track-link-neighbors "$TITLE_HOOK_LOGO_MOTION_TRACK_LINK_NEIGHBORS" \
        --logo-motion-track-min-distance "$TITLE_HOOK_LOGO_MOTION_TRACK_MIN_DISTANCE" \
        --logo-motion-track-pad-px "$TITLE_HOOK_LOGO_MOTION_TRACK_PAD_PX" \
        --logo-motion-track-link-opacity "$TITLE_HOOK_LOGO_MOTION_TRACK_LINK_OPACITY" \
        --logo-motion-track-refresh "$TITLE_HOOK_LOGO_MOTION_TRACK_REFRESH" \
        --logo-motion-track-decay "$TITLE_HOOK_LOGO_MOTION_TRACK_DECAY" \
        --logo-opacity "$TITLE_HOOK_LOGO_OPACITY" \
        --background-dim "$TITLE_HOOK_BACKGROUND_DIM" \
        --title-layer-dim "$TITLE_HOOK_TITLE_LAYER_DIM" \
        --text-align "$TITLE_HOOK_TEXT_ALIGN" \
        --title-jitter-audio-multiplier "$TITLE_HOOK_TITLE_JITTER_AUDIO_MULTIPLIER" \
        --title-hook-sparks-rate "$TITLE_HOOK_SPARKS_RATE" \
        --title-hook-sparks-motion-threshold "$TITLE_HOOK_SPARKS_MOTION_THRESHOLD" \
        --title-hook-sparks-opacity "$TITLE_HOOK_SPARKS_OPACITY" \
        ${sparks_flag} \
        1>&2

    [[ -f "$target" ]] || die "Title hook stage did not produce output: $target"

    write_cache_signature "$target" "$titlehook_sig"
    echo "[title-hook] staged clip: $target" >&2
    echo "$target"
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
            --sampling-mode "$DIVVY_SAMPLING_MODE" --sample-seconds "$sample_seconds" #--truncate-to-full-clips
        )

        if [[ "$DIVVY_SAMPLING_MODE" == "groove" ]]; then
            cmd+=(--bpm "$DIVVY_GROOVE_BPM")
        fi

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
    echo "--- 60s highlight (START) ---"
    local divvy_dst="$OUTDIR/${STEM}_highlights_60s_overlay.mp4"

    run_divvy_uniform_highlights "$divvy_dst" 60 "$DIVVY_60S_START_SAMPLE_SECONDS" "$DIVVY_60S_START_SEGMENTS" ""

    local logo tag target
    logo="$LOGO_START"
    tag="$(basename "${logo%.*}")"
    local source_for_effects="$divvy_dst"
    local reels_dst="$OUTDIR/${STEM}_highlights_60s_overlay_reels.mp4"
    source_for_effects="$(run_optional_reels_overlay_on_clip "$divvy_dst" "$reels_dst" "60s-start")"
    local source_for_logo="$source_for_effects"
    local glitch_dst="$OUTDIR/${STEM}_highlights_60s_overlay_glitchfield_${GLITCHFIELD_PRESET}.mp4"
    source_for_logo="$(run_optional_glitchfield_on_clip "$source_for_effects" "$glitch_dst" "60s-start")"

    target="$(with_logo_suffix "$OUTDIR/${STEM}_highlights_60s_overlay_logo.mp4" "$tag")"
    local logo_stage="$target"
    if [[ "$ENABLE_PARTICLE_SPARKS_STAGE" -eq 1 ]]; then
        logo_stage="${target%.mp4}_pre_sparks.mp4"
    fi
    local dvdlogo_profile="cinema_local_track_v1_start083"
    local dvdlogo_sig
    dvdlogo_sig="$(dvdlogo_cache_signature "$dvdlogo_profile" "$source_for_logo" "$logo")"

    if should_rebuild "$logo_stage" --dep "$source_for_logo" --dep "$logo" --dep "$DVDLOGO" --sig "$dvdlogo_sig"; then
        python3 "$DVDLOGO" "$source_for_logo" "$logo" \
            --speed 0 --logo-scale .4 --logo-rotate-speed 0 --trails 0.85 --opacity .5 \
            --audio-reactive-glow 1.0 --audio-reactive-scale 0.5 --audio-reactive-gain 2.0 \
            --edge-margin-px 0 --reels-local-overlay false --voidstar-preset cinema \
            --local-point-track true --local-point-track-scale 1.2 --local-point-track-pad-px 0 \
            --local-point-track-max-points 256 --local-point-track-radius 256 --local-point-track-min-distance 128 \
            --local-point-track-refresh 8 --local-point-track-opacity 0.77 --local-point-track-decay 0.77 \
            --local-point-track-link-neighbors 8 --local-point-track-link-thickness 1 \
            --local-point-track-link-opacity .77 --voidstar-colorize true --start-x .8 --start-y .83 \
            --content-bbox-for-local false --voidstar-debug-bounds true --voidstar-debug-bounds-mode hit-glitch \
            --voidstar-debug-bounds-hit-threshold 0.8 --voidstar-debug-bounds-hit-prob 0.1 \
            --output "$logo_stage"
            write_cache_signature "$logo_stage" "$dvdlogo_sig"
    fi

    if [[ "$ENABLE_PARTICLE_SPARKS_STAGE" -eq 1 ]]; then
        run_optional_particle_sparks_on_clip "$logo_stage" "$target" "60s-start-post-logo" >/dev/null
    fi

    local final_target="$target"
    local title_hook_target="${target%.mp4}_titlehook.mp4"
    final_target="$(run_optional_title_hook_on_clip "$target" "$title_hook_target" "60s-start-title-hook" "hi60s")"
    final_target="$(finalize_target_output_name "$final_target" "hi60s")"

    copy_to_gdrive_if_enabled "$final_target"
}

run_180s_start() {
    echo "--- 180s highlight (START) ---"
    local divvy_dst="$OUTDIR/${STEM}_highlights_180s_overlay.mp4"

    run_divvy_uniform_highlights "$divvy_dst" 180 "$DIVVY_180S_START_SAMPLE_SECONDS" "$DIVVY_180S_START_SEGMENTS" ""

    local logo tag target
    logo="$LOGO_START"
    tag="$(basename "${logo%.*}")"
    local source_for_effects="$divvy_dst"
    local reels_dst="$OUTDIR/${STEM}_highlights_180s_overlay_reels.mp4"
    source_for_effects="$(run_optional_reels_overlay_on_clip "$divvy_dst" "$reels_dst" "180s-start")"
    local source_for_logo="$source_for_effects"
    local glitch_dst="$OUTDIR/${STEM}_highlights_180s_overlay_glitchfield_${GLITCHFIELD_PRESET}.mp4"
    source_for_logo="$(run_optional_glitchfield_on_clip "$source_for_effects" "$glitch_dst" "180s-start")"

    target="$(with_logo_suffix "$OUTDIR/${STEM}_highlights_180s_overlay_logo.mp4" "$tag")"
    local logo_stage="$target"
    if [[ "$ENABLE_PARTICLE_SPARKS_STAGE" -eq 1 ]]; then
        logo_stage="${target%.mp4}_pre_sparks.mp4"
    fi
    local dvdlogo_profile="cinema_local_track_v1_start083"
    local dvdlogo_sig
    dvdlogo_sig="$(dvdlogo_cache_signature "$dvdlogo_profile" "$source_for_logo" "$logo")"

    if should_rebuild "$logo_stage" --dep "$source_for_logo" --dep "$logo" --dep "$DVDLOGO" --sig "$dvdlogo_sig"; then
        python3 "$DVDLOGO" "$source_for_logo" "$logo" \
            --speed 0 --logo-scale .4 --logo-rotate-speed 0 --trails 0.85 --opacity .5 \
            --audio-reactive-glow 1.0 --audio-reactive-scale 0.5 --audio-reactive-gain 2.0 \
            --edge-margin-px 0 --reels-local-overlay false --voidstar-preset cinema \
            --local-point-track true --local-point-track-scale 1.2 --local-point-track-pad-px 0 \
            --local-point-track-max-points 256 --local-point-track-radius 256 --local-point-track-min-distance 128 \
            --local-point-track-refresh 8 --local-point-track-opacity 0.77 --local-point-track-decay 0.77 \
            --local-point-track-link-neighbors 8 --local-point-track-link-thickness 1 \
            --local-point-track-link-opacity .77 --voidstar-colorize true --start-x .8 --start-y .83 \
            --content-bbox-for-local false --voidstar-debug-bounds true --voidstar-debug-bounds-mode hit-glitch \
            --voidstar-debug-bounds-hit-threshold 0.8 --voidstar-debug-bounds-hit-prob 0.1 \
            --output "$logo_stage"
            write_cache_signature "$logo_stage" "$dvdlogo_sig"
    fi

    if [[ "$ENABLE_PARTICLE_SPARKS_STAGE" -eq 1 ]]; then
        run_optional_particle_sparks_on_clip "$logo_stage" "$target" "180s-start-post-logo" >/dev/null
    fi

    local final_target="$target"
    local title_hook_target="${target%.mp4}_titlehook.mp4"
    local hook_duration
    hook_duration="$(title_hook_duration_for_target_seconds 180)"
    final_target="$(run_optional_title_hook_on_clip "$target" "$title_hook_target" "180s-start-title-hook" "hi180s" "$hook_duration")"
    final_target="$(finalize_target_output_name "$final_target" "hi180s")"

    copy_to_gdrive_if_enabled "$final_target"
}

run_full() {
    echo "--- FULL (YouTube) ---"
    local base_overlay="$BASE_REELS_OVERLAY"
    require_file "BASE_REELS_OVERLAY" "$base_overlay"

    local source_for_effects="$base_overlay"
    local reels_dst="$OUTDIR/${STEM}_full_overlay_reels.mp4"
    source_for_effects="$(run_optional_reels_overlay_on_clip "$base_overlay" "$reels_dst" "full")"
    local source_for_logo="$source_for_effects"

    local logo tag target
    logo="$LOGO_START"
    tag="$(basename "${logo%.*}")"
    target="$(with_logo_suffix "$OUTDIR/${STEM}_full_overlay_logo.mp4" "$tag")"
    local logo_stage="$target"
    if [[ "$ENABLE_PARTICLE_SPARKS_STAGE" -eq 1 ]]; then
        logo_stage="${target%.mp4}_pre_sparks.mp4"
    fi
    local dvdlogo_profile="cinema_local_track_v1_start077"
    local dvdlogo_sig
    dvdlogo_sig="$(dvdlogo_cache_signature "$dvdlogo_profile" "$source_for_logo" "$logo")"

    if should_rebuild "$logo_stage" --dep "$source_for_logo" --dep "$logo" --dep "$DVDLOGO" --sig "$dvdlogo_sig"; then
        python3 "$DVDLOGO" "$source_for_logo" "$logo" \
            --speed 0 --logo-scale .4 --logo-rotate-speed 0 --trails 0.85 --opacity .5 \
            --audio-reactive-glow 1.0 --audio-reactive-scale 0.5 --audio-reactive-gain 2.0 \
            --edge-margin-px 0 --reels-local-overlay false --voidstar-preset cinema \
            --local-point-track true --local-point-track-scale 1.2 --local-point-track-pad-px 0 \
            --local-point-track-max-points 256 --local-point-track-radius 256 --local-point-track-min-distance 128 \
            --local-point-track-refresh 8 --local-point-track-opacity 0.77 --local-point-track-decay 0.77 \
            --local-point-track-link-neighbors 8 --local-point-track-link-thickness 1 \
            --local-point-track-link-opacity .77 --voidstar-colorize true --start-x .77 --start-y .79 \
            --content-bbox-for-local false --voidstar-debug-bounds true --voidstar-debug-bounds-mode hit-glitch \
            --voidstar-debug-bounds-hit-threshold 0.8 --voidstar-debug-bounds-hit-prob 0.1 \
            --output "$logo_stage"
            write_cache_signature "$logo_stage" "$dvdlogo_sig"
    fi

    if [[ "$ENABLE_PARTICLE_SPARKS_STAGE" -eq 1 ]]; then
        run_optional_particle_sparks_on_clip "$logo_stage" "$target" "full-post-logo" >/dev/null
    fi

    local final_target="$target"
    local title_hook_target="${target%.mp4}_titlehook.mp4"
    local hook_duration
    hook_duration="$(title_hook_duration_for_target_seconds "$INPUT_DURATION_SECONDS")"
    final_target="$(run_optional_title_hook_on_clip "$target" "$title_hook_target" "full-title-hook" "full" "$hook_duration")"
    final_target="$(finalize_target_output_name "$final_target" "full")"

    copy_to_gdrive_if_enabled "$final_target"
}

run_60s_end() {
    echo "--- 60s highlight (END) ---"
    local divvy_dst="$OUTDIR/${STEM}_highlights_60t_overlay.mp4"

    run_divvy_uniform_highlights "$divvy_dst" 60 "$DIVVY_60S_END_SAMPLE_SECONDS" "$DIVVY_60S_END_SEGMENTS" "end"

    local logo tag target
    logo="$LOGO_END"
    tag="$(basename "${logo%.*}")"
    local source_for_effects="$divvy_dst"
    local reels_dst="$OUTDIR/${STEM}_highlights_60t_overlay_reels.mp4"
    source_for_effects="$(run_optional_reels_overlay_on_clip "$divvy_dst" "$reels_dst" "60s-end")"
    local source_for_logo="$source_for_effects"
    local glitch_dst="$OUTDIR/${STEM}_highlights_60t_overlay_glitchfield_${GLITCHFIELD_PRESET}.mp4"
    source_for_logo="$(run_optional_glitchfield_on_clip "$source_for_effects" "$glitch_dst" "60s-end")"

    target="$(with_logo_suffix "$OUTDIR/${STEM}_highlights_60t_overlay_logo.mp4" "$tag")"
    local logo_stage="$target"
    if [[ "$ENABLE_PARTICLE_SPARKS_STAGE" -eq 1 ]]; then
        logo_stage="${target%.mp4}_pre_sparks.mp4"
    fi
    local dvdlogo_profile="cinema_local_track_v1_start083"
    local dvdlogo_sig
    dvdlogo_sig="$(dvdlogo_cache_signature "$dvdlogo_profile" "$source_for_logo" "$logo")"

    if should_rebuild "$logo_stage" --dep "$source_for_logo" --dep "$logo" --dep "$DVDLOGO" --sig "$dvdlogo_sig"; then
        python3 "$DVDLOGO" "$source_for_logo" "$logo" \
            --speed 0 --logo-scale .4 --logo-rotate-speed 0 --trails 0.85 --opacity .5 \
            --audio-reactive-glow 1.0 --audio-reactive-scale 0.5 --audio-reactive-gain 2.0 \
            --edge-margin-px 0 --reels-local-overlay false --voidstar-preset cinema \
            --local-point-track true --local-point-track-scale 1.2 --local-point-track-pad-px 0 \
            --local-point-track-max-points 256 --local-point-track-radius 256 --local-point-track-min-distance 128 \
            --local-point-track-refresh 8 --local-point-track-opacity 0.77 --local-point-track-decay 0.77 \
            --local-point-track-link-neighbors 8 --local-point-track-link-thickness 1 \
            --local-point-track-link-opacity .77 --voidstar-colorize true --start-x .8 --start-y .83 \
            --content-bbox-for-local false --voidstar-debug-bounds true --voidstar-debug-bounds-mode hit-glitch \
            --voidstar-debug-bounds-hit-threshold 0.8 --voidstar-debug-bounds-hit-prob 0.1 \
            --output "$logo_stage"
            write_cache_signature "$logo_stage" "$dvdlogo_sig"
    fi

    if [[ "$ENABLE_PARTICLE_SPARKS_STAGE" -eq 1 ]]; then
        run_optional_particle_sparks_on_clip "$logo_stage" "$target" "60s-end-post-logo" >/dev/null
    fi

    local final_target="$target"
    local title_hook_target="${target%.mp4}_titlehook.mp4"
    local hook_duration
    hook_duration="$(title_hook_duration_for_target_seconds 60)"
    final_target="$(run_optional_title_hook_on_clip "$target" "$title_hook_target" "60s-end-title-hook" "hi60t" "$hook_duration")"
    final_target="$(finalize_target_output_name "$final_target" "hi60t")"

    copy_to_gdrive_if_enabled "$final_target"
}

run_180s_end() {
    echo "--- 180s highlight (END) ---"
    local divvy_dst="$OUTDIR/${STEM}_highlights_180t_overlay.mp4"

    run_divvy_uniform_highlights "$divvy_dst" 180 "$DIVVY_180S_END_SAMPLE_SECONDS" "$DIVVY_180S_END_SEGMENTS" "end"

    local logo tag target
    logo="$LOGO_END"
    tag="$(basename "${logo%.*}")"
    local source_for_effects="$divvy_dst"
    local reels_dst="$OUTDIR/${STEM}_highlights_180t_overlay_reels.mp4"
    source_for_effects="$(run_optional_reels_overlay_on_clip "$divvy_dst" "$reels_dst" "180s-end")"
    local source_for_logo="$source_for_effects"
    local glitch_dst="$OUTDIR/${STEM}_highlights_180t_overlay_glitchfield_${GLITCHFIELD_PRESET}.mp4"
    source_for_logo="$(run_optional_glitchfield_on_clip "$source_for_effects" "$glitch_dst" "180s-end")"

    target="$(with_logo_suffix "$OUTDIR/${STEM}_highlights_180t_overlay_logo.mp4" "$tag")"
    local logo_stage="$target"
    if [[ "$ENABLE_PARTICLE_SPARKS_STAGE" -eq 1 ]]; then
        logo_stage="${target%.mp4}_pre_sparks.mp4"
    fi
    local dvdlogo_profile="cinema_local_track_v1_start077"
    local dvdlogo_sig
    dvdlogo_sig="$(dvdlogo_cache_signature "$dvdlogo_profile" "$source_for_logo" "$logo")"

    if should_rebuild "$logo_stage" --dep "$source_for_logo" --dep "$logo" --dep "$DVDLOGO" --sig "$dvdlogo_sig"; then
        python3 "$DVDLOGO" "$source_for_logo" "$logo" \
            --speed 0 --logo-scale .4 --logo-rotate-speed 0 --trails 0.85 --opacity .5 \
            --audio-reactive-glow 1.0 --audio-reactive-scale 0.5 --audio-reactive-gain 2.0 \
            --edge-margin-px 0 --reels-local-overlay false --voidstar-preset cinema \
            --local-point-track true --local-point-track-scale 1.2 --local-point-track-pad-px 0 \
            --local-point-track-max-points 256 --local-point-track-radius 256 --local-point-track-min-distance 128 \
            --local-point-track-refresh 8 --local-point-track-opacity 0.77 --local-point-track-decay 0.77 \
            --local-point-track-link-neighbors 8 --local-point-track-link-thickness 1 \
            --local-point-track-link-opacity .77 --voidstar-colorize true --start-x .77 --start-y .79 \
            --content-bbox-for-local false --voidstar-debug-bounds true --voidstar-debug-bounds-mode hit-glitch \
            --voidstar-debug-bounds-hit-threshold 0.8 --voidstar-debug-bounds-hit-prob 0.1 \
            --output "$logo_stage"
            write_cache_signature "$logo_stage" "$dvdlogo_sig"
    fi

    if [[ "$ENABLE_PARTICLE_SPARKS_STAGE" -eq 1 ]]; then
        run_optional_particle_sparks_on_clip "$logo_stage" "$target" "180s-end-post-logo" >/dev/null
    fi

    local final_target="$target"
    local title_hook_target="${target%.mp4}_titlehook.mp4"
    local hook_duration
    hook_duration="$(title_hook_duration_for_target_seconds 180)"
    final_target="$(run_optional_title_hook_on_clip "$target" "$title_hook_target" "180s-end-title-hook" "hi180t" "$hook_duration")"
    final_target="$(finalize_target_output_name "$final_target" "hi180t")"

    copy_to_gdrive_if_enabled "$final_target"
}

main() {
    FORCE="$FORCE_DEFAULT"
    INPUT_VIDEO=""; OUTDIR=""
    START_SECONDS="$START_SECONDS_DEFAULT"; YOUTUBE_FULL_SECONDS="$YOUTUBE_FULL_SECONDS_DEFAULT"; DETECT_AUDIO_START_END="$DETECT_AUDIO_START_END_DEFAULT"; CPS="$CPS_DEFAULT"; GLITCH_SECONDS="$GLITCH_SECONDS_DEFAULT"; LOOP_SEAM_SECONDS="$LOOP_SEAM_SECONDS_DEFAULT"
    DIVVY_SAMPLING_MODE="$DIVVY_SAMPLING_MODE_DEFAULT"
    DIVVY_GROOVE_BPM="$DIVVY_GROOVE_BPM_DEFAULT"
    DIVVY_60S_START_SAMPLE_SECONDS="$DIVVY_60S_START_SAMPLE_SECONDS_DEFAULT"
    DIVVY_60S_START_SEGMENTS="$DIVVY_60S_START_SEGMENTS_DEFAULT"
    DIVVY_180S_START_SAMPLE_SECONDS="$DIVVY_180S_START_SAMPLE_SECONDS_DEFAULT"
    DIVVY_180S_START_SEGMENTS="$DIVVY_180S_START_SEGMENTS_DEFAULT"
    DIVVY_60S_END_SAMPLE_SECONDS="$DIVVY_60S_END_SAMPLE_SECONDS_DEFAULT"
    DIVVY_60S_END_SEGMENTS="$DIVVY_60S_END_SEGMENTS_DEFAULT"
    DIVVY_180S_END_SAMPLE_SECONDS="$DIVVY_180S_END_SAMPLE_SECONDS_DEFAULT"
    DIVVY_180S_END_SEGMENTS="$DIVVY_180S_END_SEGMENTS_DEFAULT"
    PREVIEW_REELS_OVERLAY="$PREVIEW_REELS_OVERLAY_DEFAULT"
    BASE_REELS_OVERLAY_PREBUILT="$BASE_REELS_OVERLAY_PREBUILT_DEFAULT"
    REELS_BOOTSTRAP_EXISTING_CACHE="$REELS_BOOTSTRAP_EXISTING_CACHE_DEFAULT"
    LOGO_START="$LOGO_START_DEFAULT"
    LOGO_END="$LOGO_END_DEFAULT"
    USE_REELS_CACHE="$USE_REELS_CACHE_DEFAULT"
    REELS_CACHE_MODE="$REELS_CACHE_MODE_DEFAULT"
    USE_PRE_REELS_GLITCHFIELD_CACHE="$USE_PRE_REELS_GLITCHFIELD_CACHE_DEFAULT"
    USE_GLITCHFIELD_CACHE="$USE_GLITCHFIELD_CACHE_DEFAULT"
    USE_PARTICLE_SPARKS_CACHE="$USE_PARTICLE_SPARKS_CACHE_DEFAULT"
    USE_TITLE_HOOK_CACHE="$USE_TITLE_HOOK_CACHE_DEFAULT"
    PARTICLE_SPARKS_MAX_POINTS="$PARTICLE_SPARKS_MAX_POINTS_DEFAULT"
    PARTICLE_SPARKS_POINT_MIN_DISTANCE="$PARTICLE_SPARKS_POINT_MIN_DISTANCE_DEFAULT"
    PARTICLE_SPARKS_MAX_LIVE_SPARKS="$PARTICLE_SPARKS_MAX_LIVE_SPARKS_DEFAULT"
    PARTICLE_SPARKS_MOTION_THRESHOLD="$PARTICLE_SPARKS_MOTION_THRESHOLD_DEFAULT"
    PARTICLE_SPARKS_RATE="$PARTICLE_SPARKS_RATE_DEFAULT"
    PARTICLE_SPARKS_SIZE="$PARTICLE_SPARKS_SIZE_DEFAULT"
    PARTICLE_SPARKS_LIFE_FRAMES="$PARTICLE_SPARKS_LIFE_FRAMES_DEFAULT"
    PARTICLE_SPARKS_SPEED="$PARTICLE_SPARKS_SPEED_DEFAULT"
    PARTICLE_SPARKS_JITTER="$PARTICLE_SPARKS_JITTER_DEFAULT"
    PARTICLE_SPARKS_OPACITY="$PARTICLE_SPARKS_OPACITY_DEFAULT"
    PARTICLE_SPARKS_AUDIO_GAIN="$PARTICLE_SPARKS_AUDIO_GAIN_DEFAULT"
    PARTICLE_SPARKS_AUDIO_SMOOTH="$PARTICLE_SPARKS_AUDIO_SMOOTH_DEFAULT"
    PARTICLE_SPARKS_COLOR_MODE="$PARTICLE_SPARKS_COLOR_MODE_DEFAULT"
    PARTICLE_SPARKS_EMMONS_SPIN_SPEED="$PARTICLE_SPARKS_EMMONS_SPIN_SPEED_DEFAULT"
    PARTICLE_SPARKS_REPEL_STRENGTH="$PARTICLE_SPARKS_REPEL_STRENGTH_DEFAULT"
    PARTICLE_SPARKS_REPEL_RADIUS="$PARTICLE_SPARKS_REPEL_RADIUS_DEFAULT"
    PARTICLE_SPARKS_COLOR_RGB="$PARTICLE_SPARKS_COLOR_RGB_DEFAULT"
    PARTICLE_SPARKS_FLOOD_IN_OUT="$PARTICLE_SPARKS_FLOOD_IN_OUT_DEFAULT"
    PARTICLE_SPARKS_FLOOD_SECONDS="$PARTICLE_SPARKS_FLOOD_SECONDS_DEFAULT"
    PARTICLE_SPARKS_FLOOD_SPAWN_MULT="$PARTICLE_SPARKS_FLOOD_SPAWN_MULT_DEFAULT"
    PARTICLE_SPARKS_FLOOD_EXTRA_SOURCES="$PARTICLE_SPARKS_FLOOD_EXTRA_SOURCES_DEFAULT"
    PARTICLE_SPARKS_FLOOD_VELOCITY_MULT="$PARTICLE_SPARKS_FLOOD_VELOCITY_MULT_DEFAULT"
    TITLE_HOOK_DURATION="$TITLE_HOOK_DURATION_DEFAULT"
    TITLE_HOOK_DURATION_600="$TITLE_HOOK_DURATION_600_DEFAULT"
    TITLE_HOOK_FADE_OUT_DURATION="$TITLE_HOOK_FADE_OUT_DURATION_DEFAULT"
    TITLE_HOOK_TITLE="$TITLE_HOOK_TITLE_DEFAULT"
    TITLE_HOOK_SECONDARY_TEXT="$TITLE_HOOK_SECONDARY_TEXT_DEFAULT"
    TITLE_HOOK_LOGO="$TITLE_HOOK_LOGO_DEFAULT"
    TITLE_HOOK_LOGO_ALPHA_THRESHOLD="$TITLE_HOOK_LOGO_ALPHA_THRESHOLD_DEFAULT"
    TITLE_HOOK_LOGO_INTENSITY="$TITLE_HOOK_LOGO_INTENSITY_DEFAULT"
    TITLE_HOOK_LOGO_IDLE_WIGGLE="$TITLE_HOOK_LOGO_IDLE_WIGGLE_DEFAULT"
    TITLE_HOOK_LOGO_X_RATIO="$TITLE_HOOK_LOGO_X_RATIO_DEFAULT"
    TITLE_HOOK_LOGO_Y_RATIO="$TITLE_HOOK_LOGO_Y_RATIO_DEFAULT"
    TITLE_HOOK_LOGO_MOTION_TRACK_SCALE="$TITLE_HOOK_LOGO_MOTION_TRACK_SCALE_DEFAULT"
    TITLE_HOOK_LOGO_MOTION_TRACK_RADIUS="$TITLE_HOOK_LOGO_MOTION_TRACK_RADIUS_DEFAULT"
    TITLE_HOOK_LOGO_MOTION_TRACK_LINK_NEIGHBORS="$TITLE_HOOK_LOGO_MOTION_TRACK_LINK_NEIGHBORS_DEFAULT"
    TITLE_HOOK_LOGO_MOTION_TRACK_MIN_DISTANCE="$TITLE_HOOK_LOGO_MOTION_TRACK_MIN_DISTANCE_DEFAULT"
    TITLE_HOOK_LOGO_MOTION_TRACK_PAD_PX="$TITLE_HOOK_LOGO_MOTION_TRACK_PAD_PX_DEFAULT"
    TITLE_HOOK_LOGO_MOTION_TRACK_LINK_OPACITY="$TITLE_HOOK_LOGO_MOTION_TRACK_LINK_OPACITY_DEFAULT"
    TITLE_HOOK_LOGO_MOTION_TRACK_REFRESH="$TITLE_HOOK_LOGO_MOTION_TRACK_REFRESH_DEFAULT"
    TITLE_HOOK_LOGO_MOTION_TRACK_DECAY="$TITLE_HOOK_LOGO_MOTION_TRACK_DECAY_DEFAULT"
    TITLE_HOOK_LOGO_OPACITY="$TITLE_HOOK_LOGO_OPACITY_DEFAULT"
    TITLE_HOOK_BACKGROUND_DIM="$TITLE_HOOK_BACKGROUND_DIM_DEFAULT"
    TITLE_HOOK_TITLE_LAYER_DIM="$TITLE_HOOK_TITLE_LAYER_DIM_DEFAULT"
    TITLE_HOOK_TEXT_ALIGN="$TITLE_HOOK_TEXT_ALIGN_DEFAULT"
    TITLE_HOOK_TITLE_JITTER_AUDIO_MULTIPLIER="$TITLE_HOOK_TITLE_JITTER_AUDIO_MULTIPLIER_DEFAULT"
    TITLE_HOOK_SPARKS="$TITLE_HOOK_SPARKS_DEFAULT"
    TITLE_HOOK_SPARKS_RATE="$TITLE_HOOK_SPARKS_RATE_DEFAULT"
    TITLE_HOOK_SPARKS_MOTION_THRESHOLD="$TITLE_HOOK_SPARKS_MOTION_THRESHOLD_DEFAULT"
    TITLE_HOOK_SPARKS_OPACITY="$TITLE_HOOK_SPARKS_OPACITY_DEFAULT"
    JOBS="$JOBS_DEFAULT"
    PIPELINE_MODE="$PIPELINE_MODE_DEFAULT"
    ENABLE_GDRIVE_COPY="$ENABLE_GDRIVE_COPY_DEFAULT"
    GDRIVE_OUTDIR="$GDRIVE_OUTDIR_DEFAULT"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --force|-f) FORCE=1; shift ;;
            --end-only) PIPELINE_MODE="end-only"; shift ;;
            --mode) PIPELINE_MODE="$2"; shift 2 ;;
            --input) INPUT_VIDEO="$2"; shift 2 ;;
            --outdir) OUTDIR="$2"; shift 2 ;;
            --start-seconds) START_SECONDS="$2"; shift 2 ;;
            --youtube-full-seconds) YOUTUBE_FULL_SECONDS="$2"; shift 2 ;;
            --detect-audio-start-end) DETECT_AUDIO_START_END=1; shift ;;
            --no-detect-audio-start-end) DETECT_AUDIO_START_END=0; shift ;;
            --skip-reels-overlay) ENABLE_REELS_OVERLAY_STEP=0; shift ;;
            --use-reels-overlay) ENABLE_REELS_OVERLAY_STEP=1; shift ;;
            --reels-cache-mode) REELS_CACHE_MODE="$2"; shift 2 ;;
            --base-reels-overlay-prebuilt) BASE_REELS_OVERLAY_PREBUILT="$2"; shift 2 ;;
            --no-reels-bootstrap-existing-cache) REELS_BOOTSTRAP_EXISTING_CACHE=0; shift ;;
            --no-pre-reels-glitchfield) ENABLE_PRE_REELS_GLITCHFIELD_STAGE=0; shift ;;
            --pre-reels-glitchfield-preset) PRE_REELS_GLITCHFIELD_PRESET="$2"; shift 2 ;;
            --pre-reels-glitchfield-seed) PRE_REELS_GLITCHFIELD_SEED="$2"; shift 2 ;;
            --pre-reels-glitchfield-min-gate-period) PRE_REELS_GLITCHFIELD_MIN_GATE_PERIOD="$2"; shift 2 ;;
            --pre-reels-glitchfield-custom-args) PRE_REELS_GLITCHFIELD_CUSTOM_ARGS="$2"; shift 2 ;;
            --no-pre-reels-glitchfield-cache) USE_PRE_REELS_GLITCHFIELD_CACHE=0; shift ;;
            --use-particle-sparks) ENABLE_PARTICLE_SPARKS_STAGE=1; shift ;;
            --skip-particle-sparks) ENABLE_PARTICLE_SPARKS_STAGE=0; shift ;;
            --no-particle-sparks-cache) USE_PARTICLE_SPARKS_CACHE=0; shift ;;
            --use-title-hook) ENABLE_TITLE_HOOK_STAGE=1; shift ;;
            --skip-title-hook) ENABLE_TITLE_HOOK_STAGE=0; shift ;;
            --no-title-hook-cache) USE_TITLE_HOOK_CACHE=0; shift ;;
            --particle-sparks-max-points) PARTICLE_SPARKS_MAX_POINTS="$2"; shift 2 ;;
            --particle-sparks-motion-threshold) PARTICLE_SPARKS_MOTION_THRESHOLD="$2"; shift 2 ;;
            --particle-sparks-rate) PARTICLE_SPARKS_RATE="$2"; shift 2 ;;
            --particle-sparks-size) PARTICLE_SPARKS_SIZE="$2"; shift 2 ;;
            --particle-sparks-life-frames) PARTICLE_SPARKS_LIFE_FRAMES="$2"; shift 2 ;;
            --particle-sparks-speed) PARTICLE_SPARKS_SPEED="$2"; shift 2 ;;
            --particle-sparks-opacity) PARTICLE_SPARKS_OPACITY="$2"; shift 2 ;;
            --particle-sparks-audio-gain) PARTICLE_SPARKS_AUDIO_GAIN="$2"; shift 2 ;;
            --particle-sparks-color-mode) PARTICLE_SPARKS_COLOR_MODE="$2"; shift 2 ;;
            --particle-sparks-color-rgb) PARTICLE_SPARKS_COLOR_RGB="$2"; shift 2 ;;
            --particle-sparks-flood-in-out) PARTICLE_SPARKS_FLOOD_IN_OUT="$2"; shift 2 ;;
            --particle-sparks-flood-seconds) PARTICLE_SPARKS_FLOOD_SECONDS="$2"; shift 2 ;;
            --particle-sparks-flood-spawn-mult) PARTICLE_SPARKS_FLOOD_SPAWN_MULT="$2"; shift 2 ;;
            --particle-sparks-flood-extra-sources) PARTICLE_SPARKS_FLOOD_EXTRA_SOURCES="$2"; shift 2 ;;
            --particle-sparks-flood-velocity-mult) PARTICLE_SPARKS_FLOOD_VELOCITY_MULT="$2"; shift 2 ;;
            --cps) CPS="$2"; shift 2 ;;
            --divvy-sampling-mode) DIVVY_SAMPLING_MODE="$2"; shift 2 ;;
            --divvy-groove-bpm) DIVVY_GROOVE_BPM="$2"; shift 2 ;;
            --preview-reels-overlay) PREVIEW_REELS_OVERLAY=1; shift ;;
            --preview-no-reels-overlay) PREVIEW_REELS_OVERLAY=0; shift ;;
            --glitch-seconds) GLITCH_SECONDS="$2"; shift 2 ;;
            --loop-seam-seconds) LOOP_SEAM_SECONDS="$2"; shift 2 ;;
            --jobs|-j) JOBS="$2"; shift 2 ;;
            --no-reels-cache) USE_REELS_CACHE=0; shift ;;
            --no-glitchfield-cache) USE_GLITCHFIELD_CACHE=0; shift ;;
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
    PIPELINE_LOG_TAG="${PIPELINE_LOG_TAG:-$PIPELINE_LOG_TAG_DEFAULT}"
    export VOIDSTAR_LOG_PREFIX="$PIPELINE_LOG_TAG"

    local pipeline_start_epoch pipeline_end_epoch
    pipeline_start_epoch="$(date +%s)"
    echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"

    if [[ -z "${OUTDIR}" ]]; then
        if [[ -n "$OUTDIR_DEFAULT" ]]; then
            OUTDIR=$(eval echo "$OUTDIR_DEFAULT")
        else
            OUTDIR=$(eval echo "~/WinVideos/atomism")
        fi
    else
        OUTDIR=$(eval echo "$OUTDIR")
    fi

    if [[ -n "${BASE_REELS_OVERLAY_PREBUILT:-}" ]]; then
        BASE_REELS_OVERLAY_PREBUILT=$(eval echo "$BASE_REELS_OVERLAY_PREBUILT")
    fi

    if [[ "$ENABLE_GDRIVE_COPY" -eq 1 && -n "${GDRIVE_OUTDIR:-}" ]]; then
        GDRIVE_OUTDIR=$(eval echo "$GDRIVE_OUTDIR")
    fi

    initialize_gdrive_mount_once

    case "$REELS_CACHE_MODE" in
        base|per-target) ;;
        *) die "Unknown --reels-cache-mode value: $REELS_CACHE_MODE (use base|per-target)" ;;
    esac

    case "$DIVVY_SAMPLING_MODE" in
        minute-averages|n-averages|recursive-halves|uniform-spread|groove) ;;
        *) die "Unknown --divvy-sampling-mode value: $DIVVY_SAMPLING_MODE (use minute-averages|n-averages|recursive-halves|uniform-spread|groove)" ;;
    esac

    case "${PARTICLE_SPARKS_FLOOD_IN_OUT}" in
        1|true|TRUE|True|yes|YES|on|ON) PARTICLE_SPARKS_FLOOD_IN_OUT=1 ;;
        0|false|FALSE|False|no|NO|off|OFF) PARTICLE_SPARKS_FLOOD_IN_OUT=0 ;;
        *) die "Unknown --particle-sparks-flood-in-out value: ${PARTICLE_SPARKS_FLOOD_IN_OUT} (use 1|0|true|false)" ;;
    esac

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
    echo "Args:  start=${start_dbg}s full=${full_dbg}s detect_audio=${detect_dbg} cps=${CPS} divvy_mode=${DIVVY_SAMPLING_MODE} divvy_bpm=${DIVVY_GROOVE_BPM} glitch=${GLITCH_SECONDS}s loop_seam=${LOOP_SEAM_SECONDS}s"
    echo "Perf:  pre_reels_glitchfield=${ENABLE_PRE_REELS_GLITCHFIELD_STAGE} pre_reels_glitchfield_cache=${USE_PRE_REELS_GLITCHFIELD_CACHE} reels_overlay=${ENABLE_REELS_OVERLAY_STEP} reels_cache=${USE_REELS_CACHE} reels_cache_mode=${REELS_CACHE_MODE} particle_sparks=${ENABLE_PARTICLE_SPARKS_STAGE} particle_sparks_cache=${USE_PARTICLE_SPARKS_CACHE} particle_sparks_color_mode=${PARTICLE_SPARKS_COLOR_MODE} title_hook=${ENABLE_TITLE_HOOK_STAGE} title_hook_cache=${USE_TITLE_HOOK_CACHE} glitchfield=${ENABLE_GLITCHFIELD_STAGE} glitchfield_cache=${USE_GLITCHFIELD_CACHE} jobs=${JOBS}"
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
    TITLE_HOOK_SCRIPT=$(find_script "voidstar_title_hook.py")
    GLITCHFIELD=$(find_script "glitchfield.py")
    PARTICLE_SPARKS=$(find_script "voidstar_particle_sparks.py")
    require_file "DIVVY" "$DIVVY"
    require_file "REELS_OVERLAY" "$REELS_OVERLAY"
    require_file "DVDLOGO" "$DVDLOGO"
    require_file "GLITCHFIELD" "$GLITCHFIELD"
    require_file "PARTICLE_SPARKS" "$PARTICLE_SPARKS"
    if [[ "$ENABLE_TITLE_HOOK_STAGE" -eq 1 ]]; then
        require_file "TITLE_HOOK_SCRIPT" "$TITLE_HOOK_SCRIPT"
    fi

    LOGO_START="$(eval echo "$LOGO_START")"
    LOGO_END="$(eval echo "$LOGO_END")"
    TITLE_HOOK_LOGO="$(eval echo "$TITLE_HOOK_LOGO")"
    require_file "LOGO_START" "$LOGO_START"
    require_file "LOGO_END" "$LOGO_END"
    if [[ "$ENABLE_TITLE_HOOK_STAGE" -eq 1 ]]; then
        require_file "TITLE_HOOK_LOGO" "$TITLE_HOOK_LOGO"
    fi
    echo "Using logo mapping:"
    echo "  *s/start -> $LOGO_START"
    echo "  *t/end   -> $LOGO_END"

    BASE_REELS_OVERLAY="$OUTDIR/${STEM}_reels_base_overlay.mp4"
    if [[ -n "${BASE_REELS_OVERLAY_PREBUILT:-}" ]]; then
        BASE_REELS_OVERLAY="$BASE_REELS_OVERLAY_PREBUILT"
    fi
    REELS_INPUT_VIDEO="$INPUT_VIDEO"
    local pre_reels_glitch_dst="$OUTDIR/${STEM}_pre_reels_glitchfield_${PRE_REELS_GLITCHFIELD_PRESET}.mp4"
    REELS_INPUT_VIDEO="$(run_optional_pre_reels_glitchfield "$INPUT_VIDEO" "$pre_reels_glitch_dst")"
    if [[ "$ENABLE_REELS_OVERLAY_STEP" -eq 1 && "$REELS_CACHE_MODE" == "base" ]]; then
        build_base_reels_overlay
    elif [[ "$ENABLE_REELS_OVERLAY_STEP" -eq 1 ]]; then
        echo "[pipeline] reels overlay in per-target mode (base precompute skipped)"
        BASE_REELS_OVERLAY="$REELS_INPUT_VIDEO"
    else
        echo "[pipeline] reels overlay step bypassed (using pre-reels source video directly)"
        BASE_REELS_OVERLAY="$REELS_INPUT_VIDEO"
    fi

    local mode_csv mode_part
    local -a mode_parts
    local mode_preview=0 mode_custom=0 mode_end_only=0 mode_all=0
    mode_csv="${PIPELINE_MODE// /}"
    IFS=',' read -r -a mode_parts <<< "$mode_csv"
    for mode_part in "${mode_parts[@]}"; do
        case "$mode_part" in
            preview) mode_preview=1 ;;
            custom) mode_custom=1 ;;
            end-only) mode_end_only=1 ;;
            all) mode_all=1 ;;
            "") ;;
            *) die "Unknown --mode token: $mode_part (use preview|custom|end-only|all, comma-separated)" ;;
        esac
    done

    if (( mode_preview == 1 )); then
        echo "--- Preview phase (quick 60s start/end) ---"
        local reels_overlay_saved="$ENABLE_REELS_OVERLAY_STEP"
        if [[ "$PREVIEW_REELS_OVERLAY" -eq 0 ]]; then
            ENABLE_REELS_OVERLAY_STEP=0
        fi
        run_60s_start
        run_60s_end
        ENABLE_REELS_OVERLAY_STEP="$reels_overlay_saved"
    fi

    declare -a TARGETS=()
    if (( mode_custom == 1 )); then
        (( RUN_60S_START == 1 )) && TARGETS+=(run_60s_start)
        (( RUN_60S_END == 1 )) && TARGETS+=(run_60s_end)
        (( RUN_180S_START == 1 )) && TARGETS+=(run_180s_start)
        (( RUN_180S_END == 1 )) && TARGETS+=(run_180s_end)
        (( RUN_FULL == 1 )) && TARGETS+=(run_full)
        (( ${#TARGETS[@]} > 0 )) || die "PIPELINE_MODE includes custom but no RUN_* targets enabled"
    elif (( mode_end_only == 1 )); then
        TARGETS=(run_60s_end)
    elif (( mode_all == 1 )); then
        TARGETS=(run_60s_start run_60s_end run_180s_start run_180s_end run_full)
    fi

    if (( ${#TARGETS[@]} > 0 )); then
        echo "Targets: ${TARGETS[*]}"

        if (( JOBS <= 1 )); then
            for fn in "${TARGETS[@]}"; do "$fn"; done
        else
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
        fi
    elif (( mode_preview == 0 )); then
        die "No targets selected for PIPELINE_MODE=$PIPELINE_MODE"
    fi

    pipeline_end_epoch="$(date +%s)"
    echo "End:   $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Elapsed: $((pipeline_end_epoch - pipeline_start_epoch))s"
}

main "$@"
