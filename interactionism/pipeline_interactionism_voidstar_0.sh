#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ATOMISM_BASE="$ROOT_DIR/atomism/pipeline_atomism_voidstar_base.sh"
CONFIG_FILE="$SCRIPT_DIR/config/pipeline_interactionism_voidstar_0.conf.sh"

PROJECT_DIR_DEFAULT="~/WinVideos/interactionism"
SOURCES_DIR_DEFAULT="~/WinVideos/interactionism/sources"
OUTDIR_DEFAULT="~/WinVideos/interactionism"
COMBINED_INPUT_DEFAULT="~/WinVideos/interactionism/interactionism_sources_combined.mp4"

PROJECT_DIR=""
SOURCES_DIR=""
OUTDIR=""
COMBINED_INPUT=""
REBUILD_COMBINED=0
RUN_INDIVIDUAL=0
INDIVIDUAL_ONLY=0

declare -a BASE_ARGS=()
declare -a SOURCE_FILES=()

die() { echo "Error: $*" >&2; exit 1; }
require_cmd() { command -v "$1" >/dev/null 2>&1 || die "Missing command: $1"; }

usage() {
    cat <<'EOF'
interactionism pipeline (multi-source, atomism-based)

Defaults:
  project-dir:    ~/WinVideos/interactionism
  sources-dir:    ~/WinVideos/interactionism/sources
  combined-input: ~/WinVideos/interactionism/interactionism_sources_combined.mp4
  outdir:         ~/WinVideos/interactionism

Behavior:
  - Builds a single combined source clip from all videos under sources-dir.
  - Runs the standard 60s/180s/full target series via atomism base pipeline.
  - Uses interactionism config defaults (shobud particle theme, reels overlay off).

Options:
  --project-dir PATH       Project folder root.
  --sources-dir PATH       Folder containing source clips (recursive scan).
  --combined-input PATH    Path for generated combined source clip.
  --outdir PATH            Output folder for combined-series outputs.
  --rebuild-combined       Force rebuild of normalized/combined source clip.
  --individual             Also render full target series for each source clip.
  --individual-only        Skip combined render and only render per-source series.
  --no-individual          Disable per-source renders.
  -h, --help               Show this help.

Any other args are forwarded to atomism base pipeline (for example: --mode, --jobs, --force).
EOF
}

expand_path() {
    local raw="$1"
    case "$raw" in
        "~")
            printf '%s\n' "$HOME"
            ;;
        "~/"*)
            printf '%s/%s\n' "$HOME" "${raw:2}"
            ;;
        *)
            printf '%s\n' "$raw"
            ;;
    esac
}

video_list_from_sources() {
    local dir="$1"
    mapfile -t SOURCE_FILES < <(
        find "$dir" -type f \( \
            -iname '*.mp4' -o \
            -iname '*.mov' -o \
            -iname '*.mkv' -o \
            -iname '*.m4v' -o \
            -iname '*.avi' -o \
            -iname '*.webm' \
        \) | sort
    )
}

source_has_audio() {
    local src="$1"
    ffprobe -v error -select_streams a -show_entries stream=index -of csv=p=0 "$src" | grep -q .
}

combined_is_fresh() {
    local target="$1"
    [[ -f "$target" ]] || return 1

    local src
    for src in "${SOURCE_FILES[@]}"; do
        if [[ "$src" -nt "$target" ]]; then
            return 1
        fi
    done
    return 0
}

build_combined_clip() {
    local target="$1"
    local work_dir="$PROJECT_DIR/_interactionism_tmp"
    local norm_dir="$work_dir/normalized"
    local concat_list="$work_dir/concat_list.txt"

    mkdir -p "$work_dir" "$norm_dir"

    if [[ "$REBUILD_COMBINED" -eq 0 ]] && combined_is_fresh "$target"; then
        echo "[interactionism] combined source is up-to-date: $target"
        return 0
    fi

    echo "[interactionism] normalizing ${#SOURCE_FILES[@]} source clips..."
    : > "$concat_list"

    local idx=0
    local src norm base
    for src in "${SOURCE_FILES[@]}"; do
        idx=$((idx + 1))
        base="$(basename "${src%.*}")"
        norm="$norm_dir/$(printf '%04d' "$idx")_${base}.mp4"

        if [[ "$REBUILD_COMBINED" -eq 0 && -f "$norm" && "$norm" -nt "$src" ]]; then
            echo "[interactionism] reuse normalized: $norm"
        else
            echo "[interactionism] normalize [$idx/${#SOURCE_FILES[@]}]: $src"
            if source_has_audio "$src"; then
                ffmpeg -y -hide_banner -loglevel error \
                    -i "$src" \
                    -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30" \
                    -map 0:v:0 -map 0:a:0 \
                    -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p \
                    -c:a aac -ar 48000 -ac 2 \
                    "$norm"
            else
                ffmpeg -y -hide_banner -loglevel error \
                    -i "$src" -f lavfi -i anullsrc=r=48000:cl=stereo \
                    -shortest \
                    -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30" \
                    -map 0:v:0 -map 1:a:0 \
                    -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p \
                    -c:a aac -ar 48000 -ac 2 \
                    "$norm"
            fi
        fi

        local escaped_norm
        escaped_norm=${norm//\'/\'\\\'\'}
        printf "file '%s'\n" "$escaped_norm" >> "$concat_list"
    done

    echo "[interactionism] concatenating normalized clips -> $target"
    mkdir -p "$(dirname "$target")"

    if ! ffmpeg -y -hide_banner -loglevel error \
        -f concat -safe 0 -i "$concat_list" \
        -c copy \
        "$target"; then
        echo "[interactionism] concat copy failed, retrying with re-encode"
        ffmpeg -y -hide_banner -loglevel error \
            -f concat -safe 0 -i "$concat_list" \
            -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p \
            -c:a aac -ar 48000 -ac 2 \
            "$target"
    fi
}

run_series_for_input() {
    local input_video="$1"
    local outdir="$2"

    mkdir -p "$outdir"
    echo "[interactionism] rendering series for: $input_video"
    "$ATOMISM_BASE" \
        --config "$CONFIG_FILE" \
        --input "$input_video" \
        --outdir "$outdir" \
        --skip-reels-overlay \
        "${BASE_ARGS[@]}"
}

main() {
    require_cmd ffmpeg
    require_cmd ffprobe

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --project-dir)
                [[ $# -ge 2 ]] || die "--project-dir requires a value"
                PROJECT_DIR="$2"
                shift 2
                ;;
            --sources-dir)
                [[ $# -ge 2 ]] || die "--sources-dir requires a value"
                SOURCES_DIR="$2"
                shift 2
                ;;
            --combined-input)
                [[ $# -ge 2 ]] || die "--combined-input requires a value"
                COMBINED_INPUT="$2"
                shift 2
                ;;
            --outdir)
                [[ $# -ge 2 ]] || die "--outdir requires a value"
                OUTDIR="$2"
                shift 2
                ;;
            --rebuild-combined)
                REBUILD_COMBINED=1
                shift
                ;;
            --individual)
                RUN_INDIVIDUAL=1
                shift
                ;;
            --individual-only)
                RUN_INDIVIDUAL=1
                INDIVIDUAL_ONLY=1
                shift
                ;;
            --no-individual)
                RUN_INDIVIDUAL=0
                INDIVIDUAL_ONLY=0
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                BASE_ARGS+=("$1")
                shift
                ;;
        esac
    done

    PROJECT_DIR="$(expand_path "${PROJECT_DIR:-$PROJECT_DIR_DEFAULT}")"
    SOURCES_DIR="$(expand_path "${SOURCES_DIR:-$SOURCES_DIR_DEFAULT}")"
    OUTDIR="$(expand_path "${OUTDIR:-$OUTDIR_DEFAULT}")"
    COMBINED_INPUT="$(expand_path "${COMBINED_INPUT:-$COMBINED_INPUT_DEFAULT}")"

    [[ -x "$ATOMISM_BASE" ]] || [[ -f "$ATOMISM_BASE" ]] || die "Missing atomism base pipeline: $ATOMISM_BASE"
    [[ -f "$CONFIG_FILE" ]] || die "Missing interactionism config: $CONFIG_FILE"
    [[ -d "$SOURCES_DIR" ]] || die "Sources folder not found: $SOURCES_DIR"

    video_list_from_sources "$SOURCES_DIR"
    [[ ${#SOURCE_FILES[@]} -gt 0 ]] || die "No source videos found under: $SOURCES_DIR"

    echo "[interactionism] project: $PROJECT_DIR"
    echo "[interactionism] sources: $SOURCES_DIR"
    echo "[interactionism] clips found: ${#SOURCE_FILES[@]}"

    if [[ "$INDIVIDUAL_ONLY" -eq 0 ]]; then
        build_combined_clip "$COMBINED_INPUT"
        run_series_for_input "$COMBINED_INPUT" "$OUTDIR"
    fi

    if [[ "$RUN_INDIVIDUAL" -eq 1 ]]; then
        local idx=0
        local src stem indiv_out
        for src in "${SOURCE_FILES[@]}"; do
            idx=$((idx + 1))
            stem="$(basename "${src%.*}")"
            indiv_out="$OUTDIR/individual/$(printf '%03d' "$idx")_${stem}"
            run_series_for_input "$src" "$indiv_out"
        done
    fi

    echo "[interactionism] complete"
}

main "$@"
