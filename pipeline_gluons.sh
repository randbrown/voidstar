#!/usr/bin/env bash
# pipeline_gluons.sh
# Pipeline for generating target videos for the gluons_voidstar_0.mp4 project
#
# Usage:
#   ./pipeline_gluons.sh                # build all targets
#   ./pipeline_gluons.sh preview        # build only 60s START target
#   ./pipeline_gluons.sh --end-only     # build only END-anchored targets (60/90/180)
#   ./pipeline_gluons.sh --force        # rebuild even if up-to-date

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

# ----------------------------
# Targets
# ----------------------------

# 1. 60s highlight (START anchor)
run_60s_start() {
    echo "--- 60s highlight (START) ---"
    local target="$OUTDIR/gluons_voidstar_0_highlights_60s_overlay_logo.mp4"
    should_rebuild "$target" || return 0

    python3 "$DIVVY" highlights "$INPUT_VIDEO" \
        --start-seconds 120 --youtube-full-seconds 600 --target-length-seconds 60 \
        --video-encoder libx264 --preset medium --out-dir "$OUTDIR" \
        --glitch-seconds 2 --glitch-style vuwind --cps 0.5 --loop-seam-seconds 2 \
        --sampling-mode uniform-spread --n-segments 8 --sample-seconds 8 --truncate-to-full-clips

    python3 "$REELS_OVERLAY" "$OUTDIR/gluons_voidstar_0__highlights__mode-uniform-spread__start-120s__full-600s__target-60s.mp4" \
        --min-det-conf 0.05 --min-trk-conf 0.05 --draw-ids true \
        --smear true --smear-frames 17 --smear-decay 0.99 \
        --trail true --trail-alpha .999 --beat-sync true \
        --velocity-color true --velocity-color-mult 10

    rename_output "$OUTDIR/gluons_voidstar_0__highlights__mode-uniform-spread__start-120s__full-600s__target-60s_fps30_mc2_det0p05_trk0p05_trail1_ta1p00_tlotrue_scan1_velcolor_ids_beatsync_smear.mp4" \
        "$OUTDIR/gluons_voidstar_0_highlights_60s_overlay.mp4"

    python3 "$DVDLOGO" "$OUTDIR/gluons_voidstar_0_highlights_60s_overlay.mp4" "$LOGO_IMG" \
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
    local target="$OUTDIR/gluons_voidstar_0_highlights_90s_overlay_logo.mp4"
    should_rebuild "$target" || return 0

    python3 "$DIVVY" highlights "$INPUT_VIDEO" \
        --start-seconds 0 --youtube-full-seconds 834 --target-length-seconds 90 \
        --video-encoder libx264 --preset medium --out-dir "$OUTDIR" \
        --glitch-seconds 2 --glitch-style vuwind --cps 0.5 --loop-seam-seconds 2 \
        --sampling-mode uniform-spread --sample-seconds 16 --truncate-to-full-clips

    python3 "$REELS_OVERLAY" "$OUTDIR/gluons_voidstar_0__highlights__mode-uniform-spread__start-0s__full-834s__target-90s.mp4" \
        --min-det-conf 0.05 --min-trk-conf 0.05 --draw-ids true \
        --smear true --smear-frames 17 --smear-decay 0.99 \
        --trail true --trail-alpha .999 --beat-sync true \
        --velocity-color true --velocity-color-mult 10

    rename_output "$OUTDIR/gluons_voidstar_0__highlights__mode-uniform-spread__start-0s__full-834s__target-90s_fps30_mc2_det0p05_trk0p05_trail1_ta1p00_tlotrue_scan1_velcolor_ids_beatsync_smear.mp4" \
        "$OUTDIR/gluons_voidstar_0_highlights_90s_overlay.mp4"

    python3 "$DVDLOGO" "$OUTDIR/gluons_voidstar_0_highlights_90s_overlay.mp4" "$LOGO_IMG" \
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
    local target="$OUTDIR/gluons_voidstar_0_highlights_180s_overlay_logo.mp4"
    should_rebuild "$target" || return 0

    python3 "$DIVVY" highlights "$INPUT_VIDEO" \
        --start-seconds 0 --youtube-full-seconds 834 --target-length-seconds 180 \
        --video-encoder libx264 --preset medium --out-dir "$OUTDIR" \
        --glitch-seconds 2 --glitch-style vuwind --cps 0.5 --loop-seam-seconds 2 \
        --sampling-mode uniform-spread --n-segments 6 --sample-seconds 32 --truncate-to-full-clips

    python3 "$REELS_OVERLAY" "$OUTDIR/gluons_voidstar_0__highlights__mode-uniform-spread__start-0s__full-834s__target-180s.mp4" \
        --min-det-conf 0.05 --min-trk-conf 0.05 --draw-ids true \
        --smear true --smear-frames 17 --smear-decay 0.99 \
        --trail true --trail-alpha .999 --beat-sync true \
        --velocity-color true --velocity-color-mult 10

    rename_output "$OUTDIR/gluons_voidstar_0__highlights__mode-uniform-spread__start-0s__full-834s__target-180s_fps30_mc2_det0p05_trk0p05_trail1_ta1p00_tlotrue_scan1_velcolor_ids_beatsync_smear.mp4" \
        "$OUTDIR/gluons_voidstar_0_highlights_180s_overlay.mp4"

    python3 "$DVDLOGO" "$OUTDIR/gluons_voidstar_0_highlights_180s_overlay.mp4" "$LOGO_IMG" \
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
    local target="$OUTDIR/gluons_voidstar_0_full_overlay_logo.mp4"
    local prebuilt_overlay="$(eval echo \"${PREBUILT_FULL_OVERLAY:-~/WinVideos/gluons_voidstar_0_full_overlay.mp4}\")"
    local overlay_output="$OUTDIR/gluons_voidstar_0_full_overlay.mp4"

    # If prebuilt overlay exists and is newer than the script, use it
    if [[ -f "$prebuilt_overlay" ]]; then
        if [[ "$0" -nt "$prebuilt_overlay" ]]; then
            echo "Prebuilt overlay $prebuilt_overlay is older than script, will rerun overlay."
        else
            echo "Using prebuilt overlay: $prebuilt_overlay"
            cp -f "$prebuilt_overlay" "$overlay_output"
            python3 "$DVDLOGO" "$overlay_output" "$LOGO_IMG" \
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
            return 0
        fi
    fi

    should_rebuild "$overlay_output" || return 0

    python3 "$REELS_OVERLAY" "$INPUT_VIDEO" \
        --min-det-conf 0.05 --min-trk-conf 0.05 --draw-ids true \
        --smear true --smear-frames 17 --smear-decay 0.99 \
        --trail true --trail-alpha .999 --beat-sync true \
        --velocity-color true --velocity-color-mult 10

    rename_output "$OUTDIR/gluons_voidstar_0_fps30_mc2_det0p05_trk0p05_trail1_ta1p00_tlotrue_scan1_velcolor_ids_beatsync_smear.mp4" \
        "$overlay_output"

    python3 "$DVDLOGO" "$overlay_output" "$LOGO_IMG" \
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
    local target="$OUTDIR/gluons_voidstar_0_highlights_60t_overlay_logo.mp4"
    should_rebuild "$target" || return 0

    python3 "$DIVVY" highlights "$INPUT_VIDEO" \
        --start-seconds 120 --youtube-full-seconds 600 --target-length-seconds 60 \
        --video-encoder libx264 --preset medium --out-dir "$OUTDIR" \
        --glitch-seconds 2 --glitch-style vuwind --cps 0.5 --loop-seam-seconds 2 \
        --sampling-mode uniform-spread --n-segments 8 --sample-seconds 8 --sample-anchor end \
        --truncate-to-full-clips

    rename_output "$OUTDIR/gluons_voidstar_0__highlights__mode-uniform-spread__start-120s__full-600s__target-60s.mp4" \
        "$OUTDIR/gluons_voidstar_0_highlights_60t.mp4"

    python3 "$REELS_OVERLAY" "$OUTDIR/gluons_voidstar_0_highlights_60t.mp4" \
        --min-det-conf 0.05 --min-trk-conf 0.05 --draw-ids true \
        --smear true --smear-frames 17 --smear-decay 0.99 \
        --trail true --trail-alpha .999 --beat-sync true \
        --velocity-color true --velocity-color-mult 10 \
        --output "$OUTDIR/gluons_voidstar_0_highlights_60t_overlay.mp4"

    rename_output "$OUTDIR/gluons_voidstar_0_highlights_60t_fps30_mc2_det0p05_trk0p05_trail1_ta1p00_tlotrue_scan1_velcolor_ids_beatsync_smear.mp4" \
        "$OUTDIR/gluons_voidstar_0_highlights_60t_overlay.mp4"

    python3 "$DVDLOGO" "$OUTDIR/gluons_voidstar_0_highlights_60t_overlay.mp4" "$LOGO_IMG" \
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
    local target="$OUTDIR/gluons_voidstar_0_highlights_90t_overlay_logo.mp4"
    should_rebuild "$target" || return 0

    python3 "$DIVVY" highlights "$INPUT_VIDEO" \
        --start-seconds 0 --youtube-full-seconds 834 --target-length-seconds 90 \
        --video-encoder libx264 --preset medium --out-dir "$OUTDIR" \
        --glitch-seconds 2 --glitch-style vuwind --cps 0.5 --loop-seam-seconds 2 \
        --sampling-mode uniform-spread --sample-seconds 16 --sample-anchor end --truncate-to-full-clips

    rename_output "$OUTDIR/gluons_voidstar_0__highlights__mode-uniform-spread__start-0s__full-834s__target-90s.mp4" \
        "$OUTDIR/gluons_voidstar_0_highlights_90t.mp4"

    python3 "$REELS_OVERLAY" "$OUTDIR/gluons_voidstar_0_highlights_90t.mp4" \
        --min-det-conf 0.05 --min-trk-conf 0.05 --draw-ids true \
        --smear true --smear-frames 17 --smear-decay 0.99 \
        --trail true --trail-alpha .999 --beat-sync true \
        --velocity-color true --velocity-color-mult 10 \
        --output "$OUTDIR/gluons_voidstar_0_highlights_90t_overlay.mp4"

    rename_output "$OUTDIR/gluons_voidstar_0_highlights_90t_fps30_mc2_det0p05_trk0p05_trail1_ta1p00_tlotrue_scan1_velcolor_ids_beatsync_smear.mp4" \
        "$OUTDIR/gluons_voidstar_0_highlights_90t_overlay.mp4"

    python3 "$DVDLOGO" "$OUTDIR/gluons_voidstar_0_highlights_90t_overlay.mp4" "$LOGO_IMG" \
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
    local target="$OUTDIR/gluons_voidstar_0_highlights_180t_overlay_logo.mp4"
    should_rebuild "$target" || return 0

    python3 "$DIVVY" highlights "$INPUT_VIDEO" \
        --start-seconds 0 --youtube-full-seconds 834 --target-length-seconds 180 \
        --video-encoder libx264 --preset medium --out-dir "$OUTDIR" \
        --glitch-seconds 2 --glitch-style vuwind --cps 0.5 --loop-seam-seconds 2 \
        --sampling-mode uniform-spread --n-segments 6 --sample-seconds 32 --sample-anchor end --truncate-to-full-clips

    rename_output "$OUTDIR/gluons_voidstar_0__highlights__mode-uniform-spread__start-0s__full-834s__target-180s.mp4" \
        "$OUTDIR/gluons_voidstar_0_highlights_180t.mp4"

    python3 "$REELS_OVERLAY" "$OUTDIR/gluons_voidstar_0_highlights_180t.mp4" \
        --min-det-conf 0.05 --min-trk-conf 0.05 --draw-ids true \
        --smear true --smear-frames 17 --smear-decay 0.99 \
        --trail true --trail-alpha .999 --beat-sync true \
        --velocity-color true --velocity-color-mult 10 \
        --output "$OUTDIR/gluons_voidstar_0_highlights_180t_overlay.mp4"

    rename_output "$OUTDIR/gluons_voidstar_0_highlights_180t_fps30_mc2_det0p05_trk0p05_trail1_ta1p00_tlotrue_scan1_velcolor_ids_beatsync_smear.mp4" \
        "$OUTDIR/gluons_voidstar_0_highlights_180t_overlay.mp4"

    python3 "$DVDLOGO" "$OUTDIR/gluons_voidstar_0_highlights_180t_overlay.mp4" "$LOGO_IMG" \
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
    INPUT_VIDEO=$(eval echo "${INPUT_VIDEO:-~/WinVideos/gluons_voidstar_0.mp4}")
    OUTDIR=$(eval echo "${OUTDIR:-~/WinVideos/gluons_voidstar_0/}")

    # Auto-detect required scripts and logo image
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
    find_logo() {
        local found
        found=$(find "$PROJECT_ROOT" -type f -iname "*logo*.png" | head -n 1)
        if [[ -z "$found" ]]; then
            die "Could not find logo image (*.png with 'logo' in name) in $PROJECT_ROOT"
        fi
        echo "$found"
    }
    DIVVY=$(find_script "divvy.py")
    REELS_OVERLAY=$(find_script "reels_cv_overlay.py")
    DVDLOGO=$(find_script "voidstar_dvd_logo.py")
    LOGO_IMG=$(find_logo)

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --force|-f) FORCE=1; shift ;;
            --end-only) END_ONLY=1; shift ;;
            preview)    PREVIEW=1; shift ;;
            --input)    INPUT_VIDEO="$2"; shift 2 ;;
            --outdir)   OUTDIR="$2"; shift 2 ;;
            -h|--help)
                cat <<'EOF'
Usage: ./pipeline_gluons.sh [preview] [--end-only] [--force] [--input <video_path>] [--outdir <output_dir>]

  preview     Build only the 60s START target (fast iteration)
  --end-only  Build only the END-anchored targets (60/90/180)
  --force     Rebuild even if targets are up-to-date
  --input     Specify input video file location (default: ~/WinVideos/gluons_voidstar_0.mp4)
  --outdir    Specify output directory (default: ~/WinVideos/gluons_voidstar_0/)
EOF
                return 0
                ;;
            *) die "Unknown arg: $1 (try --help)" ;;
        esac
    done

    mkdir -p "$OUTDIR"

    # Preflight
    require_cmd python3
    require_file "INPUT_VIDEO" "$INPUT_VIDEO"
    require_file "DIVVY" "$DIVVY"
    require_file "REELS_OVERLAY" "$REELS_OVERLAY"
    require_file "DVDLOGO" "$DVDLOGO"
    require_file "LOGO_IMG" "$LOGO_IMG"

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
