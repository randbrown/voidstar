#!/usr/bin/env python3
"""
voidstar_motion_mesh.py

Voidstar-style OpenCV overlay:
- Motion-gated feature tracking (Lucas–Kanade optical flow)
- Delaunay triangulation mesh (constellation glass)
- Audio-reactive modulation (RMS + bass + simple beat pulses)
- Trail buffer w/ decay
- FFmpeg-based trim + mux (keeps output naming + status logs)

Example:
  python voidstar_motion_mesh.py input.mp4 --start 60 --duration 20 --fps 30 --trail true --trail-alpha 0.92 --glow 7
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.io import wavfile


# ----------------------------
# Logging helpers
# ----------------------------

def log(msg: str) -> None:
    print(f"[voidstar] {msg}", flush=True)


def run_cmd(cmd: List[str]) -> None:
    log("▶ " + " ".join(shlex.quote(c) for c in cmd))
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {p.returncode}")


def safe_slug(s: str) -> str:
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^a-zA-Z0-9._-]+", "", s)
    return s


def fmt_num(x: float, digits: int = 2) -> str:
    # 12.34 -> "12p34" (matches your usual style)
    return f"{x:.{digits}f}".replace(".", "p")


def parse_rgb(s: str) -> Tuple[int, int, int]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError("overlay color must be 'R,G,B'")
    r, g, b = (int(parts[0]), int(parts[1]), int(parts[2]))
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    return r, g, b


# ----------------------------
# Audio analysis
# ----------------------------

@dataclass
class AudioFrameFeatures:
    rms: np.ndarray         # shape [nframes]
    bass: np.ndarray        # shape [nframes]
    beat_pulse: np.ndarray  # shape [nframes], 0..1


def extract_wav_mono_48k(input_path: str, start: float, duration: float, wav_path: str) -> None:
    # Extract the SAME segment we will mux, to keep sync clean.
    # -accurate_seek behavior varies; this is generally "good enough" and consistent.
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", str(start),
        "-t", str(duration),
        "-i", input_path,
        "-vn",
        "-ac", "1",
        "-ar", "48000",
        wav_path,
    ]
    run_cmd(cmd)


def audio_features_for_fps(wav_path: str, fps: float) -> AudioFrameFeatures:
    sr, x = wavfile.read(wav_path)
    if x.ndim > 1:
        x = x[:, 0]
    # normalize to float32 [-1,1]
    if x.dtype == np.int16:
        xf = x.astype(np.float32) / 32768.0
    elif x.dtype == np.int32:
        xf = x.astype(np.float32) / 2147483648.0
    else:
        xf = x.astype(np.float32)
        m = np.max(np.abs(xf)) + 1e-9
        xf = xf / m

    hop = max(1, int(round(sr / fps)))
    win = max(hop * 2, 2048)
    win = int(2 ** math.ceil(math.log2(win)))  # power of 2
    window = np.hanning(win).astype(np.float32)

    nframes = max(1, int(math.ceil(len(xf) / hop)))
    rms = np.zeros(nframes, dtype=np.float32)
    bass = np.zeros(nframes, dtype=np.float32)

    # Frequency bins
    freqs = np.fft.rfftfreq(win, d=1.0 / sr)
    bass_band = (freqs >= 20) & (freqs < 120)  # sub+bass
    all_band = (freqs >= 20) & (freqs < 8000)

    # Simple spectral-flux-ish beat pulse
    prev_mag = None
    flux = np.zeros(nframes, dtype=np.float32)

    for i in range(nframes):
        a = i * hop
        b = a + win
        seg = np.zeros(win, dtype=np.float32)
        if a < len(xf):
            seg_len = min(win, len(xf) - a)
            seg[:seg_len] = xf[a:a + seg_len]
        seg *= window

        # RMS
        rms[i] = float(np.sqrt(np.mean(seg * seg) + 1e-9))

        # FFT magnitude
        mag = np.abs(np.fft.rfft(seg))
        total = float(np.sum(mag[all_band]) + 1e-9)
        bass[i] = float(np.sum(mag[bass_band]) / total)

        if prev_mag is not None:
            d = mag - prev_mag
            d[d < 0] = 0
            flux[i] = float(np.sum(d))
        prev_mag = mag

    # Normalize features
    def norm(v: np.ndarray) -> np.ndarray:
        lo = np.percentile(v, 5)
        hi = np.percentile(v, 95)
        vv = (v - lo) / (hi - lo + 1e-9)
        return np.clip(vv, 0.0, 1.0).astype(np.float32)

    rmsn = norm(rms)
    bassn = norm(bass)
    fluxn = norm(flux)

    # Beat pulse: peak-pick flux with refractory + softness
    beat = np.zeros_like(fluxn)
    refractory = 0
    for i in range(1, len(fluxn) - 1):
        if refractory > 0:
            refractory -= 1
            continue
        if fluxn[i] > 0.65 and fluxn[i] > fluxn[i - 1] and fluxn[i] > fluxn[i + 1]:
            beat[i] = 1.0
            refractory = int(0.18 * fps)  # ~180ms lockout

    # Spread beat pulse a bit (so it feels less “1 frame”)
    k = max(1, int(round(0.10 * fps)))  # 100ms kernel
    kernel = np.exp(-np.linspace(0, 2.5, k)).astype(np.float32)
    beat_soft = np.convolve(beat, kernel, mode="same")
    beat_soft = np.clip(beat_soft, 0.0, 1.0).astype(np.float32)

    return AudioFrameFeatures(rms=rmsn, bass=bassn, beat_pulse=beat_soft)


# ----------------------------
# Visual overlay core
# ----------------------------

@dataclass
class Track:
    tid: int
    pts: List[Tuple[float, float]]  # recent positions
    age: int
    miss: int


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def blend_add(dst: np.ndarray, src: np.ndarray, alpha: float) -> np.ndarray:
    # dst/src uint8 BGR
    if alpha <= 0:
        return dst
    if alpha >= 1:
        out = cv2.add(dst, src)
        return out
    srca = (src.astype(np.float32) * alpha).astype(np.uint8)
    out = cv2.add(dst, srca)
    return out


def apply_glow(img: np.ndarray, ksize: int) -> np.ndarray:
    ksize = int(max(0, ksize))
    if ksize <= 1:
        return img
    if ksize % 2 == 0:
        ksize += 1
    blur = cv2.GaussianBlur(img, (ksize, ksize), 0)
    out = cv2.addWeighted(img, 1.0, blur, 1.0, 0)
    return out


def draw_delaunay_lines(
    frame_shape: Tuple[int, int],
    points: List[Tuple[float, float]],
    thickness: int,
    color_bgr: Tuple[int, int, int],
) -> np.ndarray:
    h, w = frame_shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    if len(points) < 3:
        return overlay

    rect = (0, 0, w, h)
    subdiv = cv2.Subdiv2D(rect)

    for (x, y) in points:
        # Subdiv2D can throw if points are out-of-bounds; clamp them
        xx = max(0.0, min(float(w - 1), float(x)))
        yy = max(0.0, min(float(h - 1), float(y)))
        subdiv.insert((xx, yy))

    tris = subdiv.getTriangleList()
    # Each tri is [x1,y1,x2,y2,x3,y3]
    for t in tris:
        x1, y1, x2, y2, x3, y3 = map(float, t)
        if not (0 <= x1 < w and 0 <= x2 < w and 0 <= x3 < w and 0 <= y1 < h and 0 <= y2 < h and 0 <= y3 < h):
            continue
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        p3 = (int(round(x3)), int(round(y3)))
        cv2.line(overlay, p1, p2, color_bgr, thickness, cv2.LINE_AA)
        cv2.line(overlay, p2, p3, color_bgr, thickness, cv2.LINE_AA)
        cv2.line(overlay, p3, p1, color_bgr, thickness, cv2.LINE_AA)

    return overlay


# ----------------------------
# Main pipeline
# ----------------------------

def build_output_name(
    input_path: str,
    fps: float,
    start: float,
    duration: float,
    max_points: int,
    motion_thresh: int,
    trail_alpha: float,
    glow: int,
    velocity_color: bool,
) -> str:
    p = Path(input_path)
    stem = p.stem
    tag = (
        f"_vmmesh"
        f"_fps{fmt_num(fps,2)}"
        f"_s{fmt_num(start,2)}"
        f"_d{fmt_num(duration,2)}"
        f"_mp{max_points}"
        f"_mt{motion_thresh}"
        f"_ta{fmt_num(trail_alpha,3)}"
        f"_g{glow}"
        f"_vc{int(velocity_color)}"
    )
    out = safe_slug(stem + tag) + ".mp4"
    return str(p.with_name(out))


def make_trimmed_video(
    input_path: str,
    start: float,
    duration: float,
    fps: float,
    tmp_video_path: str,
    width: Optional[int],
    height: Optional[int],
    crf: int,
    preset: str,
) -> None:
    vf = [f"fps={fps}:round=near"]
    if width and height:
        # keep aspect, fit inside WxH, then pad to exact size
        vf.append(f"scale={width}:{height}:force_original_aspect_ratio=decrease")
        vf.append(f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2")
    vf_str = ",".join(vf)

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", str(start),
        "-t", str(duration),
        "-i", input_path,
        "-vf", vf_str,
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-profile:v", "baseline",
        "-level", "4.1",
        "-crf", str(crf),
        "-preset", preset,
        tmp_video_path,
    ]
    run_cmd(cmd)


def mux_audio_back(
    input_path: str,
    start: float,
    duration: float,
    processed_video_path: str,
    output_path: str,
    audio_bitrate: str,
) -> None:
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", str(start),
        "-t", str(duration),
        "-i", input_path,
        "-i", processed_video_path,
        "-map", "1:v:0",
        "-map", "0:a:0?",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        "-shortest",
        output_path,
    ]
    run_cmd(cmd)


def main() -> int:
    ap = argparse.ArgumentParser(description="Voidstar Motion Mesh (motion-gated constellation + Delaunay).")
    ap.add_argument("input", help="Input video file")
    ap.add_argument("--start", type=float, default=0.0, help="Start time (seconds)")
    ap.add_argument("--duration", type=float, default=10.0, help="Duration (seconds)")
    ap.add_argument("--fps", type=float, default=30.0, help="Output FPS")

    ap.add_argument("--width", type=int, default=0, help="Optional output width (pads after scale)")
    ap.add_argument("--height", type=int, default=0, help="Optional output height (pads after scale)")

    ap.add_argument("--max-points", type=int, default=180, help="Max tracked points")
    ap.add_argument("--min-distance", type=int, default=10, help="Min distance between new points")
    ap.add_argument("--quality", type=float, default=0.01, help="gFTT quality level")

    ap.add_argument("--motion-thresh", type=int, default=22, help="Base motion threshold (0-255)")
    ap.add_argument("--motion-dilate", type=int, default=2, help="Dilate iterations for motion mask")
    ap.add_argument("--motion-blur", type=int, default=3, help="Blur kernel for motion energy (odd)")
    ap.add_argument("--spawn-per-second", type=float, default=8.0, help="New points spawn rate (per second)")
    ap.add_argument("--beat-boost", type=float, default=0.7, help="How much beat pulse boosts spawning/intensity")

    ap.add_argument("--trail", type=lambda s: s.lower() in ("1", "true", "yes", "y"), default=True, help="Enable trails")
    ap.add_argument("--trail-alpha", type=float, default=0.92, help="Trail decay (0..1) higher=longer trails")

    ap.add_argument("--thickness", type=int, default=1, help="Base line thickness")
    ap.add_argument("--bass-thickness", type=int, default=2, help="Extra thickness at strong bass")
    ap.add_argument("--alpha", type=float, default=0.9, help="Base overlay alpha (0..1)")
    ap.add_argument("--glow", type=int, default=7, help="Glow blur kernel size (odd-ish). 0 disables.")

    ap.add_argument("--overlay-color", type=str, default="255,255,255", help="RGB overlay color when velocity-color=false")
    ap.add_argument("--velocity-color", type=lambda s: s.lower() in ("1", "true", "yes", "y"), default=False,
                    help="Colorize lines by point velocity")

    ap.add_argument("--crf", type=int, default=18, help="x264 CRF for temp encode")
    ap.add_argument("--preset", type=str, default="veryfast", help="x264 preset for temp encode")
    ap.add_argument("--audio-bitrate", type=str, default="192k", help="AAC bitrate for muxed audio")

    ap.add_argument("--output", type=str, default="", help="Output file (default: auto-named)")

    args = ap.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        log(f"ERROR: input not found: {input_path}")
        return 2

    width = args.width if args.width > 0 else None
    height = args.height if args.height > 0 else None

    overlay_rgb = parse_rgb(args.overlay_color)
    # OpenCV uses BGR
    overlay_bgr = (overlay_rgb[2], overlay_rgb[1], overlay_rgb[0])

    out_path = args.output.strip() or build_output_name(
        input_path=input_path,
        fps=args.fps,
        start=args.start,
        duration=args.duration,
        max_points=args.max_points,
        motion_thresh=args.motion_thresh,
        trail_alpha=args.trail_alpha,
        glow=args.glow,
        velocity_color=args.velocity_color,
    )

    t0 = time.time()
    log("starting")
    log(f"input={input_path}")
    log(f"start={args.start:.3f}s duration={args.duration:.3f}s fps={args.fps:.3f}")

    with tempfile.TemporaryDirectory(prefix="voidstar_") as td:
        tmp_trim = str(Path(td) / "trim_silent.mp4")
        tmp_wav = str(Path(td) / "audio.wav")
        tmp_out = str(Path(td) / "processed.mp4")

        # 1) Create trimmed silent temp video (constant fps, optional pad/scale)
        make_trimmed_video(
            input_path=input_path,
            start=args.start,
            duration=args.duration,
            fps=args.fps,
            tmp_video_path=tmp_trim,
            width=width,
            height=height,
            crf=args.crf,
            preset=args.preset,
        )

        # 2) Extract audio segment and compute features aligned to fps
        extract_wav_mono_48k(input_path, args.start, args.duration, tmp_wav)
        feats = audio_features_for_fps(tmp_wav, args.fps)

        # 3) Process frames
        cap = cv2.VideoCapture(tmp_trim)
        if not cap.isOpened():
            log("ERROR: failed to open temp video via cv2")
            return 3

        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or args.fps)
        log(f"temp video: {W}x{H} frames={n_total} fps={fps:.3f}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(tmp_out, fourcc, fps, (W, H))
        if not vw.isOpened():
            log("ERROR: failed to open VideoWriter")
            return 4

        prev_gray = None
        tracks: List[Track] = []
        next_tid = 1

        # persistent trail buffer
        trail = np.zeros((H, W, 3), dtype=np.uint8)

        # motion energy buffer
        motion_acc = np.zeros((H, W), dtype=np.uint8)

        spawn_budget = 0.0
        spawn_per_frame = args.spawn_per_second / max(1.0, fps)

        lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )

        t_frames = time.time()
        last_report = time.time()

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Audio index aligned to frame
            ai = min(frame_idx, len(feats.rms) - 1)
            rms = float(feats.rms[ai])
            bass = float(feats.bass[ai])
            beat = float(feats.beat_pulse[ai])

            # Motion mask (prefer moving elements)
            if prev_gray is None:
                prev_gray = gray.copy()

            diff = cv2.absdiff(gray, prev_gray)

            k = args.motion_blur
            if k > 0:
                if k % 2 == 0:
                    k += 1
                diff = cv2.GaussianBlur(diff, (k, k), 0)

            # audio-reactive threshold: bass lowers threshold slightly so more motion lights up
            thr = int(args.motion_thresh - 8 * bass - 6 * beat)
            thr = max(1, min(255, thr))
            _, mot = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)

            if args.motion_dilate > 0:
                mot = cv2.dilate(mot, None, iterations=args.motion_dilate)

            # Optional: keep a bit of persistence in motion energy
            motion_acc = cv2.addWeighted(motion_acc, 0.85, mot, 0.40, 0).astype(np.uint8)

            # Track update
            # Prepare points from existing tracks
            p0 = []
            track_map = []
            for ti, tr in enumerate(tracks):
                if tr.pts:
                    p0.append(tr.pts[-1])
                    track_map.append(ti)

            if p0:
                p0_np = np.array(p0, dtype=np.float32).reshape(-1, 1, 2)
                p1_np, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0_np, None, **lk_params)
                st = st.reshape(-1)
                p1_np = p1_np.reshape(-1, 2)

                # Update tracks
                for j, okj in enumerate(st):
                    tr = tracks[track_map[j]]
                    if okj:
                        x, y = float(p1_np[j, 0]), float(p1_np[j, 1])
                        if 0 <= x < W and 0 <= y < H:
                            tr.pts.append((x, y))
                            if len(tr.pts) > 24:
                                tr.pts = tr.pts[-24:]
                            tr.age += 1
                            tr.miss = 0
                        else:
                            tr.miss += 1
                    else:
                        tr.miss += 1

                # Drop dead tracks
                tracks = [tr for tr in tracks if tr.miss <= 6 and tr.age <= 6000 and len(tr.pts) > 0]

            # Spawn new points (motion-gated)
            spawn_budget += spawn_per_frame * (1.0 + args.beat_boost * beat)
            # also allow extra spawn on strong bass
            spawn_budget += 0.25 * bass

            if spawn_budget >= 1.0:
                # Want points only in motion regions
                mask = motion_acc
                # Guard: if mask is basically empty, fall back to whole frame
                if np.mean(mask) < 2.0:
                    mask = None

                need = max(0, args.max_points - len(tracks))
                want = min(int(spawn_budget), max(0, need))
                spawn_budget -= float(want)

                if want > 0:
                    # Use gFTT on gray, with mask so we avoid static labels
                    corners = cv2.goodFeaturesToTrack(
                        gray,
                        maxCorners=want,
                        qualityLevel=float(args.quality),
                        minDistance=int(args.min_distance),
                        mask=mask,
                        blockSize=7,
                        useHarrisDetector=False,
                    )
                    if corners is not None:
                        for c in corners:
                            x, y = float(c[0][0]), float(c[0][1])
                            tracks.append(Track(tid=next_tid, pts=[(x, y)], age=1, miss=0))
                            next_tid += 1
                            if len(tracks) >= args.max_points:
                                break

            # Collect representative points (latest positions)
            pts = [(tr.pts[-1][0], tr.pts[-1][1]) for tr in tracks if tr.pts]

            # Line thickness and alpha modulated by audio
            thick = int(args.thickness + round(args.bass_thickness * bass + 1.0 * beat))
            thick = max(1, min(8, thick))
            alpha = clamp01(args.alpha * (0.65 + 0.55 * rms + 0.45 * beat))

            # Build mesh overlay
            if args.velocity_color and len(tracks) >= 3:
                # Color by average velocity magnitude (cheap but looks cool)
                vmag = []
                for tr in tracks:
                    if len(tr.pts) >= 2:
                        x0, y0 = tr.pts[-2]
                        x1, y1 = tr.pts[-1]
                        vmag.append(math.hypot(x1 - x0, y1 - y0))
                v = float(np.mean(vmag) if vmag else 0.0)
                v = min(12.0, v)
                # Map v to hue-ish in BGR space (simple)
                # (Keep it “voidstar subtle”: mostly white with slight tint)
                tint = int(30 + 12 * v)
                col = (min(255, overlay_bgr[0] + tint), min(255, overlay_bgr[1] + tint), min(255, overlay_bgr[2]))
                overlay_mesh = draw_delaunay_lines((H, W), pts, thick, col)
            else:
                overlay_mesh = draw_delaunay_lines((H, W), pts, thick, overlay_bgr)

            # Extra “pulse” flash on beat (adds crispness)
            if beat > 0.2:
                overlay_mesh = cv2.addWeighted(overlay_mesh, 1.0, overlay_mesh, 0.6 * beat, 0)

            # Glow pass
            overlay_mesh = apply_glow(overlay_mesh, args.glow)

            # Trail handling
            if args.trail:
                # decay
                decay = clamp01(args.trail_alpha)
                trail = (trail.astype(np.float32) * decay).astype(np.uint8)
                # add overlay into trail
                trail = blend_add(trail, overlay_mesh, alpha=alpha)
                out = cv2.add(frame, trail)
            else:
                out = cv2.add(frame, (overlay_mesh.astype(np.float32) * alpha).astype(np.uint8))

            vw.write(out)

            prev_gray = gray
            frame_idx += 1

            # Status logs
            if frame_idx % 60 == 0:
                now = time.time()
                dt = now - last_report
                elapsed = now - t_frames
                cur_fps = 60.0 / max(1e-9, dt)
                prog = frame_idx / max(1, n_total)
                eta = (elapsed / max(1e-6, prog)) - elapsed if prog > 0 else 0.0
                log(f"frames={frame_idx} fps={cur_fps:.2f} prog={prog*100:.1f}% eta={eta:.1f}s tracks={len(tracks)}")
                last_report = now

        cap.release()
        vw.release()

        # 4) Mux original audio segment back in
        mux_audio_back(
            input_path=input_path,
            start=args.start,
            duration=args.duration,
            processed_video_path=tmp_out,
            output_path=out_path,
            audio_bitrate=args.audio_bitrate,
        )

    elapsed_total = time.time() - t0
    log(f"done: {out_path}")
    log(f"elapsed: {elapsed_total:.2f}s")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("interrupted")
        raise
    except Exception as e:
        log(f"ERROR: {e}")
        raise
