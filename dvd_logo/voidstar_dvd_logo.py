#!/usr/bin/env python3
"""
VoidStar DVD Logo Overlay

Overlays a transparent PNG logo that bounces like a classic DVD screensaver.
Keeps source resolution/FPS, preserves source audio stream, and tries to keep
the same video codec family as the input.
"""

import argparse
import glob
import json
import math
import os
import shlex
import random
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import cv2
import numpy as np


def bool_flag(v: str) -> bool:
    v = v.strip().lower()
    if v in ("1", "true", "t", "yes", "y", "on"):
        return True
    if v in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {v}")


def get_username() -> str:
    return os.environ.get("USER") or os.environ.get("USERNAME") or "brown"


def default_videos_dir() -> Path:
    return Path(f"/mnt/c/users/{get_username()}/Videos")


def resolve_media_path(raw: str, default_dir: Path) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    # Explicit relative paths (./ or ../) should resolve from cwd.
    if raw.startswith("./") or raw.startswith("../"):
        return (Path.cwd() / p).resolve()
    return default_dir / p


def resolve_input_paths(raw: str, default_dir: Path) -> list[Path]:
    # Wildcards support: *.mp4, clip_??.mp4, etc.
    if glob.has_magic(raw):
        p = Path(raw).expanduser()
        if p.is_absolute():
            pattern = str(p)
        elif raw.startswith("./") or raw.startswith("../"):
            pattern = str((Path.cwd() / p).resolve())
        else:
            pattern = str(default_dir / p)

        matches = []
        for m in sorted(glob.glob(pattern)):
            mp = Path(m)
            if mp.is_file() and mp.suffix.lower() == ".mp4":
                matches.append(mp.resolve())
        return matches

    return [resolve_media_path(raw, default_dir)]


def ffprobe_streams(input_path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(input_path),
    ]
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)


def build_audio_envelopes_for_fps(
    input_path: Path,
    start: float,
    duration: float | None,
    fps: float,
    smooth: float,
    gain: float,
    bass_hz: float,
    target_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    tmp_wav = Path(tempfile.mkstemp(prefix="voidstar_dvd_env_", suffix=".wav")[1])
    try:
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(input_path)]
        if start > 0:
            cmd += ["-ss", str(start)]
        if duration is not None and duration > 0:
            cmd += ["-t", str(duration)]
        cmd += ["-vn", "-ac", "1", "-ar", "48000", "-f", "wav", str(tmp_wav)]
        subprocess.run(cmd, check=True)

        with wave.open(str(tmp_wav), "rb") as wf:
            sr = wf.getframerate()
            nch = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            nframes = wf.getnframes()
            raw = wf.readframes(nframes)

        if sampwidth != 2:
            z = np.zeros((0,), dtype=np.float32)
            return z, z

        data = np.frombuffer(raw, dtype=np.int16)
        if nch > 1:
            data = data.reshape(-1, nch)[:, 0]
        samples = data.astype(np.float32) / 32768.0

        if samples.size == 0:
            z = np.zeros((0,), dtype=np.float32)
            return z, z

        spf = sr / max(1e-9, fps)
        if target_frames is not None and target_frames > 0:
            n_out = target_frames
        else:
            n_out = max(1, int(math.ceil(samples.size / spf)))

        env = np.zeros((n_out,), dtype=np.float32)
        bass_env = np.zeros((n_out,), dtype=np.float32)
        bass_cut = float(np.clip(bass_hz, 20.0, (sr * 0.5) - 1.0))
        for i in range(n_out):
            a = int(i * spf)
            b = int((i + 1) * spf)
            if a >= samples.size:
                break
            chunk = samples[a:min(b, samples.size)]
            if chunk.size:
                env[i] = float(np.sqrt(np.mean(chunk * chunk)))

                # Low-frequency energy estimate for kick/snare-driven pulse.
                spec = np.fft.rfft(chunk)
                freqs = np.fft.rfftfreq(chunk.size, d=1.0 / sr)
                mask = freqs <= bass_cut
                if np.any(mask):
                    bass_env[i] = float(np.sqrt(np.mean(np.abs(spec[mask]) ** 2)))

        p95 = float(np.percentile(env, 95)) if env.size else 0.0
        norm = max(1e-6, p95)
        env = (env / norm) * float(gain)

        bass_p95 = float(np.percentile(bass_env, 95)) if bass_env.size else 0.0
        bass_norm = max(1e-6, bass_p95)
        bass_env = (bass_env / bass_norm) * float(gain)

        a = float(np.clip(smooth, 0.0, 0.9999))
        prev = 0.0
        for i in range(env.size):
            prev = a * prev + (1.0 - a) * float(env[i])
            env[i] = prev

        prev_b = 0.0
        for i in range(bass_env.size):
            prev_b = a * prev_b + (1.0 - a) * float(bass_env[i])
            bass_env[i] = prev_b

        return (
            np.clip(env, 0.0, 2.0).astype(np.float32),
            np.clip(bass_env, 0.0, 2.0).astype(np.float32),
        )
    finally:
        tmp_wav.unlink(missing_ok=True)


def choose_video_encoder(src_codec: str, codec_override: str) -> str:
    if codec_override != "auto":
        return codec_override

    mapping = {
        "h264": "libx264",
        "hevc": "libx265",
        "mpeg4": "mpeg4",
        "vp9": "libvpx-vp9",
        "av1": "libsvtav1",
    }
    return mapping.get(src_codec, "libx264")


def make_output_filename(input_path: Path, args: argparse.Namespace) -> str:
    parts = [
        input_path.stem,
        "dvdlogo",
        f"spd{args.speed:g}",
        f"sx{args.start_x:.3f}",
        f"sy{args.start_y:.3f}",
        f"sc{args.logo_scale:.3f}",
        f"em{args.edge_margin_px:g}",
        f"ang{args.angle_deg:g}",
        f"st{args.start:g}",
    ]
    if args.end_x is not None:
        parts.append(f"ex{args.end_x:.3f}")
    if args.end_y is not None:
        parts.append(f"ey{args.end_y:.3f}")
    if args.duration > 0:
        parts.append(f"dur{args.duration:g}")
    if args.logo_width_px > 0:
        parts.append(f"lwp{args.logo_width_px}")
    if args.trails > 0:
        parts.append(f"tr{args.trails:.2f}")
    if abs(float(args.opacity) - 1.0) > 1e-9:
        parts.append(f"op{args.opacity:.2f}")
    if args.audio_reactive_glow > 0:
        parts.append(f"arg{args.audio_reactive_glow:.2f}")
    if args.audio_reactive_scale > 0:
        parts.append(f"ars{args.audio_reactive_scale:.2f}")
    if args.voidstar_energy > 0:
        parts.append(f"ve{args.voidstar_energy:.2f}")

    safe = "_".join(parts).replace("/", "-").replace(" ", "")
    return f"{safe}{input_path.suffix}"


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def reflect_1d_interval(pos: float, vel: float, lo: float, hi: float) -> tuple[float, float]:
    """Advance by one step and reflect inside [lo, hi] with overshoot handling."""
    if hi <= lo:
        return (lo + hi) * 0.5, 0.0

    x = pos + vel
    v = vel
    for _ in range(4):
        if x < lo:
            x = 2.0 * lo - x
            v = abs(v)
            continue
        if x > hi:
            x = 2.0 * hi - x
            v = -abs(v)
            continue
        break

    x = clamp(x, lo, hi)
    return x, v


def overlay_rgba(
    frame: np.ndarray,
    logo_bgr: np.ndarray,
    logo_alpha: np.ndarray,
    x: int,
    y: int,
    opacity: float = 1.0,
) -> None:
    h, w = frame.shape[:2]
    lh, lw = logo_bgr.shape[:2]

    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w, x + lw)
    y1 = min(h, y + lh)

    if x0 >= x1 or y0 >= y1:
        return

    lx0 = x0 - x
    ly0 = y0 - y
    lx1 = lx0 + (x1 - x0)
    ly1 = ly0 + (y1 - y0)

    roi = frame[y0:y1, x0:x1].astype(np.float32)
    lroi = logo_bgr[ly0:ly1, lx0:lx1].astype(np.float32)
    aroi = logo_alpha[ly0:ly1, lx0:lx1].astype(np.float32)
    if opacity < 1.0:
        aroi *= opacity
    aroi = aroi[..., None]

    out = lroi * aroi + roi * (1.0 - aroi)
    frame[y0:y1, x0:x1] = out.astype(np.uint8)


def overlay_tinted_rgba(
    frame: np.ndarray,
    logo_bgr: np.ndarray,
    logo_alpha: np.ndarray,
    x: int,
    y: int,
    tint_bgr: np.ndarray,
    opacity: float,
) -> None:
    tinted = np.clip(
        logo_bgr.astype(np.float32) * tint_bgr.reshape(1, 1, 3).astype(np.float32),
        0,
        255,
    ).astype(np.uint8)
    overlay_rgba(frame, tinted, logo_alpha, x, y, opacity)


def rotate_logo_rgba(logo_rgba: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = logo_rgba.shape[:2]
    cx = w * 0.5
    cy = h * 0.5

    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cosv = abs(M[0, 0])
    sinv = abs(M[0, 1])
    new_w = max(1, int((h * sinv) + (w * cosv)))
    new_h = max(1, int((h * cosv) + (w * sinv)))

    M[0, 2] += (new_w * 0.5) - cx
    M[1, 2] += (new_h * 0.5) - cy

    rotated = cv2.warpAffine(
        logo_rgba,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return rotated[:, :, :3], rotated[:, :, 3].astype(np.float32) / 255.0


def stamp_rgba_into_layer(
    layer_bgr: np.ndarray,
    layer_alpha: np.ndarray,
    logo_bgr: np.ndarray,
    logo_alpha: np.ndarray,
    x: int,
    y: int,
    opacity: float,
) -> None:
    h, w = layer_alpha.shape[:2]
    lh, lw = logo_bgr.shape[:2]

    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w, x + lw)
    y1 = min(h, y + lh)
    if x0 >= x1 or y0 >= y1:
        return

    lx0 = x0 - x
    ly0 = y0 - y
    lx1 = lx0 + (x1 - x0)
    ly1 = ly0 + (y1 - y0)

    src_bgr = logo_bgr[ly0:ly1, lx0:lx1].astype(np.float32)
    src_a = (logo_alpha[ly0:ly1, lx0:lx1] * opacity).astype(np.float32)
    src_a = np.clip(src_a, 0.0, 1.0)
    src_a3 = src_a[..., None]

    dst_bgr = layer_bgr[y0:y1, x0:x1]
    dst_a = layer_alpha[y0:y1, x0:x1]

    layer_bgr[y0:y1, x0:x1] = src_bgr * src_a3 + dst_bgr * (1.0 - src_a3)
    layer_alpha[y0:y1, x0:x1] = src_a + dst_a * (1.0 - src_a)


def blend_layer_onto_frame(frame: np.ndarray, layer_bgr: np.ndarray, layer_alpha: np.ndarray) -> None:
    a3 = np.clip(layer_alpha, 0.0, 1.0)[..., None]
    out = layer_bgr * a3 + frame.astype(np.float32) * (1.0 - a3)
    frame[:, :, :] = np.clip(out, 0, 255).astype(np.uint8)


def format_eta(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "--:--:--"
    s = int(seconds)
    hh = s // 3600
    mm = (s % 3600) // 60
    ss = s % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def hue_to_bgr_tint(hue_deg: float) -> np.ndarray:
    h = int((hue_deg % 360.0) * (179.0 / 360.0))
    hsv = np.array([[[h, 220, 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32)
    return np.clip(bgr / 255.0, 0.0, 1.0)


def run_checked(cmd: list[str]) -> None:
    print("[voidstar] ▶", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Overlay bouncing PNG logo on a source video.")
    ap.add_argument("input", help="Input video path (bare names resolve under /mnt/c/users/<user>/Videos; ./ and ../ resolve from cwd)")
    ap.add_argument("logo", help="Transparent PNG logo path (bare names resolve under /mnt/c/users/<user>/Videos; ./ and ../ resolve from cwd)")

    ap.add_argument("--output", default=None, help="Output path (default: auto name in /mnt/c/users/<user>/Videos)")
    ap.add_argument("--out-dir", default=str(default_videos_dir()), help="Default output directory")

    ap.add_argument("--speed", type=float, default=220.0, help="Logo speed in pixels/second")
    ap.add_argument("--angle-deg", type=float, default=None, help="Initial movement angle in degrees (default: random)")
    ap.add_argument("--start-x", type=float, default=0.50, help="Starting X center position in normalized frame coords [0..1]")
    ap.add_argument("--start-y", type=float, default=0.50, help="Starting Y center position in normalized frame coords [0..1]")
    ap.add_argument("--end-x", type=float, default=None, help="Optional target X center [0..1] used to set initial heading")
    ap.add_argument("--end-y", type=float, default=None, help="Optional target Y center [0..1] used to set initial heading")

    ap.add_argument("--logo-scale", type=float, default=0.18, help="Logo width as fraction of frame width")
    ap.add_argument("--logo-width-px", type=int, default=0, help="Absolute logo width in pixels (overrides --logo-scale if >0)")
    ap.add_argument("--logo-rotate-speed", type=float, default=0.0, help="Continuous logo rotation speed in degrees/second (0 disables)")
    ap.add_argument("--logo-rotate-start-deg", type=float, default=0.0, help="Initial logo rotation angle in degrees")
    ap.add_argument("--edge-margin-px", type=float, default=0.0, help="Signed edge margin in pixels. Positive stays inside edge; negative allows overlap.")
    ap.add_argument("--trails", type=float, default=0.0, help="Ghost trail strength [0..1]. 0 disables, 1 is strongest.")
    ap.add_argument("--opacity", type=float, default=1.0, help="Main logo opacity [0..1]. 1.0 is fully opaque.")
    ap.add_argument("--audio-reactive-glow", type=float, default=0.0, help="Audio-reactive glow strength [0..2]. 0 disables.")
    ap.add_argument("--audio-reactive-scale", type=float, default=0.06, help="Subtle audio-reactive logo scale amount [0..0.5].")
    ap.add_argument("--audio-reactive-gain", type=float, default=2.0, help="Audio envelope gain for reactive glow.")
    ap.add_argument("--audio-reactive-smooth", type=float, default=0.88, help="Audio envelope smoothing [0..0.9999].")
    ap.add_argument("--audio-reactive-bass-hz", type=float, default=300.0, help="Upper frequency (Hz) for bass-triggered scaling.")
    ap.add_argument("--audio-glow-blur", type=float, default=8.0, help="Glow blur sigma in pixels.")
    ap.add_argument("--voidstar-energy", type=float, default=1.0, help="Extra VoidStar FX intensity [0..3].")
    ap.add_argument("--voidstar-hue-rate", type=float, default=22.0, help="Hue cycling speed in cycles/sec for VoidStar FX.")
    ap.add_argument("--voidstar-chroma", type=float, default=3.0, help="Chromatic split amount in pixels for VoidStar FX.")
    ap.add_argument("--voidstar-jitter", type=float, default=1.2, help="Jitter amount in pixels for VoidStar FX.")
    ap.add_argument("--voidstar-bloom", type=float, default=0.55, help="Bloom strength [0..2] for VoidStar FX.")
    ap.add_argument("--reels-local-overlay", type=bool_flag, default=False, help="Run reels_cv_overlay on a local region around logo only, then composite back.")
    ap.add_argument("--reels-script-path", default=None, help="Optional path to reels_cv_overlay.py (default: auto-detect in sibling folder)")
    ap.add_argument("--reels-local-pad-px", type=int, default=120, help="Padding around logo motion bounds for local reels processing")
    ap.add_argument("--reels-local-args", default="", help="Extra args string passed to reels_cv_overlay.py")

    ap.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    ap.add_argument("--duration", type=float, default=0.0, help="Duration in seconds (0 = to end)")
    ap.add_argument("--perfect-loop", type=bool_flag, default=True, help="Force final frame logo position to match start for seamless loops.")

    ap.add_argument("--codec", default="auto", help="Output video encoder (auto/libx264/libx265/mpeg4/libvpx-vp9/libsvtav1)")
    ap.add_argument("--preset", default="medium", help="Encoder preset")
    ap.add_argument("--crf", type=int, default=18, help="Quality CRF for CRF-based encoders")
    ap.add_argument("--log-interval", type=float, default=1.0, help="Progress print interval in seconds")

    args = ap.parse_args()

    if args.angle_deg is None:
        args.angle_deg = random.uniform(0.0, 360.0)

    videos_dir = default_videos_dir()
    logo_path = resolve_media_path(args.logo, videos_dir)
    input_paths = resolve_input_paths(args.input, videos_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_paths:
        raise FileNotFoundError(f"No matching .mp4 files found for input pattern: {args.input}")

    if len(input_paths) > 1:
        if args.output:
            raise ValueError("--output cannot be used with wildcard input. Use --out-dir for batch processing.")

        script_path = str(Path(__file__).resolve())
        passthrough_args = sys.argv[3:]
        print(f"[voidstar] matched {len(input_paths)} input files")

        for i, input_path in enumerate(input_paths, start=1):
            print(f"[voidstar] batch {i}/{len(input_paths)} input={input_path}")
            cmd = [sys.executable, script_path, str(input_path), str(logo_path), *passthrough_args]
            subprocess.run(cmd, check=True)
        return

    input_path = input_paths[0]

    if args.output:
        output_path = resolve_media_path(args.output, out_dir)
    else:
        output_path = out_dir / make_output_filename(input_path, args)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if not logo_path.exists():
        raise FileNotFoundError(f"Logo not found: {logo_path}")

    probe = ffprobe_streams(input_path)
    video_stream = next((s for s in probe.get("streams", []) if s.get("codec_type") == "video"), None)
    if video_stream is None:
        raise RuntimeError("No video stream found in input.")
    src_codec = video_stream.get("codec_name", "")
    src_bitrate = video_stream.get("bit_rate")
    enc = choose_video_encoder(src_codec, args.codec)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    start_frame = max(0, int(round(args.start * fps)))
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    if args.duration > 0:
        total_frames = max(1, int(round(args.duration * fps)))
    elif source_total > 0:
        total_frames = max(1, source_total - start_frame)
    else:
        total_frames = -1

    logo_rgba = cv2.imread(str(logo_path), cv2.IMREAD_UNCHANGED)
    if logo_rgba is None:
        raise RuntimeError(f"Could not read logo file: {logo_path}")
    if logo_rgba.ndim != 3 or logo_rgba.shape[2] != 4:
        raise RuntimeError("Logo PNG must contain alpha channel (RGBA).")

    if args.logo_width_px > 0:
        logo_w = int(args.logo_width_px)
    else:
        logo_w = max(8, int(round(frame_w * max(0.01, args.logo_scale))))
    src_h, src_w = logo_rgba.shape[:2]
    logo_h = max(8, int(round(logo_w * (src_h / max(1, src_w)))))

    logo_resized = cv2.resize(logo_rgba, (logo_w, logo_h), interpolation=cv2.INTER_AREA)
    base_logo_bgr = logo_resized[:, :, :3]
    base_logo_alpha = logo_resized[:, :, 3].astype(np.float32) / 255.0

    cx0 = clamp(args.start_x, 0.0, 1.0) * frame_w
    cy0 = clamp(args.start_y, 0.0, 1.0) * frame_h
    cx = clamp(cx0, logo_w * 0.5, max(logo_w * 0.5, frame_w - logo_w * 0.5))
    cy = clamp(cy0, logo_h * 0.5, max(logo_h * 0.5, frame_h - logo_h * 0.5))
    loop_start_cx = cx
    loop_start_cy = cy

    theta = math.radians(args.angle_deg)
    if args.end_x is not None and args.end_y is not None:
        tx = clamp(args.end_x, 0.0, 1.0) * frame_w
        ty = clamp(args.end_y, 0.0, 1.0) * frame_h
        dx = tx - cx
        dy = ty - cy
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            theta = math.atan2(dy, dx)

    px_per_frame = max(0.0, args.speed) / max(1e-9, fps)
    vx = math.cos(theta) * px_per_frame
    vy = math.sin(theta) * px_per_frame

    margin_px = float(args.edge_margin_px)
    trails_strength = clamp(float(args.trails), 0.0, 1.0)
    overlay_opacity = clamp(float(args.opacity), 0.0, 1.0)
    reactive_glow_strength = clamp(float(args.audio_reactive_glow), 0.0, 2.0)
    reactive_scale_strength = clamp(float(args.audio_reactive_scale), 0.0, 0.5)
    voidstar_energy = clamp(float(args.voidstar_energy), 0.0, 3.0)
    trail_decay = 0.80 + 0.18 * trails_strength
    trail_opacity = 0.10 + 0.30 * trails_strength
    trail_bgr = None
    trail_alpha = None

    local_reels_enabled = bool(args.reels_local_overlay)
    roi_min_x = frame_w
    roi_min_y = frame_h
    roi_max_x = 0
    roi_max_y = 0
    local_pad = max(0, int(args.reels_local_pad_px))

    audio_env = None
    bass_env = None
    if reactive_glow_strength > 0.0 or reactive_scale_strength > 0.0:
        print("[voidstar] analyzing audio for reactive effects...")
        audio_env, bass_env = build_audio_envelopes_for_fps(
            input_path=input_path,
            start=float(args.start),
            duration=(float(args.duration) if args.duration > 0 else None),
            fps=fps,
            smooth=float(args.audio_reactive_smooth),
            gain=float(args.audio_reactive_gain),
            bass_hz=float(args.audio_reactive_bass_hz),
            target_frames=(total_frames if total_frames > 0 else None),
        )
        print(f"[voidstar] reactive envelopes frames={len(audio_env)}")

    tmp_video = output_path.with_name(output_path.stem + "__video.mp4")

    ffmpeg_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{frame_w}x{frame_h}",
        "-r",
        f"{fps}",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        enc,
    ]

    if enc in {"libx264", "libx265", "libsvtav1"}:
        ffmpeg_cmd += ["-preset", args.preset, "-crf", str(args.crf)]
    elif src_bitrate and src_bitrate.isdigit():
        ffmpeg_cmd += ["-b:v", src_bitrate]

    ffmpeg_cmd += ["-pix_fmt", "yuv420p", str(tmp_video)]

    enc_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    t0 = time.time()
    last_log = t0
    processed = 0

    while True:
        if total_frames > 0 and processed >= total_frames:
            break

        ok, frame = cap.read()
        if not ok:
            break

        is_last_loop_frame = bool(args.perfect_loop and total_frames > 0 and processed == (total_frames - 1))
        phase_idx = 0 if is_last_loop_frame else processed

        if abs(args.logo_rotate_speed) > 1e-9 or abs(args.logo_rotate_start_deg) > 1e-9:
            angle = args.logo_rotate_start_deg + args.logo_rotate_speed * (phase_idx / max(1e-9, fps))
            logo_bgr, logo_alpha = rotate_logo_rgba(logo_resized, angle)
        else:
            logo_bgr, logo_alpha = base_logo_bgr, base_logo_alpha

        if bass_env is not None and phase_idx < len(bass_env) and reactive_scale_strength > 0.0:
            bass_level = float(np.clip(bass_env[phase_idx], 0.0, 1.0))
            scale_mult = 1.0 + (reactive_scale_strength * bass_level)
            if abs(scale_mult - 1.0) > 1e-5:
                scaled_w = max(1, int(round(logo_bgr.shape[1] * scale_mult)))
                scaled_h = max(1, int(round(logo_bgr.shape[0] * scale_mult)))
                logo_bgr = cv2.resize(logo_bgr, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
                logo_alpha = cv2.resize(logo_alpha, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
                logo_alpha = np.clip(logo_alpha, 0.0, 1.0)

        cur_w = logo_bgr.shape[1]
        cur_h = logo_bgr.shape[0]
        half_w = cur_w * 0.5
        half_h = cur_h * 0.5

        lo_x = half_w + margin_px
        hi_x = frame_w - half_w - margin_px
        lo_y = half_h + margin_px
        hi_y = frame_h - half_h - margin_px

        if is_last_loop_frame:
            # Force endpoint to starting center for seamless loop wrap.
            cx = clamp(loop_start_cx, lo_x, max(lo_x, hi_x))
            cy = clamp(loop_start_cy, lo_y, max(lo_y, hi_y))
        else:
            # Keep center valid for this frame's rotated dimensions.
            cx = clamp(cx, lo_x, max(lo_x, hi_x))
            cy = clamp(cy, lo_y, max(lo_y, hi_y))

            # Move + reflect using current frame's true logo bounds.
            cx, vx = reflect_1d_interval(cx, vx, lo_x, hi_x)
            cy, vy = reflect_1d_interval(cy, vy, lo_y, hi_y)

        draw_x = int(round(cx - half_w))
        draw_y = int(round(cy - half_h))

        pulse = 0.0
        if bass_env is not None and phase_idx < len(bass_env):
            pulse = float(np.clip(bass_env[phase_idx], 0.0, 1.0))

        if local_reels_enabled:
            x0 = max(0, draw_x - local_pad)
            y0 = max(0, draw_y - local_pad)
            x1 = min(frame_w, draw_x + cur_w + local_pad)
            y1 = min(frame_h, draw_y + cur_h + local_pad)
            if x1 > x0 and y1 > y0:
                roi_min_x = min(roi_min_x, x0)
                roi_min_y = min(roi_min_y, y0)
                roi_max_x = max(roi_max_x, x1)
                roi_max_y = max(roi_max_y, y1)

        if trails_strength > 0.0:
            if trail_bgr is None or trail_alpha is None:
                trail_bgr = np.zeros((frame_h, frame_w, 3), dtype=np.float32)
                trail_alpha = np.zeros((frame_h, frame_w), dtype=np.float32)
            trail_bgr *= trail_decay
            trail_alpha *= trail_decay
            stamp_rgba_into_layer(
                trail_bgr,
                trail_alpha,
                logo_bgr,
                logo_alpha,
                draw_x,
                draw_y,
                trail_opacity,
            )
            blend_layer_onto_frame(frame, trail_bgr, trail_alpha)

        if audio_env is not None and phase_idx < len(audio_env):
            level = float(np.clip(audio_env[phase_idx], 0.0, 1.0))
            if level > 1e-4:
                sigma = max(0.5, float(args.audio_glow_blur))
                glow_alpha = cv2.GaussianBlur(logo_alpha, (0, 0), sigmaX=sigma, sigmaY=sigma)
                peak = float(np.max(glow_alpha))
                if peak > 1e-8:
                    glow_alpha = glow_alpha / peak
                    glow_alpha = np.clip(glow_alpha * (0.55 * reactive_glow_strength * level), 0.0, 1.0)
                    overlay_rgba(frame, logo_bgr, glow_alpha, draw_x, draw_y, 1.0)

        if voidstar_energy > 0.0:
            t = phase_idx / max(1e-9, fps)
            eng = voidstar_energy * (0.65 + 0.75 * pulse)

            chroma_px = max(0, int(round(float(args.voidstar_chroma) * eng)))
            jitter_px = float(args.voidstar_jitter) * eng
            jx = int(round(np.random.uniform(-jitter_px, jitter_px))) if jitter_px > 0 else 0
            jy = int(round(np.random.uniform(-jitter_px, jitter_px))) if jitter_px > 0 else 0

            hue = t * float(args.voidstar_hue_rate) * 360.0
            tint_a = hue_to_bgr_tint(hue)
            tint_b = hue_to_bgr_tint(hue + 120.0)

            if chroma_px > 0:
                overlay_tinted_rgba(
                    frame,
                    logo_bgr,
                    logo_alpha,
                    draw_x - chroma_px + jx,
                    draw_y + jy,
                    np.array([1.0, 0.40, 0.40], dtype=np.float32) * tint_a,
                    opacity=min(1.0, 0.22 * eng),
                )
                overlay_tinted_rgba(
                    frame,
                    logo_bgr,
                    logo_alpha,
                    draw_x + chroma_px + jx,
                    draw_y + jy,
                    np.array([0.40, 0.70, 1.0], dtype=np.float32) * tint_b,
                    opacity=min(1.0, 0.22 * eng),
                )

            bloom_alpha = cv2.GaussianBlur(
                logo_alpha,
                (0, 0),
                sigmaX=max(0.5, 3.0 + 2.2 * eng),
                sigmaY=max(0.5, 3.0 + 2.2 * eng),
            )
            peak = float(np.max(bloom_alpha))
            if peak > 1e-8:
                bloom_alpha = np.clip(bloom_alpha / peak, 0.0, 1.0)
                bloom_alpha = np.clip(bloom_alpha * (float(args.voidstar_bloom) * 0.30 * eng), 0.0, 1.0)
                overlay_tinted_rgba(frame, logo_bgr, bloom_alpha, draw_x + jx, draw_y + jy, tint_a, 1.0)

        overlay_rgba(frame, logo_bgr, logo_alpha, draw_x, draw_y, overlay_opacity)

        if enc_proc.stdin is None:
            raise RuntimeError("Encoder stdin is unavailable.")
        enc_proc.stdin.write(frame.tobytes())
        processed += 1

        now = time.time()
        if now - last_log >= max(0.1, args.log_interval):
            elapsed = now - t0
            proc_fps = processed / max(elapsed, 1e-9)
            if total_frames > 0:
                remain = max(0, total_frames - processed)
                eta = remain / max(proc_fps, 1e-9)
                pct = 100.0 * processed / max(1, total_frames)
                print(
                    f"[voidstar] frame={processed}/{total_frames} ({pct:.1f}%) "
                    f"fps={proc_fps:.2f} eta={format_eta(eta)}"
                )
            else:
                print(f"[voidstar] frame={processed} fps={proc_fps:.2f} eta=--:--:--")
            last_log = now

    cap.release()

    if enc_proc.stdin:
        enc_proc.stdin.close()
    enc_proc.wait()

    mux_cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if args.start > 0:
        mux_cmd += ["-ss", str(args.start)]
    mux_cmd += ["-i", str(input_path)]
    if args.duration > 0:
        mux_cmd += ["-t", str(args.duration)]
    mux_cmd += [
        "-i",
        str(tmp_video),
        "-map",
        "1:v:0",
        "-map",
        "0:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-shortest",
        str(output_path),
    ]
    subprocess.run(mux_cmd, check=True)

    tmp_video.unlink(missing_ok=True)

    if local_reels_enabled and roi_max_x > roi_min_x and roi_max_y > roi_min_y:
        if args.reels_script_path:
            reels_script = resolve_media_path(args.reels_script_path, Path.cwd())
        else:
            reels_script = Path(__file__).resolve().parents[1] / "reels_cv_overlay" / "reels_cv_overlay.py"

        if not reels_script.exists():
            raise FileNotFoundError(f"reels_cv_overlay script not found: {reels_script}")

        crop_w = roi_max_x - roi_min_x
        crop_h = roi_max_y - roi_min_y

        local_in = output_path.with_name(output_path.stem + "__reels_local_in.mp4")
        local_out = output_path.with_name(output_path.stem + "__reels_local_out.mp4")
        final_out = output_path.with_name(output_path.stem + "__reels_local_final.mp4")

        print(
            f"[voidstar] local reels region x={roi_min_x} y={roi_min_y} "
            f"w={crop_w} h={crop_h}"
        )

        run_checked([
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(output_path),
            "-vf",
            f"crop={crop_w}:{crop_h}:{roi_min_x}:{roi_min_y}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            str(args.crf),
            "-c:a",
            "copy",
            str(local_in),
        ])

        reels_extra = shlex.split(args.reels_local_args) if args.reels_local_args else []
        run_checked([
            sys.executable,
            str(reels_script),
            str(local_in),
            "-o",
            str(local_out),
            *reels_extra,
        ])

        run_checked([
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(output_path),
            "-i",
            str(local_out),
            "-filter_complex",
            f"[0:v][1:v]overlay={roi_min_x}:{roi_min_y}:shortest=1[v]",
            "-map",
            "[v]",
            "-map",
            "0:a:0?",
            "-c:v",
            "libx264",
            "-preset",
            args.preset,
            "-crf",
            str(args.crf),
            "-c:a",
            "copy",
            str(final_out),
        ])

        final_out.replace(output_path)
        local_in.unlink(missing_ok=True)
        local_out.unlink(missing_ok=True)

    elapsed = time.time() - t0
    print(f"[voidstar] done={output_path}")
    print(f"[voidstar] frames={processed} elapsed={elapsed:.2f}s")


if __name__ == "__main__":
    main()
