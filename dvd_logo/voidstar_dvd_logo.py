#!/usr/bin/env python3
"""
VoidStar DVD Logo Overlay

Overlays a transparent PNG logo that bounces like a classic DVD screensaver.
Keeps source resolution/FPS, preserves source audio stream, and tries to keep
the same video codec family as the input.
"""

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np


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

    safe = "_".join(parts).replace("/", "-").replace(" ", "")
    return f"{safe}{input_path.suffix}"


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def overlay_rgba(frame: np.ndarray, logo_bgr: np.ndarray, logo_alpha: np.ndarray, x: int, y: int) -> None:
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
    aroi = logo_alpha[ly0:ly1, lx0:lx1].astype(np.float32)[..., None]

    out = lroi * aroi + roi * (1.0 - aroi)
    frame[y0:y1, x0:x1] = out.astype(np.uint8)


def format_eta(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "--:--:--"
    s = int(seconds)
    hh = s // 3600
    mm = (s % 3600) // 60
    ss = s % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Overlay bouncing PNG logo on a source video.")
    ap.add_argument("input", help="Input video path (bare names resolve under /mnt/c/users/<user>/Videos; ./ and ../ resolve from cwd)")
    ap.add_argument("logo", help="Transparent PNG logo path (bare names resolve under /mnt/c/users/<user>/Videos; ./ and ../ resolve from cwd)")

    ap.add_argument("--output", default=None, help="Output path (default: auto name in /mnt/c/users/<user>/Videos)")
    ap.add_argument("--out-dir", default=str(default_videos_dir()), help="Default output directory")

    ap.add_argument("--speed", type=float, default=220.0, help="Logo speed in pixels/second")
    ap.add_argument("--angle-deg", type=float, default=37.0, help="Initial movement angle in degrees")
    ap.add_argument("--start-x", type=float, default=0.50, help="Starting X center position in normalized frame coords [0..1]")
    ap.add_argument("--start-y", type=float, default=0.50, help="Starting Y center position in normalized frame coords [0..1]")
    ap.add_argument("--end-x", type=float, default=None, help="Optional target X center [0..1] used to set initial heading")
    ap.add_argument("--end-y", type=float, default=None, help="Optional target Y center [0..1] used to set initial heading")

    ap.add_argument("--logo-scale", type=float, default=0.18, help="Logo width as fraction of frame width")
    ap.add_argument("--logo-width-px", type=int, default=0, help="Absolute logo width in pixels (overrides --logo-scale if >0)")

    ap.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    ap.add_argument("--duration", type=float, default=0.0, help="Duration in seconds (0 = to end)")

    ap.add_argument("--codec", default="auto", help="Output video encoder (auto/libx264/libx265/mpeg4/libvpx-vp9/libsvtav1)")
    ap.add_argument("--preset", default="medium", help="Encoder preset")
    ap.add_argument("--crf", type=int, default=18, help="Quality CRF for CRF-based encoders")
    ap.add_argument("--log-interval", type=float, default=1.0, help="Progress print interval in seconds")

    args = ap.parse_args()

    videos_dir = default_videos_dir()
    input_path = resolve_media_path(args.input, videos_dir)
    logo_path = resolve_media_path(args.logo, videos_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    logo_bgr = logo_resized[:, :, :3]
    logo_alpha = logo_resized[:, :, 3].astype(np.float32) / 255.0

    max_x = max(0.0, frame_w - logo_w)
    max_y = max(0.0, frame_h - logo_h)

    cx0 = clamp(args.start_x, 0.0, 1.0) * frame_w
    cy0 = clamp(args.start_y, 0.0, 1.0) * frame_h
    x = clamp(cx0 - logo_w * 0.5, 0.0, max_x)
    y = clamp(cy0 - logo_h * 0.5, 0.0, max_y)

    theta = math.radians(args.angle_deg)
    if args.end_x is not None and args.end_y is not None:
        tx = clamp(args.end_x, 0.0, 1.0) * frame_w
        ty = clamp(args.end_y, 0.0, 1.0) * frame_h
        dx = tx - (x + logo_w * 0.5)
        dy = ty - (y + logo_h * 0.5)
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            theta = math.atan2(dy, dx)

    px_per_frame = max(0.0, args.speed) / max(1e-9, fps)
    vx = math.cos(theta) * px_per_frame
    vy = math.sin(theta) * px_per_frame

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

        overlay_rgba(frame, logo_bgr, logo_alpha, int(round(x)), int(round(y)))

        if enc_proc.stdin is None:
            raise RuntimeError("Encoder stdin is unavailable.")
        enc_proc.stdin.write(frame.tobytes())
        processed += 1

        x += vx
        y += vy

        for _ in range(4):
            bounced = False
            if x < 0.0:
                x = -x
                vx = abs(vx)
                bounced = True
            elif x > max_x:
                x = 2.0 * max_x - x
                vx = -abs(vx)
                bounced = True

            if y < 0.0:
                y = -y
                vy = abs(vy)
                bounced = True
            elif y > max_y:
                y = 2.0 * max_y - y
                vy = -abs(vy)
                bounced = True

            if not bounced:
                break

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

    elapsed = time.time() - t0
    print(f"[voidstar] done={output_path}")
    print(f"[voidstar] frames={processed} elapsed={elapsed:.2f}s")


if __name__ == "__main__":
    main()
