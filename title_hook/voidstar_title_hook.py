#!/usr/bin/env python3
"""
VoidStar title hook overlay builder.

Creates a high-visibility start/end title treatment with glitch text, optional logo,
audio-reactive intensity, and mirrored end behavior for short-form vertical video.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import cv2
import numpy as np


def die(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(1)


def require_cmd(name: str) -> None:
    if not shutil_which(name):
        die(f"Missing required command: {name}")


def shutil_which(name: str) -> str | None:
    from shutil import which

    return which(name)


def ffprobe_info(input_path: Path) -> dict:
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


def parse_fps(value: str | None) -> float:
    if not value or value in {"0/0", "N/A"}:
        return 30.0
    if "/" in value:
        n, d = value.split("/", 1)
        dn = float(d)
        return float(n) / dn if dn != 0 else 30.0
    return float(value)


def slug(text: str, max_len: int = 10) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "", text.strip().lower())
    if not cleaned:
        return "na"
    return cleaned[:max_len]


def normalize_text_arg(text: str, newline_token: str) -> str:
    value = str(text)
    if newline_token:
        value = value.replace(newline_token, "\n")
    value = value.replace("\\n", "\n").replace("\\r", "\r")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    return value


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def choose_encoder(src_codec: str) -> str:
    mapping = {
        "h264": "libx264",
        "hevc": "libx265",
        "mpeg4": "mpeg4",
        "vp9": "libvpx-vp9",
        "av1": "libsvtav1",
    }
    return mapping.get(src_codec, "libx264")


def make_default_output_path(input_path: Path, outdir: Path, args: argparse.Namespace) -> Path:
    name = (
        f"{input_path.stem}_th"
        f"_d{args.duration:g}"
        f"_f{args.fade_out_duration:g}"
        f"_li{args.logo_intensity:g}"
        f"_t{slug(args.title, 8)}"
        f"_s{slug(args.secondary_text, 8)}"
        f"{input_path.suffix}"
    )
    return outdir / name


def build_audio_envelope(input_path: Path, fps: float, target_frames: int) -> np.ndarray:
    fd, wav_name = tempfile.mkstemp(prefix="voidstar_title_hook_", suffix=".wav")
    os.close(fd)
    wav_path = Path(wav_name)
    try:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "48000",
            "-acodec",
            "pcm_s16le",
            str(wav_path),
        ]
        subprocess.run(cmd, check=True)

        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            nch = wf.getnchannels()
            nframes = wf.getnframes()
            data = wf.readframes(nframes)

        samples = np.frombuffer(data, dtype=np.int16)
        if nch > 1:
            samples = samples.reshape(-1, nch)[:, 0]
        if samples.size == 0:
            return np.zeros((target_frames,), dtype=np.float32)

        samples = samples.astype(np.float32) / 32768.0
        samples_per_frame = sr / max(1e-9, fps)
        env = np.zeros((target_frames,), dtype=np.float32)

        for i in range(target_frames):
            a = int(i * samples_per_frame)
            b = int((i + 1) * samples_per_frame)
            if a >= samples.size:
                break
            chunk = samples[a:min(b, samples.size)]
            if chunk.size:
                env[i] = float(np.sqrt(np.mean(chunk * chunk)))

        p95 = float(np.percentile(env, 95)) if env.size else 0.0
        norm = max(1e-6, p95)
        env = env / norm

        smooth = np.zeros_like(env)
        alpha = 0.86
        prev = 0.0
        for i, v in enumerate(env):
            prev = alpha * prev + (1.0 - alpha) * float(v)
            smooth[i] = prev

        return np.clip(smooth, 0.0, 2.5).astype(np.float32)
    finally:
        wav_path.unlink(missing_ok=True)


def ease_in_out(x: float) -> float:
    v = max(0.0, min(1.0, x))
    return v * v * (3.0 - 2.0 * v)


def start_alpha(t: float, window_duration: float, fade_out: float) -> float:
    if t < 0 or t > window_duration:
        return 0.0
    hold = max(0.0, window_duration - fade_out)
    if t <= hold:
        return 1.0
    if fade_out <= 1e-6:
        return 0.0
    return max(0.0, 1.0 - ((t - hold) / fade_out))


def end_alpha(t: float, total_duration: float, window_duration: float, fade_in: float) -> float:
    window_start = max(0.0, total_duration - window_duration)
    if t < window_start or t > total_duration:
        return 0.0
    if fade_in <= 1e-6:
        return 1.0
    u = t - window_start
    if u < fade_in:
        return ease_in_out(u / fade_in)
    return 1.0


def alpha_blend(dst: np.ndarray, src: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = np.clip(alpha, 0.0, 1.0).astype(np.float32)
    if a.ndim == 2:
        a = a[:, :, None]
    return np.clip(dst.astype(np.float32) * (1.0 - a) + src.astype(np.float32) * a, 0, 255).astype(np.uint8)


def overlay_rgba(frame: np.ndarray, rgba: np.ndarray, x: int, y: int, opacity: float) -> None:
    h, w = frame.shape[:2]
    lh, lw = rgba.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w, x + lw)
    y1 = min(h, y + lh)
    if x0 >= x1 or y0 >= y1:
        return

    roi = frame[y0:y1, x0:x1]
    sub = rgba[y0 - y : y1 - y, x0 - x : x1 - x]
    rgb = sub[:, :, :3].astype(np.float32)
    a = (sub[:, :, 3].astype(np.float32) / 255.0) * float(opacity)
    out = roi.astype(np.float32) * (1.0 - a[:, :, None]) + rgb * a[:, :, None]
    frame[y0:y1, x0:x1] = np.clip(out, 0, 255).astype(np.uint8)


def hue_to_bgr_tint(hue_deg: float) -> np.ndarray:
    h = int((hue_deg % 360.0) * (179.0 / 360.0))
    hsv = np.array([[[h, 220, 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32)
    return np.clip(bgr / 255.0, 0.0, 1.0)


def build_title_layer(
    width: int,
    height: int,
    title: str,
    secondary: str,
    base_alpha: float,
    dim_strength: float,
    audio_level: float,
    frame_idx: int,
    rng: np.random.Generator,
    text_margin_ratio: float,
    title_max_height_ratio: float,
    secondary_max_height_ratio: float,
) -> np.ndarray:
    layer = np.zeros((height, width, 3), dtype=np.uint8)

    if base_alpha <= 0:
        return layer

    dim_level = float(np.clip(dim_strength, 0.0, 1.0)) * base_alpha
    dim = np.full_like(layer, int(255 * dim_level), dtype=np.uint8)
    layer = alpha_blend(layer, dim, np.full((height, width), 1.0, dtype=np.float32))

    band_strength = 0.18 + 0.22 * min(1.8, audio_level)
    for y in range(0, height, 4):
        value = int(max(0, min(255, 15 + 55 * band_strength * (0.5 + 0.5 * math.sin((y + frame_idx * 2) * 0.045)))))
        layer[y : y + 1, :, :] = np.maximum(layer[y : y + 1, :, :], value)

    noise = rng.normal(0, 14 + 22 * min(2.0, audio_level), size=(height, width, 1)).astype(np.float32)
    layer = np.clip(layer.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    title_font = cv2.FONT_HERSHEY_DUPLEX
    subtitle_font = cv2.FONT_HERSHEY_SIMPLEX

    title_scale = max(1.2, min(4.4, width / 300.0))
    secondary_scale = max(0.55, title_scale * 0.48)

    def fit_text_block(
        lines: list[str],
        font: int,
        start_scale: float,
        min_scale: float,
        thickness_mul: float,
        line_spacing: float,
        max_width: int,
        max_height: int,
    ) -> tuple[float, int, int, list[tuple[tuple[int, int], int]]]:
        scale = start_scale
        for _ in range(40):
            thickness = max(1, int(scale * thickness_mul))
            sizes = [cv2.getTextSize(line if line else " ", font, scale, thickness) for line in lines]
            max_w = max(sz[0][0] for sz in sizes)
            line_step = max(1, int(max(sz[1] + bl for sz, bl in sizes) * line_spacing))
            block_h = max(1, (len(lines) - 1) * line_step + max(sz[1] for sz, _ in sizes))
            if (max_w <= max_width and block_h <= max_height) or scale <= min_scale + 1e-6:
                return scale, thickness, line_step, sizes
            scale = max(min_scale, scale * 0.92)

        thickness = max(1, int(scale * thickness_mul))
        sizes = [cv2.getTextSize(line if line else " ", font, scale, thickness) for line in lines]
        line_step = max(1, int(max(sz[1] + bl for sz, bl in sizes) * line_spacing))
        block_h = max(1, (len(lines) - 1) * line_step + max(sz[1] for sz, _ in sizes))
        return scale, thickness, line_step, sizes

    cx = width // 2
    title_lines = title.split("\n") if title else [""]
    secondary_lines = secondary.split("\n") if secondary else [""]

    text_margin_px = int(width * float(np.clip(text_margin_ratio, 0.02, 0.25)))
    max_text_w = max(50, width - (2 * text_margin_px))

    title_max_h = max(40, int(height * float(np.clip(title_max_height_ratio, 0.08, 0.40))))
    secondary_max_h = max(30, int(height * float(np.clip(secondary_max_height_ratio, 0.06, 0.45))))

    title_scale, title_th, title_line_step, title_sizes = fit_text_block(
        lines=title_lines,
        font=title_font,
        start_scale=title_scale,
        min_scale=0.72,
        thickness_mul=2.1,
        line_spacing=1.24,
        max_width=max_text_w,
        max_height=title_max_h,
    )
    secondary_scale, secondary_th, secondary_line_step, secondary_sizes = fit_text_block(
        lines=secondary_lines,
        font=subtitle_font,
        start_scale=secondary_scale,
        min_scale=0.42,
        thickness_mul=1.8,
        line_spacing=1.18,
        max_width=max_text_w,
        max_height=secondary_max_h,
    )

    title_block_h = max(1, (len(title_lines) - 1) * title_line_step + max(sz[1] for sz, _ in title_sizes))
    secondary_block_h = max(1, (len(secondary_lines) - 1) * secondary_line_step + max(sz[1] for sz, _ in secondary_sizes))

    title_center_y = int(height * 0.48)
    secondary_center_y = int(height * 0.63)
    title_block_y0 = title_center_y - title_block_h // 2 + max(sz[1] for sz, _ in title_sizes)
    secondary_block_y0 = secondary_center_y - secondary_block_h // 2 + max(sz[1] for sz, _ in secondary_sizes)

    jitter = int(3 + 12 * min(2.0, audio_level))
    ox = int(rng.integers(-jitter, jitter + 1))
    oy = int(rng.integers(-jitter, jitter + 1))

    red = (40, 30, 245)
    cyan = (245, 245, 40)
    white = (245, 245, 245)
    secondary_color = (190, 190, 190)

    for i, line in enumerate(title_lines):
        line_for_size = line if line else " "
        line_size, _ = cv2.getTextSize(line_for_size, title_font, title_scale, title_th)
        title_x = cx - line_size[0] // 2
        title_y = title_block_y0 + i * title_line_step
        cv2.putText(layer, line, (title_x - 3 + ox, title_y + oy), title_font, title_scale, cyan, title_th, cv2.LINE_AA)
        cv2.putText(layer, line, (title_x + 3 + ox, title_y - oy), title_font, title_scale, red, title_th, cv2.LINE_AA)
        cv2.putText(layer, line, (title_x + ox, title_y + oy), title_font, title_scale, white, title_th + 1, cv2.LINE_AA)

    for i, line in enumerate(secondary_lines):
        line_for_size = line if line else " "
        line_size, _ = cv2.getTextSize(line_for_size, subtitle_font, secondary_scale, secondary_th)
        subtitle_x = cx - line_size[0] // 2
        subtitle_y = secondary_block_y0 + i * secondary_line_step
        cv2.putText(
            layer,
            line,
            (subtitle_x - 2, subtitle_y),
            subtitle_font,
            secondary_scale,
            (70, 70, 70),
            secondary_th + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            layer,
            line,
            (subtitle_x, subtitle_y),
            subtitle_font,
            secondary_scale,
            secondary_color,
            secondary_th,
            cv2.LINE_AA,
        )

    if (frame_idx % 5) == 0:
        gline_y = int(height * (0.35 + 0.3 * rng.random()))
        gline_h = int(2 + 6 * rng.random())
        x_shift = int((rng.random() - 0.5) * width * 0.09)
        band = layer[gline_y : min(height, gline_y + gline_h), :, :].copy()
        layer[gline_y : min(height, gline_y + gline_h), :, :] = np.roll(band, x_shift, axis=1)

    return layer


def resize_logo_rgba(logo_rgba: np.ndarray, target_w: int) -> np.ndarray:
    h, w = logo_rgba.shape[:2]
    if w <= 0:
        return logo_rgba
    scale = target_w / float(w)
    nh = max(1, int(h * scale))
    return cv2.resize(logo_rgba, (target_w, nh), interpolation=cv2.INTER_AREA)


def apply_dvd_local_point_track(
    frame: np.ndarray,
    point_track_layer: np.ndarray,
    track_prev_gray: np.ndarray | None,
    track_points: np.ndarray,
    logo_x: int,
    logo_y: int,
    logo_w: int,
    logo_h: int,
    phase_idx: int,
    fps: float,
    pulse: float,
    scale: float,
    pad_px: float,
    max_points: int,
    quality: float,
    min_distance: float,
    refresh: int,
    radius: float,
    link_neighbors: int,
    link_thickness: int,
    link_opacity: float,
    opacity: float,
    decay: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_h, frame_w = frame.shape[:2]
    gray_now = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    local_base_cx = logo_x + (logo_w * 0.5)
    local_base_cy = logo_y + (logo_h * 0.5)
    bw = max(8, int(round((logo_w * scale) + (2.0 * pad_px))))
    bh = max(8, int(round((logo_h * scale) + (2.0 * pad_px))))
    bx0 = max(0, int(round(local_base_cx - bw * 0.5)))
    by0 = max(0, int(round(local_base_cy - bh * 0.5)))
    bx1 = min(frame_w, bx0 + bw)
    by1 = min(frame_h, by0 + bh)

    point_track_layer[:by0, :, :] = 0.0
    point_track_layer[by1:, :, :] = 0.0
    point_track_layer[by0:by1, :bx0, :] = 0.0
    point_track_layer[by0:by1, bx1:, :] = 0.0
    point_track_layer[by0:by1, bx0:bx1, :] *= decay

    tracked = np.empty((0, 2), dtype=np.float32)
    if track_prev_gray is not None and track_points.size > 0:
        p1, st, _ = cv2.calcOpticalFlowPyrLK(
            track_prev_gray,
            gray_now,
            track_points,
            None,
            winSize=(21, 21),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        if p1 is not None and st is not None:
            cand = p1[st.reshape(-1) == 1].reshape(-1, 2)
            if cand.size > 0:
                keep = (
                    (cand[:, 0] >= bx0)
                    & (cand[:, 0] < bx1)
                    & (cand[:, 1] >= by0)
                    & (cand[:, 1] < by1)
                )
                tracked = cand[keep]

    need_seed = (phase_idx % refresh == 0) or (tracked.shape[0] < max(8, max_points // 3))
    if need_seed and bx1 > bx0 and by1 > by0:
        mask = np.zeros_like(gray_now, dtype=np.uint8)
        mask[by0:by1, bx0:bx1] = 255
        max_new = max(0, max_points - tracked.shape[0])
        if max_new > 0:
            pts_new = cv2.goodFeaturesToTrack(
                gray_now,
                maxCorners=max_new,
                qualityLevel=quality,
                minDistance=min_distance,
                mask=mask,
                blockSize=7,
            )
            if pts_new is not None:
                new_pts = pts_new.reshape(-1, 2).astype(np.float32)
                tracked = np.vstack([tracked, new_pts]) if tracked.size else new_pts

    if tracked.shape[0] > max_points:
        tracked = tracked[:max_points]
    track_points = tracked.reshape(-1, 1, 2).astype(np.float32) if tracked.size else np.empty((0, 1, 2), dtype=np.float32)

    if tracked.shape[0] > 0:
        t = phase_idx / max(1e-9, fps)
        tint = hue_to_bgr_tint(t * 22.0 * 360.0 + 120.0 * pulse)
        color = tuple(int(c) for c in (tint * 255.0))
        canvas = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

        n = tracked.shape[0]
        r2 = radius * radius
        if n >= 2:
            dx = tracked[:, 0][:, None] - tracked[:, 0][None, :]
            dy = tracked[:, 1][:, None] - tracked[:, 1][None, :]
            d2 = dx * dx + dy * dy
            for i in range(n):
                nbrs = np.where((d2[i] > 0) & (d2[i] <= r2))[0]
                if nbrs.size:
                    nbrs = nbrs[np.argsort(d2[i, nbrs])[:link_neighbors]]
                    x0, y0 = int(round(tracked[i, 0])), int(round(tracked[i, 1]))
                    for j in nbrs:
                        if j <= i:
                            continue
                        x1, y1 = int(round(tracked[j, 0])), int(round(tracked[j, 1]))
                        cv2.line(canvas, (x0, y0), (x1, y1), color, link_thickness, cv2.LINE_AA)

        for p in tracked:
            cv2.circle(canvas, (int(round(p[0])), int(round(p[1]))), 2, color, -1, cv2.LINE_AA)

        point_track_layer[:, :, :] = np.clip(
            point_track_layer + canvas.astype(np.float32) * (0.35 + 0.75 * pulse) * link_opacity,
            0,
            255,
        )
        frame[:, :, :] = np.clip(
            frame.astype(np.float32) + point_track_layer * opacity,
            0,
            255,
        ).astype(np.uint8)

    track_prev_gray = gray_now
    return track_prev_gray, track_points, point_track_layer


def build_video_encoder_cmd(
    width: int,
    height: int,
    fps: float,
    encoder: str,
    bitrate: int,
    temp_video: Path,
) -> list[str]:
    cmd = [
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
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        encoder,
    ]

    if bitrate > 0:
        cmd += ["-b:v", str(bitrate), "-maxrate", str(bitrate), "-bufsize", str(max(bitrate * 2, 1_000_000))]
    else:
        if encoder in {"libx264", "libx265"}:
            cmd += ["-crf", "18", "-preset", "slow"]
        elif encoder == "libvpx-vp9":
            cmd += ["-b:v", "0", "-crf", "30"]

    cmd += ["-pix_fmt", "yuv420p", str(temp_video)]
    return cmd


def mux_original_audio(temp_video: Path, input_path: Path, output_path: Path) -> None:
    cmd_copy = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(temp_video),
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-shortest",
        str(output_path),
    ]
    copy_proc = subprocess.run(cmd_copy)
    if copy_proc.returncode == 0:
        return

    cmd_reencode_audio = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(temp_video),
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "320k",
        "-shortest",
        str(output_path),
    ]
    subprocess.run(cmd_reencode_audio, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create VoidStar start/end glitch title hooks on a video.")
    parser.add_argument("input_video", help="Input video path")
    parser.add_argument("--output", default="", help="Output video path (optional)")
    parser.add_argument("--outdir", default="", help="Output directory (defaults to input directory)")
    parser.add_argument("--title", default="VOIDSTAR", help="Primary title text")
    parser.add_argument("--secondary-text", default="AUDIO • MOTION • GLITCH", help="Secondary title text")
    parser.add_argument("--newline-token", default="|", help="Token interpreted as newline in title fields (also supports literal \\n)")
    parser.add_argument("--background-dim", type=float, default=0.52, help="Background dim amount [0..1], lower keeps more source visible")
    parser.add_argument("--title-layer-dim", type=float, default=0.50, help="Gray title-layer shade amount [0..1] behind glitch text")
    parser.add_argument("--text-margin-ratio", type=float, default=0.08, help="Horizontal text margins as frame-width ratio [0.02..0.25]")
    parser.add_argument("--title-max-height-ratio", type=float, default=0.22, help="Max title block height as frame-height ratio")
    parser.add_argument("--secondary-max-height-ratio", type=float, default=0.24, help="Max secondary block height as frame-height ratio")
    parser.add_argument("--duration", type=float, default=3.2, help="Hook duration in seconds for start and end windows")
    parser.add_argument("--fade-out-duration", type=float, default=1.0, help="Fade-out at start and mirrored fade-in at end")
    parser.add_argument("--logo", default="", help="Optional logo file with alpha channel")
    parser.add_argument("--logo-width-ratio", type=float, default=0.94, help="Logo width ratio relative to frame width")
    parser.add_argument("--logo-x-ratio", type=float, default=None, help="Logo center X position [0..1] (default keeps current center behavior)")
    parser.add_argument("--logo-y-ratio", type=float, default=None, help="Logo center Y position [0..1] (default keeps current title-hook placement)")
    parser.add_argument("--logo-opacity", type=float, default=0.82, help="Base logo opacity")
    parser.add_argument("--logo-intensity", type=float, default=1.35, help="Logo effect intensity")
    parser.add_argument("--logo-motion-track-scale", type=float, default=2.0, help="Scale factor for local motion-track bbox around logo")
    parser.add_argument("--logo-motion-track-pad-px", type=float, default=0.0, help="Signed pixel padding for local motion-track bbox")
    parser.add_argument("--logo-motion-track-max-points", type=int, default=90, help="Max number of tracked feature points")
    parser.add_argument("--logo-motion-track-quality", type=float, default=0.01, help="Corner quality level for local feature detection")
    parser.add_argument("--logo-motion-track-min-distance", type=float, default=7.0, help="Minimum tracked-point spacing in pixels")
    parser.add_argument("--logo-motion-track-refresh", type=int, default=6, help="Frames between local feature reseeding")
    parser.add_argument("--logo-motion-track-radius", type=float, default=80.0, help="Connection radius for constellation links")
    parser.add_argument("--logo-motion-track-link-neighbors", type=int, default=3, help="Max nearest neighbors to connect per point")
    parser.add_argument("--logo-motion-track-link-thickness", type=int, default=1, help="Line thickness for point connections")
    parser.add_argument("--logo-motion-track-link-opacity", type=float, default=1.0, help="Multiplier for connection intensity [0..2]")
    parser.add_argument("--logo-motion-track-opacity", type=float, default=0.72, help="Overlay opacity for local point-track effect [0..1]")
    parser.add_argument("--logo-motion-track-decay", type=float, default=0.90, help="Trail decay for local point-track layer [0..0.999]")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")

    args = parser.parse_args()

    if args.duration <= 0:
        die("--duration must be > 0")
    if args.fade_out_duration < 0:
        die("--fade-out-duration must be >= 0")

    title_text = normalize_text_arg(args.title, args.newline_token)
    secondary_text = normalize_text_arg(args.secondary_text, args.newline_token)

    input_path = Path(args.input_video).expanduser().resolve()
    if not input_path.exists():
        die(f"Input video not found: {input_path}")

    require_cmd("ffmpeg")
    require_cmd("ffprobe")

    output_dir = Path(args.outdir).expanduser().resolve() if args.outdir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = make_default_output_path(input_path, output_dir, args)

    probe = ffprobe_info(input_path)
    streams = probe.get("streams", [])
    format_info = probe.get("format", {})
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not video_stream:
        die("No video stream found in input")

    src_codec = str(video_stream.get("codec_name", "h264"))
    fps = parse_fps(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate"))
    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))
    duration = float(video_stream.get("duration") or format_info.get("duration") or 0.0)
    if duration <= 0:
        die("Could not determine input duration")

    bitrate_text = video_stream.get("bit_rate") or format_info.get("bit_rate") or "0"
    try:
        bitrate = int(float(bitrate_text))
    except Exception:
        bitrate = 0

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        die(f"Failed to open input video: {input_path}")

    cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if cap_w > 0 and cap_h > 0:
        width, height = cap_w, cap_h

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        frame_count = max(1, int(round(duration * fps)))

    encoder = choose_encoder(src_codec)

    fd, temp_name = tempfile.mkstemp(prefix="voidstar_title_hook_video_", suffix=".mp4")
    os.close(fd)
    temp_video = Path(temp_name)

    logo_rgba = None
    if args.logo:
        logo_path = Path(args.logo).expanduser().resolve()
        if not logo_path.exists():
            die(f"Logo file not found: {logo_path}")
        logo_rgba = cv2.imread(str(logo_path), cv2.IMREAD_UNCHANGED)
        if logo_rgba is None:
            die(f"Failed to read logo image: {logo_path}")
        if logo_rgba.ndim != 3 or logo_rgba.shape[2] == 3:
            bgr = logo_rgba[:, :, :3] if logo_rgba.ndim == 3 else cv2.cvtColor(logo_rgba, cv2.COLOR_GRAY2BGR)
            alpha = np.full((bgr.shape[0], bgr.shape[1], 1), 255, dtype=np.uint8)
            logo_rgba = np.concatenate([bgr, alpha], axis=2)

    print(f"[voidstar-title-hook] input:  {input_path}")
    print(f"[voidstar-title-hook] output: {output_path}")
    print(f"[voidstar-title-hook] size:   {width}x{height} @ {fps:.3f} fps")
    print(f"[voidstar-title-hook] dur:    {duration:.2f}s  hook={args.duration:.2f}s fade={args.fade_out_duration:.2f}s")
    print(f"[voidstar-title-hook] encoder:{encoder} bitrate={bitrate if bitrate > 0 else 'auto-crf'}")

    audio_env = build_audio_envelope(input_path, fps=fps, target_frames=frame_count)

    enc_cmd = build_video_encoder_cmd(width, height, fps, encoder, bitrate, temp_video)
    ffmpeg_proc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE)

    rng = np.random.default_rng(args.seed)
    track_prev_gray = None
    track_points = np.empty((0, 1, 2), dtype=np.float32)
    point_track_layer = np.zeros((height, width, 3), dtype=np.float32)

    motion_track_scale = float(np.clip(args.logo_motion_track_scale, 0.1, 8.0))
    motion_track_pad_px = float(np.clip(args.logo_motion_track_pad_px, -2000.0, 2000.0))
    motion_track_max_points = max(4, int(args.logo_motion_track_max_points))
    motion_track_quality = float(np.clip(args.logo_motion_track_quality, 1e-4, 0.2))
    motion_track_min_distance = float(np.clip(args.logo_motion_track_min_distance, 1.0, 80.0))
    motion_track_refresh = max(1, int(args.logo_motion_track_refresh))
    motion_track_radius = float(np.clip(args.logo_motion_track_radius, 8.0, 500.0))
    motion_track_link_neighbors = max(1, int(args.logo_motion_track_link_neighbors))
    motion_track_link_thickness = max(1, int(args.logo_motion_track_link_thickness))
    motion_track_link_opacity = float(np.clip(args.logo_motion_track_link_opacity, 0.0, 2.0))
    motion_track_opacity = float(np.clip(args.logo_motion_track_opacity, 0.0, 1.0))
    motion_track_decay = float(np.clip(args.logo_motion_track_decay, 0.0, 0.999))
    background_dim = float(np.clip(args.background_dim, 0.0, 1.0))
    title_layer_dim = float(np.clip(args.title_layer_dim, 0.0, 1.0))
    text_margin_ratio = float(np.clip(args.text_margin_ratio, 0.02, 0.25))
    title_max_height_ratio = float(np.clip(args.title_max_height_ratio, 0.08, 0.40))
    secondary_max_height_ratio = float(np.clip(args.secondary_max_height_ratio, 0.06, 0.45))

    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            t = idx / fps
            a0 = start_alpha(t, args.duration, args.fade_out_duration)
            a1 = end_alpha(t, duration, args.duration, args.fade_out_duration)
            hook_alpha = max(a0, a1)

            audio_level = float(audio_env[idx]) if idx < len(audio_env) else float(audio_env[-1] if len(audio_env) else 0.0)

            if hook_alpha > 0.0:
                motion_boost = min(1.0, (0.4 + audio_level * 0.45) * hook_alpha)
                dark = np.clip(frame.astype(np.float32) * (1.0 - 0.34 * background_dim * hook_alpha), 0, 255).astype(np.uint8)
                frame = alpha_blend(frame, dark, np.full((height, width), background_dim * hook_alpha, dtype=np.float32))

                title_layer = build_title_layer(
                    width=width,
                    height=height,
                    title=title_text,
                    secondary=secondary_text,
                    base_alpha=hook_alpha,
                    dim_strength=title_layer_dim,
                    audio_level=audio_level,
                    frame_idx=idx,
                    rng=rng,
                    text_margin_ratio=text_margin_ratio,
                    title_max_height_ratio=title_max_height_ratio,
                    secondary_max_height_ratio=secondary_max_height_ratio,
                )
                frame = alpha_blend(frame, title_layer, np.full((height, width), 0.68 * hook_alpha, dtype=np.float32))

                if logo_rgba is not None:
                    pulse = 1.0 + (0.025 + 0.03 * args.logo_intensity) * min(2.0, audio_level)
                    pulse += 0.012 * math.sin(idx * 0.23)
                    logo_w = max(1, int(width * args.logo_width_ratio * pulse))
                    logo_scaled = resize_logo_rgba(logo_rgba, logo_w)

                    lh, lw = logo_scaled.shape[:2]
                    cx_ratio = 0.5 if args.logo_x_ratio is None else clamp(float(args.logo_x_ratio), 0.0, 1.0)
                    cy_ratio = None if args.logo_y_ratio is None else clamp(float(args.logo_y_ratio), 0.0, 1.0)
                    x = int(round((width * cx_ratio) - (lw * 0.5)))
                    if cy_ratio is None:
                        y = int(height * 0.50 - lh * 0.56)
                    else:
                        y = int(round((height * cy_ratio) - (lh * 0.5)))

                    glow = logo_scaled.copy()
                    glow[:, :, 0] = np.clip(glow[:, :, 0].astype(np.float32) * (1.05 + 0.35 * audio_level), 0, 255).astype(np.uint8)
                    glow[:, :, 1] = np.clip(glow[:, :, 1].astype(np.float32) * (1.08 + 0.45 * audio_level), 0, 255).astype(np.uint8)
                    glow[:, :, 2] = np.clip(glow[:, :, 2].astype(np.float32) * (1.08 + 0.52 * audio_level), 0, 255).astype(np.uint8)
                    overlay_rgba(frame, glow, x, y, opacity=float(min(1.0, args.logo_opacity * hook_alpha * (0.70 + 0.45 * audio_level))))

                    ch_x = int(2 + 4 * min(2.0, audio_level) * args.logo_intensity)
                    if ch_x > 0:
                        shifted = np.roll(logo_scaled, ch_x, axis=1)
                        shifted[:, :, 1] = 0
                        shifted[:, :, 0] = 0
                        overlay_rgba(frame, shifted, x - ch_x, y, opacity=float(0.22 * hook_alpha))

                    overlay_rgba(frame, logo_scaled, x, y, opacity=float(args.logo_opacity * hook_alpha))
                    pulse = float(np.clip(audio_level, 0.0, 1.0))
                    track_prev_gray, track_points, point_track_layer = apply_dvd_local_point_track(
                        frame=frame,
                        point_track_layer=point_track_layer,
                        track_prev_gray=track_prev_gray,
                        track_points=track_points,
                        logo_x=x,
                        logo_y=y,
                        logo_w=lw,
                        logo_h=lh,
                        phase_idx=idx,
                        fps=fps,
                        pulse=pulse,
                        scale=motion_track_scale,
                        pad_px=motion_track_pad_px,
                        max_points=motion_track_max_points,
                        quality=motion_track_quality,
                        min_distance=motion_track_min_distance,
                        refresh=motion_track_refresh,
                        radius=motion_track_radius,
                        link_neighbors=motion_track_link_neighbors,
                        link_thickness=motion_track_link_thickness,
                        link_opacity=motion_track_link_opacity,
                        opacity=motion_track_opacity,
                        decay=motion_track_decay,
                    )

                if (idx % 4) == 0 and motion_boost > 0.2:
                    gy = int(height * (0.3 + 0.4 * rng.random()))
                    gh = int(2 + 7 * rng.random())
                    shift = int((rng.random() - 0.5) * width * (0.03 + 0.06 * motion_boost))
                    chunk = frame[gy : min(height, gy + gh), :, :].copy()
                    frame[gy : min(height, gy + gh), :, :] = np.roll(chunk, shift, axis=1)
            else:
                track_prev_gray = None
                track_points = np.empty((0, 1, 2), dtype=np.float32)
                point_track_layer[:, :, :] = 0.0

            if ffmpeg_proc.stdin is None:
                die("Encoder pipe closed unexpectedly")
            ffmpeg_proc.stdin.write(frame.tobytes())
            idx += 1

        if ffmpeg_proc.stdin is not None:
            ffmpeg_proc.stdin.close()
        rc = ffmpeg_proc.wait()
        if rc != 0:
            die("FFmpeg video encoding failed")

        mux_original_audio(temp_video, input_path, output_path)
        print(f"[voidstar-title-hook] done: {output_path}")

    finally:
        cap.release()
        if ffmpeg_proc.stdin and not ffmpeg_proc.stdin.closed:
            ffmpeg_proc.stdin.close()
        if ffmpeg_proc.poll() is None:
            ffmpeg_proc.terminate()
            ffmpeg_proc.wait(timeout=3)
        temp_video.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
