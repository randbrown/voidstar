#!/usr/bin/env python3
"""
VoidStar title hook overlay builder.

Creates a high-visibility start/end title treatment with glitch text, optional logo,
audio-reactive intensity, and mirrored end behavior for short-form vertical video.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import dataclass
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


def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


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


def alpha_content_bbox(alpha: np.ndarray, threshold: float) -> tuple[int, int, int, int]:
    if alpha.size == 0:
        return 0, 0, 1, 1
    m = alpha > threshold
    if not np.any(m):
        h, w = alpha.shape[:2]
        return 0, 0, max(1, w), max(1, h)
    ys, xs = np.where(m)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def logo_content_rect_in_frame(
    logo_rgba: np.ndarray,
    logo_x: int,
    logo_y: int,
    alpha_threshold: float = 0.18,
) -> tuple[int, int, int, int]:
    if logo_rgba.ndim != 3 or logo_rgba.shape[2] < 4:
        h, w = logo_rgba.shape[:2]
        return logo_x, logo_y, w, h
    alpha = logo_rgba[:, :, 3].astype(np.float32) / 255.0
    ax0, ay0, ax1, ay1 = alpha_content_bbox(alpha, float(np.clip(alpha_threshold, 0.0, 1.0)))
    return logo_x + ax0, logo_y + ay0, max(1, ax1 - ax0), max(1, ay1 - ay0)


def build_localized_dim_mask(
    width: int,
    height: int,
    text_line_rects: list[tuple[int, int, int, int]],
    logo_rect: tuple[int, int, int, int] | None,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.float32)

    if text_line_rects:
        pad_x = max(10, int(round(width * 0.012)))
        pad_y = max(8, int(round(height * 0.010)))
        for tx, ty, tw, th in text_line_rects:
            if tw <= 0 or th <= 0:
                continue
            x0 = max(0, tx - pad_x)
            y0 = max(0, ty - pad_y)
            x1 = min(width, tx + tw + pad_x)
            y1 = min(height, ty + th + pad_y)
            cv2.rectangle(mask, (x0, y0), (x1, y1), 1.0, thickness=-1)

    if logo_rect is not None:
        lx, ly, lw, lh = logo_rect
        if lw > 0 and lh > 0:
            logo_pad_x = max(6, int(round(lw * 0.10)))
            logo_pad_y = max(6, int(round(lh * 0.12)))
            x0 = max(0, lx - logo_pad_x)
            y0 = max(0, ly - logo_pad_y)
            x1 = min(width, lx + lw + logo_pad_x)
            y1 = min(height, ly + lh + logo_pad_y)
            cv2.rectangle(mask, (x0, y0), (x1, y1), 1.0, thickness=-1)

    if np.any(mask > 0.0):
        blur_k = max(3, int(round(min(width, height) * 0.0035)))
        if (blur_k % 2) == 0:
            blur_k += 1
        mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)
        mask = np.clip(mask * 1.22, 0.0, 1.0)
    return np.clip(mask, 0.0, 1.0)


def hue_to_bgr_tint(hue_deg: float) -> np.ndarray:
    h = int((hue_deg % 360.0) * (179.0 / 360.0))
    hsv = np.array([[[h, 220, 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32)
    return np.clip(bgr / 255.0, 0.0, 1.0)


def blend_bgr(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    w = clamp01(t)
    return (
        int(round((1.0 - w) * a[0] + w * b[0])),
        int(round((1.0 - w) * a[1] + w * b[1])),
        int(round((1.0 - w) * a[2] + w * b[2])),
    )


def audio_level_to_bgr(audio_level: float) -> tuple[int, int, int]:
    level = clamp01(audio_level)
    hue = (2.0 / 3.0) * (1.0 - level)
    sat = 1.0
    val = 1.0
    r_f, g_f, b_f = colorsys.hsv_to_rgb(hue, sat, val)
    return int(round(b_f * 255)), int(round(g_f * 255)), int(round(r_f * 255))


@dataclass
class HookSpark:
    x: float
    y: float
    vx: float
    vy: float
    life: int
    max_life: int
    radius: int
    color_bgr: tuple[int, int, int]
    charge: float = 0.0


def apply_hook_antiparticles(
    frame: np.ndarray,
    prev_gray: np.ndarray | None,
    prev_points: np.ndarray,
    sparks: list[HookSpark],
    frame_idx: int,
    audio_level: float,
    rng: np.random.Generator,
    max_points: int,
    track_refresh: int,
    motion_threshold: float,
    spark_rate: float,
    spark_life_frames: int,
    spark_speed: float,
    spark_jitter: float,
    spark_size: float,
    spark_opacity: float,
) -> tuple[np.ndarray, np.ndarray, list[HookSpark], np.ndarray]:
    frame_h, frame_w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    movers: list[tuple[float, float, float]] = []

    antiparticle_palette_bgr = [
        (255, 255, 90),
        (255, 120, 40),
        (255, 90, 255),
        (140, 255, 255),
        (255, 70, 170),
        (220, 130, 255),
    ]

    if prev_gray is not None and prev_points.size > 0:
        p1, st, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            gray,
            prev_points,
            None,
            winSize=(21, 21),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        if p1 is not None and st is not None:
            good_new = p1[st.reshape(-1) == 1].reshape(-1, 2)
            good_old = prev_points[st.reshape(-1) == 1].reshape(-1, 2)
            if good_new.size > 0 and good_old.size > 0:
                motion = np.linalg.norm(good_new - good_old, axis=1)
                for i, speed in enumerate(motion):
                    if speed >= motion_threshold:
                        x, y = good_new[i]
                        if 0 <= x < frame_w and 0 <= y < frame_h:
                            movers.append((float(x), float(y), float(speed)))
                prev_points = good_new.reshape(-1, 1, 2).astype(np.float32)

    need_reseed = (frame_idx % max(1, track_refresh) == 0) or (prev_points.shape[0] < max(12, max_points // 4))
    if need_reseed:
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=max(16, max_points),
            qualityLevel=0.01,
            minDistance=6,
            blockSize=7,
        )
        if pts is not None and pts.size > 0:
            prev_points = pts.astype(np.float32)

    reactive_mult = 1.0 + (0.8 * audio_level)
    spawn_prob = clamp01(spark_rate * (0.35 + (0.65 * reactive_mult)))
    spawn_prob = clamp01(spawn_prob * 1.25)

    for x, y, speed in movers:
        if float(rng.random()) > spawn_prob:
            continue
        n_spawn = 1
        if audio_level > 0.8 and float(rng.random()) < 0.35:
            n_spawn = 2
        if float(rng.random()) < 0.45:
            n_spawn += 1

        for _ in range(n_spawn):
            angle = float(rng.uniform(0.0, 2.0 * math.pi))
            base_speed = spark_speed * (0.65 + 0.45 * min(3.0, speed / 4.0))
            vel = base_speed * reactive_mult
            vx = math.cos(angle) * vel + float(rng.uniform(-spark_jitter, spark_jitter))
            vy = math.sin(angle) * vel + float(rng.uniform(-spark_jitter, spark_jitter))
            life = max(2, int(round(spark_life_frames * (0.8 + 0.5 * audio_level))))
            radius = max(1, int(round(spark_size * (0.8 + 0.6 * audio_level))))

            speed_level = clamp01(speed / 6.0)
            anti_energy = clamp01((0.72 * audio_level) + (0.28 * speed_level) + float(rng.uniform(-0.08, 0.08)))
            palette_color = antiparticle_palette_bgr[int(rng.integers(0, len(antiparticle_palette_bgr)))]
            audio_color = audio_level_to_bgr(anti_energy)
            spark_color_bgr = blend_bgr(palette_color, audio_color, 0.58 + (0.30 * anti_energy))

            charge = 1.0 if float(rng.random()) < 0.5 else -1.0
            sparks.append(
                HookSpark(
                    x=x,
                    y=y,
                    vx=vx,
                    vy=vy,
                    life=life,
                    max_life=life,
                    radius=radius,
                    color_bgr=spark_color_bgr,
                    charge=charge,
                )
            )

            if float(rng.random()) < 0.8:
                anti_vx = -vx + float(rng.uniform(-0.45, 0.45))
                anti_vy = -vy + float(rng.uniform(-0.45, 0.45))
                anti_life = max(2, int(round(life * float(rng.uniform(0.85, 1.15)))))
                anti_radius = max(1, int(round(radius * float(rng.uniform(0.85, 1.2)))))
                complement_color = (
                    max(0, min(255, 255 - spark_color_bgr[0] + int(rng.integers(-25, 26)))),
                    max(0, min(255, 255 - spark_color_bgr[1] + int(rng.integers(-25, 26)))),
                    max(0, min(255, 255 - spark_color_bgr[2] + int(rng.integers(-25, 26)))),
                )
                anti_audio_color = audio_level_to_bgr(clamp01(audio_level + float(rng.uniform(0.05, 0.2))))
                anti_color = blend_bgr(complement_color, anti_audio_color, 0.42 + (0.30 * clamp01(audio_level)))
                sparks.append(
                    HookSpark(
                        x=x + float(rng.uniform(-1.5, 1.5)),
                        y=y + float(rng.uniform(-1.5, 1.5)),
                        vx=anti_vx,
                        vy=anti_vy,
                        life=anti_life,
                        max_life=anti_life,
                        radius=anti_radius,
                        color_bgr=anti_color,
                        charge=-charge,
                    )
                )

    overlay = np.zeros_like(frame, dtype=np.uint8)
    alive: list[HookSpark] = []

    for sp in sparks:
        cx = frame_w * 0.5
        cy = frame_h * 0.5
        dx = sp.x - cx
        dy = sp.y - cy
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > 1e-6:
            swirl = 0.15 * sp.charge / max(28.0, dist)
            sp.vx += -dy * swirl
            sp.vy += dx * swirl

        sp.x += sp.vx
        sp.y += sp.vy
        sp.vx *= 0.972
        sp.vy *= 0.972
        sp.life -= 1

        if sp.life <= 0:
            continue
        if sp.x < -8 or sp.x >= frame_w + 8 or sp.y < -8 or sp.y >= frame_h + 8:
            continue

        life_ratio = sp.life / max(1, sp.max_life)
        radius = max(1, int(round(sp.radius * (0.65 + 0.7 * life_ratio))))
        cv2.circle(overlay, (int(round(sp.x)), int(round(sp.y))), radius, sp.color_bgr, -1, cv2.LINE_AA)
        if radius >= 2 and life_ratio > 0.35:
            cv2.circle(overlay, (int(round(sp.x)), int(round(sp.y))), 1, (255, 255, 255), -1, cv2.LINE_AA)
        alive.append(sp)

    alpha = clamp01(spark_opacity * (0.55 + 0.75 * min(1.2, audio_level)))
    if alpha > 0:
        frame = cv2.addWeighted(frame, 1.0, overlay, alpha, 0.0)

    return gray, prev_points, alive, frame


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
    text_align: str,
    title_jitter_audio_multiplier: float,
    font_family: str,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    layer = np.zeros((height, width, 3), dtype=np.uint8)
    text_line_rects: list[tuple[int, int, int, int]] = []

    if base_alpha <= 0:
        return layer, text_line_rects

    dim_level = float(np.clip(dim_strength, 0.0, 1.0)) * base_alpha
    dim = np.full_like(layer, int(255 * dim_level), dtype=np.uint8)
    layer = alpha_blend(layer, dim, np.full((height, width), 1.0, dtype=np.float32))

    if dim_strength > 1e-6:
        band_strength = 0.18 + 0.22 * min(1.8, audio_level)
        for y in range(0, height, 4):
            value = int(max(0, min(255, 15 + 55 * band_strength * (0.5 + 0.5 * math.sin((y + frame_idx * 2) * 0.045)))))
            layer[y : y + 1, :, :] = np.maximum(layer[y : y + 1, :, :], value)

        noise = rng.normal(0, 14 + 22 * min(2.0, audio_level), size=(height, width, 1)).astype(np.float32)
        layer = np.clip(layer.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if font_family == "monospace":
        title_font = cv2.FONT_HERSHEY_PLAIN
        subtitle_font = cv2.FONT_HERSHEY_PLAIN
    else:
        title_font = cv2.FONT_HERSHEY_DUPLEX
        subtitle_font = cv2.FONT_HERSHEY_SIMPLEX

    title_scale = max(2.0, min(8.0, width / 165.0))

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
    secondary_start_scale = max(0.42, title_scale * 0.78)
    secondary_scale, secondary_th, secondary_line_step, secondary_sizes = fit_text_block(
        lines=secondary_lines,
        font=subtitle_font,
        start_scale=secondary_start_scale,
        min_scale=0.42,
        thickness_mul=1.8,
        line_spacing=1.18,
        max_width=max_text_w,
        max_height=secondary_max_h,
    )
    if secondary_scale >= title_scale:
        secondary_scale = max(0.42, title_scale * 0.96)
        secondary_th = max(1, int(secondary_scale * 1.8))
        secondary_sizes = [
            cv2.getTextSize(line if line else " ", subtitle_font, secondary_scale, secondary_th)
            for line in secondary_lines
        ]
        secondary_line_step = max(1, int(max(sz[1] + bl for sz, bl in secondary_sizes) * 1.18))

    title_block_h = max(1, (len(title_lines) - 1) * title_line_step + max(sz[1] for sz, _ in title_sizes))
    secondary_block_h = max(1, (len(secondary_lines) - 1) * secondary_line_step + max(sz[1] for sz, _ in secondary_sizes))
    title_block_w = max(sz[0] for sz, _ in title_sizes)
    secondary_block_w = max(sz[0] for sz, _ in secondary_sizes)

    title_center_y = int(height * 0.48)
    secondary_center_y = int(height * 0.63)
    title_top = title_center_y - title_block_h // 2
    secondary_top = secondary_center_y - secondary_block_h // 2

    top_margin = int(height * 0.05)
    bottom_margin = int(height * 0.05)
    min_gap = max(8, int(height * 0.02))

    secondary_top = max(secondary_top, title_top + title_block_h + min_gap)

    bottom_limit = height - bottom_margin
    overflow = (secondary_top + secondary_block_h) - bottom_limit
    if overflow > 0:
        title_top -= overflow
        secondary_top -= overflow

    if title_top < top_margin:
        shift = top_margin - title_top
        title_top += shift
        secondary_top += shift

    secondary_top = max(secondary_top, title_top + title_block_h + min_gap)
    secondary_top = min(secondary_top, max(top_margin, bottom_limit - secondary_block_h))

    if secondary_top < title_top + title_block_h + min_gap:
        need = (title_top + title_block_h + min_gap) - secondary_top
        title_top = max(top_margin, title_top - need)
        secondary_top = min(max(top_margin, title_top + title_block_h + min_gap), max(top_margin, bottom_limit - secondary_block_h))

    title_block_y0 = title_top + max(sz[1] for sz, _ in title_sizes)
    secondary_block_y0 = secondary_top + max(sz[1] for sz, _ in secondary_sizes)

    jitter_multiplier = max(0.0, float(title_jitter_audio_multiplier))
    jitter_audio = min(2.0, max(0.0, audio_level * jitter_multiplier))
    jitter_base = 0 if jitter_multiplier <= 1e-6 else 3
    jitter = int(jitter_base + 12 * jitter_audio)
    ox = int(rng.integers(-jitter, jitter + 1))
    oy = int(rng.integers(-jitter, jitter + 1))

    align_mode = text_align if text_align in {"center", "left"} else "center"
    shared_block_w = max(title_block_w, secondary_block_w)
    shared_left = max(text_margin_px, min(width - text_margin_px - shared_block_w, cx - shared_block_w // 2))

    red = (40, 30, 245)
    cyan = (245, 245, 40)
    white = (245, 245, 245)
    secondary_color = (190, 190, 190)

    for i, line in enumerate(title_lines):
        line_for_size = line if line else " "
        line_size, line_baseline = cv2.getTextSize(line_for_size, title_font, title_scale, title_th)
        if align_mode == "left":
            title_x = int(shared_left)
        else:
            title_x = cx - line_size[0] // 2
        title_y = title_block_y0 + i * title_line_step
        cv2.putText(layer, line, (title_x - 3 + ox, title_y + oy), title_font, title_scale, cyan, title_th, cv2.LINE_AA)
        cv2.putText(layer, line, (title_x + 3 + ox, title_y - oy), title_font, title_scale, red, title_th, cv2.LINE_AA)
        cv2.putText(layer, line, (title_x + ox, title_y + oy), title_font, title_scale, white, title_th + 1, cv2.LINE_AA)
        text_line_rects.append(
            (
                int(title_x + ox - 7),
                int(title_y + oy - line_size[1] - 7),
                int(line_size[0] + 14),
                int(line_size[1] + line_baseline + 14),
            )
        )

    for i, line in enumerate(secondary_lines):
        line_for_size = line if line else " "
        line_size, line_baseline = cv2.getTextSize(line_for_size, subtitle_font, secondary_scale, secondary_th)
        if align_mode == "left":
            subtitle_x = int(shared_left)
        else:
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
        text_line_rects.append(
            (
                int(subtitle_x - 5),
                int(subtitle_y - line_size[1] - 6),
                int(line_size[0] + 10),
                int(line_size[1] + line_baseline + 12),
            )
        )

    if (frame_idx % 5) == 0:
        gline_y = int(height * (0.35 + 0.3 * rng.random()))
        gline_h = int(2 + 6 * rng.random())
        x_shift = int((rng.random() - 0.5) * width * 0.09)
        band = layer[gline_y : min(height, gline_y + gline_h), :, :].copy()
        layer[gline_y : min(height, gline_y + gline_h), :, :] = np.roll(band, x_shift, axis=1)

    return layer, text_line_rects


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
    parser.add_argument("--background-dim", type=float, default=0.2, help="Background dim amount [0..1], lower keeps more source visible")
    parser.add_argument("--title-layer-dim", type=float, default=0.1, help="Gray title-layer shade amount [0..1] behind glitch text")
    parser.add_argument("--text-margin-ratio", type=float, default=0.05, help="Horizontal text margins as frame-width ratio [0.02..0.25]")
    parser.add_argument("--text-align", choices=["center", "left"], default="center", help="Text alignment mode for title and secondary blocks")
    parser.add_argument("--font-family", choices=["classic", "monospace"], default="classic", help="Title text font family style")
    parser.add_argument("--title-jitter-audio-multiplier", type=float, default=1.0, help="Multiplier for audio reactivity driving title text jitter")
    parser.add_argument("--title-max-height-ratio", type=float, default=0.66, help="Max title block height as frame-height ratio")
    parser.add_argument("--secondary-max-height-ratio", type=float, default=0.66, help="Max secondary block height as frame-height ratio")
    parser.add_argument("--duration", type=float, default=3.2, help="Hook duration in seconds for start and end windows")
    parser.add_argument("--fade-out-duration", type=float, default=1.0, help="Fade-out at start and mirrored fade-in at end")
    parser.add_argument("--logo", default="", help="Optional logo file with alpha channel")
    parser.add_argument("--logo-width-ratio", type=float, default=0.94, help="Logo width ratio relative to frame width")
    parser.add_argument("--logo-x-ratio", type=float, default=None, help="Logo center X position [0..1] (default keeps current center behavior)")
    parser.add_argument("--logo-y-ratio", type=float, default=None, help="Logo center Y position [0..1] (default keeps current title-hook placement)")
    parser.add_argument("--logo-opacity", type=float, default=0.82, help="Base logo opacity")
    parser.add_argument("--logo-alpha-threshold", type=float, default=0.18, help="Alpha cutoff [0..1] for logo content bounds (higher trims faint edges)")
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
    parser.add_argument("--title-hook-sparks", action="store_true", help="Enable antiparticles sparks only during title-hook windows")
    parser.add_argument("--title-hook-sparks-max-points", type=int, default=128, help="Max tracked points for hook sparks")
    parser.add_argument("--title-hook-sparks-track-refresh", type=int, default=5, help="Frames between hook-sparks feature reseeding")
    parser.add_argument("--title-hook-sparks-motion-threshold", type=float, default=0.25, help="Min motion to emit hook sparks")
    parser.add_argument("--title-hook-sparks-rate", type=float, default=0.25, help="Hook spark spawn rate per moving point")
    parser.add_argument("--title-hook-sparks-life-frames", type=int, default=30, help="Hook spark lifetime in frames")
    parser.add_argument("--title-hook-sparks-speed", type=float, default=0.25, help="Hook spark speed in px/frame")
    parser.add_argument("--title-hook-sparks-jitter", type=float, default=1.1, help="Hook spark velocity jitter")
    parser.add_argument("--title-hook-sparks-size", type=float, default=2.2, help="Hook spark radius base")
    parser.add_argument("--title-hook-sparks-opacity", type=float, default=0.50, help="Hook sparks overlay opacity [0..1]")
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

    print("[voidstar-title-hook] phase: audio envelope extraction...")
    audio_env_t0 = time.monotonic()
    audio_env = build_audio_envelope(input_path, fps=fps, target_frames=frame_count)
    print(f"[voidstar-title-hook] phase: audio envelope ready in {time.monotonic() - audio_env_t0:.2f}s")

    enc_cmd = build_video_encoder_cmd(width, height, fps, encoder, bitrate, temp_video)
    ffmpeg_proc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE)
    print("[voidstar-title-hook] phase: frame processing + encode...")

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
    hook_sparks_enabled = bool(args.title_hook_sparks)
    hook_sparks_max_points = max(16, int(args.title_hook_sparks_max_points))
    hook_sparks_track_refresh = max(1, int(args.title_hook_sparks_track_refresh))
    hook_sparks_motion_threshold = float(np.clip(args.title_hook_sparks_motion_threshold, 0.0, 20.0))
    hook_sparks_rate = float(np.clip(args.title_hook_sparks_rate, 0.0, 1.0))
    hook_sparks_life_frames = max(2, int(args.title_hook_sparks_life_frames))
    hook_sparks_speed = float(np.clip(args.title_hook_sparks_speed, 0.01, 40.0))
    hook_sparks_jitter = float(np.clip(args.title_hook_sparks_jitter, 0.0, 8.0))
    hook_sparks_size = float(np.clip(args.title_hook_sparks_size, 0.5, 20.0))
    hook_sparks_opacity = float(np.clip(args.title_hook_sparks_opacity, 0.0, 1.0))
    background_dim = float(np.clip(args.background_dim, 0.0, 1.0))
    title_layer_dim = float(np.clip(args.title_layer_dim, 0.0, 1.0))
    text_margin_ratio = float(np.clip(args.text_margin_ratio, 0.02, 0.25))
    text_align = args.text_align
    font_family = args.font_family
    title_jitter_audio_multiplier = max(0.0, float(args.title_jitter_audio_multiplier))
    title_max_height_ratio = float(np.clip(args.title_max_height_ratio, 0.08, 0.40))
    secondary_max_height_ratio = float(np.clip(args.secondary_max_height_ratio, 0.06, 0.45))
    logo_alpha_threshold = float(np.clip(args.logo_alpha_threshold, 0.0, 1.0))

    hook_sparks_prev_gray = None
    hook_sparks_prev_points = np.empty((0, 1, 2), dtype=np.float32)
    hook_sparks: list[HookSpark] = []

    try:
        idx = 0
        progress_interval_frames = max(1, int(round(fps)))
        encode_t0 = time.monotonic()
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

                title_layer, text_line_rects = build_title_layer(
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
                    text_align=text_align,
                    title_jitter_audio_multiplier=title_jitter_audio_multiplier,
                    font_family=font_family,
                )

                logo_scaled = None
                logo_x = 0
                logo_y = 0
                logo_w = 0
                logo_h = 0
                logo_dim_rect = None
                if logo_rgba is not None:
                    logo_pulse = 1.0 + (0.025 + 0.03 * args.logo_intensity) * min(2.0, audio_level)
                    logo_pulse += 0.012 * math.sin(idx * 0.23)
                    logo_w = max(1, int(width * args.logo_width_ratio * logo_pulse))
                    logo_scaled = resize_logo_rgba(logo_rgba, logo_w)
                    logo_h, logo_w = logo_scaled.shape[:2]

                    cx_ratio = 0.5 if args.logo_x_ratio is None else clamp(float(args.logo_x_ratio), 0.0, 1.0)
                    cy_ratio = None if args.logo_y_ratio is None else clamp(float(args.logo_y_ratio), 0.0, 1.0)
                    logo_x = int(round((width * cx_ratio) - (logo_w * 0.5)))
                    if cy_ratio is None:
                        logo_y = int(height * 0.50 - logo_h * 0.56)
                    else:
                        logo_y = int(round((height * cy_ratio) - (logo_h * 0.5)))
                    logo_dim_rect = logo_content_rect_in_frame(
                        logo_scaled,
                        logo_x,
                        logo_y,
                        alpha_threshold=logo_alpha_threshold,
                    )

                if background_dim > 1e-6:
                    local_dim_mask = build_localized_dim_mask(
                        width=width,
                        height=height,
                        text_line_rects=text_line_rects,
                        logo_rect=logo_dim_rect,
                    )
                    dim_alpha = np.clip(local_dim_mask * (1.30 * background_dim * hook_alpha), 0.0, 1.0)
                    if np.any(dim_alpha > 1e-6):
                        black = np.zeros_like(frame, dtype=np.uint8)
                        frame = alpha_blend(frame, black, dim_alpha)

                title_alpha_map = np.full((height, width), 0.68 * hook_alpha, dtype=np.float32)
                if title_layer_dim <= 1e-6:
                    text_mask = cv2.cvtColor(title_layer, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                    text_mask = np.clip(text_mask * 1.25, 0.0, 1.0)
                    title_alpha_map *= text_mask
                frame = alpha_blend(frame, title_layer, title_alpha_map)

                if logo_scaled is not None:
                    glow = logo_scaled.copy()
                    glow[:, :, 0] = np.clip(glow[:, :, 0].astype(np.float32) * (1.05 + 0.35 * audio_level), 0, 255).astype(np.uint8)
                    glow[:, :, 1] = np.clip(glow[:, :, 1].astype(np.float32) * (1.08 + 0.45 * audio_level), 0, 255).astype(np.uint8)
                    glow[:, :, 2] = np.clip(glow[:, :, 2].astype(np.float32) * (1.08 + 0.52 * audio_level), 0, 255).astype(np.uint8)
                    overlay_rgba(frame, glow, logo_x, logo_y, opacity=float(min(1.0, args.logo_opacity * hook_alpha * (0.70 + 0.45 * audio_level))))

                    ch_x = int(2 + 4 * min(2.0, audio_level) * args.logo_intensity)
                    if ch_x > 0:
                        shifted = np.roll(logo_scaled, ch_x, axis=1)
                        shifted[:, :, 1] = 0
                        shifted[:, :, 0] = 0
                        overlay_rgba(frame, shifted, logo_x - ch_x, logo_y, opacity=float(0.22 * hook_alpha))

                    overlay_rgba(frame, logo_scaled, logo_x, logo_y, opacity=float(args.logo_opacity * hook_alpha))
                    pulse = float(np.clip(audio_level, 0.0, 1.0))
                    track_prev_gray, track_points, point_track_layer = apply_dvd_local_point_track(
                        frame=frame,
                        point_track_layer=point_track_layer,
                        track_prev_gray=track_prev_gray,
                        track_points=track_points,
                        logo_x=logo_x,
                        logo_y=logo_y,
                        logo_w=logo_w,
                        logo_h=logo_h,
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

                if hook_sparks_enabled:
                    hook_sparks_prev_gray, hook_sparks_prev_points, hook_sparks, frame = apply_hook_antiparticles(
                        frame=frame,
                        prev_gray=hook_sparks_prev_gray,
                        prev_points=hook_sparks_prev_points,
                        sparks=hook_sparks,
                        frame_idx=idx,
                        audio_level=audio_level,
                        rng=rng,
                        max_points=hook_sparks_max_points,
                        track_refresh=hook_sparks_track_refresh,
                        motion_threshold=hook_sparks_motion_threshold,
                        spark_rate=hook_sparks_rate,
                        spark_life_frames=hook_sparks_life_frames,
                        spark_speed=hook_sparks_speed,
                        spark_jitter=hook_sparks_jitter,
                        spark_size=hook_sparks_size,
                        spark_opacity=hook_sparks_opacity,
                    )
            else:
                track_prev_gray = None
                track_points = np.empty((0, 1, 2), dtype=np.float32)
                point_track_layer[:, :, :] = 0.0
                hook_sparks_prev_gray = None
                hook_sparks_prev_points = np.empty((0, 1, 2), dtype=np.float32)
                hook_sparks = []

            if ffmpeg_proc.stdin is None:
                die("Encoder pipe closed unexpectedly")
            ffmpeg_proc.stdin.write(frame.tobytes())
            idx += 1

            if idx == 1 or idx % progress_interval_frames == 0 or idx >= frame_count:
                elapsed = max(1e-6, time.monotonic() - encode_t0)
                proc_fps = idx / elapsed
                remain = max(0, frame_count - idx)
                eta = remain / max(1e-6, proc_fps)
                pct = (100.0 * idx / max(1, frame_count))
                if hook_sparks_enabled:
                    print(
                        f"[voidstar-title-hook] frame={idx}/{frame_count} ({pct:.1f}%) fps={proc_fps:.2f} eta={eta:.1f}s sparks={len(hook_sparks)}",
                        flush=True,
                    )
                else:
                    print(
                        f"[voidstar-title-hook] frame={idx}/{frame_count} ({pct:.1f}%) fps={proc_fps:.2f} eta={eta:.1f}s",
                        flush=True,
                    )

        if ffmpeg_proc.stdin is not None:
            ffmpeg_proc.stdin.close()
        rc = ffmpeg_proc.wait()
        if rc != 0:
            die("FFmpeg video encoding failed")

        print("[voidstar-title-hook] phase: mux original audio...")
        mux_t0 = time.monotonic()
        mux_original_audio(temp_video, input_path, output_path)
        print(f"[voidstar-title-hook] phase: mux finished in {time.monotonic() - mux_t0:.2f}s")
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
