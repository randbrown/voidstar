#!/usr/bin/env python3

from __future__ import annotations

import argparse
import colorsys
import math
import os
import random
import shlex
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def log(msg: str) -> None:
    prefix = os.environ.get("VOIDSTAR_LOG_PREFIX", "voidstar")
    print(f"[{prefix}] {msg}", flush=True)


def run_cmd(cmd: List[str], heartbeat_label: str | None = None, heartbeat_interval_sec: float = 1.0) -> None:
    log("▶ " + " ".join(shlex.quote(c) for c in cmd))
    if heartbeat_label is None:
        p = subprocess.run(cmd)
        if p.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {p.returncode}")
        return

    proc = subprocess.Popen(cmd)
    start = time.time()
    interval = max(0.2, float(heartbeat_interval_sec))
    while True:
        rc = proc.poll()
        if rc is not None:
            if rc != 0:
                raise RuntimeError(f"Command failed with exit code {rc}")
            return
        elapsed = time.time() - start
        log(f"{heartbeat_label} ... elapsed={elapsed:.1f}s")
        time.sleep(interval)


def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def parse_bool(text: str) -> bool:
    return str(text).strip().lower() in {"1", "true", "yes", "on"}


def safe_slug(text: str) -> str:
    out = text.strip().replace(" ", "_")
    return "".join(ch for ch in out if ch.isalnum() or ch in "._-")


def gpu_available() -> bool:
    try:
        p = subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return p.returncode == 0
    except Exception:
        return False


def ffmpeg_has_encoder(name: str) -> bool:
    try:
        p = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        return name in (p.stdout or "")
    except Exception:
        return False


def pick_video_encoder(mode: str) -> str:
    mode = mode.lower().strip()
    if mode in {"libx264", "hevc_nvenc", "h264_nvenc"}:
        return mode
    if mode != "auto":
        raise ValueError("--video-encoder must be auto|libx264|h264_nvenc|hevc_nvenc")

    if gpu_available() and ffmpeg_has_encoder("hevc_nvenc"):
        return "hevc_nvenc"
    if gpu_available() and ffmpeg_has_encoder("h264_nvenc"):
        return "h264_nvenc"
    return "libx264"


def output_path_for(input_path: Path, out_dir: Path, args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output).expanduser().resolve()

    stem = input_path.stem
    suffix = (
        f"sparks_sr{args.spark_rate:.2f}_sp{args.spark_speed:.1f}_"
        f"life{args.spark_life_frames}_mg{args.motion_threshold:.1f}_"
        f"ag{args.audio_reactive_gain:.2f}"
    ).replace(".", "p")
    name = f"{safe_slug(stem)}_{suffix}.mp4"
    return out_dir / name


def parse_rgb(rgb_text: str) -> Tuple[int, int, int]:
    try:
        parts = [int(x.strip()) for x in rgb_text.split(",")]
    except Exception as exc:
        raise ValueError(f"Invalid --color-rgb value: {rgb_text!r}") from exc

    if len(parts) != 3:
        raise ValueError(f"Invalid --color-rgb value (need R,G,B): {rgb_text!r}")

    r = max(0, min(255, parts[0]))
    g = max(0, min(255, parts[1]))
    b = max(0, min(255, parts[2]))
    return r, g, b


def rgb_to_bgr(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    r, g, b = rgb
    return b, g, r


def blend_bgr(a: Tuple[int, int, int], b: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    w = clamp01(t)
    return (
        int(round((1.0 - w) * a[0] + w * b[0])),
        int(round((1.0 - w) * a[1] + w * b[1])),
        int(round((1.0 - w) * a[2] + w * b[2])),
    )


def apply_color_temperature_bgr(color_bgr: Tuple[int, int, int], temperature: float) -> Tuple[int, int, int]:
    temp = max(-1.0, min(1.0, float(temperature)))
    if abs(temp) < 1e-6:
        return color_bgr

    b, g, r = color_bgr
    if temp > 0.0:
        b = int(round(b * (1.0 - (0.60 * temp))))
        g = int(round(g * (1.0 - (0.22 * temp))))
        r = int(round(r + ((255 - r) * (0.30 * temp))))
    else:
        cool = -temp
        r = int(round(r * (1.0 - (0.55 * cool))))
        g = int(round(g * (1.0 - (0.12 * cool))))
        b = int(round(b + ((255 - b) * (0.34 * cool))))

    return (
        max(0, min(255, b)),
        max(0, min(255, g)),
        max(0, min(255, r)),
    )


def audio_level_to_bgr(audio_level: float) -> Tuple[int, int, int]:
    level = clamp01(audio_level)
    hue = (2.0 / 3.0) * (1.0 - level)
    sat = 1.0
    val = 1.0
    r_f, g_f, b_f = colorsys.hsv_to_rgb(hue, sat, val)
    return int(round(b_f * 255)), int(round(g_f * 255)), int(round(r_f * 255))


def extract_audio_envelope(input_path: Path, start: float, duration: float, fps: float, gain: float, smooth: float) -> np.ndarray:
    with tempfile.NamedTemporaryFile(prefix="voidstar_sparks_audio_", suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)

    try:
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", str(max(0.0, start)),
        ]
        if duration > 0:
            cmd += ["-t", str(duration)]
        cmd += [
            "-i", str(input_path),
            "-vn", "-ac", "1", "-ar", "48000", str(wav_path),
        ]
        run_cmd(cmd, heartbeat_label="audio envelope extraction", heartbeat_interval_sec=0.75)

        with wave.open(str(wav_path), "rb") as wavf:
            sr = wavf.getframerate()
            n = wavf.getnframes()
            data = wavf.readframes(n)

        if not data:
            return np.zeros(1, dtype=np.float32)

        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        hop = max(1, int(round(sr / max(1e-6, fps))))
        nframes = max(1, int(math.ceil(len(samples) / hop)))

        env = np.zeros(nframes, dtype=np.float32)
        for i in range(nframes):
            a = i * hop
            b = min(len(samples), a + hop)
            seg = samples[a:b]
            if seg.size == 0:
                env[i] = 0.0
            else:
                env[i] = float(np.sqrt(np.mean(seg * seg) + 1e-10))

        lo = np.percentile(env, 5)
        hi = np.percentile(env, 95)
        env = np.clip((env - lo) / (hi - lo + 1e-9), 0.0, 1.0)

        alpha = clamp01(smooth)
        for i in range(1, len(env)):
            env[i] = (alpha * env[i - 1]) + ((1.0 - alpha) * env[i])

        env = np.clip(env * max(0.0, gain), 0.0, 2.0).astype(np.float32)
        return env
    finally:
        try:
            wav_path.unlink(missing_ok=True)
        except Exception:
            pass


@dataclass
class Spark:
    x: float
    y: float
    vx: float
    vy: float
    life: int
    max_life: int
    radius: int
    color_bgr: Tuple[int, int, int]
    charge: float = 0.0
    shape_kind: str = "dot"
    angle: float = 0.0
    ang_vel: float = 0.0
    sides: int = 4
    stroke: int = 1
    target_x: float = 0.0
    target_y: float = 0.0
    phase: float = 0.0


def draw_abstract_form(
    canvas: np.ndarray,
    x: float,
    y: float,
    radius: int,
    color_bgr: Tuple[int, int, int],
    shape_kind: str,
    angle: float,
    sides: int,
    stroke: int,
) -> None:
    cx = int(round(x))
    cy = int(round(y))
    r = max(1, int(radius))
    thick = max(1, int(stroke))

    if shape_kind == "ring":
        cv2.circle(canvas, (cx, cy), r, color_bgr, thick, cv2.LINE_AA)
        if r >= 4:
            cv2.circle(canvas, (cx, cy), max(1, r // 2), color_bgr, 1, cv2.LINE_AA)
        return

    if shape_kind == "diamond":
        pts = np.array(
            [
                [cx, cy - r],
                [cx + r, cy],
                [cx, cy + r],
                [cx - r, cy],
            ],
            dtype=np.int32,
        )
        cv2.polylines(canvas, [pts], True, color_bgr, thick, cv2.LINE_AA)
        return

    n = max(3, int(sides))
    pts_list: list[list[int]] = []
    for i in range(n):
        t = angle + ((2.0 * math.pi * i) / n)
        px = int(round(cx + (math.cos(t) * r)))
        py = int(round(cy + (math.sin(t) * r)))
        pts_list.append([px, py])
    pts = np.array(pts_list, dtype=np.int32)
    cv2.polylines(canvas, [pts], True, color_bgr, thick, cv2.LINE_AA)

    if r >= 6 and n >= 4:
        inner_scale = 0.55
        inner_list: list[list[int]] = []
        for i in range(n):
            t = -angle + ((2.0 * math.pi * i) / n)
            px = int(round(cx + (math.cos(t) * (r * inner_scale))))
            py = int(round(cy + (math.sin(t) * (r * inner_scale))))
            inner_list.append([px, py])
        inner = np.array(inner_list, dtype=np.int32)
        cv2.polylines(canvas, [inner], True, color_bgr, 1, cv2.LINE_AA)


def draw_electrostatic_arc(
    canvas: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color_bgr: Tuple[int, int, int],
    life_ratio: float,
    audio_level: float,
) -> None:
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    dist = math.sqrt((dx * dx) + (dy * dy))
    if dist < 2.0:
        cv2.line(canvas, (x1, y1), (x2, y2), color_bgr, 1, cv2.LINE_AA)
        return

    nx = -dy / dist
    ny = dx / dist
    dir_x = dx / dist
    dir_y = dy / dist

    segs = max(4, min(11, int(round(dist / 14.0)) + int(round(2.0 * clamp01(audio_level)))))
    amp = min(22.0, max(2.5, dist * (0.10 + (0.10 * clamp01(audio_level)))))
    amp *= 0.55 + (0.70 * clamp01(life_ratio))

    pts: list[tuple[int, int]] = [(x1, y1)]
    for i in range(1, segs):
        t = i / max(1, segs)
        envelope = max(0.0, 1.0 - abs((2.0 * t) - 1.0))
        perp = random.uniform(-amp, amp) * envelope
        along = random.uniform(-2.0, 2.0)
        px = (x1 + (dx * t)) + (nx * perp) + (dir_x * along)
        py = (y1 + (dy * t)) + (ny * perp) + (dir_y * along)
        pts.append((int(round(px)), int(round(py))))
    pts.append((x2, y2))

    arc = np.array(pts, dtype=np.int32)
    glow_color = blend_bgr(color_bgr, (255, 245, 230), 0.18 + (0.28 * clamp01(audio_level)))
    cv2.polylines(canvas, [arc], False, glow_color, 2, cv2.LINE_AA)
    cv2.polylines(canvas, [arc], False, color_bgr, 1, cv2.LINE_AA)

    if len(pts) >= 5 and random.random() < (0.22 + (0.30 * clamp01(audio_level))):
        branch_idx = random.randint(2, len(pts) - 3)
        bx, by = pts[branch_idx]
        branch_len = random.uniform(4.0, min(18.0, dist * 0.35))
        branch_angle = math.atan2(dy, dx) + random.uniform(-1.2, 1.2)
        ex = int(round(bx + (math.cos(branch_angle) * branch_len)))
        ey = int(round(by + (math.sin(branch_angle) * branch_len)))
        cv2.line(canvas, (bx, by), (ex, ey), color_bgr, 1, cv2.LINE_AA)


def draw_neuron_curve(
    canvas: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    base_color_bgr: Tuple[int, int, int],
    pulse_color_bgr: Tuple[int, int, int],
    travel_t: float,
    phase: float,
    life_ratio: float,
    audio_level: float,
    curvature: float,
    pulse_boost: float,
) -> tuple[int, int]:
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    dist = math.sqrt((dx * dx) + (dy * dy))
    if dist < 2.0:
        cv2.line(canvas, (x1, y1), (x2, y2), base_color_bgr, 1, cv2.LINE_AA)
        return x2, y2

    nx = -dy / dist
    ny = dx / dist
    mid_x = 0.5 * (x1 + x2)
    mid_y = 0.5 * (y1 + y2)
    curvature_strength = max(0.0, float(curvature))
    base_amp = max(3.0, dist * (0.06 + (0.11 * clamp01(audio_level))))
    amp = base_amp * (0.65 + (0.85 * curvature_strength))
    amp *= 0.55 + (0.55 * clamp01(life_ratio))
    amp_cap = 28.0 + (42.0 * max(0.0, curvature_strength - 1.0))
    amp = min(amp_cap, amp)
    bend = math.sin(phase * 0.8) * amp
    ctrl_x = mid_x + (nx * bend)
    ctrl_y = mid_y + (ny * bend)

    samples = max(10, min(36, int(round((dist / 7.0) * (0.9 + (0.25 * curvature_strength))))))
    pts: list[tuple[int, int]] = []
    for i in range(samples + 1):
        t = i / max(1, samples)
        omt = 1.0 - t
        px = (omt * omt * x1) + (2.0 * omt * t * ctrl_x) + (t * t * x2)
        py = (omt * omt * y1) + (2.0 * omt * t * ctrl_y) + (t * t * y2)
        pts.append((int(round(px)), int(round(py))))

    curve = np.array(pts, dtype=np.int32)
    glow_color = blend_bgr(base_color_bgr, (245, 245, 245), 0.18 + (0.35 * clamp01(audio_level)))
    cv2.polylines(canvas, [curve], False, glow_color, 2, cv2.LINE_AA)
    cv2.polylines(canvas, [curve], False, base_color_bgr, 1, cv2.LINE_AA)

    t = clamp01(travel_t)
    omt = 1.0 - t
    pulse_x = (omt * omt * x1) + (2.0 * omt * t * ctrl_x) + (t * t * x2)
    pulse_y = (omt * omt * y1) + (2.0 * omt * t * ctrl_y) + (t * t * y2)
    px_i = int(round(pulse_x))
    py_i = int(round(pulse_y))

    pulse_strength = clamp01(0.35 + (0.65 * clamp01(audio_level)))
    pb = max(0.0, float(pulse_boost))
    head_r = max(2, int(round(1.0 + (1.2 * pb) + (1.2 * pulse_strength))))
    glow_mix = clamp01(0.35 + (0.45 * pulse_strength))
    glow_color = blend_bgr(pulse_color_bgr, (245, 245, 245), glow_mix)
    cv2.circle(canvas, (px_i, py_i), head_r + 1, glow_color, -1, cv2.LINE_AA)
    cv2.circle(canvas, (px_i, py_i), head_r, pulse_color_bgr, -1, cv2.LINE_AA)

    tail_steps = 3
    for i in range(1, tail_steps + 1):
        dt = (0.06 * i) * (0.85 + (0.45 * clamp01(pb)))
        tt = max(0.0, t - dt)
        omt_t = 1.0 - tt
        tx = (omt_t * omt_t * x1) + (2.0 * omt_t * tt * ctrl_x) + (tt * tt * x2)
        ty = (omt_t * omt_t * y1) + (2.0 * omt_t * tt * ctrl_y) + (tt * tt * y2)
        tr = max(1, head_r - i)
        tail_color = blend_bgr(base_color_bgr, pulse_color_bgr, max(0.15, 0.55 - (0.12 * i)))
        cv2.circle(canvas, (int(round(tx)), int(round(ty))), tr, tail_color, -1, cv2.LINE_AA)
    return px_i, py_i


def draw_neuron_node(
    canvas: np.ndarray,
    x: float,
    y: float,
    radius: int,
    color_bgr: Tuple[int, int, int],
    phase: float,
    angle: float,
    pulse_strength: float,
    audio_level: float,
    irregularity_scale: float,
) -> None:
    cx = int(round(x))
    cy = int(round(y))
    r = max(1, int(radius))
    if r <= 2:
        cv2.circle(canvas, (cx, cy), r, color_bgr, -1, cv2.LINE_AA)
        return

    n_pts = max(7, min(13, 7 + r))
    irregularity = 0.08 + (0.12 * clamp01(audio_level)) + (0.08 * clamp01(pulse_strength))
    irregularity *= max(0.0, float(irregularity_scale))
    pts: list[list[int]] = []
    for i in range(n_pts):
        t = (2.0 * math.pi * i) / n_pts
        wobble = (
            0.55 * math.sin((phase * 1.9) + (i * 1.37))
            + 0.45 * math.cos((phase * 1.15) - (i * 0.91))
        )
        rr = max(1.0, r * (1.0 + (wobble * irregularity)))
        px = int(round(cx + (math.cos(t + angle) * rr)))
        py = int(round(cy + (math.sin(t + angle) * rr)))
        pts.append([px, py])

    poly = np.array(pts, dtype=np.int32)
    cv2.fillPoly(canvas, [poly], color_bgr, cv2.LINE_AA)
    rim = blend_bgr(color_bgr, (245, 245, 245), 0.18 + (0.22 * clamp01(pulse_strength)))
    cv2.polylines(canvas, [poly], True, rim, 1, cv2.LINE_AA)


def draw_emmons_atom(
    canvas: np.ndarray,
    x: float,
    y: float,
    radius: int,
    color_bgr: Tuple[int, int, int],
    angle: float,
    phase: float,
    line_thickness_scale: float,
) -> None:
    cx = int(round(x))
    cy = int(round(y))
    r = max(3, int(radius))

    orbit_a = max(4, int(round(r * 1.20)))
    orbit_b = max(2, int(round(r * 0.48)))
    orbit_thick = max(1, int(round(r * max(0.005, float(line_thickness_scale)))))
    electron_r = max(1, int(round(r * 0.17)))
    nucleus_r = max(1, int(round(r * 0.24)))

    glow_color = blend_bgr(color_bgr, (240, 240, 240), 0.20)
    base_deg = math.degrees(angle)
    for orbit_idx, orbit_offset_deg in enumerate((0.0, 60.0, 120.0)):
        orbit_deg = base_deg + orbit_offset_deg
        cv2.ellipse(
            canvas,
            (cx, cy),
            (orbit_a, orbit_b),
            orbit_deg,
            0,
            360,
            glow_color,
            orbit_thick,
            cv2.LINE_AA,
        )
        cv2.ellipse(
            canvas,
            (cx, cy),
            (orbit_a, orbit_b),
            orbit_deg,
            0,
            360,
            color_bgr,
            orbit_thick,
            cv2.LINE_AA,
        )

        t = phase + (orbit_idx * 2.15)
        ex0 = orbit_a * math.cos(t)
        ey0 = orbit_b * math.sin(t)
        th = math.radians(orbit_deg)
        ex = int(round(cx + (ex0 * math.cos(th)) - (ey0 * math.sin(th))))
        ey = int(round(cy + (ex0 * math.sin(th)) + (ey0 * math.cos(th))))
        cv2.circle(canvas, (ex, ey), electron_r + 1, glow_color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (ex, ey), electron_r, color_bgr, -1, cv2.LINE_AA)

    cv2.circle(canvas, (cx, cy), nucleus_r + 1, glow_color, -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), nucleus_r, color_bgr, -1, cv2.LINE_AA)


def draw_shobud_suit(
    canvas: np.ndarray,
    x: float,
    y: float,
    radius: int,
    color_bgr: Tuple[int, int, int],
    suit: str,
    angle: float,
    outline_only: bool,
    stroke_scale: float,
    edge_shimmer: float,
    edge_width: float,
    shimmer_phase: float,
) -> None:
    cx = float(x)
    cy = float(y)
    r = max(3.0, float(radius))
    h, w = canvas.shape[:2]
    cos_t = math.cos(angle)
    sin_t = math.sin(angle)

    stroke = max(1, int(round(max(0.0, float(stroke_scale)))))
    shimmer_base = clamp01(float(edge_shimmer))
    shimmer_wave = 0.5 + (0.5 * math.sin(float(shimmer_phase)))
    shimmer_amt = clamp01(shimmer_base * (0.35 + (0.65 * shimmer_wave)))
    shimmer_color = blend_bgr(color_bgr, (245, 245, 245), 0.55 + (0.35 * shimmer_amt))
    shimmer_edge_enabled = float(edge_width) > 0.0
    shimmer_thickness = max(1, int(round(max(0.0, float(edge_width)) + (1.5 * shimmer_amt))))

    pad = int(round((1.35 * r) + max(stroke, shimmer_thickness) + 4))
    cx_i = int(round(cx))
    cy_i = int(round(cy))
    x0 = max(0, cx_i - pad)
    y0 = max(0, cy_i - pad)
    x1 = min(w, cx_i + pad + 1)
    y1 = min(h, cy_i + pad + 1)
    if x0 >= x1 or y0 >= y1:
        return

    roi_w = x1 - x0
    roi_h = y1 - y0
    roi = canvas[y0:y1, x0:x1]
    mask = np.zeros((roi_h, roi_w), dtype=np.uint8)

    local_cx = cx - x0
    local_cy = cy - y0

    def rot(px: float, py: float) -> tuple[int, int]:
        rx = local_cx + (px * cos_t) - (py * sin_t)
        ry = local_cy + (px * sin_t) + (py * cos_t)
        return int(round(rx)), int(round(ry))

    if suit == "diamond":
        pts = np.array(
            [
                rot(0.0, -1.15 * r),
                rot(0.88 * r, 0.0),
                rot(0.0, 1.15 * r),
                rot(-0.88 * r, 0.0),
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [pts], 255, cv2.LINE_AA)
    elif suit == "heart":
        lcx, lcy = rot(-0.42 * r, -0.22 * r)
        rcx, rcy = rot(0.42 * r, -0.22 * r)
        bottom = rot(0.0, 1.05 * r)
        left_knot = rot(-0.92 * r, 0.15 * r)
        right_knot = rot(0.92 * r, 0.15 * r)
        tri = np.array([left_knot, bottom, right_knot], dtype=np.int32)
        rad = max(2, int(round(0.58 * r)))
        cv2.circle(mask, (lcx, lcy), rad, 255, -1, cv2.LINE_AA)
        cv2.circle(mask, (rcx, rcy), rad, 255, -1, cv2.LINE_AA)
        cv2.fillPoly(mask, [tri], 255, cv2.LINE_AA)
    elif suit == "club":
        c1 = rot(0.0, -0.58 * r)
        c2 = rot(-0.60 * r, 0.05 * r)
        c3 = rot(0.60 * r, 0.05 * r)
        rad = max(2, int(round(0.52 * r)))
        stem_tri = np.array(
            [
                rot(0.0, 1.18 * r),
                rot(-0.26 * r, 0.55 * r),
                rot(0.26 * r, 0.55 * r),
            ],
            dtype=np.int32,
        )
        cv2.circle(mask, c1, rad, 255, -1, cv2.LINE_AA)
        cv2.circle(mask, c2, rad, 255, -1, cv2.LINE_AA)
        cv2.circle(mask, c3, rad, 255, -1, cv2.LINE_AA)
        cv2.fillPoly(mask, [stem_tri], 255, cv2.LINE_AA)
    else:
        top_l = rot(-0.42 * r, 0.22 * r)
        top_r = rot(0.42 * r, 0.22 * r)
        bottom = rot(0.0, -1.05 * r)
        left_knot = rot(-0.92 * r, -0.15 * r)
        right_knot = rot(0.92 * r, -0.15 * r)
        tri = np.array([left_knot, bottom, right_knot], dtype=np.int32)
        rad = max(2, int(round(0.58 * r)))
        stem_tri = np.array(
            [
                rot(0.0, 1.16 * r),
                rot(-0.24 * r, 0.53 * r),
                rot(0.24 * r, 0.53 * r),
            ],
            dtype=np.int32,
        )
        cv2.circle(mask, top_l, rad, 255, -1, cv2.LINE_AA)
        cv2.circle(mask, top_r, rad, 255, -1, cv2.LINE_AA)
        cv2.fillPoly(mask, [tri], 255, cv2.LINE_AA)
        cv2.fillPoly(mask, [stem_tri], 255, cv2.LINE_AA)

    needs_edge = outline_only or (shimmer_amt > 0.0 and shimmer_edge_enabled)

    if not outline_only:
        color_layer = np.zeros_like(roi, dtype=np.uint8)
        color_layer[:, :] = color_bgr
        cv2.copyTo(color_layer, mask, roi)

    if not needs_edge:
        return

    contours_info = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
    if not contours:
        return

    if outline_only:
        cv2.drawContours(roi, contours, -1, color_bgr, stroke, cv2.LINE_AA)

    if shimmer_amt > 0.0 and shimmer_edge_enabled:
        cv2.drawContours(roi, contours, -1, shimmer_color, shimmer_thickness, cv2.LINE_AA)


def find_nearest_neuron_target(
    x: float,
    y: float,
    sparks: List[Spark],
    max_distance: float,
    min_distance: float = 8.0,
) -> tuple[float, float] | None:
    max_d = max(0.0, float(max_distance))
    min_d = max(0.0, float(min_distance))
    best_dist: float | None = None
    best_xy: tuple[float, float] | None = None

    for sp in sparks:
        if sp.life <= 0 or sp.shape_kind != "neuron":
            continue
        dx = float(sp.x - x)
        dy = float(sp.y - y)
        d = math.sqrt((dx * dx) + (dy * dy))
        if d <= min_d:
            continue
        if max_d > 0.0 and d > max_d:
            continue
        if best_dist is None or d < best_dist:
            best_dist = d
            best_xy = (float(sp.x), float(sp.y))

    return best_xy


def find_nearest_neuron_targets(
    x: float,
    y: float,
    sparks: List[Spark],
    max_distance: float,
    count: int,
    min_distance: float = 8.0,
) -> list[tuple[float, float]]:
    max_d = max(0.0, float(max_distance))
    min_d = max(0.0, float(min_distance))
    k = max(1, int(count))

    candidates: list[tuple[float, float, float]] = []
    for sp in sparks:
        if sp.life <= 0 or sp.shape_kind != "neuron":
            continue
        dx = float(sp.x - x)
        dy = float(sp.y - y)
        d = math.sqrt((dx * dx) + (dy * dy))
        if d <= min_d:
            continue
        if max_d > 0.0 and d > max_d:
            continue
        candidates.append((d, float(sp.x), float(sp.y)))

    candidates.sort(key=lambda item: item[0])
    return [(cx, cy) for _, cx, cy in candidates[:k]]


def has_live_neuron_near(
    x: float,
    y: float,
    sparks: List[Spark],
    tolerance: float,
) -> bool:
    tol = max(1.0, float(tolerance))
    tol2 = tol * tol
    for sp in sparks:
        if sp.life <= 0 or sp.shape_kind != "neuron":
            continue
        dx = float(sp.x - x)
        dy = float(sp.y - y)
        if (dx * dx) + (dy * dy) <= tol2:
            return True
    return False


def abstract_form_spawn_policy(
    x: float,
    y: float,
    radius: int,
    sparks: List[Spark],
    contain_epsilon: float,
    overlap_epsilon: float,
    replacement_scale: float,
) -> tuple[bool, list[int]]:
    """
    Lightweight spatial policy for abstract forms.
    - Reject new form if it is contained by or heavily overlaps an existing form.
    - If new form fully contains smaller existing forms, mark those for removal.
    """
    r_new = max(1.0, float(radius))
    reject = False
    remove_idx: list[int] = []

    contain_eps = max(0.10, min(1.20, float(contain_epsilon)))
    overlap_eps = max(0.10, min(1.20, float(overlap_epsilon)))
    replace_scale = max(1.00, float(replacement_scale))

    for i, sp in enumerate(sparks):
        if sp.life <= 0:
            continue
        if sp.shape_kind == "dot":
            continue

        dx = float(x - sp.x)
        dy = float(y - sp.y)
        d = math.sqrt((dx * dx) + (dy * dy))
        r_old = max(1.0, float(sp.radius))

        old_contains_new = (d + r_new) <= (r_old * contain_eps)
        if old_contains_new:
            reject = True
            break

        too_much_overlap = d < ((r_new + r_old) * overlap_eps)
        if too_much_overlap:
            reject = True
            break

        new_contains_old = (d + r_old) <= (r_new * contain_eps)
        if new_contains_old and r_new > (r_old * replace_scale):
            remove_idx.append(i)

    return (not reject), remove_idx


def main() -> None:
    ap = argparse.ArgumentParser(description="Voidstar particle sparks overlay (motion + audio reactive).")
    ap.add_argument("input", help="Input video path")
    ap.add_argument("--output", default=None, help="Output .mp4 path")
    ap.add_argument("--out-dir", default=None, help="Output directory when --output is not provided")

    ap.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    ap.add_argument("--duration", type=float, default=0.0, help="Duration in seconds (0 = to end)")

    ap.add_argument("--video-encoder", default="auto", help="auto|libx264|h264_nvenc|hevc_nvenc")
    ap.add_argument("--preset", default="fast", help="ffmpeg preset")
    ap.add_argument("--crf", type=int, default=19, help="CRF for CPU encoder mode")

    ap.add_argument("--max-points", type=int, default=220, help="Maximum tracked points")
    ap.add_argument(
        "--max-live-sparks",
        type=int,
        default=0,
        help="Maximum live particles (0=auto from max-points*4)",
    )
    ap.add_argument("--point-min-distance", type=float, default=6.0, help="Minimum spacing between tracked points (pixels)")
    ap.add_argument("--track-refresh", type=int, default=5, help="Frames between reseeding features")
    ap.add_argument("--motion-threshold", type=float, default=1.2, help="Minimum pixel motion to emit sparks")
    ap.add_argument("--spark-rate", type=float, default=0.65, help="Spark spawn rate per moving point")
    ap.add_argument("--spark-life-frames", type=int, default=18, help="Particle lifetime in frames")
    ap.add_argument("--spark-speed", type=float, default=3.2, help="Base particle speed in px/frame")
    ap.add_argument("--spark-jitter", type=float, default=1.1, help="Random velocity jitter")
    ap.add_argument("--spark-size", type=float, default=2.2, help="Base spark radius in pixels")
    ap.add_argument("--spark-opacity", type=float, default=0.70, help="Overlay opacity multiplier")
    ap.add_argument(
        "--spark-color-temperature",
        type=float,
        default=0.0,
        help="Spark color temperature shift (-1.0=cooler, 0=neutral, +1.0=warmer)",
    )
    ap.add_argument(
        "--electric-ray-boost",
        type=float,
        default=1.0,
        help="Electric mode streak/ray intensity multiplier (0.0+; default 1.0)",
    )
    ap.add_argument(
        "--neurons-style",
        default="voidstar",
        help="Neurons style: voidstar|white|cyan|audio|biolum|sunset",
    )
    ap.add_argument(
        "--neurons-link-max-distance",
        type=float,
        default=0.0,
        help="Neurons: max distance to link to another node (0=unlimited)",
    )
    ap.add_argument(
        "--neurons-curvature",
        type=float,
        default=1.0,
        help="Neurons: curved link bend amount (0=straight, 1=default)",
    )
    ap.add_argument(
        "--neurons-pulse-speed",
        type=float,
        default=1.0,
        help="Neurons: pulse travel speed multiplier along links",
    )
    ap.add_argument(
        "--neurons-pulse-boost",
        type=float,
        default=1.0,
        help="Neurons: pulse visibility/intensity multiplier",
    )
    ap.add_argument(
        "--neurons-connections",
        type=int,
        default=1,
        help="Neurons: nearest-neighbor connections per neuron",
    )
    ap.add_argument(
        "--neurons-node-irregularity",
        type=float,
        default=1.0,
        help="Neurons: node shape irregularity multiplier (0=smooth circle, 1=default)",
    )
    ap.add_argument(
        "--emmons-line-thickness",
        type=float,
        default=0.04,
        help="Emmons: atom line thickness scale (default 0.04)",
    )
    ap.add_argument(
        "--emmons-spin-speed",
        type=float,
        default=0.06,
        help="Emmons: per-frame spin phase speed (default 0.06)",
    )
    ap.add_argument(
        "--shobud-render",
        default="filled",
        help="ShoBud: suit render mode filled|outline",
    )
    ap.add_argument(
        "--shobud-stroke-width",
        type=float,
        default=1.0,
        help="ShoBud: outline stroke width in pixels (min 1)",
    )
    ap.add_argument(
        "--shobud-rotation-speed",
        type=float,
        default=0.028,
        help="ShoBud: per-frame logo rotation speed",
    )
    ap.add_argument(
        "--shobud-wobble",
        type=float,
        default=0.0,
        help="ShoBud: vintage steel wobble amount (0=off)",
    )
    ap.add_argument(
        "--shobud-edge-shimmer",
        type=float,
        default=0.0,
        help="ShoBud: edge shimmer intensity (0=off)",
    )
    ap.add_argument(
        "--shobud-edge-width",
        type=float,
        default=1.0,
        help="ShoBud: shimmer edge width in pixels (0 disables shimmer contour)",
    )
    ap.add_argument(
        "--shobud-palette",
        default="classic",
        help="ShoBud: color palette classic|cream-cherry",
    )
    ap.add_argument(
        "--shobud-dark-suit-color",
        default="black",
        help="ShoBud: non-red suit color black|white",
    )

    ap.add_argument("--flood-in-out", type=str, default="false", help="true|false, add mirrored particle bursts at clip start/end")
    ap.add_argument("--flood-seconds", type=float, default=2.0, help="Duration in seconds for flood burst at start and end")
    ap.add_argument("--flood-spawn-mult", type=float, default=3.0, help="Peak multiplier for spawn probability during flood window")
    ap.add_argument("--flood-extra-sources", type=int, default=180, help="Peak extra random spawn sources per frame during flood")
    ap.add_argument("--flood-velocity-mult", type=float, default=1.35, help="Velocity multiplier at flood peak")

    ap.add_argument("--audio-reactive", type=str, default="true", help="true|false")
    ap.add_argument("--audio-reactive-gain", type=float, default=1.35, help="Audio intensity gain")
    ap.add_argument("--audio-reactive-smooth", type=float, default=0.70, help="Audio envelope smoothing")

    ap.add_argument(
        "--color-mode",
        default="white",
        help=(
            "white|rgb|random|audio-intensity|antiparticles|abstract-forms|abstract-shapes|"
            "electric|electrostatic|neurons|strings|emmons|shobud"
        ),
    )
    ap.add_argument("--color-rgb", default="255,255,255", help="Spark color as R,G,B (used when --color-mode rgb)")
    ap.add_argument(
        "--abstract-contain-epsilon",
        type=float,
        default=0.85,
        help="Abstract forms: containment strictness (lower=stricter anti-nesting, default 0.85)",
    )
    ap.add_argument(
        "--abstract-overlap-epsilon",
        type=float,
        default=0.78,
        help="Abstract forms: overlap threshold (lower=more spacing, default 0.78)",
    )
    ap.add_argument(
        "--abstract-replacement-scale",
        type=float,
        default=1.10,
        help="Abstract forms: new shape must be this much larger to evict contained smaller forms",
    )
    ap.add_argument("--log-interval", type=float, default=1.0, help="Progress print interval")

    args = ap.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_path_for(input_path, out_dir, args)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if frame_w <= 0 or frame_h <= 0:
        raise RuntimeError("Could not determine video dimensions")

    start_sec = max(0.0, float(args.start))
    start_frame = int(round(start_sec * src_fps))
    total_available = max(0, src_frames - start_frame) if src_frames > 0 else 0

    if args.duration > 0:
        total_frames = int(round(args.duration * src_fps))
        total_frames = max(1, total_frames)
        if total_available > 0:
            total_frames = min(total_frames, total_available)
    else:
        total_frames = total_available if total_available > 0 else 0

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    encoder = pick_video_encoder(args.video_encoder)
    color_mode = str(args.color_mode).strip().lower().replace("_", "-")
    if color_mode == "abstract-shapes":
        color_mode = "abstract-forms"
    if color_mode not in {
        "white",
        "rgb",
        "random",
        "audio-intensity",
        "antiparticles",
        "abstract-forms",
        "electric",
        "electrostatic",
        "neurons",
        "strings",
        "emmons",
        "shobud",
    }:
        raise ValueError(
            "--color-mode must be "
            "white|rgb|random|audio-intensity|antiparticles|abstract-forms|abstract-shapes|"
            "electric|electrostatic|neurons|strings|emmons|shobud"
        )

    rgb_color = parse_rgb(args.color_rgb)
    neurons_style = str(args.neurons_style).strip().lower().replace("_", "-")
    if neurons_style not in {"voidstar", "white", "cyan", "audio", "biolum", "sunset"}:
        raise ValueError("--neurons-style must be voidstar|white|cyan|audio|biolum|sunset")
    neurons_link_max_distance = max(0.0, float(args.neurons_link_max_distance))
    neurons_curvature = max(0.0, float(args.neurons_curvature))
    neurons_pulse_speed = max(0.0, float(args.neurons_pulse_speed))
    neurons_pulse_boost = max(0.0, float(args.neurons_pulse_boost))
    neurons_connections = max(1, min(8, int(args.neurons_connections)))
    neurons_node_irregularity = max(0.0, float(args.neurons_node_irregularity))
    emmons_line_thickness = max(0.0, float(args.emmons_line_thickness))
    emmons_spin_speed = max(0.0, float(args.emmons_spin_speed))
    shobud_render = str(args.shobud_render).strip().lower().replace("_", "-")
    if shobud_render not in {"filled", "outline"}:
        raise ValueError("--shobud-render must be filled|outline")
    shobud_stroke_width = max(0.0, float(args.shobud_stroke_width))
    shobud_rotation_speed = max(0.0, float(args.shobud_rotation_speed))
    shobud_wobble = max(0.0, float(args.shobud_wobble))
    shobud_edge_shimmer = max(0.0, float(args.shobud_edge_shimmer))
    shobud_edge_width = max(0.0, float(args.shobud_edge_width))
    shobud_palette = str(args.shobud_palette).strip().lower().replace("_", "-")
    if shobud_palette not in {"classic", "cream-cherry"}:
        raise ValueError("--shobud-palette must be classic|cream-cherry")
    shobud_dark_suit_color = str(args.shobud_dark_suit_color).strip().lower().replace("_", "-")
    if shobud_dark_suit_color not in {"black", "white"}:
        raise ValueError("--shobud-dark-suit-color must be black|white")

    white_bgr = (255, 255, 255)
    rgb_mode_bgr = rgb_to_bgr(rgb_color)
    random_palette_bgr = [
        (255, 80, 80),
        (255, 180, 80),
        (255, 255, 80),
        (100, 255, 120),
        (80, 235, 255),
        (180, 120, 255),
        (255, 120, 220),
    ]
    antiparticle_palette_bgr = [
        (255, 255, 90),
        (255, 120, 40),
        (255, 90, 255),
        (140, 255, 255),
        (255, 70, 170),
        (220, 130, 255),
    ]
    abstract_palette_bgr = [
        (255, 230, 120),
        (255, 170, 95),
        (240, 120, 255),
        (140, 255, 230),
        (120, 190, 255),
        (255, 120, 190),
        (220, 255, 140),
    ]
    electric_palette_bgr = [
        (8, 86, 255),
        (12, 108, 255),
        (18, 132, 255),
        (26, 156, 255),
        (36, 182, 255),
    ]
    electrostatic_palette_bgr = [
        (255, 245, 210),
        (255, 200, 120),
        (240, 220, 170),
        (255, 175, 90),
    ]
    neuron_palette_bgr = [
        (255, 180, 80),
        (255, 140, 120),
        (245, 210, 150),
        (220, 160, 255),
        (200, 240, 255),
    ]
    neuron_white_palette_bgr = [
        (245, 245, 245),
        (255, 255, 255),
        (235, 235, 235),
    ]
    neuron_cyan_palette_bgr = [
        (255, 245, 170),
        (255, 225, 135),
        (245, 205, 120),
        (230, 200, 90),
    ]
    neuron_biolum_palette_bgr = [
        (255, 220, 115),
        (240, 190, 130),
        (220, 170, 175),
        (200, 160, 255),
        (170, 210, 255),
    ]
    neuron_sunset_palette_bgr = [
        (255, 140, 130),
        (240, 110, 165),
        (215, 170, 255),
        (170, 220, 255),
    ]
    string_palette_bgr = [
        (255, 130, 230),
        (255, 220, 120),
        (130, 235, 255),
        (160, 255, 170),
        (255, 150, 120),
    ]
    emmons_palette_bgr = [
        (55, 165, 248),
        (60, 70, 245),
        (185, 140, 95),
        (85, 210, 248),
    ]
    shobud_red_bgr = (30, 35, 220) if shobud_palette == "classic" else (42, 58, 205)
    shobud_dark_bgr = (24, 24, 24)
    if shobud_dark_suit_color == "white":
        shobud_dark_bgr = (248, 248, 248) if shobud_palette == "classic" else (220, 233, 245)
    shobud_suits = ["heart", "club", "diamond", "spade"]
    raw_live_cap = int(args.max_live_sparks)
    if raw_live_cap > 0:
        live_spark_cap = max(1, raw_live_cap)
        live_spark_cap_source = "cli"
    else:
        live_spark_cap = max(4, int(args.max_points) * 4)
        live_spark_cap_source = "auto(max-points*4)"

    log(f"input={input_path}")
    log(f"output={output_path}")
    log(f"resolution={frame_w}x{frame_h} fps={src_fps:.2f} encoder={encoder}")
    if color_mode == "rgb":
        log(f"color_mode={color_mode} color_rgb={rgb_color[0]},{rgb_color[1]},{rgb_color[2]}")
    else:
        log(f"color_mode={color_mode}")
    log(f"live_spark_cap={live_spark_cap} source={live_spark_cap_source}")
    if color_mode == "neurons":
        log(f"neurons_style={neurons_style}")
        if neurons_link_max_distance > 0.0:
            log(f"neurons_link_max_distance={neurons_link_max_distance:.1f}")
        else:
            log("neurons_link_max_distance=unlimited")
        log(
            f"neurons_curvature={neurons_curvature:.2f} "
            f"neurons_pulse_speed={neurons_pulse_speed:.2f} "
            f"neurons_pulse_boost={neurons_pulse_boost:.2f} "
            f"neurons_connections={neurons_connections} "
            f"neurons_node_irregularity={neurons_node_irregularity:.2f}"
        )
    if color_mode == "emmons":
        log(
            f"emmons_line_thickness={emmons_line_thickness:.3f} "
            f"emmons_spin_speed={emmons_spin_speed:.3f}"
        )
    if color_mode == "shobud":
        log(
            f"shobud_render={shobud_render} "
            f"shobud_stroke_width={shobud_stroke_width:.2f} "
            f"shobud_rotation_speed={shobud_rotation_speed:.3f} "
            f"shobud_wobble={shobud_wobble:.2f} "
            f"shobud_edge_shimmer={shobud_edge_shimmer:.2f} "
            f"shobud_edge_width={shobud_edge_width:.2f} "
            f"shobud_palette={shobud_palette} "
            f"shobud_dark_suit_color={shobud_dark_suit_color}"
        )

    audio_reactive = parse_bool(args.audio_reactive)
    spark_color_temperature = max(-1.0, min(1.0, float(args.spark_color_temperature)))
    electric_ray_boost = max(0.0, float(args.electric_ray_boost))
    flood_in_out = parse_bool(args.flood_in_out)
    flood_seconds = max(0.0, float(args.flood_seconds))
    flood_spawn_mult = max(1.0, float(args.flood_spawn_mult))
    flood_extra_sources = max(0, int(args.flood_extra_sources))
    flood_velocity_mult = max(1.0, float(args.flood_velocity_mult))

    if abs(spark_color_temperature) > 1e-6:
        log(f"spark_color_temperature={spark_color_temperature:+.2f}")
    if color_mode == "electric" and abs(electric_ray_boost - 1.0) > 1e-6:
        log(f"electric_ray_boost={electric_ray_boost:.2f}")

    if flood_in_out:
        log(
            "flood_in_out=true "
            f"flood_seconds={flood_seconds:.2f} "
            f"flood_spawn_mult={flood_spawn_mult:.2f} "
            f"flood_extra_sources={flood_extra_sources} "
            f"flood_velocity_mult={flood_velocity_mult:.2f}"
        )
    if audio_reactive:
        log("analyzing audio envelope...")
        env = extract_audio_envelope(
            input_path,
            start_sec,
            float(args.duration),
            src_fps,
            float(args.audio_reactive_gain),
            float(args.audio_reactive_smooth),
        )
        log(f"audio envelope ready frames={len(env)}")
    else:
        env = np.zeros(max(1, total_frames if total_frames > 0 else 1), dtype=np.float32)
        log("audio reactive disabled; using flat envelope")

    tmp_video = output_path.with_name(output_path.stem + "__video.mp4")

    ffmpeg_cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{frame_w}x{frame_h}",
        "-r", f"{src_fps}",
        "-i", "pipe:0",
        "-an",
        "-c:v", encoder,
    ]
    if encoder == "libx264":
        ffmpeg_cmd += ["-preset", args.preset, "-crf", str(args.crf)]
    else:
        ffmpeg_cmd += ["-preset", args.preset]
    ffmpeg_cmd += ["-pix_fmt", "yuv420p", str(tmp_video)]

    log("starting video encoder subprocess...")
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    log("video encoder started")

    prev_gray = None
    prev_points = np.empty((0, 1, 2), dtype=np.float32)
    sparks: List[Spark] = []

    processed = 0
    t0 = time.time()
    last_log = t0
    slow_frame_threshold_sec = 2.0
    render_heartbeat_interval = max(0.5, float(args.log_interval))

    while True:
        if total_frames > 0 and processed >= total_frames:
            break

        ok, frame = cap.read()
        if not ok:
            break

        frame_start = time.time()
        last_render_heartbeat = frame_start

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        movers: List[Tuple[float, float, float]] = []

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
                    for idx, speed in enumerate(motion):
                        if speed >= float(args.motion_threshold):
                            x, y = good_new[idx]
                            if 0 <= x < frame_w and 0 <= y < frame_h:
                                movers.append((float(x), float(y), float(speed)))
                    prev_points = good_new.reshape(-1, 1, 2).astype(np.float32)

        need_reseed = (processed % max(1, int(args.track_refresh)) == 0) or (prev_points.shape[0] < max(12, int(args.max_points) // 4))
        if need_reseed:
            pts = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=max(16, int(args.max_points)),
                qualityLevel=0.01,
                minDistance=max(1.0, float(args.point_min_distance)),
                blockSize=7,
            )
            if pts is not None and pts.size > 0:
                prev_points = pts.astype(np.float32)

        audio_level = float(env[min(processed, len(env) - 1)]) if len(env) > 0 else 0.0
        reactive_mult = 1.0 + (0.8 * audio_level)

        flood_strength = 0.0
        if flood_in_out and flood_seconds > 0.0:
            elapsed_sec = processed / max(1e-6, src_fps)
            if elapsed_sec < flood_seconds:
                flood_strength = max(flood_strength, 1.0 - (elapsed_sec / flood_seconds))
            if total_frames > 0:
                remaining_frames = max(0, total_frames - processed - 1)
                remaining_sec = remaining_frames / max(1e-6, src_fps)
                if remaining_sec < flood_seconds:
                    flood_strength = max(flood_strength, 1.0 - (remaining_sec / flood_seconds))

        spawn_prob = clamp01(float(args.spark_rate) * (0.35 + (0.65 * reactive_mult)))
        if flood_strength > 0.0:
            spawn_prob = clamp01(spawn_prob * (1.0 + (flood_spawn_mult - 1.0) * flood_strength))
        if color_mode == "antiparticles":
            spawn_prob = clamp01(spawn_prob * 1.25)
        elif color_mode == "abstract-forms":
            spawn_prob = clamp01(spawn_prob * 0.95)
        elif color_mode == "electric":
            spawn_prob = clamp01(spawn_prob * 1.35)
        elif color_mode == "electrostatic":
            spawn_prob = clamp01(spawn_prob * 0.82)
        elif color_mode == "neurons":
            spawn_prob = clamp01(spawn_prob * 0.74)
        elif color_mode == "strings":
            spawn_prob = clamp01(spawn_prob * 0.92)
        elif color_mode == "emmons":
            spawn_prob = clamp01(spawn_prob * 0.82)
        elif color_mode == "shobud":
            spawn_prob = clamp01(spawn_prob * 0.72)

        if flood_strength > 0.0 and flood_extra_sources > 0:
            extra_count = int(round(flood_extra_sources * flood_strength))
            extra_speed = max(float(args.motion_threshold), 1.0) * (1.0 + (0.75 * flood_strength))
            for _ in range(extra_count):
                movers.append(
                    (
                        random.uniform(0.0, frame_w - 1.0),
                        random.uniform(0.0, frame_h - 1.0),
                        extra_speed,
                    )
                )

        for x, y, speed in movers:
            if len(sparks) >= live_spark_cap:
                break
            if random.random() > spawn_prob:
                continue
            n_spawn = 1
            if color_mode != "shobud" and audio_level > 0.8 and random.random() < 0.35:
                n_spawn = 2
            if color_mode == "antiparticles" and random.random() < 0.45:
                n_spawn += 1
            elif color_mode == "abstract-forms" and audio_level > 0.65 and random.random() < 0.40:
                n_spawn += 1
            elif color_mode == "electric" and random.random() < (0.40 + (0.25 * audio_level)):
                n_spawn += random.randint(1, 2)
            elif color_mode == "electrostatic" and random.random() < (0.22 + (0.20 * audio_level)):
                n_spawn += 1
            elif color_mode == "neurons" and random.random() < (0.18 + (0.24 * audio_level)):
                n_spawn += 1
            elif color_mode == "strings" and random.random() < (0.34 + (0.18 * audio_level)):
                n_spawn += 1
            elif color_mode == "emmons" and random.random() < (0.22 + (0.18 * audio_level)):
                n_spawn += 1
            elif color_mode == "shobud" and random.random() < (0.18 + (0.14 * audio_level)):
                n_spawn += 1
            remaining_slots = max(0, live_spark_cap - len(sparks))
            if remaining_slots <= 0:
                continue
            n_spawn = min(n_spawn, remaining_slots)
            for _ in range(n_spawn):
                angle = random.uniform(0.0, 2.0 * math.pi)
                base_speed = float(args.spark_speed) * (0.65 + 0.45 * min(3.0, speed / 4.0))
                vel = base_speed * reactive_mult
                if flood_strength > 0.0:
                    vel *= 1.0 + ((flood_velocity_mult - 1.0) * flood_strength)
                vx = math.cos(angle) * vel + random.uniform(-float(args.spark_jitter), float(args.spark_jitter))
                vy = math.sin(angle) * vel + random.uniform(-float(args.spark_jitter), float(args.spark_jitter))
                life = max(2, int(round(float(args.spark_life_frames) * (0.8 + 0.5 * audio_level))))
                radius = max(1, int(round(float(args.spark_size) * (0.8 + 0.6 * audio_level))))
                if color_mode == "white":
                    spark_color_bgr = white_bgr
                elif color_mode == "rgb":
                    spark_color_bgr = rgb_mode_bgr
                elif color_mode == "random":
                    spark_color_bgr = random.choice(random_palette_bgr)
                elif color_mode == "audio-intensity":
                    spark_audio_level = clamp01(audio_level + random.uniform(-0.12, 0.12))
                    spark_color_bgr = audio_level_to_bgr(spark_audio_level)
                elif color_mode == "abstract-forms":
                    speed_level = clamp01(speed / 6.5)
                    form_energy = clamp01((0.58 * audio_level) + (0.42 * speed_level) + random.uniform(-0.10, 0.10))
                    palette_color = random.choice(abstract_palette_bgr)
                    audio_color = audio_level_to_bgr(form_energy)
                    spark_color_bgr = blend_bgr(palette_color, audio_color, 0.50 + (0.38 * form_energy))
                elif color_mode == "electric":
                    speed_level = clamp01(speed / 7.0)
                    electric_energy = clamp01((0.55 * audio_level) + (0.45 * speed_level) + random.uniform(-0.08, 0.12))
                    palette_color = random.choice(electric_palette_bgr)
                    hot_core = apply_color_temperature_bgr((24, 174, 255), spark_color_temperature)
                    spark_color_bgr = blend_bgr(palette_color, hot_core, 0.30 + (0.62 * electric_energy))
                elif color_mode == "electrostatic":
                    static_energy = clamp01((0.70 * audio_level) + (0.30 * clamp01(speed / 6.5)) + random.uniform(-0.05, 0.10))
                    palette_color = random.choice(electrostatic_palette_bgr)
                    audio_color = audio_level_to_bgr(clamp01(static_energy + 0.08))
                    spark_color_bgr = blend_bgr(palette_color, audio_color, 0.38 + (0.35 * static_energy))
                elif color_mode == "neurons":
                    neuron_energy = clamp01((0.64 * audio_level) + (0.36 * clamp01(speed / 6.8)) + random.uniform(-0.10, 0.08))
                    if neurons_style == "white":
                        palette_color = random.choice(neuron_white_palette_bgr)
                        audio_color = (245, 245, 245)
                        spark_color_bgr = blend_bgr(palette_color, audio_color, 0.35 + (0.35 * neuron_energy))
                    elif neurons_style == "cyan":
                        palette_color = random.choice(neuron_cyan_palette_bgr)
                        audio_color = apply_color_temperature_bgr(audio_level_to_bgr(clamp01(neuron_energy + 0.20)), -0.65)
                        spark_color_bgr = blend_bgr(palette_color, audio_color, 0.38 + (0.42 * neuron_energy))
                    elif neurons_style == "audio":
                        palette_color = apply_color_temperature_bgr(audio_level_to_bgr(neuron_energy), -0.22)
                        audio_color = apply_color_temperature_bgr(audio_level_to_bgr(clamp01(neuron_energy + 0.28)), -0.30)
                        spark_color_bgr = blend_bgr(palette_color, audio_color, 0.52 + (0.34 * neuron_energy))
                    elif neurons_style == "biolum":
                        palette_color = random.choice(neuron_biolum_palette_bgr)
                        audio_color = apply_color_temperature_bgr(audio_level_to_bgr(clamp01(neuron_energy + 0.15)), -0.15)
                        spark_color_bgr = blend_bgr(palette_color, audio_color, 0.35 + (0.40 * neuron_energy))
                    elif neurons_style == "sunset":
                        palette_color = random.choice(neuron_sunset_palette_bgr)
                        audio_color = apply_color_temperature_bgr(audio_level_to_bgr(clamp01(neuron_energy + 0.10)), 0.38)
                        spark_color_bgr = blend_bgr(palette_color, audio_color, 0.35 + (0.38 * neuron_energy))
                    else:
                        palette_color = random.choice(neuron_palette_bgr)
                        audio_color = audio_level_to_bgr(clamp01(neuron_energy + 0.12))
                        spark_color_bgr = blend_bgr(palette_color, audio_color, 0.30 + (0.42 * neuron_energy))
                elif color_mode == "strings":
                    string_energy = clamp01((0.46 * audio_level) + (0.54 * clamp01(speed / 7.2)) + random.uniform(-0.08, 0.08))
                    palette_color = random.choice(string_palette_bgr)
                    audio_color = audio_level_to_bgr(string_energy)
                    spark_color_bgr = blend_bgr(palette_color, audio_color, 0.44 + (0.30 * string_energy))
                elif color_mode == "emmons":
                    spark_color_bgr = random.choice(emmons_palette_bgr)
                elif color_mode == "shobud":
                    shobud_suit = random.choice(shobud_suits)
                    spark_color_bgr = shobud_red_bgr if shobud_suit in {"heart", "diamond"} else shobud_dark_bgr
                else:
                    speed_level = clamp01(speed / 6.0)
                    anti_energy = clamp01((0.72 * audio_level) + (0.28 * speed_level) + random.uniform(-0.08, 0.08))
                    palette_color = random.choice(antiparticle_palette_bgr)
                    audio_color = audio_level_to_bgr(anti_energy)
                    spark_color_bgr = blend_bgr(palette_color, audio_color, 0.58 + (0.30 * anti_energy))

                if abs(spark_color_temperature) > 1e-6:
                    spark_color_bgr = apply_color_temperature_bgr(spark_color_bgr, spark_color_temperature)

                charge = 0.0
                if color_mode == "antiparticles":
                    charge = 1.0 if random.random() < 0.5 else -1.0

                shape_kind = "dot"
                shape_sides = 4
                shape_stroke = 1
                shape_ang_vel = 0.0
                target_x = x
                target_y = y
                phase = random.uniform(0.0, 2.0 * math.pi)
                if color_mode == "abstract-forms":
                    shape_kind = random.choice(["poly", "poly", "diamond", "ring"])
                    shape_sides = random.randint(3, 7)
                    shape_stroke = 1 if random.random() < 0.65 else 2
                    shape_ang_vel = random.uniform(-0.15, 0.15) * (0.65 + audio_level)
                    life = max(3, int(round(life * random.uniform(1.05, 1.45))))
                    radius = max(2, int(round(radius * random.uniform(1.25, 2.0))))

                    allow_spawn, remove_idx = abstract_form_spawn_policy(
                        x=x,
                        y=y,
                        radius=radius,
                        sparks=sparks,
                        contain_epsilon=float(args.abstract_contain_epsilon),
                        overlap_epsilon=float(args.abstract_overlap_epsilon),
                        replacement_scale=float(args.abstract_replacement_scale),
                    )
                    if not allow_spawn:
                        continue
                    for idx_rm in remove_idx:
                        if 0 <= idx_rm < len(sparks):
                            sparks[idx_rm].life = 0
                elif color_mode == "electric":
                    life = max(2, int(round(life * random.uniform(0.55, 0.92))))
                    radius = max(1, int(round(radius * random.uniform(0.85, 1.25))))
                    shape_kind = "electric"
                elif color_mode == "electrostatic":
                    life = max(4, int(round(life * random.uniform(0.95, 1.45))))
                    radius = max(1, int(round(radius * random.uniform(0.85, 1.15))))
                    jump_dist = min(frame_w, frame_h) * random.uniform(0.08, 0.22) * (0.85 + (0.65 * audio_level))
                    target_x = max(0.0, min(frame_w - 1.0, x + (math.cos(angle) * jump_dist)))
                    target_y = max(0.0, min(frame_h - 1.0, y + (math.sin(angle) * jump_dist)))
                    shape_kind = "arc"
                elif color_mode == "neurons":
                    life = max(6, int(round(life * random.uniform(1.15, 1.85))))
                    radius = max(1, int(round(radius * random.uniform(0.95, 1.45))))
                    target_xy = find_nearest_neuron_target(
                        x=x,
                        y=y,
                        sparks=sparks,
                        max_distance=neurons_link_max_distance,
                        min_distance=max(8.0, float(radius) * 2.0),
                    )
                    if target_xy is not None:
                        target_x, target_y = target_xy
                    else:
                        dendrite_dist = min(frame_w, frame_h) * random.uniform(0.05, 0.18)
                        dendrite_angle = angle + random.uniform(-0.9, 0.9)
                        target_x = max(0.0, min(frame_w - 1.0, x + (math.cos(dendrite_angle) * dendrite_dist)))
                        target_y = max(0.0, min(frame_h - 1.0, y + (math.sin(dendrite_angle) * dendrite_dist)))
                    shape_kind = "neuron"
                    shape_ang_vel = random.uniform(-0.10, 0.10)
                elif color_mode == "strings":
                    life = max(5, int(round(life * random.uniform(1.05, 1.7))))
                    radius = max(1, int(round(radius * random.uniform(0.75, 1.15))))
                    shape_kind = "string"
                    shape_ang_vel = random.uniform(-0.25, 0.25)
                    vel *= random.uniform(0.70, 0.95)
                    vx = math.cos(angle) * vel + random.uniform(-float(args.spark_jitter) * 0.35, float(args.spark_jitter) * 0.35)
                    vy = math.sin(angle) * vel + random.uniform(-float(args.spark_jitter) * 0.35, float(args.spark_jitter) * 0.35)
                elif color_mode == "emmons":
                    life = max(8, int(round(life * random.uniform(1.15, 1.95))))
                    radius = max(3, int(round(radius * random.uniform(1.20, 1.95))))
                    shape_kind = "emmons"
                    shape_ang_vel = random.uniform(-0.08, 0.08)
                    vel *= random.uniform(0.35, 0.65)
                    vx = math.cos(angle) * vel + random.uniform(-float(args.spark_jitter) * 0.20, float(args.spark_jitter) * 0.20)
                    vy = math.sin(angle) * vel + random.uniform(-float(args.spark_jitter) * 0.20, float(args.spark_jitter) * 0.20)
                elif color_mode == "shobud":
                    life = max(9, int(round(life * random.uniform(1.25, 2.15))))
                    radius = max(4, int(round(radius * random.uniform(1.35, 2.20))))
                    shape_kind = "shobud"
                    shape_sides = shobud_suits.index(shobud_suit)
                    shape_ang_vel = random.uniform(-0.020, 0.020)
                    vel *= random.uniform(0.22, 0.45)
                    vx = math.cos(angle) * vel + random.uniform(-float(args.spark_jitter) * 0.10, float(args.spark_jitter) * 0.10)
                    vy = math.sin(angle) * vel + random.uniform(-float(args.spark_jitter) * 0.10, float(args.spark_jitter) * 0.10)

                sparks.append(
                    Spark(
                        x=x,
                        y=y,
                        vx=vx,
                        vy=vy,
                        life=life,
                        max_life=life,
                        radius=radius,
                        color_bgr=spark_color_bgr,
                        charge=charge,
                        shape_kind=shape_kind,
                        angle=random.uniform(0.0, 2.0 * math.pi),
                        ang_vel=shape_ang_vel,
                        sides=shape_sides,
                        stroke=shape_stroke,
                        target_x=target_x,
                        target_y=target_y,
                        phase=phase,
                    )
                )

                if color_mode == "antiparticles" and random.random() < 0.8:
                    anti_vx = -vx + random.uniform(-0.45, 0.45)
                    anti_vy = -vy + random.uniform(-0.45, 0.45)
                    anti_life = max(2, int(round(life * random.uniform(0.85, 1.15))))
                    anti_radius = max(1, int(round(radius * random.uniform(0.85, 1.2))))
                    complement_color = (
                        max(0, min(255, 255 - spark_color_bgr[0] + random.randint(-25, 25))),
                        max(0, min(255, 255 - spark_color_bgr[1] + random.randint(-25, 25))),
                        max(0, min(255, 255 - spark_color_bgr[2] + random.randint(-25, 25))),
                    )
                    anti_audio_color = audio_level_to_bgr(clamp01(audio_level + random.uniform(0.05, 0.2)))
                    anti_color = blend_bgr(complement_color, anti_audio_color, 0.42 + (0.30 * clamp01(audio_level)))
                    sparks.append(
                        Spark(
                            x=x + random.uniform(-1.5, 1.5),
                            y=y + random.uniform(-1.5, 1.5),
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
        alive: List[Spark] = []

        total_sparks_this_frame = len(sparks)
        for sp_idx, sp in enumerate(sparks, start=1):
            if (sp_idx % 32) == 0:
                now_render = time.time()
                elapsed_render = now_render - frame_start
                if elapsed_render >= slow_frame_threshold_sec and (now_render - last_render_heartbeat) >= render_heartbeat_interval:
                    frame_label = f"{processed + 1}/{total_frames}" if total_frames > 0 else f"{processed + 1}"
                    log(
                        f"render heartbeat frame={frame_label} "
                        f"spark={sp_idx}/{total_sparks_this_frame} elapsed={elapsed_render:.1f}s"
                    )
                    last_render_heartbeat = now_render
            if color_mode == "antiparticles":
                cx = frame_w * 0.5
                cy = frame_h * 0.5
                dx = sp.x - cx
                dy = sp.y - cy
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > 1e-6:
                    swirl = 0.15 * sp.charge / max(28.0, dist)
                    sp.vx += -dy * swirl
                    sp.vy += dx * swirl
            elif color_mode == "abstract-forms":
                cx = frame_w * 0.5
                cy = frame_h * 0.5
                dx = sp.x - cx
                dy = sp.y - cy
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > 1e-6:
                    swirl = 0.03 / max(45.0, dist)
                    sp.vx += -dy * swirl
                    sp.vy += dx * swirl
            elif color_mode == "electric":
                sp.vx += random.uniform(-0.85, 0.85) * (0.5 + (0.75 * audio_level))
                sp.vy += random.uniform(-0.85, 0.85) * (0.5 + (0.75 * audio_level))
            elif color_mode == "electrostatic":
                tx = sp.target_x - sp.x
                ty = sp.target_y - sp.y
                dist = math.sqrt((tx * tx) + (ty * ty))
                if dist < 4.0:
                    jump_dist = min(frame_w, frame_h) * random.uniform(0.07, 0.20) * (0.9 + (0.7 * audio_level))
                    jump_angle = random.uniform(0.0, 2.0 * math.pi)
                    sp.target_x = max(0.0, min(frame_w - 1.0, sp.x + (math.cos(jump_angle) * jump_dist)))
                    sp.target_y = max(0.0, min(frame_h - 1.0, sp.y + (math.sin(jump_angle) * jump_dist)))
                elif dist > 1e-6:
                    pull = (0.20 + (0.35 * audio_level)) / max(8.0, dist)
                    sp.vx += tx * pull
                    sp.vy += ty * pull
                sp.vx += random.uniform(-0.15, 0.15)
                sp.vy += random.uniform(-0.15, 0.15)
            elif color_mode == "neurons":
                if not has_live_neuron_near(sp.target_x, sp.target_y, sparks, tolerance=max(10.0, float(sp.radius) * 3.0)):
                    target_xy = find_nearest_neuron_target(
                        x=sp.x,
                        y=sp.y,
                        sparks=sparks,
                        max_distance=neurons_link_max_distance,
                        min_distance=8.0,
                    )
                    if target_xy is not None:
                        sp.target_x, sp.target_y = target_xy

                tx = sp.target_x - sp.x
                ty = sp.target_y - sp.y
                dist = math.sqrt((tx * tx) + (ty * ty))
                if dist < 5.0 and random.random() < 0.35:
                    target_xy = find_nearest_neuron_target(
                        x=sp.x,
                        y=sp.y,
                        sparks=sparks,
                        max_distance=neurons_link_max_distance,
                        min_distance=8.0,
                    )
                    if target_xy is not None:
                        sp.target_x, sp.target_y = target_xy
                    else:
                        dendrite_dist = min(frame_w, frame_h) * random.uniform(0.04, 0.14)
                        dendrite_angle = random.uniform(0.0, 2.0 * math.pi)
                        sp.target_x = max(0.0, min(frame_w - 1.0, sp.x + (math.cos(dendrite_angle) * dendrite_dist)))
                        sp.target_y = max(0.0, min(frame_h - 1.0, sp.y + (math.sin(dendrite_angle) * dendrite_dist)))
                elif random.random() < (0.05 + (0.12 * clamp01(audio_level))):
                    target_xy = find_nearest_neuron_target(
                        x=sp.x,
                        y=sp.y,
                        sparks=sparks,
                        max_distance=neurons_link_max_distance,
                        min_distance=8.0,
                    )
                    if target_xy is not None:
                        sp.target_x, sp.target_y = target_xy
                elif dist > 1e-6:
                    pull = (0.08 + (0.18 * audio_level)) / max(10.0, dist)
                    sp.vx += tx * pull
                    sp.vy += ty * pull
                sp.vx += random.uniform(-0.08, 0.08)
                sp.vy += random.uniform(-0.08, 0.08)
            elif color_mode == "emmons":
                sp.vx += random.uniform(-0.05, 0.05) * (0.5 + (0.5 * audio_level))
                sp.vy += random.uniform(-0.05, 0.05) * (0.5 + (0.5 * audio_level))
                sp.phase += 0.12 + (0.26 * audio_level)
                sp.angle += sp.ang_vel
            elif color_mode == "shobud":
                sp.vx += random.uniform(-0.025, 0.025) * (0.4 + (0.5 * audio_level))
                sp.vy += random.uniform(-0.025, 0.025) * (0.4 + (0.5 * audio_level))
                if shobud_wobble > 0.0:
                    wobble_amp = shobud_wobble * (0.06 + (0.12 * audio_level))
                    sp.vx += math.sin((sp.phase * 0.85) + (processed * 0.032)) * wobble_amp
                    sp.vy += math.cos((sp.phase * 1.10) - (processed * 0.028)) * (wobble_amp * 0.82)
                sp.phase += 0.05 + (0.08 * audio_level)
                sp.angle += shobud_rotation_speed
            elif color_mode == "strings":
                speed_norm = math.sqrt((sp.vx * sp.vx) + (sp.vy * sp.vy))
                if speed_norm > 1e-6:
                    nx = -sp.vy / speed_norm
                    ny = sp.vx / speed_norm
                    wave = math.sin(sp.phase) * (0.20 + (0.45 * audio_level))
                    sp.vx += nx * wave
                    sp.vy += ny * wave
                sp.phase += 0.30 + (0.55 * audio_level)
                sp.angle += sp.ang_vel

            sp.x += sp.vx
            sp.y += sp.vy
            if color_mode in {"abstract-forms", "neurons", "emmons", "shobud"}:
                sp.angle += sp.ang_vel
            if color_mode == "antiparticles":
                sp.vx *= 0.972
                sp.vy *= 0.972
            elif color_mode == "abstract-forms":
                sp.vx *= 0.968
                sp.vy *= 0.968
                sp.ang_vel *= 0.985
            elif color_mode == "electric":
                sp.vx *= 0.90
                sp.vy *= 0.90
            elif color_mode == "electrostatic":
                sp.vx *= 0.93
                sp.vy *= 0.93
            elif color_mode == "neurons":
                sp.vx *= 0.955
                sp.vy *= 0.955
                sp.ang_vel *= 0.992
            elif color_mode == "emmons":
                sp.vx *= 0.964
                sp.vy *= 0.964
                sp.ang_vel *= 0.996
            elif color_mode == "shobud":
                sp.vx *= 0.976
                sp.vy *= 0.976
                sp.ang_vel *= 0.998
            elif color_mode == "strings":
                sp.vx *= 0.965
                sp.vy *= 0.965
                sp.ang_vel *= 0.996
            else:
                sp.vx *= 0.96
                sp.vy *= 0.96
            sp.life -= 1

            if sp.life <= 0:
                continue
            if sp.x < -8 or sp.x >= frame_w + 8 or sp.y < -8 or sp.y >= frame_h + 8:
                continue

            life_ratio = sp.life / max(1, sp.max_life)
            radius = max(1, int(round(sp.radius * (0.65 + 0.7 * life_ratio))))
            if color_mode == "abstract-forms":
                stroke = max(1, int(round(sp.stroke * (0.85 + 0.45 * life_ratio))))
                draw_abstract_form(
                    overlay,
                    x=sp.x,
                    y=sp.y,
                    radius=radius,
                    color_bgr=sp.color_bgr,
                    shape_kind=sp.shape_kind,
                    angle=sp.angle,
                    sides=sp.sides,
                    stroke=stroke,
                )
            elif color_mode == "electric":
                x1 = int(round(sp.x))
                y1 = int(round(sp.y))
                ray_boost = max(0.15, electric_ray_boost)
                x0 = int(round(sp.x - (sp.vx * random.uniform(2.2, 4.2) * ray_boost)))
                y0 = int(round(sp.y - (sp.vy * random.uniform(2.2, 4.2) * ray_boost)))
                glow_mix = clamp01((0.30 + (0.70 * life_ratio)) * random.uniform(0.85, 1.15))
                hot_target = apply_color_temperature_bgr((44, 190, 255), spark_color_temperature)
                hot_color = blend_bgr(sp.color_bgr, hot_target, glow_mix)
                streak_thickness = max(1, int(round((radius + 1) * (0.75 + (0.25 * ray_boost)))))
                cv2.line(overlay, (x0, y0), (x1, y1), hot_color, streak_thickness, cv2.LINE_AA)
                if life_ratio > 0.42 and random.random() < clamp01(0.62 * ray_boost):
                    bx = int(round(sp.x + random.uniform(-8.0, 8.0) * ray_boost))
                    by = int(round(sp.y + random.uniform(-8.0, 8.0) * ray_boost))
                    cv2.line(overlay, (x1, y1), (bx, by), hot_color, 1, cv2.LINE_AA)
                if life_ratio > 0.36 and random.random() < clamp01((0.36 + (0.32 * audio_level)) * ray_boost):
                    ray_count = 2 if random.random() < 0.65 else 3
                    if ray_boost > 1.0 and random.random() < clamp01(0.45 * (ray_boost - 1.0)):
                        ray_count += 1
                    for _ in range(ray_count):
                        ray_angle = random.uniform(0.0, 2.0 * math.pi)
                        ray_len = random.uniform(7.0, 18.0) * (0.9 + (0.7 * life_ratio)) * ray_boost
                        rx = int(round(sp.x + (math.cos(ray_angle) * ray_len)))
                        ry = int(round(sp.y + (math.sin(ray_angle) * ray_len)))
                        cv2.line(overlay, (x1, y1), (rx, ry), hot_color, 1, cv2.LINE_AA)
                if radius >= 1:
                    core_r = 1 if radius <= 2 else 2
                    core_color = apply_color_temperature_bgr((58, 200, 255), spark_color_temperature)
                    cv2.circle(overlay, (x1, y1), core_r, core_color, -1, cv2.LINE_AA)
            elif color_mode == "electrostatic":
                x1 = int(round(sp.x))
                y1 = int(round(sp.y))
                tx = int(round(sp.target_x))
                ty = int(round(sp.target_y))
                if life_ratio > 0.12:
                    draw_electrostatic_arc(
                        overlay,
                        x1=x1,
                        y1=y1,
                        x2=tx,
                        y2=ty,
                        color_bgr=sp.color_bgr,
                        life_ratio=life_ratio,
                        audio_level=audio_level,
                    )
                cv2.circle(overlay, (x1, y1), max(1, radius), sp.color_bgr, -1, cv2.LINE_AA)
            elif color_mode == "neurons":
                x1 = int(round(sp.x))
                y1 = int(round(sp.y))
                pulse_strength = clamp01(0.35 + (0.65 * abs(math.sin(sp.phase + (processed * 0.035)))))
                if neurons_style == "white":
                    link_color = blend_bgr(sp.color_bgr, (245, 245, 245), 0.40 + (0.42 * pulse_strength))
                    pulse_color = (255, 255, 255)
                elif neurons_style == "cyan":
                    link_color = blend_bgr(sp.color_bgr, (255, 240, 180), 0.32 + (0.45 * pulse_strength))
                    pulse_color = (255, 250, 220)
                elif neurons_style == "audio":
                    audio_pulse = apply_color_temperature_bgr(audio_level_to_bgr(clamp01(0.15 + pulse_strength)), -0.28)
                    link_color = blend_bgr(sp.color_bgr, audio_pulse, 0.45 + (0.34 * pulse_strength))
                    pulse_color = blend_bgr(audio_pulse, (245, 245, 245), 0.35)
                elif neurons_style == "biolum":
                    link_color = blend_bgr(sp.color_bgr, (235, 230, 175), 0.30 + (0.42 * pulse_strength))
                    pulse_color = (255, 235, 205)
                elif neurons_style == "sunset":
                    link_color = blend_bgr(sp.color_bgr, (220, 170, 255), 0.26 + (0.40 * pulse_strength))
                    pulse_color = (235, 210, 255)
                else:
                    link_color = blend_bgr(sp.color_bgr, (245, 245, 245), 0.25 + (0.45 * pulse_strength))
                    pulse_color = blend_bgr(sp.color_bgr, (255, 245, 225), 0.28)

                link_color = apply_color_temperature_bgr(link_color, spark_color_temperature)
                pulse_color = apply_color_temperature_bgr(pulse_color, spark_color_temperature)
                nearest_targets = find_nearest_neuron_targets(
                    x=sp.x,
                    y=sp.y,
                    sparks=sparks,
                    max_distance=neurons_link_max_distance,
                    count=neurons_connections,
                    min_distance=8.0,
                )
                if nearest_targets:
                    sp.target_x, sp.target_y = nearest_targets[0]
                else:
                    nearest_targets = [(sp.target_x, sp.target_y)]

                pulse_x = x1
                pulse_y = y1
                for idx_conn, (target_x_conn, target_y_conn) in enumerate(nearest_targets):
                    tx = int(round(target_x_conn))
                    ty = int(round(target_y_conn))
                    pulse_speed = neurons_pulse_speed * (1.0 + (0.95 * audio_level if audio_reactive else 0.0))
                    travel_t = (
                        (0.18 * processed * pulse_speed)
                        + (sp.phase * 0.12)
                        + (idx_conn * 0.19)
                    ) % 1.0
                    pulse_x, pulse_y = draw_neuron_curve(
                        overlay,
                        x1=x1,
                        y1=y1,
                        x2=tx,
                        y2=ty,
                        base_color_bgr=link_color,
                        pulse_color_bgr=pulse_color,
                        travel_t=travel_t,
                        phase=sp.phase + (processed * 0.045) + (idx_conn * 0.55),
                        life_ratio=life_ratio,
                        audio_level=audio_level,
                        curvature=neurons_curvature,
                        pulse_boost=neurons_pulse_boost * (1.0 + (0.9 * audio_level if audio_reactive else 0.0)),
                    )
                node_r = max(1, int(round(radius * (0.85 + (0.6 * pulse_strength)))))
                draw_neuron_node(
                    overlay,
                    x=sp.x,
                    y=sp.y,
                    radius=node_r,
                    color_bgr=sp.color_bgr,
                    phase=sp.phase,
                    angle=sp.angle,
                    pulse_strength=pulse_strength,
                    audio_level=audio_level,
                    irregularity_scale=neurons_node_irregularity,
                )
                if random.random() < (0.18 + (0.26 * pulse_strength)):
                    cv2.circle(overlay, (pulse_x, pulse_y), 1, pulse_color, -1, cv2.LINE_AA)
            elif color_mode == "strings":
                x1 = sp.x
                y1 = sp.y
                steps = 4
                pts: list[tuple[int, int]] = []
                speed_norm = max(1e-6, math.sqrt((sp.vx * sp.vx) + (sp.vy * sp.vy)))
                nx = -sp.vy / speed_norm
                ny = sp.vx / speed_norm
                for i in range(steps + 1):
                    t = i / max(1, steps)
                    back = t * (7.0 + (6.0 * life_ratio))
                    wave = math.sin(sp.phase - (t * math.pi * 2.0)) * (1.2 + (1.5 * audio_level))
                    px = x1 - (sp.vx * back) + (nx * wave)
                    py = y1 - (sp.vy * back) + (ny * wave)
                    pts.append((int(round(px)), int(round(py))))
                cv2.polylines(overlay, [np.array(pts, dtype=np.int32)], False, sp.color_bgr, max(1, radius), cv2.LINE_AA)
            elif color_mode == "emmons":
                atom_radius = max(3, int(round(radius * (1.0 + (0.9 * life_ratio)))))
                draw_emmons_atom(
                    overlay,
                    x=sp.x,
                    y=sp.y,
                    radius=atom_radius,
                    color_bgr=sp.color_bgr,
                    angle=sp.angle,
                    phase=sp.phase + (processed * emmons_spin_speed),
                    line_thickness_scale=emmons_line_thickness,
                )
            elif color_mode == "shobud":
                suit_idx = int(sp.sides) % len(shobud_suits)
                suit_name = shobud_suits[suit_idx]
                suit_radius = max(3, int(round(radius * (1.1 + (0.55 * life_ratio)))))
                draw_shobud_suit(
                    overlay,
                    x=sp.x,
                    y=sp.y,
                    radius=suit_radius,
                    color_bgr=sp.color_bgr,
                    suit=suit_name,
                    angle=sp.angle,
                    outline_only=(shobud_render == "outline"),
                    stroke_scale=shobud_stroke_width,
                    edge_shimmer=shobud_edge_shimmer,
                    edge_width=shobud_edge_width,
                    shimmer_phase=(sp.phase + (processed * 0.08) + (0.15 * suit_idx)),
                )
            else:
                cv2.circle(overlay, (int(round(sp.x)), int(round(sp.y))), radius, sp.color_bgr, -1, cv2.LINE_AA)
            if color_mode == "antiparticles" and radius >= 2 and life_ratio > 0.35:
                cv2.circle(overlay, (int(round(sp.x)), int(round(sp.y))), 1, (255, 255, 255), -1, cv2.LINE_AA)
            if color_mode == "abstract-forms" and radius >= 4 and life_ratio > 0.30:
                cv2.circle(overlay, (int(round(sp.x)), int(round(sp.y))), 1, (245, 245, 245), -1, cv2.LINE_AA)
            if color_mode == "electric" and radius >= 1 and life_ratio > 0.35:
                accent_color = apply_color_temperature_bgr((32, 178, 255), spark_color_temperature)
                cv2.circle(overlay, (int(round(sp.x)), int(round(sp.y))), 1, accent_color, -1, cv2.LINE_AA)
            if color_mode == "neurons" and life_ratio > 0.22 and random.random() < 0.24:
                draw_neuron_node(
                    overlay,
                    x=sp.target_x,
                    y=sp.target_y,
                    radius=1,
                    color_bgr=(255, 245, 235),
                    phase=sp.phase + 0.65,
                    angle=-sp.angle,
                    pulse_strength=life_ratio,
                    audio_level=audio_level,
                    irregularity_scale=neurons_node_irregularity,
                )
            alive.append(sp)

        sparks = alive

        alpha = clamp01(float(args.spark_opacity) * (0.55 + 0.75 * min(1.2, audio_level)))
        if color_mode == "electric":
            alpha = clamp01(alpha * (1.0 + (0.26 * max(0.0, electric_ray_boost))))
        elif color_mode == "electrostatic":
            alpha = clamp01(alpha * 1.02)
        elif color_mode == "neurons":
            alpha = clamp01(alpha * 0.95)
        elif color_mode == "strings":
            alpha = clamp01(alpha * 0.98)
        elif color_mode == "emmons":
            alpha = clamp01(alpha * 0.96)
        elif color_mode == "shobud":
            alpha = clamp01(alpha * 0.92)
        if alpha > 0:
            frame = cv2.addWeighted(frame, 1.0, overlay, alpha, 0.0)

        if proc.stdin is None:
            raise RuntimeError("Encoder stdin is unavailable")
        proc.stdin.write(frame.tobytes())

        prev_gray = gray
        processed += 1

        now = time.time()
        if now - last_log >= float(args.log_interval):
            elapsed = max(1e-6, now - t0)
            fps_now = processed / elapsed
            if total_frames > 0:
                pct = (processed / total_frames) * 100.0
                eta = (total_frames - processed) / max(1e-6, fps_now)
                log(f"frame={processed}/{total_frames} ({pct:.1f}%) fps={fps_now:.2f} eta={eta:0.1f}s sparks={len(sparks)}")
            else:
                log(f"frame={processed} fps={fps_now:.2f} sparks={len(sparks)}")
            last_log = now

    if proc.stdin is not None:
        proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Video encode failed with exit code {rc}")

    cap.release()

    mux_cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(tmp_video),
        "-ss", str(start_sec),
        "-i", str(input_path),
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "320k",
        "-shortest",
    ]
    if args.duration > 0:
        mux_cmd += ["-t", str(float(args.duration))]
    mux_cmd += [str(output_path)]
    run_cmd(mux_cmd, heartbeat_label="audio mux", heartbeat_interval_sec=0.75)

    try:
        tmp_video.unlink(missing_ok=True)
    except Exception:
        pass

    elapsed = time.time() - t0
    log(f"complete frames={processed} elapsed={elapsed:.2f}s output={output_path}")


if __name__ == "__main__":
    main()
