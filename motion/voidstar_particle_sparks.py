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
    print(f"[voidstar] {msg}", flush=True)


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
    ap.add_argument("--track-refresh", type=int, default=5, help="Frames between reseeding features")
    ap.add_argument("--motion-threshold", type=float, default=1.2, help="Minimum pixel motion to emit sparks")
    ap.add_argument("--spark-rate", type=float, default=0.65, help="Spark spawn rate per moving point")
    ap.add_argument("--spark-life-frames", type=int, default=18, help="Particle lifetime in frames")
    ap.add_argument("--spark-speed", type=float, default=3.2, help="Base particle speed in px/frame")
    ap.add_argument("--spark-jitter", type=float, default=1.1, help="Random velocity jitter")
    ap.add_argument("--spark-size", type=float, default=2.2, help="Base spark radius in pixels")
    ap.add_argument("--spark-opacity", type=float, default=0.70, help="Overlay opacity multiplier")

    ap.add_argument("--flood-in-out", type=str, default="false", help="true|false, add mirrored particle bursts at clip start/end")
    ap.add_argument("--flood-seconds", type=float, default=2.0, help="Duration in seconds for flood burst at start and end")
    ap.add_argument("--flood-spawn-mult", type=float, default=3.0, help="Peak multiplier for spawn probability during flood window")
    ap.add_argument("--flood-extra-sources", type=int, default=180, help="Peak extra random spawn sources per frame during flood")
    ap.add_argument("--flood-velocity-mult", type=float, default=1.35, help="Velocity multiplier at flood peak")

    ap.add_argument("--audio-reactive", type=str, default="true", help="true|false")
    ap.add_argument("--audio-reactive-gain", type=float, default=1.35, help="Audio intensity gain")
    ap.add_argument("--audio-reactive-smooth", type=float, default=0.70, help="Audio envelope smoothing")

    ap.add_argument("--color-mode", default="white", help="white|rgb|random|audio-intensity|antiparticles|abstract-forms|abstract-shapes")
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
    if color_mode not in {"white", "rgb", "random", "audio-intensity", "antiparticles", "abstract-forms"}:
        raise ValueError("--color-mode must be white|rgb|random|audio-intensity|antiparticles|abstract-forms|abstract-shapes")

    rgb_color = parse_rgb(args.color_rgb)
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

    log(f"input={input_path}")
    log(f"output={output_path}")
    log(f"resolution={frame_w}x{frame_h} fps={src_fps:.2f} encoder={encoder}")
    if color_mode == "rgb":
        log(f"color_mode={color_mode} color_rgb={rgb_color[0]},{rgb_color[1]},{rgb_color[2]}")
    else:
        log(f"color_mode={color_mode}")

    audio_reactive = parse_bool(args.audio_reactive)
    flood_in_out = parse_bool(args.flood_in_out)
    flood_seconds = max(0.0, float(args.flood_seconds))
    flood_spawn_mult = max(1.0, float(args.flood_spawn_mult))
    flood_extra_sources = max(0, int(args.flood_extra_sources))
    flood_velocity_mult = max(1.0, float(args.flood_velocity_mult))

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
    else:
        env = np.zeros(max(1, total_frames if total_frames > 0 else 1), dtype=np.float32)

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

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    prev_gray = None
    prev_points = np.empty((0, 1, 2), dtype=np.float32)
    sparks: List[Spark] = []

    processed = 0
    t0 = time.time()
    last_log = t0

    while True:
        if total_frames > 0 and processed >= total_frames:
            break

        ok, frame = cap.read()
        if not ok:
            break

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
                minDistance=6,
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
            if random.random() > spawn_prob:
                continue
            n_spawn = 1
            if audio_level > 0.8 and random.random() < 0.35:
                n_spawn = 2
            if color_mode == "antiparticles" and random.random() < 0.45:
                n_spawn += 1
            elif color_mode == "abstract-forms" and audio_level > 0.65 and random.random() < 0.40:
                n_spawn += 1
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
                else:
                    speed_level = clamp01(speed / 6.0)
                    anti_energy = clamp01((0.72 * audio_level) + (0.28 * speed_level) + random.uniform(-0.08, 0.08))
                    palette_color = random.choice(antiparticle_palette_bgr)
                    audio_color = audio_level_to_bgr(anti_energy)
                    spark_color_bgr = blend_bgr(palette_color, audio_color, 0.58 + (0.30 * anti_energy))

                charge = 0.0
                if color_mode == "antiparticles":
                    charge = 1.0 if random.random() < 0.5 else -1.0

                shape_kind = "dot"
                shape_sides = 4
                shape_stroke = 1
                shape_ang_vel = 0.0
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

        for sp in sparks:
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

            sp.x += sp.vx
            sp.y += sp.vy
            if color_mode == "abstract-forms":
                sp.angle += sp.ang_vel
            if color_mode == "antiparticles":
                sp.vx *= 0.972
                sp.vy *= 0.972
            elif color_mode == "abstract-forms":
                sp.vx *= 0.968
                sp.vy *= 0.968
                sp.ang_vel *= 0.985
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
            else:
                cv2.circle(overlay, (int(round(sp.x)), int(round(sp.y))), radius, sp.color_bgr, -1, cv2.LINE_AA)
            if color_mode == "antiparticles" and radius >= 2 and life_ratio > 0.35:
                cv2.circle(overlay, (int(round(sp.x)), int(round(sp.y))), 1, (255, 255, 255), -1, cv2.LINE_AA)
            if color_mode == "abstract-forms" and radius >= 4 and life_ratio > 0.30:
                cv2.circle(overlay, (int(round(sp.x)), int(round(sp.y))), 1, (245, 245, 245), -1, cv2.LINE_AA)
            alive.append(sp)

        sparks = alive

        alpha = clamp01(float(args.spark_opacity) * (0.55 + 0.75 * min(1.2, audio_level)))
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
