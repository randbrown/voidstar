#!/usr/bin/env python3

from __future__ import annotations

import argparse
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


def run_cmd(cmd: List[str]) -> None:
    log("▶ " + " ".join(shlex.quote(c) for c in cmd))
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {p.returncode}")


def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


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
        run_cmd(cmd)

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

    ap.add_argument("--audio-reactive", type=str, default="true", help="true|false")
    ap.add_argument("--audio-reactive-gain", type=float, default=1.35, help="Audio intensity gain")
    ap.add_argument("--audio-reactive-smooth", type=float, default=0.70, help="Audio envelope smoothing")

    ap.add_argument("--color-rgb", default="180,235,255", help="Spark color as R,G,B")
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
    log(f"input={input_path}")
    log(f"output={output_path}")
    log(f"resolution={frame_w}x{frame_h} fps={src_fps:.2f} encoder={encoder}")

    audio_reactive = args.audio_reactive.lower() in {"1", "true", "yes", "on"}
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

    r, g, b = [int(max(0, min(255, int(x.strip())))) for x in args.color_rgb.split(",")]
    color_bgr = (b, g, r)

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
        spawn_prob = clamp01(float(args.spark_rate) * (0.35 + (0.65 * reactive_mult)))

        for x, y, speed in movers:
            if random.random() > spawn_prob:
                continue
            n_spawn = 1
            if audio_level > 0.8 and random.random() < 0.35:
                n_spawn = 2
            for _ in range(n_spawn):
                angle = random.uniform(0.0, 2.0 * math.pi)
                base_speed = float(args.spark_speed) * (0.65 + 0.45 * min(3.0, speed / 4.0))
                vel = base_speed * reactive_mult
                vx = math.cos(angle) * vel + random.uniform(-float(args.spark_jitter), float(args.spark_jitter))
                vy = math.sin(angle) * vel + random.uniform(-float(args.spark_jitter), float(args.spark_jitter))
                life = max(2, int(round(float(args.spark_life_frames) * (0.8 + 0.5 * audio_level))))
                radius = max(1, int(round(float(args.spark_size) * (0.8 + 0.6 * audio_level))))
                sparks.append(Spark(x=x, y=y, vx=vx, vy=vy, life=life, max_life=life, radius=radius))

        overlay = np.zeros_like(frame, dtype=np.uint8)
        alive: List[Spark] = []

        for sp in sparks:
            sp.x += sp.vx
            sp.y += sp.vy
            sp.vx *= 0.96
            sp.vy *= 0.96
            sp.life -= 1

            if sp.life <= 0:
                continue
            if sp.x < -8 or sp.x >= frame_w + 8 or sp.y < -8 or sp.y >= frame_h + 8:
                continue

            life_ratio = sp.life / max(1, sp.max_life)
            radius = max(1, int(round(sp.radius * (0.65 + 0.7 * life_ratio))))
            cv2.circle(overlay, (int(round(sp.x)), int(round(sp.y))), radius, color_bgr, -1, cv2.LINE_AA)
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
    run_cmd(mux_cmd)

    try:
        tmp_video.unlink(missing_ok=True)
    except Exception:
        pass

    elapsed = time.time() - t0
    log(f"complete frames={processed} elapsed={elapsed:.2f}s output={output_path}")


if __name__ == "__main__":
    main()
