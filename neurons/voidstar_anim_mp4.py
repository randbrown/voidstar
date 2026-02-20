#!/usr/bin/env python3
"""
voidstar_anim_mp4.py

MP4-only version.

Generates a perfectly-looping MP4 from a still image:
- neuron firing + crawl shimmer + temporal trails
- elliptical/angled halo swirl with rim electricity arcs + sparks
- elliptical pulse starting at the central ellipse and expanding outward

Features:
- Presets: --preset v5|v6|v7|crispy|electric
- Output directory: --out-dir
- Default output name (when --out not provided) includes key arg values

Dependencies:
  pip install pillow numpy
Optional (better ellipse fit):
  pip install opencv-python
MP4 encoding:
  ffmpeg must be in PATH

Examples:
  python voidstar_anim_mp4.py --list-presets
  python voidstar_anim_mp4.py input.png --preset v7 --out-dir renders
  python voidstar_anim_mp4.py input.png --preset v7 --out-dir renders --out myclip.mp4
  python voidstar_anim_mp4.py input.png --preset v7 --out-dir renders --out myclip   (adds .mp4)
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageChops


# ----------------------------
# Presets (defaults only; CLI flags override)
# ----------------------------
PRESETS: Dict[str, Dict[str, Any]] = {
    "v5": dict(
        glow_scale=1.0,
        crawl_x_freq=0.0,
        trail_frames=7,
        trail_decay=0.78,
        pulse_level=45.0,
        alpha_fire=215,
        alpha_halo=220,
        alpha_pulse=60,
        halo_blur=2.6,
        pulse_blur=2.3,
        pulse_ellipse_scale=0.62,
        pulse_range=0.85,
        pulse_sigma=0.055,
        twinkle=0.015,
        arc_strength=0.50,
        micro_arc_strength=0.15,
    ),
    "v6": dict(
        glow_scale=0.9,
        crawl_x_freq=0.0,
        trail_frames=6,
        trail_decay=0.76,
        pulse_level=45.0,
        alpha_fire=160,
        alpha_halo=170,
        alpha_pulse=60,
        halo_blur=2.6,
        pulse_blur=2.3,
        pulse_ellipse_scale=0.62,
        pulse_range=0.85,
        pulse_sigma=0.040,
        twinkle=0.015,
        arc_strength=0.50,
        micro_arc_strength=0.15,
    ),
    "v7": dict(
        glow_scale=1.0,
        crawl_x_freq=0.0,
        trail_frames=5,
        trail_decay=0.74,
        pulse_level=32.0,
        alpha_fire=130,
        alpha_halo=140,
        alpha_pulse=45,
        halo_blur=2.2,
        pulse_blur=2.0,
        pulse_ellipse_scale=0.62,
        pulse_range=0.85,
        pulse_sigma=0.055,
        twinkle=0.012,
        arc_strength=0.42,
        micro_arc_strength=0.12,
    ),
    "crispy": dict(
        glow_scale=0.80,
        trail_frames=4,
        trail_decay=0.70,
        pulse_level=22.0,
        alpha_fire=95,
        alpha_halo=110,
        alpha_pulse=30,
        halo_blur=1.8,
        pulse_blur=1.6,
        twinkle=0.010,
        arc_strength=0.36,
        micro_arc_strength=0.09,
    ),
    "electric": dict(
        arcs=26,
        arc_strength=0.55,
        micro_arc_strength=0.18,
        trail_frames=6,
        trail_decay=0.78,
        alpha_halo=175,
    ),
}


# ----------------------------
# Helpers
# ----------------------------
def circ_dist(a: float, b: float) -> float:
    d = abs(a - b)
    return min(d, 1.0 - d)


def wrap_angle(a: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a), np.cos(a)).astype(np.float32)


@dataclass
class EllipseFit:
    cx: float
    cy: float
    a: float        # semi-major
    b: float        # semi-minor
    ang_deg: float  # degrees


def fit_center_dark_centroid(gray: np.ndarray) -> Tuple[float, float]:
    h, w = gray.shape
    x0, x1 = int(w * 0.22), int(w * 0.78)
    y0, y1 = int(h * 0.32), int(h * 0.68)
    crop = gray[y0:y1, x0:x1]
    mask_dark = crop < 30
    ys, xs = np.nonzero(mask_dark)
    if len(xs) < 10:
        return w / 2.0, h / 2.0
    return float(xs.mean() + x0), float(ys.mean() + y0)


def try_fit_ellipse_opencv(gray: np.ndarray, cx0: float, cy0: float) -> Optional[EllipseFit]:
    try:
        import cv2  # type: ignore
    except Exception:
        return None

    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dx0 = (xx - cx0).astype(np.float32)
    dy0 = (yy - cy0).astype(np.float32)
    r0 = np.sqrt(dx0 * dx0 + dy0 * dy0)

    ann = (r0 > 160) & (r0 < 360)
    bright = (gray > 210) & ann

    m = (bright.astype(np.uint8) * 255)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    if len(c) < 5:
        return None

    (ex, ey), (MA, ma), angle = cv2.fitEllipse(c)
    a = float(max(MA, ma) / 2.0)
    b = float(min(MA, ma) / 2.0)
    return EllipseFit(cx=float(ex), cy=float(ey), a=a, b=b, ang_deg=float(angle))


def make_phase_map(rng: np.random.Generator, w: int, h: int, blur: float) -> np.ndarray:
    img = Image.fromarray((rng.random((h, w)) * 255).astype(np.uint8), mode="L")
    img = img.filter(ImageFilter.GaussianBlur(blur))
    return (np.array(img).astype(np.float32) / 255.0) * (2 * np.pi)


def which_ffmpeg() -> Optional[str]:
    return shutil.which("ffmpeg")


def ensure_even_dims(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    w2 = w - (w % 2)
    h2 = h - (h % 2)
    if (w2, h2) == (w, h):
        return img
    return img[:h2, :w2]


def resolve_out_path(out_dir: str, path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(Path(out_dir) / p)


def build_default_stem(args: argparse.Namespace) -> str:
    # Keep it readable + stable across runs; include only key knobs.
    stem = Path(args.input).stem
    preset = args.preset or "custom"
    parts = [
        stem,
        f"p{preset}",
        f"w{int(args.width)}",
        f"f{int(args.frames)}",
        f"fps{float(args.fps):.2f}",
        f"gl{float(args.glow_scale):.2f}",
        f"pl{float(args.pulse_level):.0f}",
        f"pr{float(args.pulse_range):.2f}",
        f"ps{float(args.pulse_sigma):.3f}",
        f"as{float(args.arc_strength):.2f}",
        f"tf{int(args.trail_frames)}",
        f"td{float(args.trail_decay):.2f}",
    ]
    if float(args.crawl_x_freq) != 0.0:
        parts.append(f"cx{float(args.crawl_x_freq):g}")
    return "_".join(parts)


def apply_preset_defaults(ns: argparse.Namespace, preset_name: str, argv: list[str]) -> None:
    preset = PRESETS.get(preset_name)
    if not preset:
        raise SystemExit(f"Unknown preset: {preset_name}. Use --list-presets.")

    provided = set()
    for a in argv:
        if a.startswith("--"):
            provided.add(a.split("=")[0])

    def flag_for_attr(attr: str) -> str:
        return "--" + attr.replace("_", "-")

    for attr, val in preset.items():
        if flag_for_attr(attr) not in provided:
            setattr(ns, attr, val)


import time

def write_mp4_via_ffmpeg(frames_rgb: list[np.ndarray], out_mp4: str, fps: float, crf: int, preset: str,
                         show_progress: bool = False, log_interval_s: float = 0.5) -> None:
    ff = which_ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg not found on PATH. Install ffmpeg (or add it to PATH).")

    Path(out_mp4).parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    n = len(frames_rgb)

    with tempfile.TemporaryDirectory(prefix="voidstar_frames_") as td:
        # 1) Write PNG frames
        if show_progress:
            print(f"[encode] writing {n} PNG frames to temp: {td}")
        for i, fr in enumerate(frames_rgb):
            Image.fromarray(fr).save(os.path.join(td, f"frame_{i:05d}.png"))
            if show_progress and (i == 0 or (i + 1) % max(1, n // 10) == 0 or (i + 1) == n):
                print(f"[encode] wrote PNG {i+1}/{n}")

        t1 = time.perf_counter()
        if show_progress:
            print(f"[encode] PNG write done in {t1 - t0:.2f}s")

        # 2) ffmpeg encode with progress
        # Use -progress pipe:1 so we can parse progress keys without noisy logs.
        cmd = [
            ff,
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-framerate", f"{fps:.6f}",
            "-i", os.path.join(td, "frame_%05d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", str(crf),
            "-preset", preset,
            "-movflags", "+faststart",
        ]

        if show_progress:
            cmd += ["-progress", "pipe:1", "-nostats"]

        cmd += [out_mp4]

        if not show_progress:
            subprocess.run(cmd, check=True)
        else:
            print("[encode] running ffmpeg encode...")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            last = time.perf_counter()
            cur_frame = None
            speed = None
            out_time_ms = None

            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.strip()
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k == "frame":
                    try: cur_frame = int(v)
                    except: pass
                elif k == "speed":
                    speed = v
                elif k == "out_time_ms":
                    try: out_time_ms = int(v)
                    except: pass
                elif k == "progress" and v == "continue":
                    now = time.perf_counter()
                    if now - last >= log_interval_s:
                        if cur_frame is not None:
                            pct = 100.0 * cur_frame / max(1, n)
                            extra = f", speed={speed}" if speed else ""
                            print(f"[ffmpeg] frame {cur_frame}/{n} ({pct:.1f}%)" + extra)
                        elif out_time_ms is not None:
                            print(f"[ffmpeg] out_time={out_time_ms/1e6:.2f}s" + (f", speed={speed}" if speed else ""))
                        last = now
                elif k == "progress" and v == "end":
                    break

            rc = proc.wait()
            if rc != 0:
                raise RuntimeError(f"ffmpeg failed with exit code {rc}")

        t2 = time.perf_counter()
        if show_progress:
            print(f"[encode] ffmpeg done in {t2 - t1:.2f}s")
            print(f"[encode] total export time {t2 - t0:.2f}s")


# ----------------------------
# Args
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("input", nargs="?", help="Input image (png/jpg).")
    p.add_argument("--list-presets", action="store_true", help="Print presets and exit.")
    p.add_argument("--preset", default=None, help="Preset name (v5, v6, v7, crispy, electric).")

    p.add_argument(
        "--out-dir",
        default=str(Path("~/WinVideos").expanduser()),
        help="Directory to write output into (created if missing).",
    )

    p.add_argument("--out", default=None, help="Output filename or path (.mp4 optional). If relative, joins --out-dir.")

    p.add_argument("--width", type=int, default=720, help="Resize output width (keeps aspect).")
    p.add_argument("--frames", type=int, default=40, help="Number of frames in loop.")

    # MP4 timing
    p.add_argument("--fps", type=float, default=20.0, help="FPS for MP4. Loop duration = frames/fps.")

    # ffmpeg encode settings
    p.add_argument("--crf", type=int, default=18, help="H.264 CRF (lower=better, bigger file).")
    p.add_argument("--ffpreset", default="veryfast", help="ffmpeg preset (ultrafast..veryslow).")

    p.add_argument("--seed", type=int, default=777, help="Random seed.")

    # Ellipse / alignment
    p.add_argument("--ellipse-rot-offset-deg", type=float, default=90.0,
                   help="Extra rotation added to fitted ellipse angle.")
    p.add_argument("--no-opencv-ellipse-fit", action="store_true",
                   help="Disable OpenCV ellipse fit even if cv2 is installed.")
    p.add_argument("--pulse-ellipse-scale", type=float, default=0.62,
                   help="Pulse ellipse scale relative to fitted halo ellipse (a,b).")

    # Neuron shimmer + trails
    p.add_argument("--events", type=int, default=360, help="Neuron burst event count.")
    p.add_argument("--event-sigma", type=float, default=0.020, help="Neuron burst temporal sigma.")
    p.add_argument("--trail-frames", type=int, default=5, help="Neuron trail length in frames.")
    p.add_argument("--trail-decay", type=float, default=0.74, help="Neuron trail decay per frame.")
    p.add_argument("--crawl-speed", type=float, default=2.0, help="Crawl shimmer temporal speed.")
    p.add_argument("--crawl-noise-scale", type=float, default=0.8, help="Crawl phase noise scale.")
    p.add_argument("--crawl-power", type=float, default=2.2, help="Crawl contrast/power.")
    p.add_argument("--crawl-x-freq", type=float, default=0.0,
                   help="Horizontal cycles across image (0 disables scanline drift).")

    # Halo + electricity
    p.add_argument("--arcs", type=int, default=18, help="Number of large rim arcs.")
    p.add_argument("--arc-strength", type=float, default=0.42, help="Arc mix strength into halo.")
    p.add_argument("--micro-arc-strength", type=float, default=0.12, help="Micro-arc strength.")
    p.add_argument("--halo-blur", type=float, default=2.2, help="Halo blur radius.")
    p.add_argument("--halo-intensity", type=float, default=1.0, help="Overall halo intensity scalar.")

    # Pulse
    p.add_argument("--pulse-range", type=float, default=0.85, help="How far outward pulse expands (rho units).")
    p.add_argument("--pulse-sigma", type=float, default=0.055, help="Pulse ring thickness (rho sigma).")
    p.add_argument("--pulse-intensity", type=float, default=1.0, help="Pulse intensity scalar.")
    p.add_argument("--pulse-level", type=float, default=32.0, help="Max pulse level (lower = subtler).")
    p.add_argument("--pulse-blur", type=float, default=2.0, help="Pulse blur radius.")


    p.add_argument("--ellipse-dx", type=float, default=0.0, help="Ellipse center X offset in pixels (+right).")
    p.add_argument("--ellipse-dy", type=float, default=0.0, help="Ellipse center Y offset in pixels (+down).")
    p.add_argument("--ellipse-ang-delta-deg", type=float, default=0.0, help="Extra ellipse rotation adjustment in degrees.")
    p.add_argument("--ellipse-a-scale", type=float, default=1.0, help="Scale fitted semi-major axis.")
    p.add_argument("--ellipse-b-scale", type=float, default=1.0, help="Scale fitted semi-minor axis.")
    p.add_argument("--debug-ellipse", action="store_true", help="Write a debug PNG with ellipse outline.")

    # Master “glow”
    p.add_argument("--glow-scale", type=float, default=1.0, help="Master glow scalar ( <1 = less glow ).")
    p.add_argument("--alpha-fire", type=float, default=130, help="Neuron overlay alpha clamp.")
    p.add_argument("--alpha-halo", type=float, default=140, help="Halo overlay alpha clamp.")
    p.add_argument("--alpha-pulse", type=float, default=45, help="Pulse overlay alpha clamp.")

    p.add_argument("--pulse-center-mix", type=float, default=0.28,
                help="How much of the center breathing term to mix into the pulse (0 disables).")

    p.add_argument("--fire-threshold", type=float, default=0.12,
               help="Min burst amplitude to draw (higher = fewer simultaneous fires).")


    p.add_argument("--encode-progress", action="store_true",
                help="Print progress while writing frames + during ffmpeg encode.")
    p.add_argument("--encode-log-interval", type=float, default=0.5,
                help="Seconds between ffmpeg progress prints (when --encode-progress).")

    # Twinkle
    p.add_argument("--twinkle", type=float, default=0.012, help="Global twinkle amplitude.")

    return p.parse_args()


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    args = parse_args()
    argv = sys.argv[1:]

    args.out_dir = str(Path(args.out_dir).expanduser())

    if args.list_presets:
        for k in sorted(PRESETS.keys()):
            print(k)
        return

    if not args.input:
        raise SystemExit("Missing input image. Example: python voidstar_anim_mp4.py input.png --preset v7 --out-dir renders")

    if args.preset:
        apply_preset_defaults(args, args.preset, argv)

    os.makedirs(args.out_dir, exist_ok=True)

    # Resolve output path
    if args.out:
        out_path = resolve_out_path(args.out_dir, args.out)
        if not out_path.lower().endswith(".mp4"):
            out_path += ".mp4"
    else:
        # default name embeds key arg values
        stem = build_default_stem(args)
        out_path = resolve_out_path(args.out_dir, stem + ".mp4")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))

    # Load + resize
    orig = Image.open(args.input).convert("RGBA")
    w0, h0 = orig.size
    w = int(args.width)
    h = int(h0 * w / w0)
    base = orig.resize((w, h), Image.LANCZOS).convert("RGBA")
    gray = np.array(base.convert("L"))

    # Fit center/ellipse
    cx0, cy0 = fit_center_dark_centroid(gray)
    fit = None
    if not args.no_opencv_ellipse_fit:
        fit = try_fit_ellipse_opencv(gray, cx0, cy0)
    if fit is None:
        fit = EllipseFit(
            cx=fit.cx + args.ellipse_dx,
            cy=fit.cy + args.ellipse_dy,
            a=fit.a * args.ellipse_a_scale,
            b=fit.b * args.ellipse_b_scale,
            ang_deg=fit.ang_deg,
        )
    fit_ang = fit.ang_deg + float(args.ellipse_rot_offset_deg) + float(args.ellipse_ang_delta_deg)

    if args.debug_ellipse:
        dbg = base.convert("RGBA")
        d = ImageDraw.Draw(dbg)
        ang = math.radians(fit_ang)
        cA, sA = math.cos(ang), math.sin(ang)
        pts = []
        for tt in np.linspace(0, 2 * math.pi, 720, endpoint=True):
            x = fit.cx + fit.a * math.cos(tt) * cA - fit.b * math.sin(tt) * sA
            y = fit.cy + fit.a * math.cos(tt) * sA + fit.b * math.sin(tt) * cA
            pts.append((x, y))
        d.line(pts, fill=(255, 50, 50, 255), width=4)
        debug_path = str(Path(out_path).with_suffix(".ellipse_debug.png"))
        dbg.save(debug_path)
        print(f"[voidstar_anim] wrote debug ellipse: {debug_path}")


    yy, xx = np.mgrid[0:h, 0:w]
    dx = (xx - fit.cx).astype(np.float32)
    dy = (yy - fit.cy).astype(np.float32)

    omega = 2 * np.pi
    ang = math.radians(fit_ang)
    cA, sA = math.cos(ang), math.sin(ang)
    xR = (cA * dx + sA * dy).astype(np.float32)
    yR = (-sA * dx + cA * dy).astype(np.float32)

    eps = 1e-6
    rho = np.sqrt((xR / (fit.a + eps)) ** 2 + (yR / (fit.b + eps)) ** 2).astype(np.float32)
    theta_e = np.arctan2(yR / (fit.b + eps), xR / (fit.a + eps)).astype(np.float32)

    # band/rim masks
    sigma_band1, sigma_band2 = 0.055, 0.040
    band1 = np.exp(-((rho - 1.00) ** 2) / (2 * sigma_band1 ** 2)).astype(np.float32)
    band2 = np.exp(-((rho - 1.06) ** 2) / (2 * sigma_band2 ** 2)).astype(np.float32)
    band = np.clip(0.75 * band1 + 0.55 * band2, 0, 1)

    sigma_rim = 0.020
    rim_mask = np.exp(-((rho - 1.0) ** 2) / (2 * sigma_rim ** 2)).astype(np.float32)

    # phase maps
    phase_map = make_phase_map(rng, w, h, blur=7)
    phase_map2 = make_phase_map(rng, w, h, blur=5)

    # edges for neuron network
    edges = base.convert("L").filter(ImageFilter.FIND_EDGES)
    edge_arr = np.array(edges).astype(np.float32) / 255.0
    edge_strength = np.clip((edge_arr - 0.15) / 0.85, 0, 1)
    edge_img = Image.fromarray((edge_strength * 255).astype(np.uint8), mode="L").filter(ImageFilter.GaussianBlur(1.3))
    edge_strength_blur = np.array(edge_img).astype(np.float32) / 255.0

    # sample neuron event points
    coords = np.column_stack(np.nonzero(edge_strength > 0.42))
    if coords.shape[0] < 1400:
        coords = np.column_stack(np.nonzero(edge_strength > 0.34))
    rr = np.sqrt((coords[:, 1] - fit.cx) ** 2 + (coords[:, 0] - fit.cy) ** 2)
    coords = coords[rr > 170]
    n_events = min(int(args.events), coords.shape[0])
    pick = rng.choice(coords.shape[0], size=n_events, replace=False)
    event_pos = coords[pick]
    base_ph = (np.arange(n_events, dtype=np.float32) + 0.5) / float(n_events)
    j = rng.normal(0.0, 0.25 / float(n_events), size=n_events).astype(np.float32)  # small jitter
    event_phase = (base_ph + j) % 1.0
    rng.shuffle(event_phase)


    # rim spark points
    rim_coords = np.column_stack(np.nonzero((rim_mask > 0.55) & (band > 0.20)))
    if rim_coords.shape[0] > 0:
        sel = rng.choice(rim_coords.shape[0], size=min(240, rim_coords.shape[0]), replace=False)
        rim_events = rim_coords[sel]
    else:
        rim_events = np.empty((0, 2), dtype=np.int32)
    rim_event_phase = rng.random(rim_events.shape[0]).astype(np.float32)

    # build raw neuron frames
    raw_fire: list[np.ndarray] = []
    for i in range(int(args.frames)):
        t = i / float(args.frames)
        firing = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(firing)

        for (yy0, xx0), ph in zip(event_pos, event_phase):
            dt = circ_dist(t, float(ph))
            a_t = math.exp(-(dt * dt) / (2 * (float(args.event_sigma) ** 2)))
            if a_t > float(args.fire_threshold):
                v = int(min(255, 16 + 210 * a_t))
                r0 = int(2 + 2.0 * a_t)
                draw.ellipse((int(xx0 - r0), int(yy0 - r0), int(xx0 + r0), int(yy0 + r0)), fill=v)

        crawl_phase = (omega * (float(args.crawl_speed) * t) + float(args.crawl_noise_scale) * phase_map2).astype(np.float32)
        if float(args.crawl_x_freq) != 0.0:
            crawl_phase = crawl_phase + (float(args.crawl_x_freq) * (xx / w) * 2 * np.pi).astype(np.float32)

        crawl = (0.5 + 0.5 * np.sin(crawl_phase)).astype(np.float32)
        crawl = (crawl ** float(args.crawl_power)) * edge_strength_blur
        crawl_L = Image.fromarray(np.clip(crawl * 255, 0, 255).astype(np.uint8), mode="L").filter(ImageFilter.GaussianBlur(1.4))

        raw = np.maximum(np.array(firing, dtype=np.float32), np.array(crawl_L, dtype=np.float32))
        raw_fire.append(raw)

    # trails
    fire_trail_list: list[Image.Image] = []
    for i in range(int(args.frames)):
        acc = np.zeros((h, w), dtype=np.float32)
        for k in range(int(args.trail_frames)):
            j = (i - k) % int(args.frames)
            acc += (float(args.trail_decay) ** k) * raw_fire[j]
        acc = np.clip(acc, 0, 255)
        im = Image.fromarray(acc.astype(np.uint8), mode="L").filter(ImageFilter.GaussianBlur(1.7)).filter(ImageFilter.GaussianBlur(0.6))
        fire_trail_list.append(im)

    # halo frames
    M_ARCS = int(args.arcs)
    arc_phase = rng.random(M_ARCS).astype(np.float32)
    arc_phi0 = (rng.random(M_ARCS) * 2 * np.pi - np.pi).astype(np.float32)
    arc_speed = rng.uniform(-0.55, 0.55, size=M_ARCS).astype(np.float32)
    arc_sigma_t = 0.020
    arc_sigma_phi = rng.uniform(0.06, 0.14, size=M_ARCS).astype(np.float32)

    halo_L_list: list[Image.Image] = []
    for i in range(int(args.frames)):
        t = i / float(args.frames)

        breathe = 0.016 * math.sin(omega * t)
        rho_b = rho + breathe * band1

        swirl1 = band * (0.55 + 0.45 * np.sin(6.0 * theta_e + omega * (t * 1.0) + 1.0 * phase_map + 8.5 * (rho_b - 1.0)))
        swirl2 = band * (0.55 + 0.45 * np.sin(11.0 * theta_e - omega * (t * 1.25) - 0.60 * phase_map + 14.5 * (rho_b - 1.0)))
        swirl = np.clip(0.70 * swirl1 + 0.50 * swirl2, 0, 1)

        rim = np.clip(rim_mask * (0.5 + 0.5 * np.sin(omega * (3 * t) + phase_map)), 0, 1)
        swirl = np.clip(swirl + 0.22 * rim, 0, 1)

        arcs = np.zeros((h, w), dtype=np.float32)
        for m in range(M_ARCS):
            dt = circ_dist(t, float(arc_phase[m]))
            a_t = math.exp(-(dt * dt) / (2 * (arc_sigma_t ** 2)))
            if a_t < 0.06:
                continue
            phi = float(arc_phi0[m] + (arc_speed[m] * omega * t))
            dphi = wrap_angle(theta_e - phi)
            spatial = np.exp(-(dphi * dphi) / (2 * (arc_sigma_phi[m] ** 2))).astype(np.float32)
            jag = (0.55 + 0.45 * np.sin(22.0 * theta_e + 6.0 * (rho - 1.0) + phase_map * 1.2)).astype(np.float32)
            arcs += (a_t * spatial * rim_mask * jag)

        micro = np.clip(rim_mask * (0.5 + 0.5 * np.sin(40.0 * theta_e + omega * (6.0 * t) + phase_map * 1.8)), 0, 1)
        arcs += float(args.micro_arc_strength) * micro

        sparks = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(sparks)
        for (yy0, xx0), ph in zip(rim_events, rim_event_phase):
            dt = circ_dist(t, float(ph))
            a_t = math.exp(-(dt * dt) / (2 * (0.018 ** 2)))
            if a_t > 0.18:
                v = int(min(255, 28 + 175 * a_t))
                r0 = int(2 + 1.6 * a_t)
                draw.ellipse((int(xx0 - r0), int(yy0 - r0), int(xx0 + r0), int(yy0 + r0)), fill=v)
        sparks = sparks.filter(ImageFilter.GaussianBlur(1.0))

        halo = np.clip(swirl + float(args.arc_strength) * np.clip(arcs, 0, 1), 0, 1) * float(args.halo_intensity)
        halo_L = Image.fromarray((np.clip(halo, 0, 1) * 255).astype(np.uint8), mode="L").filter(ImageFilter.GaussianBlur(float(args.halo_blur)))
        halo_L = ImageChops.add(halo_L, sparks, scale=1.0, offset=0)
        halo_L_list.append(halo_L)

    # pulse frames (starts at ellipse rim)
    a_p = fit.a * float(args.pulse_ellipse_scale)
    b_p = fit.b * float(args.pulse_ellipse_scale)
    rho_p = np.sqrt((xR / (a_p + eps)) ** 2 + (yR / (b_p + eps)) ** 2).astype(np.float32)

    pulse_L_list: list[Image.Image] = []
    for i in range(int(args.frames)):
        t = i / float(args.frames)
        wave_center = 1.0 + t * float(args.pulse_range)
        wave = np.exp(-((rho_p - wave_center) ** 2) / (2 * (float(args.pulse_sigma) ** 2))).astype(np.float32)

        fade = np.clip(1.0 - (rho_p - 1.0) / (float(args.pulse_range) + 1e-6), 0, 1).astype(np.float32)
        wave *= (fade ** 0.9)
        wave *= (rho_p >= 1.0).astype(np.float32)

        center = np.exp(-(rho_p * rho_p) / (2 * (0.58 ** 2))).astype(np.float32)
        center_breathe = center * (0.5 + 0.5 * math.sin(omega * t))

        pulse = ((1.0 - args.pulse_center_mix) * wave + args.pulse_center_mix * center_breathe) * float(args.pulse_intensity)

        pulse_L = Image.fromarray(np.clip(pulse * float(args.pulse_level), 0, 255).astype(np.uint8), mode="L")
        pulse_L = pulse_L.filter(ImageFilter.GaussianBlur(float(args.pulse_blur)))
        pulse_L_list.append(pulse_L)

    # Composite frames
    frames_rgb: list[np.ndarray] = []
    glow = float(args.glow_scale)

    for i in range(int(args.frames)):
        f_arr = np.array(fire_trail_list[i], dtype=np.uint8)
        a_fire = np.clip(f_arr.astype(np.float32) * glow, 0, float(args.alpha_fire)).astype(np.uint8)
        fire_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        fire_rgba[..., 0] = (f_arr * 0.50).astype(np.uint8)
        fire_rgba[..., 1] = (f_arr * 0.76).astype(np.uint8)
        fire_rgba[..., 2] = f_arr
        fire_rgba[..., 3] = a_fire
        fire_img = Image.fromarray(fire_rgba, mode="RGBA")

        s2 = np.array(halo_L_list[i], dtype=np.uint8)
        a_halo = np.clip(s2.astype(np.float32) * glow, 0, float(args.alpha_halo)).astype(np.uint8)
        halo_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        halo_rgba[..., 0] = (s2 * 0.72).astype(np.uint8)
        halo_rgba[..., 1] = (s2 * 0.82).astype(np.uint8)
        halo_rgba[..., 2] = s2
        halo_rgba[..., 3] = a_halo
        halo_img = Image.fromarray(halo_rgba, mode="RGBA")

        p2 = np.array(pulse_L_list[i], dtype=np.uint8)
        a_pulse = np.clip(p2.astype(np.float32) * glow, 0, float(args.alpha_pulse)).astype(np.uint8)
        pulse_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        pulse_rgba[..., 0] = (p2 * 0.82).astype(np.uint8)
        pulse_rgba[..., 1] = (p2 * 0.90).astype(np.uint8)
        pulse_rgba[..., 2] = p2
        pulse_rgba[..., 3] = a_pulse
        pulse_img = Image.fromarray(pulse_rgba, mode="RGBA")

        comp = base.copy()
        comp = ImageChops.add(comp, pulse_img, scale=1.0, offset=0)
        comp = ImageChops.add(comp, halo_img, scale=1.0, offset=0)
        comp = ImageChops.add(comp, fire_img, scale=1.0, offset=0)

        t = i / float(args.frames)
        tw = 1.0 + float(args.twinkle) * math.sin(2 * math.pi * t)
        comp_np = np.array(comp, dtype=np.float32)
        comp_np[..., :3] = np.clip(comp_np[..., :3] * tw, 0, 255)
        comp = Image.fromarray(comp_np.astype(np.uint8), mode="RGBA")

        rgb = ensure_even_dims(np.array(comp.convert("RGB")))
        frames_rgb.append(rgb)

        if (i + 1) % max(1, int(args.frames) // 10) == 0:
            print(f"[voidstar_anim] frame {i+1}/{args.frames}")

    # Encode MP4
    write_mp4_via_ffmpeg(
        frames_rgb=frames_rgb,
        out_mp4=out_path,
        fps=float(args.fps),
        crf=int(args.crf),
        preset=str(args.ffpreset),
        show_progress=bool(args.encode_progress),
        log_interval_s=float(args.encode_log_interval),
    )
    print(f"[voidstar_anim] wrote MP4  {out_path}")


if __name__ == "__main__":
    main()
