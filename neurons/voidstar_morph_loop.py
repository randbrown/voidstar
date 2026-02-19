#!/usr/bin/env python3
import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np


def fmt_float_token(x: float, ndigits: int = 3) -> str:
    """
    Convert float to a filename-safe token like 0.250 -> 0p250
    """
    s = f"{x:.{ndigits}f}"
    return s.replace(".", "p")


def default_out_path(left_path: str, right_path: str, seconds: float, fps: int,
                     cx: float, cy: float, swirl_radius: float, w: int, h: int) -> str:
    L = Path(left_path)
    R = Path(right_path)

    # base name: from left file (common in your pipelines), with a hint of the pair
    stem = L.stem
    pair = f"L{L.stem}_R{R.stem}"

    out_name = (
        f"{stem}_morph_"
        f"s{fmt_float_token(seconds,2)}_fps{fps}_"
        f"cx{fmt_float_token(cx,3)}_cy{fmt_float_token(cy,3)}_"
        f"rad{fmt_float_token(swirl_radius,3)}_"
        f"{w}x{h}_"
        f"{pair}.mp4"
    )

    # Keep output in the left image folder by default
    return str(L.parent / out_name)


def smooth_pingpong(t: float) -> float:
    """
    t in [0,1) -> 0->1->0 smoothly across the full duration.
    Perfect-loop friendly.
    """
    return 0.5 - 0.5 * math.cos(2.0 * math.pi * t)


def subtle_drift(img: np.ndarray, t: float) -> np.ndarray:
    """
    Tiny, looping drift/zoom/rotation to make the neural net feel alive.
    """
    h, w = img.shape[:2]
    cx, cy = w * 0.5, h * 0.5

    angle = 0.35 * math.sin(2 * math.pi * t)                # degrees
    scale = 1.0 + 0.005 * math.sin(2 * math.pi * (t + 0.13))
    dx = 3.0 * math.sin(2 * math.pi * (t + 0.21))
    dy = 3.0 * math.sin(2 * math.pi * (t + 0.47))

    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    M[0, 2] += dx
    M[1, 2] += dy

    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT
    )


def vortex_swirl(frame: np.ndarray, t: float, center_xy, radius_px: float) -> np.ndarray:
    """
    Swirl pixels near the voidstar center (like a processing vortex),
    masked by radius. Includes a subtle pulsing halo glow.
    """
    h, w = frame.shape[:2]
    cx, cy = center_xy

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = xx - cx
    dy = yy - cy
    r = np.sqrt(dx * dx + dy * dy) + 1e-6

    m = np.clip(1.0 - (r / radius_px), 0.0, 1.0)
    m = m * m * (3.0 - 2.0 * m)  # smoothstep

    swirl = 1.2 * math.sin(2 * math.pi * t)
    dtheta = swirl * m

    cosv = np.cos(dtheta)
    sinv = np.sin(dtheta)

    x2 = cx + dx * cosv - dy * sinv
    y2 = cy + dx * sinv + dy * cosv

    out = cv2.remap(
        frame,
        x2.astype(np.float32),
        y2.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    # loop-safe pulsing glow
    glow = (0.10 + 0.08 * (0.5 - 0.5 * math.cos(2 * math.pi * t)))  # 0.10..0.18
    blurred = cv2.GaussianBlur(out, (0, 0), sigmaX=6.0, sigmaY=6.0)
    out = cv2.addWeighted(out, 1.0, blurred, glow, 0)

    return out


def log(msg: str) -> None:
    print(f"[voidstar] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("left", help="left image (A)")
    ap.add_argument("right", help="right image (B)")
    ap.add_argument("-o", "--out", default=None, help="output .mp4 (default: auto-named)")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--center-x", type=float, default=0.5, help="0..1 fraction of width")
    ap.add_argument("--center-y", type=float, default=0.5, help="0..1 fraction of height")
    ap.add_argument("--
    
    ius", type=float, default=0.22, help="0..1 fraction of min(w,h)")
    ap.add_argument("--crf", type=int, default=18, help="(currently informational; mp4v ignores CRF)")
    ap.add_argument("--preset", default="n/a", help="(informational)")
    ap.add_argument("--status-every", type=int, default=30, help="log status every N frames")
    args = ap.parse_args()

    t0 = time.perf_counter()

    A = cv2.imread(args.left, cv2.IMREAD_COLOR)
    B = cv2.imread(args.right, cv2.IMREAD_COLOR)
    if A is None:
        raise SystemExit(f"[voidstar] ERROR: failed to load left image: {args.left}")
    if B is None:
        raise SystemExit(f"[voidstar] ERROR: failed to load right image: {args.right}")

    # Match sizes
    h = min(A.shape[0], B.shape[0])
    w = min(A.shape[1], B.shape[1])
    A = cv2.resize(A, (w, h), interpolation=cv2.INTER_AREA)
    B = cv2.resize(B, (w, h), interpolation=cv2.INTER_AREA)

    frames = int(round(args.seconds * args.fps))
    if frames <= 1:
        raise SystemExit("[voidstar] ERROR: seconds*fps must produce at least 2 frames.")

    out_path = args.out or default_out_path(
        args.left, args.right, args.seconds, args.fps,
        args.center_x, args.center_y, args.swirl_radius, w, h
    )

    cx = args.center_x * w
    cy = args.center_y * h
    radius_px = args.swirl_radius * min(w, h)

    log("starting")
    log(f"left={args.left}")
    log(f"right={args.right}")
    log(f"size={w}x{h} fps={args.fps} seconds={args.seconds:.3f} frames={frames}")
    log(f"vortex center=({args.center_x:.3f},{args.center_y:.3f}) radius={args.swirl_radius:.3f} (px={radius_px:.1f})")
    log(f"out={out_path}")

    # NOTE: OpenCV mp4v is widely available but not the best quality encoder.
    # If you want libx264 + CRF, we can render PNG frames then ffmpeg encode.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, args.fps, (w, h))
    if not vw.isOpened():
        raise SystemExit("[voidstar] ERROR: failed to open VideoWriter (mp4v). Try output .avi or install codecs.")

    # Precompute grids once for speed? (We recompute inside vortex_swirl currently.)
    # Kept simple for clarity; can optimize later if needed.

    t_write0 = time.perf_counter()
    last_status = time.perf_counter()

    for i in range(frames):
        # IMPORTANT for perfect looping: t uses /frames, NOT /(frames-1)
        t = i / frames
        mix = smooth_pingpong(t)

        A2 = subtle_drift(A, t)
        B2 = subtle_drift(B, (t + 0.37) % 1.0)

        frame = cv2.addWeighted(A2, 1.0 - mix, B2, mix, 0.0)
        frame = vortex_swirl(frame, t, (cx, cy), radius_px)

        # loop-safe sparkle/noise: deterministic seed from frame index
        noise_amp = 2.0
        n = (math.sin(2 * math.pi * (t * 3.0 + 0.11)) + 1.0) * 0.5  # 0..1
        rng = np.random.RandomState(i)  # deterministic
        noise = (rng.randn(h, w, 3) * (noise_amp * n)).astype(np.float32)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        vw.write(frame)

        if args.status_every > 0 and (i % args.status_every == 0 or i == frames - 1):
            now = time.perf_counter()
            elapsed = now - t_write0
            done = i + 1
            fps_eff = done / elapsed if elapsed > 1e-9 else 0.0
            remaining = frames - done
            eta_s = remaining / fps_eff if fps_eff > 1e-9 else float("inf")

            # avoid spamming if status_every is small; still respect user setting
            if now - last_status >= 0.2 or i == frames - 1:
                pct = (done / frames) * 100.0
                log(f"frames={done}/{frames} ({pct:5.1f}%)  speed={fps_eff:6.2f} fps  ETA={eta_s:6.1f}s")
                last_status = now

    vw.release()
    total = time.perf_counter() - t0
    log(f"done in {total:.2f}s -> {out_path}")


if __name__ == "__main__":
    main()
