#!/usr/bin/env python3
"""
voidstar_glyphfield.py
Randy-grade glyph field renderer.

UPDATED:
✓ --start
✓ --duration
✓ proper frame bounds
"""

import argparse
import time
import math
import sys
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm


# =========================
# Logging helpers
# =========================

def log(msg):
    print(f"[voidstar] {msg}", flush=True)


def format_eta(start_time, progress, total):
    if progress == 0:
        return "--:--"
    elapsed = time.time() - start_time
    rate = progress / elapsed
    remaining = (total - progress) / rate
    return time.strftime("%M:%S", time.gmtime(remaining))


# =========================
# Glyph set
# =========================

GLYPHS = np.array(list(" .·:+*#%@"))


def brightness_to_glyph_idx(gray):
    norm = gray.astype(np.float32) / 255.0
    idx = (norm * (len(GLYPHS) - 1)).astype(np.int32)
    return idx


# =========================
# CPU FALLBACK
# =========================

def run_cpu(args):
    log("mode=CPU fallback")

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError("Could not open input")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_input = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 🎯 compute frame bounds
    start_frame = int(args.start * fps)

    if args.duration is None:
        end_frame = total_frames_input
    else:
        end_frame = min(
            total_frames_input,
            start_frame + int(args.duration * fps)
        )

    frames_to_process = max(0, end_frame - start_frame)

    log(f"input={args.input}")
    log(f"fps={fps:.3f}")
    log(f"start={args.start:.3f}s ({start_frame})")
    log(f"end_frame={end_frame}")
    log(f"frames_to_process={frames_to_process}")

    # 🎯 seek
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    out_path = build_output_name(args)
    log(f"output={out_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    prev_gray = None
    start_time = time.time()

    font = cv2.FONT_HERSHEY_SIMPLEX
    cell = args.cell

    with tqdm(total=frames_to_process) as pbar:
        frame_idx = 0

        while frame_idx < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # motion emphasis
            if prev_gray is not None:
                motion = cv2.absdiff(gray, prev_gray)
                motion = cv2.GaussianBlur(motion, (5, 5), 0)
            else:
                motion = np.zeros_like(gray)

            prev_gray = gray

            # edge bias
            edges = cv2.Canny(gray, 60, 120)

            energy = (
                gray.astype(np.float32) * args.luma_weight +
                motion.astype(np.float32) * args.motion_weight +
                edges.astype(np.float32) * args.edge_weight
            )

            energy = np.clip(energy, 0, 255).astype(np.uint8)
            glyph_idx = brightness_to_glyph_idx(energy)

            out = np.zeros_like(frame)

            step = cell
            for y in range(0, height, step):
                for x in range(0, width, step):

                    if np.random.rand() > args.density:
                        continue

                    idx = glyph_idx[y, x]
                    ch = GLYPHS[idx]

                    brightness = energy[y, x] / 255.0
                    color = (
                        int(255 * brightness * 0.4),
                        int(255 * brightness * 0.7),
                        int(255 * brightness * 1.0),
                    )

                    cv2.putText(
                        out,
                        ch,
                        (x, y),
                        font,
                        args.font_scale,
                        color,
                        1,
                        cv2.LINE_AA,
                    )

            writer.write(out)

            frame_idx += 1

            if frame_idx % 10 == 0:
                eta = format_eta(start_time, frame_idx, frames_to_process)
                log(f"frames={frame_idx}/{frames_to_process} eta={eta}")

            pbar.update(1)

    cap.release()
    writer.release()

    log(f"✅ done → {out_path}")


# =========================
# GPU PATH (still stub)
# =========================

def run_gpu(args):
    try:
        import moderngl  # noqa
    except Exception:
        log("⚠️ moderngl not available — falling back to CPU")
        run_cpu(args)
        return

    log("mode=GPU (delegating to CPU for now)")
    run_cpu(args)


# =========================
# Output naming
# =========================

def build_output_name(args):
    in_path = Path(args.input)
    stem = in_path.stem
    out_dir = in_path.parent

    dur_str = "full" if args.duration is None else f"{args.duration:.2f}"

    filename = (
        f"{stem}_glyph"
        f"_st{args.start:.2f}"
        f"_dur{dur_str}"
        f"_c{args.cell}"
        f"_d{args.density:.2f}"
        f".mp4"
    )

    return str(out_dir / filename)



# =========================
# CLI
# =========================

def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("input")

    ap.add_argument("--mode", default="auto",
                    choices=["auto", "gpu", "cpu"])

    # ✅ NEW
    ap.add_argument("--start", type=float, default=0.0,
                    help="start time in seconds")

    ap.add_argument("--duration", type=float, default=None,
                    help="duration in seconds")

    ap.add_argument("--cell", type=int, default=12)
    ap.add_argument("--density", type=float, default=0.65)
    ap.add_argument("--font-scale", type=float, default=0.4)

    ap.add_argument("--motion-weight", type=float, default=1.6)
    ap.add_argument("--edge-weight", type=float, default=1.2)
    ap.add_argument("--luma-weight", type=float, default=0.8)

    return ap.parse_args()


# =========================
# Main
# =========================

def main():
    args = parse_args()

    log("starting glyph field")

    if args.mode == "cpu":
        run_cpu(args)
    elif args.mode == "gpu":
        run_gpu(args)
    else:
        try:
            import moderngl  # noqa
            run_gpu(args)
        except Exception:
            run_cpu(args)


if __name__ == "__main__":
    main()
