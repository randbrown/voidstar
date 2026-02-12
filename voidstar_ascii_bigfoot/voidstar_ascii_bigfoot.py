#!/usr/bin/env python3

import argparse
import cv2
import numpy as np
import math
from pathlib import Path

# =========================
# ASCII SETTINGS
# =========================
ASCII_CHARS = " .:-=+*#%@"
ASCII_SCALE_X = 0.55  # compensate for font aspect
BG_CHAR = " "

# =========================
# UTILS
# =========================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def to_ascii(frame, cols, rows):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    cell_w = w / cols
    cell_h = h / rows

    out = []
    for y in range(rows):
        row = ""
        for x in range(cols):
            sx = int(x * cell_w)
            sy = int(y * cell_h)
            v = gray[sy, sx]
            idx = int((v / 255) * (len(ASCII_CHARS) - 1))
            row += ASCII_CHARS[idx]
        out.append(row)
    return out


def ascii_to_frame(ascii_img, width, height):
    font = cv2.FONT_HERSHEY_PLAIN
    img = np.zeros((height, width, 3), dtype=np.uint8)
    char_w = width // len(ascii_img[0])
    char_h = height // len(ascii_img)

    for y, row in enumerate(ascii_img):
        for x, ch in enumerate(row):
            if ch != BG_CHAR:
                cv2.putText(
                    img,
                    ch,
                    (x * char_w, (y + 1) * char_h),
                    font,
                    1.0,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
    return img


# =========================
# REGION ROTATION
# =========================
def rotate_region(img, region, pivot, angle_deg):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = region
    px, py = pivot

    rx0 = int(x0 * w)
    ry0 = int(y0 * h)
    rx1 = int(x1 * w)
    ry1 = int(y1 * h)

    roi = img[ry0:ry1, rx0:rx1].copy()

    cx = int((px * w) - rx0)
    cy = int((py * h) - ry0)

    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    rotated = cv2.warpAffine(
        roi, M, (roi.shape[1], roi.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_TRANSPARENT
    )

    img[ry0:ry1, rx0:rx1] = rotated
    return img

def resize_and_center(img, out_w, out_h):
    h, w = img.shape[:2]
    scale = min(out_w / w, out_h / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    x0 = (out_w - new_w) // 2
    y0 = (out_h - new_h) // 2

    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


# =========================
# SKELETON (swagger tuned)
# =========================
SKELETON = {
    "hip":   {"pivot": (0.50, 0.62)},
    "leg_L": {"pivot": (0.44, 0.62), "region": (0.36, 0.62, 0.46, 0.92)},
    "leg_R": {"pivot": (0.56, 0.62), "region": (0.54, 0.62, 0.64, 0.92)},
    "arm_L": {"pivot": (0.40, 0.44), "region": (0.26, 0.40, 0.40, 0.68)},
    "arm_R": {"pivot": (0.60, 0.44), "region": (0.60, 0.40, 0.74, 0.68)},
    "torso": {"pivot": (0.50, 0.55), "region": (0.30, 0.30, 0.70, 0.75)},
}


# =========================
# SWAGGER ANIMATION
# =========================
def animate_swagger(base, phase):
    t = 2 * math.pi * phase
    t += 0.15 * math.sin(0.5 * t)  # micro drift

    leg_swing = 22 * math.sin(t)
    leg_drag  = 4  * math.sin(2 * t)
    arm_swing = 12 * math.sin(t + math.pi * 0.8)
    torso_sway = 6 * math.sin(t * 0.5)
    bob = int(6 * math.sin(t * 0.5))

    img = base.copy()

    img = rotate_region(img, **SKELETON["leg_L"], angle_deg= leg_swing + leg_drag)
    img = rotate_region(img, **SKELETON["leg_R"], angle_deg=-leg_swing + leg_drag)

    img = rotate_region(img, **SKELETON["arm_L"], angle_deg=-arm_swing)
    img = rotate_region(img, **SKELETON["arm_R"], angle_deg= arm_swing)

    img = rotate_region(img, **SKELETON["torso"], angle_deg=torso_sway)

    if bob != 0:
        img = np.roll(img, bob, axis=0)

    return img


# =========================
# MAIN
# =========================
def main():
    ap = argparse.ArgumentParser(description="Voidstar ASCII Sasquatch Swagger")
    ap.add_argument("image")
    ap.add_argument("--out", default="voidstar_ascii_sasquatch.mp4")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--seconds", type=int, default=10)
    ap.add_argument("--cols", type=int, default=120)
    ap.add_argument("--rows", type=int, default=60)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    src = cv2.imread(args.image)
    if src is None:
        raise FileNotFoundError(args.image)

    src = resize_and_center(src, args.width, args.height)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out, fourcc, args.fps, (args.width, args.height))

    total_frames = args.fps * args.seconds

    for i in range(total_frames):
        phase = (i / args.fps) % 1.0

        animated = animate_swagger(src, phase)
        ascii_img = to_ascii(animated, args.cols, args.rows)
        frame = ascii_to_frame(ascii_img, args.width, args.height)

        out.write(frame)

        if i % 10 == 0:
            print(f"frame {i}/{total_frames}")

    out.release()
    print("✅ Saved:", args.out)


if __name__ == "__main__":
    main()
