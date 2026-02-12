import argparse
import cv2
import numpy as np
import math

ASCII_CHARS = " .:-=+*#%@"

# =========================
# UTILS
# =========================
def resize_keep_aspect(img, target_width):
    h, w = img.shape[:2]
    scale = target_width / w
    return cv2.resize(img, (target_width, int(h * scale)))

def image_to_ascii(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    norm = gray / 255.0
    idx = (norm * (len(ASCII_CHARS) - 1)).astype(int)
    return ["".join(ASCII_CHARS[i] for i in row) for row in idx]

def ascii_to_image(lines, scale=8):
    h = len(lines)
    w = len(lines[0])
    img = np.zeros((h * scale, w * scale), np.uint8)
    for y, line in enumerate(lines):
        for x, ch in enumerate(line):
            v = int(255 * ASCII_CHARS.index(ch) / (len(ASCII_CHARS) - 1))
            img[y*scale:(y+1)*scale, x*scale:(x+1)*scale] = v
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# =========================
# SILHOUETTE MASK
# =========================
def extract_bigfoot_mask(img, thresh=90):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binary threshold
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)

    # Morphological cleanup
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Flood fill from center to isolate main figure
    h, w = mask.shape
    flood = mask.copy()
    ff_mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(flood, ff_mask, (w//2, h//2), 255)

    return flood

# =========================
# SILHOUETTE ANIMATION
# =========================
def animate_masked(img, mask, t):
    h, w = img.shape[:2]
    out = img.copy()

    walk = math.sin(2 * math.pi * t)
    counter = math.sin(2 * math.pi * t + math.pi)

    for y in range(h):
        # normalize vertical position (0 = top, 1 = bottom)
        yn = y / h

        # FOOT ZONE (planted)
        if yn > 0.85:
            dx = 0

        # LEG ZONE
        elif yn > 0.55:
            dx = int(10 * walk * (1 - (yn - 0.55) / 0.3))

        # TORSO ZONE
        elif yn > 0.30:
            dx = int(3 * counter)

        # HEAD ZONE
        else:
            dx = int(2 * walk)

        # apply shift only where masked
        row_mask = mask[y] == 255
        if not np.any(row_mask):
            continue

        src_x = np.where(row_mask)[0]
        dst_x = src_x + dx

        valid = (dst_x >= 0) & (dst_x < w)
        out[y, dst_x[valid]] = img[y, src_x[valid]]

    return out


# =========================
# MAIN
# =========================
def main():
    ap = argparse.ArgumentParser(description="Voidstar ASCII Bigfoot (masked animation)")
    ap.add_argument("image")
    ap.add_argument("-o", "--output", default="voidstar_ascii_bigfoot_masked.mp4")
    ap.add_argument("--width", type=int, default=120)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--threshold", type=int, default=90)
    args = ap.parse_args()

    src = cv2.imread(args.image)
    if src is None:
        raise FileNotFoundError(args.image)

    src = resize_keep_aspect(src, args.width)
    mask = extract_bigfoot_mask(src, args.threshold)

    ascii_sample = image_to_ascii(src)
    frame_img = ascii_to_image(ascii_sample)
    h, w = frame_img.shape[:2]

    out = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (w, h)
    )

    total_frames = int(args.fps * args.duration)

    for f in range(total_frames):
        t = f / args.fps
        anim = animate_masked(src, mask, t)
        ascii_frame = image_to_ascii(anim)
        img = ascii_to_image(ascii_frame)
        out.write(img)

    out.release()
    print("✅ Saved:", args.output)

if __name__ == "__main__":
    main()
