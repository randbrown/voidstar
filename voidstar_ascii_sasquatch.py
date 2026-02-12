import argparse
import cv2
import numpy as np
import math
from pathlib import Path

# =========================
# CONFIG
# =========================
ASCII_CHARS = " .:-=+*#%@"
DEFAULT_WIDTH = 120
DEFAULT_FPS = 24
DEFAULT_DURATION = 6.0

# =========================
# UTILS
# =========================
def crop_center_region(img, x_frac=0.5, y_frac=0.5, w_frac=0.42, h_frac=0.78):
    h, w = img.shape[:2]
    cw = int(w * w_frac)
    ch = int(h * h_frac)
    cx = int(w * x_frac)
    cy = int(h * y_frac)
    x0 = max(0, cx - cw // 2)
    y0 = max(0, cy - ch // 2)
    return img[y0:y0 + ch, x0:x0 + cw]

def resize_keep_aspect(img, target_width):
    h, w = img.shape[:2]
    aspect = h / w
    return cv2.resize(img, (target_width, int(target_width * aspect)))

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
# SKELETON (TIGHT)
# =========================
SKELETON = {
    "hip":   {"pivot": (0.50, 0.60)},
    "leg_L": {"pivot": (0.46, 0.60), "region": (0.42, 0.60, 0.50, 0.96)},
    "leg_R": {"pivot": (0.54, 0.60), "region": (0.50, 0.60, 0.58, 0.96)},
    "arm_L": {"pivot": (0.44, 0.40), "region": (0.36, 0.38, 0.46, 0.72)},
    "arm_R": {"pivot": (0.56, 0.40), "region": (0.54, 0.38, 0.64, 0.72)},
    "torso": {"pivot": (0.50, 0.52), "region": (0.42, 0.32, 0.58, 0.78)},
}

# =========================
# ANIMATION
# =========================
def animate_swagger(img, t, debug=False):
    h, w = img.shape[:2]
    out = img.copy()

    walk = math.sin(t * 2 * math.pi)
    sway = math.sin(t * math.pi)

    def region_slice(r):
        x0, y0, x1, y1 = r
        return (
            int(x0 * w), int(y0 * h),
            int(x1 * w), int(y1 * h)
        )

    def rotate_region(region, pivot, angle):
        x0, y0, x1, y1 = region_slice(region)
        px, py = int(pivot[0] * w), int(pivot[1] * h)
        roi = out[y0:y1, x0:x1].copy()

        M = cv2.getRotationMatrix2D((px - x0, py - y0), angle, 1.0)
        rotated = cv2.warpAffine(roi, M, (roi.shape[1], roi.shape[0]),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0,0,0))
        out[y0:y1, x0:x1] = rotated

    rotate_region(SKELETON["leg_L"]["region"], SKELETON["leg_L"]["pivot"],  18 * walk)
    rotate_region(SKELETON["leg_R"]["region"], SKELETON["leg_R"]["pivot"], -18 * walk)
    rotate_region(SKELETON["arm_L"]["region"], SKELETON["arm_L"]["pivot"], -22 * walk)
    rotate_region(SKELETON["arm_R"]["region"], SKELETON["arm_R"]["pivot"],  22 * walk)
    rotate_region(SKELETON["torso"]["region"], SKELETON["hip"]["pivot"],     4 * sway)

    if debug:
        for k, v in SKELETON.items():
            if "region" in v:
                x0, y0, x1, y1 = region_slice(v["region"])
                cv2.rectangle(out, (x0,y0), (x1,y1), (0,255,0), 1)
            px, py = int(v["pivot"][0]*w), int(v["pivot"][1]*h)
            cv2.circle(out, (px,py), 3, (0,0,255), -1)

    return out

# =========================
# MAIN
# =========================
def main():
    ap = argparse.ArgumentParser(description="Voidstar ASCII Sasquatch Walker")
    ap.add_argument("image", help="Ramblin Visioneer image")
    ap.add_argument("-o", "--output", default="voidstar_ascii_bigfoot.mp4")
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)
    ap.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    ap.add_argument("--debug-skeleton", action="store_true")
    args = ap.parse_args()

    src = cv2.imread(args.image)
    if src is None:
        raise FileNotFoundError(args.image)

    src = crop_center_region(src)
    src = resize_keep_aspect(src, args.width)

    total_frames = int(args.fps * args.duration)
    ascii_sample = image_to_ascii(src)
    frame_img = ascii_to_image(ascii_sample)
    h, w = frame_img.shape[:2]

    out = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (w, h)
    )

    for f in range(total_frames):
        t = f / args.fps
        anim = animate_swagger(src, t, args.debug_skeleton)
        ascii_frame = image_to_ascii(anim)
        img = ascii_to_image(ascii_frame)
        out.write(img)

    out.release()
    print("✅ Saved:", args.output)

if __name__ == "__main__":
    main()
