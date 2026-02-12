import argparse
import cv2
import numpy as np


ASCII_CHARS = np.asarray(list(" .:-=+*#%@"))


# =========================
# VIDEO WRITER (ROBUST)
# =========================
def open_video_writer(path, fps, size):
    codecs = [
        ("mp4v", "MPEG-4"),
        ("MJPG", "Motion JPEG"),
    ]
    for fourcc_str, name in codecs:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        out = cv2.VideoWriter(path, fourcc, fps, size)
        if out.isOpened():
            print(f"🎥 Using codec: {name}")
            return out
    raise RuntimeError("❌ Could not open VideoWriter")


# =========================
# BIGFOOT MASK (ROBUST)
# =========================
def extract_bigfoot_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        41,
        5,
    )

    if np.mean(mask) > 127:
        mask = cv2.bitwise_not(mask)

    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        largest = max(contours, key=cv2.contourArea)
        out = np.zeros_like(gray)
        cv2.drawContours(out, [largest], -1, 255, -1)
        return out

    # Guaranteed fallback
    edges = cv2.Canny(gray, 60, 140)
    return cv2.dilate(edges, None)


# =========================
# ASCII CONVERSION (KEY FIX)
# =========================
def frame_to_ascii(frame, mask, cols):
    h, w = frame.shape[:2]
    scale = cols / w
    rows = int(h * scale * 0.55)

    small = cv2.resize(frame, (cols, rows))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    small_mask = cv2.resize(mask, (cols, rows))

    # Strong contrast normalization
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    chars = np.full(gray.shape, " ", dtype="<U1")

    fg = small_mask > 128

    # Background: extremely subtle
    bg_levels = 3
    bg_norm = (gray / 255.0 * bg_levels).astype(int)
    chars[~fg] = ASCII_CHARS[bg_norm[~fg]]

    # Foreground (Bigfoot): full ASCII range + boost
    fg_norm = (
        gray / 255.0 * (len(ASCII_CHARS) - 1) * 1.4
    )
    fg_norm = np.clip(fg_norm, 0, len(ASCII_CHARS) - 1).astype(int)
    chars[fg] = ASCII_CHARS[fg_norm[fg]]

    return chars


def ascii_to_image(chars, char_px):
    rows, cols = chars.shape
    img = np.zeros((rows * char_px, cols * char_px, 3), np.uint8)

    for y in range(rows):
        for x in range(cols):
            c = chars[y, x]
            if c != " ":
                cv2.putText(
                    img,
                    c,
                    (x * char_px, (y + 1) * char_px),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    char_px / 20,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
    return img


# =========================
# WALK / SWAGGER WARP
# =========================
def apply_walk_warp(t, h, w):
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    stride = np.sin(2 * np.pi * t)
    sway = np.sin(2 * np.pi * t + np.pi / 2)

    dx = stride * 12 * (yy / h)
    dy = sway * 5 * (yy / h)

    return (
        (xx + dx).astype(np.float32),
        (yy + dy).astype(np.float32),
    )


# =========================
# MAIN
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default="voidstar_ascii_bigfoot.mp4")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--duration", type=float, default=10)
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--char-px", type=int, default=12)
    args = ap.parse_args()

    src = cv2.imread(args.input)
    if src is None:
        raise FileNotFoundError(args.input)

    mask = extract_bigfoot_mask(src)

    frames = int(args.duration * args.fps)
    h, w = src.shape[:2]

    out_w = args.width * args.char_px
    out_h = int((h / w) * out_w * 0.55)

    out = open_video_writer(args.output, args.fps, (out_w, out_h))

    for i in range(frames):
        t = i / args.fps

        map_x, map_y = apply_walk_warp(t, h, w)
        warped = cv2.remap(
            src, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )
        warped_mask = cv2.remap(
            mask, map_x, map_y, cv2.INTER_NEAREST
        )

        ascii_chars = frame_to_ascii(warped, warped_mask, args.width)
        img = ascii_to_image(ascii_chars, args.char_px)
        img = cv2.resize(img, (out_w, out_h))

        out.write(img)

    out.release()
    print(f"✅ Saved {frames} frames → {args.output}")


if __name__ == "__main__":
    main()
