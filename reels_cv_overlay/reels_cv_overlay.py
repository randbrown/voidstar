import argparse
import subprocess
import shutil
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp


# =========================
# DEFAULTS
# =========================
DEFAULT_OUT_W = 1080
DEFAULT_OUT_H = 1920
DEFAULT_FPS = 30.0

DEFAULT_MODEL_COMPLEXITY = 2
DEFAULT_MIN_DET_CONF = 0.5
DEFAULT_MIN_TRK_CONF = 0.5

DEFAULT_TRAIL = True
DEFAULT_TRAIL_ALPHA = 0.9

DEFAULT_SCANLINES = True
DEFAULT_SCANLINE_STRENGTH = 0.06

DEFAULT_CRF = 18
DEFAULT_PRESET = "veryfast"
DEFAULT_AUDIO_BITRATE = "320k"

DEFAULT_START = 0.0
DEFAULT_DURATION = None


# =========================
# UTILS
# =========================
def bool_flag(v: str) -> bool:
    v = v.strip().lower()
    if v in ("1", "true", "t", "yes", "y", "on"):
        return True
    if v in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {v}")


def run(cmd):
    print("▶", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


def ffprobe_duration_seconds(video_path: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe not found")
    r = subprocess.run(
        [ffprobe, "-v", "error",
         "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1",
         str(video_path)],
        stdout=subprocess.PIPE,
        text=True,
        check=True
    )
    return float(r.stdout.strip())


def fmt_float(x: float) -> str:
    return f"{x:.2f}".replace(".", "p")


def build_output_name(src: Path, args) -> Path:
    parts = []

    if args.fps:
        parts.append(f"fps{int(args.fps)}")
    if args.start > 0:
        parts.append(f"s{fmt_float(args.start)}")
    if args.duration is not None:
        parts.append(f"d{fmt_float(args.duration)}")

    parts += [
        f"mc{args.model_complexity}",
        f"det{fmt_float(args.min_det_conf)}",
        f"trk{fmt_float(args.min_trk_conf)}",
        f"trail{int(args.trail)}",
        f"ta{fmt_float(args.trail_alpha)}",
        f"scan{int(args.scanlines)}",
    ]

    if args.velocity_color:
        parts.append("velcolor")
    if args.draw_ids:
        parts.append("ids")
    if args.code_overlay:
        parts.append(f"code{args.code_overlay_order}")

    return src.with_name(f"{src.stem}_{'_'.join(parts)}.mp4")


def fit_to_reels(frame, out_w, out_h):
    h, w = frame.shape[:2]
    target_aspect = out_w / out_h
    src_aspect = w / h

    if src_aspect > target_aspect:
        new_w = int(h * target_aspect)
        x0 = (w - new_w) // 2
        crop = frame[:, x0:x0 + new_w]
    else:
        new_h = int(w / target_aspect)
        y0 = (h - new_h) // 2
        crop = frame[y0:y0 + new_h, :]

    return cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)


def apply_scanlines(frame, strength=0.06):
    out = frame.astype(np.float32)
    out[::2] *= (1.0 - strength)
    return np.clip(out, 0, 255).astype(np.uint8)


# =========================
# CODE OVERLAY
# =========================
def render_code_overlay(lines, w, h, font_scale=0.6, line_spacing=1.45):
    base_line_h = int(22 * font_scale)
    line_advance = max(1, int(base_line_h * line_spacing))

    img_h = max(h * 2, len(lines) * line_advance + h)
    img = np.zeros((img_h, w, 3), np.uint8)

    y = 30  # ✅ start code at top (no dead space)
    for line in lines:
        cv2.putText(
            img,
            line.rstrip(),
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )
        y += line_advance

    return img


def overlay_code(frame, code_img, yoff, alpha=0.85):
    h, w = frame.shape[:2]
    yoff = int(max(0, min(yoff, code_img.shape[0] - h)))
    src = code_img[yoff:yoff + h]
    return cv2.addWeighted(frame, 1.0, src, alpha, 0)


# =========================
# VELOCITY COLORING
# =========================
def velocity_to_color(v, v_min=0.0, v_max=800.0):
    t = (v - v_min) / max(1e-6, (v_max - v_min))
    t = max(0.0, min(1.0, t))

    if t < 0.5:
        a = t / 0.5
        return (int(255 * (1 - a)), int(255 * a), 0)
    else:
        a = (t - 0.5) / 0.5
        return (0, int(255 * (1 - a)), int(255 * a))


def draw_landmark_id(frame, x, y, idx, color):
    label = str(idx)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.35
    thickness = 1

    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    pad = 3
    x0 = x + 4
    y0 = y - th - 4

    cv2.rectangle(
        frame,
        (x0 - pad, y0 - pad),
        (x0 + tw + pad, y0 + th + pad),
        color,
        1
    )

    cv2.putText(
        frame,
        label,
        (x0, y0 + th),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


# =========================
# CFR INTERMEDIATE
# =========================
def make_cfr_intermediate(src, dst, fps, start, duration):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found")

    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(src)]
    if start > 0:
        cmd += ["-ss", str(start)]
    if duration is not None:
        cmd += ["-t", str(duration)]

    cmd += [
        "-vf", f"fps={fps}:round=near",
        "-fps_mode", "cfr",
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-profile:v", "baseline",
        "-crf", "18",
        "-preset", "veryfast",
        str(dst)
    ]

    run(cmd)


# =========================
# MAIN
# =========================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("input")
    ap.add_argument("-o", "--output")

    ap.add_argument("--width", type=int, default=DEFAULT_OUT_W)
    ap.add_argument("--height", type=int, default=DEFAULT_OUT_H)
    ap.add_argument("--fps", type=float, default=DEFAULT_FPS)

    ap.add_argument("--start", type=float, default=DEFAULT_START)
    ap.add_argument("--duration", type=float, default=DEFAULT_DURATION)

    ap.add_argument("--audio-wav")

    ap.add_argument(
        "--code-overlay",
        nargs="?",
        const=True,
        default=False,
        help="Overlay scrolling code. If true, defaults to this script. If a path is given, uses that file."
    )
    ap.add_argument("--code-overlay-order", choices=["before", "after"], default="after")

    ap.add_argument("--model-complexity", type=int, default=DEFAULT_MODEL_COMPLEXITY)
    ap.add_argument("--min-det-conf", type=float, default=DEFAULT_MIN_DET_CONF)
    ap.add_argument("--min-trk-conf", type=float, default=DEFAULT_MIN_TRK_CONF)

    ap.add_argument("--trail", type=bool_flag, default=DEFAULT_TRAIL)
    ap.add_argument("--trail-alpha", type=float, default=DEFAULT_TRAIL_ALPHA)

    ap.add_argument("--draw-ids", type=bool_flag, default=False)
    ap.add_argument("--velocity-color", type=bool_flag, default=False)

    ap.add_argument("--scanlines", type=bool_flag, default=DEFAULT_SCANLINES)
    ap.add_argument("--scanline-strength", type=float, default=DEFAULT_SCANLINE_STRENGTH)

    ap.add_argument("--crf", type=int, default=DEFAULT_CRF)
    ap.add_argument("--preset", default=DEFAULT_PRESET)
    ap.add_argument("--audio-bitrate", default=DEFAULT_AUDIO_BITRATE)

    args = ap.parse_args()

    src = Path(args.input)
    outp = Path(args.output) if args.output else build_output_name(src, args)

    full_duration = ffprobe_duration_seconds(src)

    tmp_cfr = outp.with_suffix(".cfr.tmp.mp4")
    make_cfr_intermediate(src, tmp_cfr, args.fps, args.start, args.duration)

    cap = cv2.VideoCapture(str(tmp_cfr))
    out = cv2.VideoWriter(
        str(outp),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.width, args.height)
    )

    # --- Code overlay setup ---
    code_img = None
    code_scroll_range = 0
    if args.code_overlay:
        code_path = Path(__file__).resolve() if args.code_overlay is True else Path(args.code_overlay)
        if not code_path.exists():
            raise FileNotFoundError(code_path)

        lines = code_path.read_text(encoding="utf-8", errors="replace").splitlines()
        code_img = render_code_overlay(lines, args.width, args.height)

        # ✅ cap scroll range so short clips don't hyper-scroll
        max_scroll = code_img.shape[0] - args.height
        code_scroll_range = max(1, min(max_scroll, args.height * 2))

    mp_pose = mp.solutions.pose
    prev_landmarks = None
    trail_buf = None
    fps = float(args.fps)
    frame_idx = 0

    with mp_pose.Pose(
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_det_conf,
        min_tracking_confidence=args.min_trk_conf
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = fit_to_reels(frame, args.width, args.height)

            if code_img is not None and args.code_overlay_order == "before":
                t = args.start + frame_idx / fps
                yoff = (t / full_duration) * code_scroll_range
                frame = overlay_code(frame, code_img, yoff)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if args.trail:
                trail_buf = np.zeros_like(frame) if trail_buf is None else (trail_buf * args.trail_alpha).astype(np.uint8)

            if res.pose_landmarks:
                h, w = frame.shape[:2]
                curr_landmarks = []
                velocities = {}

                for i, lm in enumerate(res.pose_landmarks.landmark):
                    x, y = int(lm.x * w), int(lm.y * h)
                    curr_landmarks.append((x, y))
                    velocities[i] = 0.0 if not prev_landmarks else (
                        ((x - prev_landmarks[i][0]) ** 2 + (y - prev_landmarks[i][1]) ** 2) ** 0.5 * fps
                    )

                draw_target = trail_buf if args.trail else frame

                for a, b in mp_pose.POSE_CONNECTIONS:
                    color = velocity_to_color(max(velocities[a], velocities[b])) if args.velocity_color else (255, 255, 255)
                    cv2.line(draw_target, curr_landmarks[a], curr_landmarks[b], color, 1)

                if args.draw_ids:
                    for i, (x, y) in enumerate(curr_landmarks):
                        color = velocity_to_color(velocities[i]) if args.velocity_color else (255, 255, 255)
                        draw_landmark_id(frame, x, y, i, color)

                prev_landmarks = curr_landmarks

            if args.trail:
                frame = cv2.addWeighted(frame, 1.0, trail_buf, 1.0, 0)

            if code_img is not None and args.code_overlay_order == "after":
                t = args.start + frame_idx / fps
                yoff = (t / full_duration) * code_scroll_range
                frame = overlay_code(frame, code_img, yoff)

            if args.scanlines:
                frame = apply_scanlines(frame, args.scanline_strength)

            out.write(frame)
            frame_idx += 1

    cap.release()
    out.release()
    tmp_cfr.unlink(missing_ok=True)

    print("✅ Saved:", outp)


if __name__ == "__main__":
    main()
