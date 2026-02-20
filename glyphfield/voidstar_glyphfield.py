#!/usr/bin/env python3
"""
voidstar_glyphfield.py
Voidstar glyph renderer with beat-gated audio reactivity.

Behavior (audio reactive):
- Output is ORIGINAL frames unless audio energy >= beat threshold.
- When triggered, switches to glyph mode (optionally held for N frames).
- Parameter aggressiveness scales with energy while in glyph mode.

Also:
- --start/--duration are applied consistently to video processing and audio analysis
- Audio energy is aligned per-frame (no drift)
- Original audio is muxed back losslessly

Author: Voidstar Systems 😈
"""

import argparse
import time
import subprocess
import tempfile
from pathlib import Path
import wave
import math

import numpy as np
import cv2
from tqdm import tqdm


# ============================================================
# Logging helpers
# ============================================================

def log(msg: str) -> None:
    print(f"[voidstar] {msg}", flush=True)


def format_eta(start_time: float, progress: int, total: int) -> str:
    if progress <= 0:
        return "--:--"
    elapsed = time.time() - start_time
    if elapsed <= 0:
        return "--:--"
    rate = progress / elapsed
    if rate <= 0:
        return "--:--"
    remaining = (total - progress) / rate
    return time.strftime("%M:%S", time.gmtime(max(0.0, remaining)))


def run_ffmpeg(cmd):
    log("▶ " + " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


# ============================================================
# Glyphs + palettes
# ============================================================

GLYPHS = np.array(list(" .·:+*#%@"))

PALETTES = {
    "neon": [(255, 80, 40), (255, 160, 20), (255, 255, 60), (100, 255, 180), (40, 200, 255)],
    "fire": [(20, 20, 80), (40, 80, 180), (40, 140, 255), (100, 200, 255), (180, 255, 255)],
    "ice": [(120, 40, 10), (180, 90, 20), (220, 160, 50), (255, 220, 120), (255, 255, 220)],
    "toxic": [(20, 80, 20), (40, 160, 40), (60, 220, 60), (120, 255, 120), (220, 255, 220)],
    "sunset": [(80, 20, 120), (120, 40, 200), (160, 80, 255), (120, 140, 255), (80, 200, 255)],
    "mono": [(40, 40, 40), (90, 90, 90), (150, 150, 150), (210, 210, 210), (255, 255, 255)],
    "insane": [(255, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 80, 0), (255, 0, 0)],
}


def brightness_to_glyph_idx(gray_u8: np.ndarray) -> np.ndarray:
    norm = gray_u8.astype(np.float32) / 255.0
    idx = (norm * (len(GLYPHS) - 1)).astype(np.int32)
    return idx


def parse_colors_arg(s: str):
    if not s:
        return None
    out = []
    for part in s.split(","):
        p = part.strip().lstrip("#")
        if len(p) != 6:
            continue
        r = int(p[0:2], 16)
        g = int(p[2:4], 16)
        b = int(p[4:6], 16)
        out.append((b, g, r))  # BGR
    return out if out else None


def lerp_color(c1, c2, t: float):
    t = float(np.clip(t, 0.0, 1.0))
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def shift_hue_bgr(color, degrees: float):
    px = np.uint8([[color]])
    hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[0, 0, 0] = (hsv[0, 0, 0] + degrees / 2.0) % 180.0
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
    return int(out[0]), int(out[1]), int(out[2])


def scale_sv_bgr(color, sat_mul=1.0, val_mul=1.0):
    px = np.uint8([[color]])
    hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[0, 0, 1] = np.clip(hsv[0, 0, 1] * sat_mul, 0, 255)
    hsv[0, 0, 2] = np.clip(hsv[0, 0, 2] * val_mul, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
    return int(out[0]), int(out[1]), int(out[2])


def extract_input_palette(frame, k=5):
    small = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
    pixels = small.reshape(-1, 3).astype(np.float32)
    if pixels.shape[0] < k:
        k = max(1, pixels.shape[0])

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    centers = centers.astype(np.uint8)
    counts = np.bincount(labels.flatten(), minlength=k)
    order = np.argsort(-counts)
    return [tuple(int(x) for x in centers[i]) for i in order]


def resolve_palette(name: str, custom_colors):
    if custom_colors:
        return custom_colors
    return PALETTES.get(name, PALETTES["neon"])


def blend_palettes(p1, p2, t):
    n = max(len(p1), len(p2))
    out = []
    for i in range(n):
        c1 = p1[i % len(p1)]
        c2 = p2[i % len(p2)]
        out.append(lerp_color(c1, c2, t))
    return out


def palette_pick(palette, x01: float):
    if not palette:
        return (255, 255, 255)
    i = int(np.clip(x01, 0.0, 1.0) * (len(palette) - 1))
    return palette[i]


# ============================================================
# Output naming (same folder as input)
# ============================================================

def build_output_name(args) -> str:
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


# ============================================================
# Audio energy analysis (aligned per-frame)
# ============================================================

def analyze_audio_energy(args, fps: float, frames_to_process: int) -> np.ndarray:
    """
    Extracts trimmed audio to WAV (matching --start/--duration) and returns
    per-frame normalized RMS energy aligned to video frames.

    Alignment is done using exact sample boundaries per frame:
      s0 = round(i * sr / fps)
      s1 = round((i+1) * sr / fps)
    """
    log("analyzing audio energy (aligned)")

    tmp_wav = str(Path(tempfile.gettempdir()) / f"voidstar_energy_{int(time.time())}.wav")

    # IMPORTANT: use accurate trim ordering: -i first, then -ss/-t.
    # This avoids "fast seek" offsets and aligns segment exactly.
    cmd = [
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "error",
        "-i", args.input,
        "-ss", str(args.start),
    ]
    if args.duration is not None:
        cmd += ["-t", str(args.duration)]

    # analysis wav: mono, reasonably high SR to reduce envelope error
    cmd += ["-ac", "1", "-ar", str(args.audio_sr), tmp_wav]
    run_ffmpeg(cmd)

    with wave.open(tmp_wav, "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)

    Path(tmp_wav).unlink(missing_ok=True)

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    if samples.size == 0:
        log("audio analysis: no samples found (silence?)")
        return np.zeros(frames_to_process, dtype=np.float32)

    # per-frame RMS with exact boundaries to avoid drift
    energy = np.zeros(frames_to_process, dtype=np.float32)
    for i in range(frames_to_process):
        s0 = int(round(i * sr / fps))
        s1 = int(round((i + 1) * sr / fps))
        if s0 >= samples.size:
            break
        window = samples[s0:min(s1, samples.size)]
        if window.size:
            energy[i] = float(np.sqrt(np.mean(window * window)))

    # normalize robustly (avoid one transient dominating)
    # Use percentile to make thresholding feel better across tracks.
    peak = np.percentile(energy, 99.5) if np.any(energy > 0) else 0.0
    if peak > 0:
        energy = np.clip(energy / peak, 0.0, 1.0)

    log("audio analysis complete")
    return energy


# ============================================================
# CPU renderer
# ============================================================

def run_cpu(args):
    log("mode=CPU")

    rng = np.random.default_rng(args.seed)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input: {args.input}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames_input = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if width <= 0 or height <= 0:
        raise RuntimeError("Could not read video dimensions.")

    # --------------------------------------------------------
    # frame bounds
    # --------------------------------------------------------
    start_frame = int(round(args.start * fps))
    if args.duration is None:
        end_frame = total_frames_input if total_frames_input > 0 else 10**12
    else:
        end_frame = start_frame + int(round(args.duration * fps))

    frames_to_process = max(0, end_frame - start_frame)

    log(f"input={args.input}")
    log(f"fps={fps:.3f} size={width}x{height}")
    log(f"start={args.start:.3f}s -> start_frame={start_frame}")
    log(f"frames_to_process={frames_to_process}")

    if frames_to_process <= 0:
        raise RuntimeError("Nothing to process (check --start/--duration).")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    out_path = build_output_name(args)
    log(f"output={out_path}")

    # --------------------------------------------------------
    # audio analysis (optional)
    # --------------------------------------------------------
    energy_curve = None
    if args.audio_reactive:
        energy_curve = analyze_audio_energy(args, fps, frames_to_process)

    # --------------------------------------------------------
    # temp video (video only)
    # --------------------------------------------------------
    tmp_video = str(Path(tempfile.gettempdir()) / f"voidstar_glyph_tmp_{int(time.time())}.mp4")
    log(f"temp_video={tmp_video}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, fps, (width, height))

    prev_gray = None
    t0 = time.time()

    font = cv2.FONT_HERSHEY_SIMPLEX
    base_cell = args.cell

    # hold state so glyph mode persists briefly after a trigger
    hold_left = 0
    flash_left = 0
    custom_colors = parse_colors_arg(args.colors)
    cached_input_palette = None

    with tqdm(total=frames_to_process) as pbar:
        frame_idx = 0
        while frame_idx < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break

            # --------------------------------------------
            # Decide whether glyph mode is ON for this frame
            # --------------------------------------------
            glyph_on = True  # default if not audio-reactive

            e = 0.0
            triggered = False
            if energy_curve is not None:
                e = float(energy_curve[frame_idx])

                # Trigger only when above threshold (probability-gated)
                triggered = (e >= args.beat_threshold) and (rng.random() <= args.beat_prob)

                if triggered:
                    hold_left = max(hold_left, args.glitch_hold)
                    if args.beat_flash:
                        flash_left = max(flash_left, args.beat_flash_frames)

                if hold_left > 0:
                    glyph_on = True
                    hold_left -= 1
                else:
                    glyph_on = False  # pass-through original frame

            if not glyph_on:
                # Pass-through original frame (no glyph overlay)
                writer.write(frame)
                frame_idx += 1

                if flash_left > 0:
                    flash_left -= 1

                if frame_idx % 10 == 0:
                    eta = format_eta(t0, frame_idx, frames_to_process)
                    if energy_curve is not None and frame_idx < len(energy_curve):
                        log(f"frames={frame_idx}/{frames_to_process} e={e:.3f} mode=orig eta={eta}")
                    else:
                        log(f"frames={frame_idx}/{frames_to_process} mode=orig eta={eta}")

                pbar.update(1)
                continue

            # --------------------------------------------
            # Glyph mode render
            # --------------------------------------------
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # motion
            if prev_gray is not None:
                motion = cv2.absdiff(gray, prev_gray)
                motion = cv2.GaussianBlur(motion, (5, 5), 0)
            else:
                motion = np.zeros_like(gray)
            prev_gray = gray

            edges = cv2.Canny(gray, 60, 120)

            # Audio-reactive parameter modulation (only while glyph_on)
            if energy_curve is not None:
                # "reactivity" scales aggressiveness with intensity
                # keep it stable, avoid insane spikes
                scale = 1.0 + args.reactivity * e

                motion_w = args.motion_weight * scale
                edge_w = args.edge_weight * (0.75 + 0.75 * e)
                luma_w = args.luma_weight * (0.90 + 0.35 * e)

                density = min(1.0, args.density * (0.75 + 0.65 * e))

                # optionally tighten cell size when intense
                # (smaller cell = higher detail)
                cell_local = max(args.min_cell, int(round(base_cell * (1.0 - args.cell_shrink * e))))
            else:
                motion_w = args.motion_weight
                edge_w = args.edge_weight
                luma_w = args.luma_weight
                density = args.density
                cell_local = base_cell

            # energy field
            energy_img = (
                gray.astype(np.float32) * luma_w +
                motion.astype(np.float32) * motion_w +
                edges.astype(np.float32) * edge_w
            )
            energy_img = np.clip(energy_img, 0, 255).astype(np.uint8)
            glyph_idx = brightness_to_glyph_idx(energy_img)

            # palette per frame
            t_sec = frame_idx / max(1e-6, fps)
            mode = "bw" if args.bw else args.color_mode

            if mode == "input" or (mode == "auto" and e < args.wild_threshold):
                if cached_input_palette is None or (frame_idx % max(1, args.palette_refresh) == 0):
                    cached_input_palette = extract_input_palette(frame, args.input_palette_k)
                pal_a = cached_input_palette
            elif mode == "audio":
                cool = resolve_palette("ice", custom_colors)
                hot = resolve_palette(args.palette, custom_colors)
                pal_a = blend_palettes(cool, hot, e)
            else:
                pal_a = resolve_palette(args.palette, custom_colors)

            pal_b = resolve_palette(args.palette_b, custom_colors)
            if args.palette_mix_speed > 0:
                mix_t = 0.5 * (1.0 + math.sin(2.0 * math.pi * args.palette_mix_speed * t_sec))
                palette = blend_palettes(pal_a, pal_b, mix_t)
            else:
                palette = pal_a

            if args.hue_drift != 0:
                hue_deg = args.hue_drift * t_sec
                palette = [shift_hue_bgr(c, hue_deg) for c in palette]

            out = np.zeros_like(frame)

            wild = (mode == "glitch") or (mode == "auto" and e >= args.wild_threshold)

            for y in range(0, height, cell_local):
                for x in range(0, width, cell_local):
                    if rng.random() > density:
                        continue

                    idx = glyph_idx[y, x]
                    ch = GLYPHS[idx]

                    brightness = float(energy_img[y, x]) / 255.0

                    if mode == "bw":
                        v = int(255 * brightness)
                        base = (v, v, v)
                    else:
                        base = palette_pick(palette, brightness)

                        if args.depth_color > 0 and len(palette) > 1:
                            dt = np.clip((y / max(1, height - 1)) * args.depth_color, 0.0, 1.0)
                            base = lerp_color(base, palette[-1], dt)

                        if args.motion_color > 0:
                            m = float(motion[y, x]) / 255.0
                            base = shift_hue_bgr(base, 180.0 * m * args.motion_color)

                        if wild and (rng.random() < args.glitch_prob * (0.35 + 0.65 * e)):
                            insane = resolve_palette("insane", None)
                            base = insane[int(rng.integers(0, len(insane)))]

                        if args.audio_color_reactive and energy_curve is not None:
                            base = scale_sv_bgr(base, sat_mul=1.0 + 0.9 * e, val_mul=0.9 + 0.35 * e)

                    color = (
                        int(base[0] * brightness),
                        int(base[1] * brightness),
                        int(base[2] * brightness),
                    )

                    if args.beat_flash and flash_left > 0:
                        ft = flash_left / max(1, args.beat_flash_frames)
                        color = lerp_color(color, (255, 255, 255), args.beat_flash_strength * ft)

                    cv2.putText(
                        out, ch, (x, y),
                        font, args.font_scale,
                        color, 1, cv2.LINE_AA
                    )

            writer.write(out)

            frame_idx += 1
            if flash_left > 0:
                flash_left -= 1
            if frame_idx % 10 == 0:
                eta = format_eta(t0, frame_idx, frames_to_process)
                if energy_curve is not None and frame_idx < len(energy_curve):
                    log(f"frames={frame_idx}/{frames_to_process} e={e:.3f} mode=glyph hold={hold_left} eta={eta}")
                else:
                    log(f"frames={frame_idx}/{frames_to_process} mode=glyph eta={eta}")

            pbar.update(1)

    cap.release()
    writer.release()

    # ============================================================
    # AUDIO MUX (lossless)
    # ============================================================
    log("video render complete — starting audio mux")

    tmp_audio = str(Path(tempfile.gettempdir()) / f"voidstar_audio_tmp_{int(time.time())}.aac")

    # IMPORTANT: use same accurate trim ordering for stream copy
    cmd_extract = [
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "error",
        "-i", args.input,
        "-ss", str(args.start),
    ]
    if args.duration is not None:
        cmd_extract += ["-t", str(args.duration)]

    cmd_extract += ["-vn", "-acodec", "copy", tmp_audio]
    run_ffmpeg(cmd_extract)

    cmd_mux = [
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "error",
        "-i", tmp_video,
        "-i", tmp_audio,
        "-c:v", "copy",
        "-c:a", "copy",
        "-shortest",
        out_path,
    ]
    run_ffmpeg(cmd_mux)

    try:
        Path(tmp_video).unlink(missing_ok=True)
        Path(tmp_audio).unlink(missing_ok=True)
    except Exception:
        pass

    log(f"✅ done → {out_path}")


# ============================================================
# GPU hook (future)
# ============================================================

def run_gpu(args):
    try:
        import moderngl  # noqa: F401
    except Exception:
        log("⚠️ moderngl not available — falling back to CPU")
        run_cpu(args)
        return

    log("mode=GPU (delegating to CPU for now)")
    run_cpu(args)


# ============================================================
# CLI
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("input")
    ap.add_argument("--mode", default="auto", choices=["auto", "gpu", "cpu"])

    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=None)

    ap.add_argument("--cell", type=int, default=12)
    ap.add_argument("--density", type=float, default=0.65)
    ap.add_argument("--font-scale", type=float, default=0.4)

    ap.add_argument("--motion-weight", type=float, default=1.6)
    ap.add_argument("--edge-weight", type=float, default=1.2)
    ap.add_argument("--luma-weight", type=float, default=0.8)

    # Color system
    ap.add_argument("--color-mode", default="fixed",
                    choices=["fixed", "input", "audio", "glitch", "bw", "auto"],
                    help="fixed=chosen palette, input=match source palette, audio=energy-driven blend, glitch=wild random, bw=monochrome, auto=input+wild on intensity")
    ap.add_argument("--palette", default="neon",
                    choices=list(PALETTES.keys()),
                    help="primary named palette")
    ap.add_argument("--palette-b", default="sunset",
                    choices=list(PALETTES.keys()),
                    help="secondary palette for crossfade")
    ap.add_argument("--colors", default=None,
                    help="custom comma-separated hex colors, e.g. ff0066,00ffee,ffffff")
    ap.add_argument("--bw", action="store_true",
                    help="force black/white glyph colors")
    ap.add_argument("--input-palette-k", type=int, default=5,
                    help="number of colors when extracting source palette")
    ap.add_argument("--palette-refresh", type=int, default=10,
                    help="recompute source palette every N frames")

    ap.add_argument("--audio-color-reactive", action="store_true",
                    help="boost saturation/value by audio energy")
    ap.add_argument("--glitch-prob", type=float, default=0.25,
                    help="chance of per-glyph wild color in glitch/auto mode")
    ap.add_argument("--wild-threshold", type=float, default=0.78,
                    help="energy threshold for wild mode when --color-mode auto")
    ap.add_argument("--beat-flash", action="store_true",
                    help="brief white-ish flash bursts on triggers")
    ap.add_argument("--beat-flash-frames", type=int, default=2,
                    help="length of beat flash")
    ap.add_argument("--beat-flash-strength", type=float, default=0.65,
                    help="flash intensity 0..1")

    ap.add_argument("--palette-mix-speed", type=float, default=0.0,
                    help="Hz crossfade speed between palette and palette-b (0=off)")
    ap.add_argument("--hue-drift", type=float, default=0.0,
                    help="hue rotation in degrees/sec")
    ap.add_argument("--depth-color", type=float, default=0.0,
                    help="0..1 vertical depth tinting")
    ap.add_argument("--motion-color", type=float, default=0.0,
                    help="0..1 motion-driven hue shift")

    ap.add_argument("--seed", type=int, default=None,
                    help="random seed for deterministic beat/glitch randomness")

    # Audio reactive (gated)
    ap.add_argument("--audio-reactive", action="store_true",
                    help="gate glyph effect by audio energy; original frames otherwise")
    ap.add_argument("--beat-threshold", type=float, default=0.55,
                    help="energy threshold (0..1) to trigger glyph mode")
    ap.add_argument("--beat-prob", type=float, default=0.85,
                    help="probability to trigger when above threshold")
    ap.add_argument("--glitch-hold", type=int, default=6,
                    help="hold glyph mode for N frames after trigger (prevents flicker)")
    ap.add_argument("--reactivity", type=float, default=1.0,
                    help="how strongly params scale with energy (0..2ish)")

    # Audio analysis settings
    ap.add_argument("--audio-sr", type=int, default=48000,
                    help="sample rate for analysis WAV")

    # Cell modulation
    ap.add_argument("--cell-shrink", type=float, default=0.25,
                    help="max fraction to shrink cell size at high energy (0..0.6)")
    ap.add_argument("--min-cell", type=int, default=4,
                    help="minimum cell size when shrinking")

    return ap.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    log("starting glyph field")

    if args.mode == "cpu":
        run_cpu(args)
    elif args.mode == "gpu":
        run_gpu(args)
    else:
        try:
            import moderngl  # noqa: F401
            run_gpu(args)
        except Exception:
            run_cpu(args)


if __name__ == "__main__":
    main()
