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
# Glyphs
# ============================================================

GLYPHS = np.array(list(" .·:+*#%@"))


def brightness_to_glyph_idx(gray_u8: np.ndarray) -> np.ndarray:
    norm = gray_u8.astype(np.float32) / 255.0
    idx = (norm * (len(GLYPHS) - 1)).astype(np.int32)
    return idx


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
            if energy_curve is not None:
                e = float(energy_curve[frame_idx])

                # Trigger only when above threshold (probability-gated)
                triggered = (e >= args.beat_threshold) and (np.random.rand() <= args.beat_prob)

                if triggered:
                    hold_left = max(hold_left, args.glitch_hold)

                if hold_left > 0:
                    glyph_on = True
                    hold_left -= 1
                else:
                    glyph_on = False  # pass-through original frame

            if not glyph_on:
                # Pass-through original frame (no glyph overlay)
                writer.write(frame)
                frame_idx += 1

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

            out = np.zeros_like(frame)

            for y in range(0, height, cell_local):
                for x in range(0, width, cell_local):
                    if np.random.rand() > density:
                        continue

                    idx = glyph_idx[y, x]
                    ch = GLYPHS[idx]

                    brightness = float(energy_img[y, x]) / 255.0
                    color = (
                        int(255 * brightness * 0.4),
                        int(255 * brightness * 0.7),
                        int(255 * brightness * 1.0),
                    )

                    cv2.putText(
                        out, ch, (x, y),
                        font, args.font_scale,
                        color, 1, cv2.LINE_AA
                    )

            writer.write(out)

            frame_idx += 1
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
