import argparse
import subprocess
import shutil
import tempfile
import wave
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
import time


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
DEFAULT_TRAIL_LINES_ONLY = True

DEFAULT_SCANLINES = True
DEFAULT_SCANLINE_STRENGTH = 0.06

DEFAULT_CRF = 18
DEFAULT_PRESET = "veryfast"
DEFAULT_AUDIO_BITRATE = "320k"

DEFAULT_START = 0.0
DEFAULT_DURATION = None

# Code overlay defaults
DEFAULT_CODE_ALPHA = 0.85
DEFAULT_CODE_FONT_SCALE = 0.40
DEFAULT_CODE_LINE_SPACING = 1.45
DEFAULT_CODE_TOP_MARGIN = 30
DEFAULT_CODE_SCROLL_CAP_SCREENS = 2.0  # cap to ~2 screen-heights by default

DEFAULT_AUDIO_OFFSET = 0.0  # seconds (can be negative)

DEFAULT_USE_TEMP_FILES = True

# Reactive defaults (all OFF unless enabled)
DEFAULT_BEAT_SYNC = False
DEFAULT_BEAT_GAIN = 2.0
DEFAULT_BEAT_SMOOTH = 0.88
DEFAULT_BEAT_PULSE_THRESHOLD = 0.65
DEFAULT_BEAT_PULSE_BOOST = 0.9

DEFAULT_SMEAR = False
DEFAULT_SMEAR_FRAMES = 10
DEFAULT_SMEAR_DECAY = 0.86
DEFAULT_SMEAR_MODE = "lerp"  # lerp|add|screen|max
DEFAULT_SMEAR_ON_POSE_ONLY = True

DEFAULT_GLITCH = False
DEFAULT_GLITCH_STRENGTH = 0.22
DEFAULT_GLITCH_RATE = 0.9
DEFAULT_GLITCH_RGB_SPLIT = 8
DEFAULT_GLITCH_SCAN_JITTER = 14
DEFAULT_GLITCH_DROPOUT = 0.02  # occasional band dropout probability


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

def parse_bgr_color(s: str) -> tuple[int, int, int]:
    try:
        parts = [int(x.strip()) for x in s.split(",")]
        if len(parts) != 3:
            raise ValueError
        b, g, r = parts
        return (
            max(0, min(255, b)),
            max(0, min(255, g)),
            max(0, min(255, r))
        )
    except Exception:
        raise argparse.ArgumentTypeError(
            "Color must be B,G,R format like 255,255,255"
        )


def run(cmd: list[str]) -> None:
    print("▶", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


def require_ffmpeg(tool: str) -> str:
    p = shutil.which(tool)
    if not p:
        raise RuntimeError(f"{tool} not found. Install ffmpeg package.")
    return p


# =========================
# GPU UTILITIES
# =========================
def detect_nvidia_gpu() -> bool:
    """
    Check if NVIDIA GPU is available for CUDA acceleration.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=2
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_nvenc_codec() -> str | None:
    """
    Detect best available NVIDIA encoder (hevc_nvenc > h264_nvenc).
    Returns encoder name or None if not available.
    """
    ffmpeg = require_ffmpeg("ffmpeg")
    try:
        result = subprocess.run(
            [ffmpeg, "-codecs"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5
        )
        output = result.stdout.lower()
        
        # Prefer hevc for better compression; fallback to h264
        if "hevc_nvenc" in output:
            return "hevc_nvenc"
        elif "h264_nvenc" in output:
            return "h264_nvenc"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    return None


def detect_input_codec(video_path: Path) -> tuple[str, int]:
    """
    Detect the video codec of input file.
    Returns (codec_name, bitrate_kbps) or defaults to ('h264', 5000) if detection fails.
    """
    ffprobe = require_ffmpeg("ffprobe")
    try:
        cmd = [
            ffprobe, "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,bit_rate",
            "-of", "default=noprint_wrappers=1:nokey=1:nk=1",
            str(video_path)
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
        lines = result.stdout.strip().split('\n')
        
        if len(lines) >= 2:
            codec = lines[0].strip() or "h264"
            try:
                bitrate = int(lines[1].strip()) // 1000 if lines[1].strip() else 5000
            except (ValueError, IndexError):
                bitrate = 5000
            return (codec, bitrate)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
        pass
    
    return ("h264", 5000)


def ffprobe_duration_seconds(video_path: Path) -> float:
    ffprobe = require_ffmpeg("ffprobe")
    cmd = [
        ffprobe, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1",
        str(video_path)
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    s = r.stdout.strip()
    if not s:
        raise RuntimeError("ffprobe could not read duration.")
    return float(s)


def fmt_float(x: float) -> str:
    return f"{x:.2f}".replace(".", "p")


def build_output_name(src: Path, args) -> Path:
    parts = []

    # timing
    if args.fps:
        parts.append(f"fps{int(args.fps)}")
    if args.start and args.start > 0:
        parts.append(f"s{fmt_float(args.start)}")
    if args.duration is not None:
        parts.append(f"d{fmt_float(args.duration)}")

    # pose params
    parts.append(f"mc{args.model_complexity}")
    parts.append(f"det{fmt_float(args.min_det_conf)}")
    parts.append(f"trk{fmt_float(args.min_trk_conf)}")

    # visuals
    parts.append(f"trail{int(args.trail)}")
    if args.trail:
        parts.append(f"ta{fmt_float(args.trail_alpha)}")
        parts.append(f"tlo{str(args.trail_lines_only).lower()}")

    parts.append(f"scan{int(args.scanlines)}")

    # ids/color
    if args.velocity_color:
        parts.append("velcolor")
    if args.draw_ids:
        parts.append("ids")

    # code overlay
    if args.code_overlay:
        parts.append(f"code{args.code_overlay_order}")
        parts.append(f"ca{fmt_float(args.code_alpha)}")

    # reactive toggles
    if args.beat_sync:
        parts.append("beatsync")
    if args.smear:
        parts.append("smear")
    if args.glitch:
        parts.append("glitch")

    suffix = "_" + "_".join(parts) if parts else ""
    return src.with_name(f"{src.stem}{suffix}.mp4")


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
    # Optimized: work in-place when possible
    if strength <= 0:
        return frame
    out = frame.astype(np.float32)
    out[::2] *= (1.0 - strength)  # Simplified indexing
    np.clip(out, 0, 255, out=out)  # In-place clip
    return out.astype(np.uint8)


# =========================
# VELOCITY COLORING
# =========================
def velocity_to_color(v, v_min=0.0, v_max=800.0):
    t = (v - v_min) / max(1e-6, (v_max - v_min))
    t = max(0.0, min(1.0, t))

    # BGR ramp: blue->green->red
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

    # Outline-only box (no fill)
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
# CODE OVERLAY
# =========================
def resolve_code_overlay_path(code_overlay_arg) -> Path | None:
    """
    --code-overlay can be:
      - not provided / false -> None
      - provided as flag (const=True) -> True
      - provided with 'true'/'false' string
      - provided with a filepath
    """
    if not code_overlay_arg:
        return None

    if code_overlay_arg is True:
        return Path(__file__).resolve()

    if isinstance(code_overlay_arg, str):
        s = code_overlay_arg.strip().lower()
        if s in ("1", "true", "t", "yes", "y", "on"):
            return Path(__file__).resolve()
        if s in ("0", "false", "f", "no", "n", "off"):
            return None
        return Path(code_overlay_arg)

    return None


def render_code_overlay(
    lines,
    w,
    h,
    font_scale=0.40,
    line_spacing=1.45,
    top_margin=30
):
    base_line_h = int(22 * font_scale)
    line_advance = max(1, int(base_line_h * line_spacing))

    img_h = max(h * 2, len(lines) * line_advance + h + top_margin)
    img = np.zeros((img_h, w, 3), np.uint8)

    # Start at top immediately (no dead space)
    y = int(top_margin)
    for line in lines:
        cv2.putText(
            img,
            line.rstrip("\n"),
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
    yoff = int(yoff)
    yoff = max(0, min(yoff, max(0, code_img.shape[0] - h)))
    src = code_img[yoff:yoff + h]
    return cv2.addWeighted(frame, 1.0, src, float(alpha), 0)


# =========================
# FFMPEG CFR + AUDIO MUX
# =========================
def make_cfr_intermediate(
    src: Path,
    dst: Path,
    fps: float,
    start: float,
    duration: float | None,
    use_gpu: bool = False,
):
    """
    Converts (likely VFR) phone video to true CFR before OpenCV processing.
    Put -ss AFTER -i for accurate trimming.
    """
    print(f"📹 Creating CFR intermediate from {src.name}...")
    print(f"   FPS: {fps}, Start: {start}s, Duration: {duration if duration else 'full'}")
    ffmpeg = require_ffmpeg("ffmpeg")

    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
    cmd += ["-i", str(src)]

    if start and start > 0:
        cmd += ["-ss", str(start)]
    if duration is not None:
        cmd += ["-t", str(duration)]

    cmd += [
        "-vf", f"fps={fps}:round=near",
        "-fps_mode", "cfr",
        "-an",  # video-only intermediate
    ]
    
    # Use hardware encoder if available
    nvenc_codec = None
    if use_gpu:
        nvenc_codec = get_nvenc_codec()
        if nvenc_codec:
            print(f"   Using GPU acceleration: {nvenc_codec}")
            cmd += ["-c:v", nvenc_codec]
            if nvenc_codec.startswith("hevc"):
                cmd += ["-preset", "fast"]  # fast, default, slow for HEVC
            else:
                cmd += ["-preset", "fast"]  # fast, default, slow for H.264
        else:
            print(f"   GPU not available, using CPU encoder")
            cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-profile:v", "baseline", "-level", "4.1", "-crf", "18", "-preset", "veryfast"]
    else:
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-profile:v", "baseline", "-level", "4.1", "-crf", "18", "-preset", "veryfast"]
    
    cmd += [str(dst)]
    run(cmd)
    print("✓ CFR intermediate created")


def mux_audio_player_safe(
    video_noaudio: Path,
    src_video: Path,
    audio_wav: Path | None,
    out_mp4: Path,
    audio_start: float,
    audio_duration: float | None,
    crf: int,
    preset: str,
    audio_bitrate: str,
    use_gpu: bool = False,
    input_codec: str = "h264",
    input_bitrate: int = 5000,
):
    """
    Mux audio back in (external WAV preferred, else source video audio).
    Sample-accurate trim using atrim + asetpts.
    Encodes output using same codec/bitrate as input for similar file size.
    """
    audio_src = "external WAV" if audio_wav else "source video"
    print(f"🎵 Muxing audio from {audio_src}...")
    print(f"   Audio offset: {audio_start}s")
    print(f"   Output codec: {input_codec} @ {input_bitrate}kbps (matched to input)")
    ffmpeg = require_ffmpeg("ffmpeg")

    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]

    # processed video
    cmd += ["-i", str(video_noaudio)]

    # audio source
    if audio_wav:
        cmd += ["-i", str(audio_wav)]
        audio_input_idx = 1
    else:
        cmd += ["-i", str(src_video)]
        audio_input_idx = 1

    af = []
    if audio_start > 0:
        af.append(f"atrim=start={audio_start}")
    if audio_duration is not None:
        af.append(f"atrim=duration={audio_duration}")
    af.append("asetpts=PTS-STARTPTS")

    cmd += ["-filter_complex", f"[{audio_input_idx}:a]{','.join(af)}[a]"]
    cmd += ["-map", "0:v:0", "-map", "[a]"]

    cmd += ["-movflags", "+faststart"]
    
    # Select encoder based on input codec
    encoder_used = "CPU"
    
    # Try GPU first if available
    if use_gpu:
        nvenc_codec = get_nvenc_codec()
        if nvenc_codec and ("hevc" in input_codec.lower() or "h264" in input_codec.lower()):
            print(f"   Using GPU encoder: {nvenc_codec}")
            cmd += ["-c:v", nvenc_codec]
            if "hevc" in nvenc_codec.lower():
                cmd += ["-preset", "fast", "-rc", "vbr", "-cq", str(max(18, min(28, 23)))]  # Match quality to input
            else:
                cmd += ["-preset", "fast", "-rc", "vbr", "-cq", str(max(18, min(28, 23)))]
            cmd += ["-b:v", f"{input_bitrate}k"]
            encoder_used = "GPU (NVENC)"
        else:
            # Fall back to CPU - match input codec
            if "hevc" in input_codec.lower():
                cmd += ["-c:v", "libx265", "-preset", "fast", "-crf", str(max(18, min(28, 23)))]
            else:
                cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(max(18, min(28, 23)))]
            cmd += ["-pix_fmt", "yuv420p", "-b:v", f"{input_bitrate}k"]
    else:
        # CPU mode - match input codec
        if "hevc" in input_codec.lower():
            cmd += ["-c:v", "libx265", "-preset", "fast"]
        else:
            cmd += ["-c:v", "libx264", "-preset", preset]
        cmd += ["-pix_fmt", "yuv420p", "-b:v", f"{input_bitrate}k"]
    
    cmd += [
        "-c:a", "aac",
        "-b:a", str(audio_bitrate),
        "-ar", "48000",
        "-shortest",
        str(out_mp4),
    ]
    run(cmd)
    print(f"✓ Audio muxed successfully (using {encoder_used})")


# =========================
# AUDIO ANALYSIS (beat-sync)
# =========================
def extract_audio_wav_segment(
    src_media: Path,
    dst_wav: Path,
    start: float,
    duration: float | None,
    sr: int = 48000,
):
    """
    Extract a mono 48k PCM wav segment using ffmpeg.
    Works for input video OR wav files (re-encodes to consistent format).
    """
    print(f"🎧 Extracting audio segment for beat-sync analysis...")
    ffmpeg = require_ffmpeg("ffmpeg")
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]

    # Accurate trim: -ss after -i
    cmd += ["-i", str(src_media)]
    if start and start > 0:
        cmd += ["-ss", str(start)]
    if duration is not None:
        cmd += ["-t", str(duration)]

    cmd += [
        "-vn",
        "-ac", "1",
        "-ar", str(sr),
        "-f", "wav",
        str(dst_wav)
    ]
    run(cmd)


def read_wav_mono_int16(wav_path: Path) -> tuple[np.ndarray, int]:
    """
    Read WAV (PCM) into float32 [-1,1], returns (samples, sample_rate).
    """
    with wave.open(str(wav_path), "rb") as wf:
        nch = wf.getnchannels()
        sr = wf.getframerate()
        sampwidth = wf.getsampwidth()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    if sampwidth != 2:
        # enforce 16-bit output from ffmpeg; if something slips through, fail loud
        raise RuntimeError(f"WAV sample width {sampwidth} bytes not supported; expected 16-bit PCM.")

    data = np.frombuffer(raw, dtype=np.int16)

    if nch > 1:
        data = data.reshape(-1, nch)
        data = data[:, 0]

    x = data.astype(np.float32) / 32768.0
    return x, sr


def build_envelope_at_fps(
    wav_samples: np.ndarray,
    sr: int,
    fps: float,
    n_frames: int,
    smooth: float,
    gain: float,
) -> np.ndarray:
    """
    Build an RMS envelope aligned to frame count.
    Returns array length n_frames, roughly in [0..1+] after gain.
    """
    if n_frames <= 0:
        return np.zeros((0,), dtype=np.float32)

    # RMS per frame window
    spf = sr / float(fps)  # samples per frame
    env = np.zeros((n_frames,), dtype=np.float32)

    for i in range(n_frames):
        a = int(i * spf)
        b = int((i + 1) * spf)
        if a >= len(wav_samples):
            break
        chunk = wav_samples[a:min(b, len(wav_samples))]
        if chunk.size == 0:
            v = 0.0
        else:
            v = float(np.sqrt(np.mean(chunk * chunk)))
        env[i] = v

    # Normalize (robust) then apply gain
    p95 = float(np.percentile(env, 95)) if env.size else 0.0
    norm = max(1e-6, p95)
    env = env / norm
    env = env * float(gain)

    # Exponential smoothing
    s = np.zeros_like(env)
    a = float(np.clip(smooth, 0.0, 0.9999))
    prev = 0.0
    for i in range(env.size):
        prev = a * prev + (1.0 - a) * float(env[i])
        s[i] = prev

    return s


def pulse_from_env(env_val: float, threshold: float) -> float:
    """
    Convert envelope value to a 0..1 pulse scalar above threshold.
    """
    th = float(threshold)
    if env_val <= th:
        return 0.0
    return float(np.clip((env_val - th) / max(1e-6, (1.0 - th)), 0.0, 1.0))


# =========================
# TEMPORAL SMEAR
# =========================
def blend_temporal(buffer: deque[np.ndarray], current: np.ndarray, decay: float, mode: str) -> np.ndarray:
    """
    Blend current with previous frames in buffer using given mode.
    buffer: previous frames (most recent at right)
    """
    if not buffer:
        return current

    mode = (mode or "lerp").lower()
    decay = float(np.clip(decay, 0.0, 0.9999))

    # Build weights: newer frames contribute more
    # buffer order: oldest..newest
    weights = []
    for i in range(len(buffer)):
        # i=0 oldest, i=len-1 newest
        age = len(buffer) - i
        weights.append(decay ** age)

    w_cur = 1.0
    w_hist = float(np.sum(weights))
    denom = w_cur + w_hist if mode == "lerp" else 1.0

    cur = current.astype(np.float32)

    if mode == "max":
        out = cur.copy()
        for fr in buffer:
            out = np.maximum(out, fr.astype(np.float32))
        return np.clip(out, 0, 255).astype(np.uint8)

    if mode == "add":
        out = cur.copy()
        for fr, w in zip(buffer, weights):
            out += fr.astype(np.float32) * w
        return np.clip(out, 0, 255).astype(np.uint8)

    if mode == "screen":
        # screen: 1 - (1-a)(1-b) in [0..1]
        out = cur / 255.0
        for fr, w in zip(buffer, weights):
            b = (fr.astype(np.float32) / 255.0) * w
            out = 1.0 - (1.0 - out) * (1.0 - np.clip(b, 0.0, 1.0))
        return np.clip(out * 255.0, 0, 255).astype(np.uint8)

    # default: lerp weighted average
    out = cur * w_cur
    for fr, w in zip(buffer, weights):
        out += fr.astype(np.float32) * w
    out /= float(denom)
    return np.clip(out, 0, 255).astype(np.uint8)


# =========================
# GLITCH
# =========================
def apply_glitch(
    frame: np.ndarray,
    t: float,
    strength: float,
    rate: float,
    rgb_split_px: int,
    scan_jitter_px: int,
    dropout_prob: float,
) -> np.ndarray:
    """
    Fast glitch pass:
      - RGB split (animated)
      - random scanline jitter bands
      - occasional dropout band
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return frame

    h, w = frame.shape[:2]
    out = frame.copy()

    # Animated phase
    phase = np.sin(2.0 * np.pi * float(rate) * float(t))
    dx = int(phase * rgb_split_px * strength)

    b, g, r = cv2.split(out)
    # Slightly different offsets for channels
    r2 = np.roll(r, dx, axis=1)
    b2 = np.roll(b, -dx, axis=1)
    g2 = g
    out = cv2.merge([b2, g2, r2])

    # Scanline jitter bands
    n_bands = int(1 + 6 * strength)
    for _ in range(n_bands):
        if np.random.rand() > (0.35 * strength + 0.05):
            continue
        y0 = np.random.randint(0, h)
        band_h = np.random.randint(2, max(3, int(20 * strength)))
        y1 = min(h, y0 + band_h)
        jitter = int(np.random.randint(-scan_jitter_px, scan_jitter_px + 1) * strength)
        out[y0:y1, :, :] = np.roll(out[y0:y1, :, :], jitter, axis=1)

    # Occasional dropout band
    if np.random.rand() < float(dropout_prob) * strength:
        y0 = np.random.randint(0, h)
        band_h = np.random.randint(8, max(10, int(80 * strength)))
        y1 = min(h, y0 + band_h)
        val = 0 if np.random.rand() < 0.5 else 255
        out[y0:y1, :, :] = val

    return out


# =========================
# MAIN
# =========================
def main():
    start_wall_time = time.time()
    
    print("="*60)
    print("🎬 REELS CV OVERLAY - Starting Processing")
    print("="*60)

    ap = argparse.ArgumentParser(
        description="Pose overlay + optional code overlay with CFR pipeline + v1.1 reactive (beat-sync, smear, glitch)."
    )
    
    ap.add_argument("input", help="Input phone video")
    ap.add_argument("-o", "--output", default=None, help="Output mp4 path")

    ap.add_argument("--width", type=int, default=DEFAULT_OUT_W)
    ap.add_argument("--height", type=int, default=DEFAULT_OUT_H)
    ap.add_argument("--fps", type=float, default=DEFAULT_FPS)

    ap.add_argument("--start", type=float, default=DEFAULT_START, help="Start time in seconds (relative to FULL source)")
    ap.add_argument("--duration", type=float, default=DEFAULT_DURATION, help="Duration in seconds")

    ap.add_argument("--audio-wav", default=None, help="Optional external WAV to mux (overrides input audio).")
    ap.add_argument("--audio-offset", type=float, default=DEFAULT_AUDIO_OFFSET,
                    help="Seconds to offset audio relative to video trim (negative starts audio earlier).")

    ap.add_argument("--use-temp-files", type=bool_flag, default=DEFAULT_USE_TEMP_FILES,
                    help="Use WSL-native temp folder for intermediates for better performance (default true).")


    ap.add_argument(
        "--code-overlay",
        nargs="?",
        const=True,
        default=False,
        help="Overlay scrolling code. If true/flag, defaults to this script. If a path is given, uses that file."
    )
    ap.add_argument("--code-overlay-order", choices=["before", "after"], default="after")

    # code overlay tuning
    ap.add_argument("--code-alpha", type=float, default=DEFAULT_CODE_ALPHA)
    ap.add_argument("--code-font-scale", type=float, default=DEFAULT_CODE_FONT_SCALE)
    ap.add_argument("--code-line-spacing", type=float, default=DEFAULT_CODE_LINE_SPACING)
    ap.add_argument("--code-top-margin", type=int, default=DEFAULT_CODE_TOP_MARGIN)
    ap.add_argument("--code-scroll-cap-screens", type=float, default=DEFAULT_CODE_SCROLL_CAP_SCREENS,
                    help="Caps code scroll distance to N screen-heights (helps short clips). 0 disables cap.")

    ap.add_argument("--model-complexity", type=int, choices=[0, 1, 2], default=DEFAULT_MODEL_COMPLEXITY)
    ap.add_argument("--min-det-conf", type=float, default=DEFAULT_MIN_DET_CONF)
    ap.add_argument("--min-trk-conf", type=float, default=DEFAULT_MIN_TRK_CONF)

    ap.add_argument("--trail", type=bool_flag, default=DEFAULT_TRAIL)
    ap.add_argument("--trail-alpha", type=float, default=DEFAULT_TRAIL_ALPHA)
    ap.add_argument("--trail-lines-only", type=bool_flag, default=DEFAULT_TRAIL_LINES_ONLY)

    ap.add_argument("--draw-ids", type=bool_flag, default=False)
    ap.add_argument("--velocity-color", type=bool_flag, default=False)

    ap.add_argument(
        "--overlay-color",
        type=parse_bgr_color,
        default="255,255,255",
        help="Overlay color as B,G,R when velocity-color is false (default 255,255,255)"
    )

    ap.add_argument("--scanlines", type=bool_flag, default=DEFAULT_SCANLINES)
    ap.add_argument("--scanline-strength", type=float, default=DEFAULT_SCANLINE_STRENGTH)

    ap.add_argument("--crf", type=int, default=DEFAULT_CRF)
    ap.add_argument("--preset", type=str, default=DEFAULT_PRESET)
    ap.add_argument("--audio-bitrate", type=str, default=DEFAULT_AUDIO_BITRATE)
    ap.add_argument("--match-input-codec", type=bool_flag, default=True,
                    help="Match output codec/bitrate to input for similar file size (default true).")

    # v1.1 reactive flags
    ap.add_argument("--beat-sync", type=bool_flag, default=DEFAULT_BEAT_SYNC,
                    help="Enable audio-reactive modulation (envelope follower).")
    ap.add_argument("--beat-gain", type=float, default=DEFAULT_BEAT_GAIN)
    ap.add_argument("--beat-smooth", type=float, default=DEFAULT_BEAT_SMOOTH)
    ap.add_argument("--beat-pulse-threshold", type=float, default=DEFAULT_BEAT_PULSE_THRESHOLD)
    ap.add_argument("--beat-pulse-boost", type=float, default=DEFAULT_BEAT_PULSE_BOOST)

    ap.add_argument("--smear", type=bool_flag, default=DEFAULT_SMEAR)
    ap.add_argument("--smear-frames", type=int, default=DEFAULT_SMEAR_FRAMES)
    ap.add_argument("--smear-decay", type=float, default=DEFAULT_SMEAR_DECAY)
    ap.add_argument("--smear-mode", choices=["lerp", "add", "screen", "max"], default=DEFAULT_SMEAR_MODE)
    ap.add_argument("--smear-on-pose-only", type=bool_flag, default=DEFAULT_SMEAR_ON_POSE_ONLY)

    ap.add_argument("--glitch", type=bool_flag, default=DEFAULT_GLITCH)
    ap.add_argument("--glitch-strength", type=float, default=DEFAULT_GLITCH_STRENGTH)
    ap.add_argument("--glitch-rate", type=float, default=DEFAULT_GLITCH_RATE)
    ap.add_argument("--glitch-rgb-split", type=int, default=DEFAULT_GLITCH_RGB_SPLIT)
    ap.add_argument("--glitch-scan-jitter", type=int, default=DEFAULT_GLITCH_SCAN_JITTER)
    ap.add_argument("--glitch-dropout", type=float, default=DEFAULT_GLITCH_DROPOUT)

    ap.add_argument("--gpu", type=bool_flag, default=None,
                    help="Use NVIDIA GPU acceleration if available (auto-detect by default).")

    args = ap.parse_args()

    # Detect GPU capability (auto-detect if not explicitly set)
    has_gpu = detect_nvidia_gpu()
    use_gpu = args.gpu if args.gpu is not None else has_gpu
    
    if use_gpu:
        if has_gpu:
            print(f"\n✅ NVIDIA GPU detected and enabled")
        else:
            print(f"\n⚠️  GPU explicitly requested but not detected, falling back to CPU")
            use_gpu = False
    else:
        print(f"\n⚙️  Using CPU mode")

    # Convert color string to tuple if needed
    if isinstance(args.overlay_color, str):
        args.overlay_color = parse_bgr_color(args.overlay_color)

    src = Path(args.input)
    if not src.exists():
        raise FileNotFoundError(src)

    outp = Path(args.output) if args.output else build_output_name(src, args)
    
    print(f"\n💻 Input: {src.name}")
    print(f"💾 Output: {outp.name}")
    print(f"\n⚙️  Processing Mode: {'GPU (NVIDIA)' if use_gpu else 'CPU'}")
    print(f"\n⚙️  Settings:")
    print(f"   Resolution: {args.width}x{args.height} @ {args.fps} fps")
    print(f"   Start: {args.start}s, Duration: {args.duration if args.duration else 'full'}")
    print(f"   Model complexity: {args.model_complexity}")
    print(f"   Trail: {args.trail}, Scanlines: {args.scanlines}")
    if args.code_overlay:
        print(f"   Code overlay: ON ({args.code_overlay_order})")
    if args.beat_sync:
        print(f"   Beat-sync: ON")
    if args.smear:
        print(f"   Smear: ON ({args.smear_mode}, {args.smear_frames} frames)")
    if args.glitch:
        print(f"   Glitch: ON")

    # Full-source duration for code scroll normalization
    print(f"\n🔍 Analyzing source video...")
    full_duration = ffprobe_duration_seconds(src)
    print(f"   Source duration: {full_duration:.2f}s")
    
    # Detect input codec to match output file size
    input_codec, input_bitrate = detect_input_codec(src)
    print(f"   Input codec: {input_codec} @ {input_bitrate}kbps")

    # --- Determine audio start for mux and for analysis (matches what you will hear) ---
    audio_start = float(args.start) + float(args.audio_offset)
    if audio_start < 0:
        audio_start = 0.0


    # ---- Temp file location logic ----
    if args.use_temp_files:
        temp_root = Path(tempfile.gettempdir()) / "reels_cv_overlay_tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        print(f"\n📁 Using temp directory: {temp_root}")

        tmp_cfr = temp_root / f"{outp.stem}.cfr.tmp.mp4"
        tmp_noaudio = temp_root / f"{outp.stem}.noaudio.tmp.mp4"
    else:
        print(f"\n📁 Using local directory for temp files")
        tmp_cfr = outp.with_suffix(".cfr.tmp.mp4")
        tmp_noaudio = outp.with_suffix(".noaudio.tmp.mp4")

    # 1) CFR intermediate (ffmpeg clock master)
    print(f"\n{'='*60}")
    print("STEP 1: Creating CFR Intermediate")
    print(f"{'='*60}")
    make_cfr_intermediate(
        src=src,
        dst=tmp_cfr,
        fps=args.fps,
        start=args.start,
        duration=args.duration,
        use_gpu=use_gpu,
    )

    print(f"\n{'='*60}")
    print("STEP 2: Initializing Video Processing")
    print(f"{'='*60}")
    print(f"📹 Opening CFR intermediate...")
    cap = cv2.VideoCapture(str(tmp_cfr))

    if not cap.isOpened():
        raise RuntimeError("Could not open CFR intermediate.")

    fps = float(args.fps)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        # fallback
        frame_count = int(round((args.duration or 0.0) * fps)) if args.duration else 0
    
    print(f"✓ Video opened successfully")
    print(f"   Total frames to process: {frame_count}")
    print(f"   Duration: {frame_count/fps:.2f}s")

    # Use rawvideo for faster intermediate writing (no encoding overhead)
    out = cv2.VideoWriter(
        str(tmp_noaudio),
        cv2.VideoWriter_fourcc(*"FFV1"),  # FFV1 is lossless and faster than mp4v
        fps,
        (args.width, args.height)
    )

    # --- Code overlay setup ---
    code_img = None
    code_scroll_range = 0
    code_path = resolve_code_overlay_path(args.code_overlay)
    if code_path is not None:
        if not code_path.exists():
            raise FileNotFoundError(code_path)

        print(f"\n📝 Setting up code overlay from {code_path.name}...")
        lines = code_path.read_text(encoding="utf-8", errors="replace").splitlines()
        print(f"   Lines of code: {len(lines)}")
        code_img = render_code_overlay(
            lines,
            args.width,
            args.height,
            font_scale=args.code_font_scale,
            line_spacing=args.code_line_spacing,
            top_margin=args.code_top_margin,
        )

        max_scroll = max(0, code_img.shape[0] - args.height)
        if args.code_scroll_cap_screens and args.code_scroll_cap_screens > 0:
            cap_px = int(args.height * float(args.code_scroll_cap_screens))
            code_scroll_range = max(1, min(max_scroll, cap_px))
        else:
            code_scroll_range = max(1, max_scroll)
        print(f"   Code overlay image: {code_img.shape[0]}x{code_img.shape[1]}")
        print(f"   Scroll range: {code_scroll_range}px")
        print("✓ Code overlay ready")

    # --- Beat-sync envelope (optional) ---
    env = None
    tmp_env_wav = None
    if args.beat_sync:
        print(f"\n🎵 Setting up beat-sync envelope...")
        # Default to input video audio; override with --audio-wav
        analysis_src = Path(args.audio_wav) if args.audio_wav else src
        if not analysis_src.exists():
            raise FileNotFoundError(analysis_src)

        # Extract trimmed segment to a consistent wav for analysis
        tmp_env_wav = Path(tempfile.mkstemp(prefix="voidstar_env_", suffix=".wav")[1])
        extract_audio_wav_segment(
            src_media=analysis_src,
            dst_wav=tmp_env_wav,
            start=audio_start,              # match mux timing
            duration=args.duration,         # match clip length
            sr=48000,
        )

        print(f"   Building audio envelope...")
        samples, sr = read_wav_mono_int16(tmp_env_wav)
        env = build_envelope_at_fps(
            wav_samples=samples,
            sr=sr,
            fps=fps,
            n_frames=frame_count if frame_count > 0 else max(1, int(round((args.duration or 0) * fps))),
            smooth=float(args.beat_smooth),
            gain=float(args.beat_gain),
        )
        print(f"   Envelope frames: {len(env)}")
        print("✓ Beat-sync envelope ready")

    # --- Temporal buffers (optional) ---
    smear_buf = deque(maxlen=max(0, int(args.smear_frames))) if args.smear else None

    # Handle MediaPipe Pose
    try:
        # Try legacy API (MediaPipe < 0.10)
        mp_pose = mp.solutions.pose
        use_new_mediapipe_api = False
        pose_context = None
    except (AttributeError, ImportError):
        # New API detected but not easily usable in this script
        # The new API requires downloading model files and has a different interface
        print("\n❌ MediaPipe API Compatibility Issue:")
        print("   This script was designed for MediaPipe with the legacy 'solutions' API.")
        print("   Your system has a newer MediaPipe version without that API.")
        print("\n   Options:")
        print("   1. Reinstall older MediaPipe: pip install 'mediapipe<0.10'")
        print("      (May not have pre-built wheels for your system)")
        print("   2. Use Docker/WSL with a compatible version")
        print("\n   Sorry for the inconvenience - the new MediaPipe API requires")
        print("   significant refactoring of the pose detection code.")
        raise RuntimeError("Incompatible MediaPipe version. Legacy solutions API not available.")
    
    trail_buf = None
    prev_landmarks = None
    frame_idx = 0
    start_time_full = float(args.start)
    
    # Pre-allocate reusable arrays for performance
    pose_layer = None
    if args.smear and args.smear_on_pose_only:
        pose_layer = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    
    print(f"\n{'='*60}")
    print("STEP 3: Processing Frames with Pose Detection")
    print(f"{'='*60}")
    print(f"🤖 Initializing MediaPipe Pose...")
    
    # Use legacy context manager
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=args.model_complexity,
        enable_segmentation=False,
        min_detection_confidence=args.min_det_conf,
        min_tracking_confidence=args.min_trk_conf
    ) as pose:
        print("✓ Pose estimator initialized")
        print(f"\n🎞️  Processing frames...")
        
        last_progress = -1
        
        # Pre-compute constants for performance
        beat_pulse_threshold = float(args.beat_pulse_threshold) if args.beat_sync else 0.0
        full_duration_inv = 1.0 / full_duration if full_duration > 0 else 0.0
        fps_inv = 1.0 / fps

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Progress indicator every 10%
            if frame_count > 0:
                progress = int((frame_idx / frame_count) * 10)
                if progress > last_progress:
                    pct = progress * 10
                    elapsed = time.time() - start_wall_time
                    fps_actual = frame_idx / elapsed if elapsed > 0 else 0
                    print(f"   {pct}% complete ({frame_idx}/{frame_count} frames, {fps_actual:.1f} fps)")
                    last_progress = progress

            frame = fit_to_reels(frame, args.width, args.height)

            # Beat modulation scalar for this frame
            mod = float(env[frame_idx]) if (env is not None and frame_idx < len(env)) else 0.0
            pulse = pulse_from_env(mod, beat_pulse_threshold) if args.beat_sync else 0.0

            # code overlay BEFORE pose drawing
            if code_img is not None and args.code_overlay_order == "before":
                # Optimized: use pre-computed constants
                t_full = start_time_full + (frame_idx * fps_inv)
                progress = max(0.0, min(1.0, t_full * full_duration_inv))
                yoff = progress * code_scroll_range
                # subtle code "breathing" if beat-sync is on
                code_alpha_eff = float(args.code_alpha) * (1.0 + 0.15 * pulse)
                frame = overlay_code(frame, code_img, yoff, alpha=code_alpha_eff)

            # Color conversion is expensive; do it once per frame
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Trails buffer
            if args.trail:
                if trail_buf is None:
                    trail_buf = np.zeros_like(frame)
                else:
                    # reactive: smear/trail breathes slightly on pulse
                    # Optimized: use in-place operations to avoid extra allocations
                    alpha_eff = float(args.trail_alpha)
                    if args.beat_sync:
                        alpha_eff = float(np.clip(alpha_eff - 0.15 * pulse, 0.0, 0.999))
                    # In-place multiplication is faster
                    np.multiply(trail_buf, alpha_eff, out=trail_buf, casting='unsafe')

            # We'll build a "pose layer" if we need pose-only smear
            # Reuse pre-allocated array for performance
            if args.smear and args.smear_on_pose_only:
                pose_layer.fill(0)  # Clear the pre-allocated array instead of creating new

            if res.pose_landmarks:
                h, w = frame.shape[:2]
                curr_landmarks = []
                velocities = {}

                for i, lm in enumerate(res.pose_landmarks.landmark):
                    x, y = int(lm.x * w), int(lm.y * h)
                    curr_landmarks.append((x, y))
                    if prev_landmarks:
                        px, py = prev_landmarks[i]
                        velocities[i] = ((x - px) ** 2 + (y - py) ** 2) ** 0.5 * fps
                    else:
                        velocities[i] = 0.0

                # Choose draw targets
                # Lines go to trail buffer (if enabled) else to frame
                draw_target = trail_buf if (args.trail and trail_buf is not None) else frame

                # Beat-synced line thickness modulation
                # base 1, pulse adds up to +2 (1..3)
                thickness = 1 + (2 if pulse > 0.75 else (1 if pulse > 0.35 else 0))

                # connections
                for a, b in mp_pose.POSE_CONNECTIONS:
                    xa, ya = curr_landmarks[a]
                    xb, yb = curr_landmarks[b]
                    if args.velocity_color:
                        color = velocity_to_color(max(velocities[a], velocities[b]))
                    else:
                        color = args.overlay_color

                    cv2.line(draw_target, (xa, ya), (xb, yb), color, thickness)

                    if pose_layer is not None:
                        cv2.line(pose_layer, (xa, ya), (xb, yb), color, thickness)

                # joints when not lines-only
                if not args.trail_lines_only:
                    for i, (x, y) in enumerate(curr_landmarks):
                        if args.velocity_color:
                            color = velocity_to_color(velocities[i])
                        else:
                            color = args.overlay_color
                        cv2.circle(draw_target, (x, y), 2 + (1 if pulse > 0.6 else 0), color, -1, lineType=cv2.LINE_AA)
                        if pose_layer is not None:
                            cv2.circle(pose_layer, (x, y), 2 + (1 if pulse > 0.6 else 0), color, -1, lineType=cv2.LINE_AA)

                # ids on the main frame (not on trail buffer)
                if args.draw_ids:
                    for i, (x, y) in enumerate(curr_landmarks):
                        if args.velocity_color:
                            color = velocity_to_color(velocities[i])
                        else:
                            color = args.overlay_color
                        draw_landmark_id(frame, x, y, i, color)

                prev_landmarks = curr_landmarks

            # Composite trail (optionally through pose-only smear)
            if args.trail and trail_buf is not None:
                if args.smear and args.smear_on_pose_only and smear_buf is not None:
                    # feed pose-ish layer: prefer pose_layer, else trail_buf
                    layer = pose_layer if pose_layer is not None else trail_buf
                    # push and blend
                    smear_buf.append(layer.copy())
                    smeared = blend_temporal(smear_buf, layer, decay=args.smear_decay, mode=args.smear_mode)
                    frame = cv2.addWeighted(frame, 1.0, smeared, 1.0, 0)
                else:
                    frame = cv2.addWeighted(frame, 1.0, trail_buf, 1.0, 0)

            # code overlay AFTER pose drawing
            if code_img is not None and args.code_overlay_order == "after":
                # Optimized: use pre-computed constants
                t_full = start_time_full + (frame_idx * fps_inv)
                progress = max(0.0, min(1.0, t_full * full_duration_inv))
                yoff = progress * code_scroll_range
                code_alpha_eff = float(args.code_alpha) * (1.0 + 0.15 * pulse)
                frame = overlay_code(frame, code_img, yoff, alpha=code_alpha_eff)

            # Full-frame temporal smear (if enabled and NOT pose-only)
            if args.smear and not args.smear_on_pose_only and smear_buf is not None:
                smear_buf.append(frame.copy())
                # reactive: more smear on pulse by reducing decay
                decay_eff = float(args.smear_decay)
                if args.beat_sync:
                    decay_eff = float(np.clip(decay_eff - 0.25 * pulse, 0.0, 0.999))
                frame = blend_temporal(smear_buf, frame, decay=decay_eff, mode=args.smear_mode)

            # Glitch (reactive burst on pulse)
            if args.glitch:
                t_clip = frame_idx * fps_inv  # Optimized: use pre-computed constant
                strength_eff = float(args.glitch_strength)
                if args.beat_sync:
                    strength_eff = float(np.clip(strength_eff + float(args.beat_pulse_boost) * pulse, 0.0, 1.0))
                frame = apply_glitch(
                    frame=frame,
                    t=t_clip,
                    strength=strength_eff,
                    rate=float(args.glitch_rate),
                    rgb_split_px=int(args.glitch_rgb_split),
                    scan_jitter_px=int(args.glitch_scan_jitter),
                    dropout_prob=float(args.glitch_dropout),
                )

            # Scanlines (breathes slightly on pulse)
            if args.scanlines:
                scan_eff = float(args.scanline_strength)
                if args.beat_sync:
                    scan_eff = float(np.clip(scan_eff + 0.08 * pulse, 0.0, 0.25))
                frame = apply_scanlines(frame, scan_eff)

            out.write(frame)
            frame_idx += 1
        
        print(f"   100% complete ({frame_idx}/{frame_count} frames)")

        cap.release()
        out.release()
        print("\n✓ Frame processing complete")

    # 3) Mux audio (external WAV preferred) with offset
    print(f"\n{'='*60}")
    print("STEP 4: Final Audio Muxing")
    print(f"{'='*60}")
    audio_wav = Path(args.audio_wav) if args.audio_wav else None
    if audio_wav and not audio_wav.exists():
        raise FileNotFoundError(audio_wav)

    mux_audio_player_safe(
        video_noaudio=tmp_noaudio,
        src_video=src,
        audio_wav=audio_wav,
        out_mp4=(temp_root / outp.name if args.use_temp_files else outp),
        audio_start=audio_start,
        audio_duration=args.duration,
        crf=args.crf,
        preset=args.preset,
        audio_bitrate=args.audio_bitrate,
        use_gpu=use_gpu,
        input_codec=input_codec if args.match_input_codec else "h264",
        input_bitrate=input_bitrate if args.match_input_codec else 5000,
    )

    # cleanup
    print(f"\n🧹 Cleaning up temporary files...")
    # cleanup intermediates
    tmp_cfr.unlink(missing_ok=True)
    tmp_noaudio.unlink(missing_ok=True)
    if tmp_env_wav is not None:
        tmp_env_wav.unlink(missing_ok=True)

    # if temp mode was used, copy final result back
    if args.use_temp_files:
        final_temp = temp_root / outp.name
        if final_temp.exists():
            print(f"📦 Copying final output to {outp}...")
            shutil.copy2(str(final_temp), str(outp))
            try:
                final_temp.unlink()
            except Exception:
                pass

    elapsed = time.time() - start_wall_time
    mins = int(elapsed // 60)
    secs = elapsed % 60

    print(f"\n{'='*60}")
    print("✅ PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"📂 Output file: {outp}")
    print(f"📊 Processed {frame_idx} frames")
    print(f"⏱️  Total runtime: {mins}m {secs:.1f}s")
    if frame_idx > 0:
        avg_fps = frame_idx / elapsed
        print(f"⚡ Average processing speed: {avg_fps:.2f} fps")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
