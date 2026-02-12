#!/usr/bin/env python3
import subprocess
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path

INPUT = Path("input.mp4")
OUTPUT = Path("output_cinematic.mp4")

FPS = 30
WINDOW_SEC = 0.25      # analysis window
MIN_SEGMENT = 0.75     # seconds
SILENCE_DB = -40

def run(cmd):
    print("▶", " ".join(cmd))
    subprocess.run(cmd, check=True)

def extract_audio(video, wav):
    run([
        "ffmpeg", "-y", "-i", str(video),
        "-vn", "-ac", "1", "-ar", "48000",
        str(wav)
    ])

def energy_curve(wav):
    audio, sr = sf.read(wav)
    audio = audio.astype(np.float32)

    win = int(sr * WINDOW_SEC)

    # Square once
    sq = audio * audio

    # Cumulative sum (fast!)
    csum = np.cumsum(np.insert(sq, 0, 0))

    # Sliding RMS in O(N)
    rms = np.sqrt((csum[win:] - csum[:-win]) / win)

    # Normalize robustly
    p95 = np.percentile(rms, 95)
    if p95 > 0:
        rms /= p95

    return rms


def energy_to_speed(e):
    if e < 0.04: return 3.0
    if e < 0.12: return 2.0
    if e < 0.35: return 1.25
    return 1.0

def build_segments(energy):
    segments = []
    t = 0.0
    last_speed = energy_to_speed(energy[0])
    start = 0.0

    for i, e in enumerate(energy):
        speed = energy_to_speed(e)
        if speed != last_speed:
            dur = t - start
            if dur >= MIN_SEGMENT:
                segments.append((start, dur, last_speed))
                start = t
                last_speed = speed
        t += WINDOW_SEC

    segments.append((start, t - start, last_speed))
    return segments

def render_segments(video, segments, tmpdir):
    out_files = []

    for i, (start, dur, speed) in enumerate(segments):
        out = tmpdir / f"seg_{i:04d}.mp4"
        run([
            "ffmpeg", "-y",
            "-ss", f"{start:.2f}",
            "-t", f"{dur:.2f}",
            "-i", str(video),
            "-filter_complex",
            f"[0:v]setpts=PTS/{speed}[v];"
            f"[0:a]atempo={min(speed,2.0)}[a]",
            "-map", "[v]", "-map", "[a]",
            "-movflags", "+faststart",
            "-pix_fmt", "yuv420p",
            str(out)
        ])
        out_files.append(out)

    return out_files

def concat(files, out):
    listfile = out.with_suffix(".txt")
    with open(listfile, "w") as f:
        for x in files:
            f.write(f"file '{x.resolve()}'\n")

    run([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(listfile),
        "-c", "copy",
        str(out)
    ])

def main():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        wav = td / "audio.wav"

        extract_audio(INPUT, wav)
        energy = energy_curve(wav)
        segments = build_segments(energy)
        files = render_segments(INPUT, segments, td)
        concat(files, OUTPUT)

    print("✅ Done:", OUTPUT)

if __name__ == "__main__":
    main()
