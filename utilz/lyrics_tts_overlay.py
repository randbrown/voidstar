#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import edge_tts


@dataclass
class VideoInfo:
    duration: float
    width: int
    height: int
    has_audio: bool


@dataclass
class LyricSegment:
    index: int
    text: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def run_cmd(cmd: list[str], *, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
    )


def ffprobe_json(input_path: Path) -> dict[str, Any]:
    result = run_cmd(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(input_path),
        ],
        capture_output=True,
    )
    return json.loads(result.stdout)


def get_video_info(input_path: Path) -> VideoInfo:
    data = ffprobe_json(input_path)
    duration = float(data["format"]["duration"])
    video_stream = next((s for s in data["streams"] if s.get("codec_type") == "video"), None)
    if video_stream is None:
        raise ValueError(f"No video stream found: {input_path}")
    width = int(video_stream["width"])
    height = int(video_stream["height"])
    has_audio = any(s.get("codec_type") == "audio" for s in data["streams"])
    return VideoInfo(duration=duration, width=width, height=height, has_audio=has_audio)


def get_media_duration(path: Path) -> float:
    result = run_cmd(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
    )
    return float(result.stdout.strip())


def read_lyrics_lines(path: Path) -> list[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        raise ValueError(f"No non-empty lyric lines found in {path}")
    return lines


def parse_color(text: str) -> tuple[int, int, int]:
    value = text.strip().lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected RRGGBB color, got: {text!r}")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def rgb_to_ass_bgr_hex(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"&H{b:02X}{g:02X}{r:02X}&"


def seconds_to_ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int(round((seconds - math.floor(seconds)) * 100.0))
    if centis >= 100:
        centis = 0
        secs += 1
    if secs >= 60:
        secs = 0
        minutes += 1
    if minutes >= 60:
        minutes = 0
        hours += 1
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def ass_escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}").replace("\n", r"\N")


def ffmpeg_filter_path(path: Path) -> str:
    value = str(path)
    value = value.replace("\\", r"\\")
    value = value.replace("'", r"\'")
    return f"'{value}'"


def build_even_segments(lines: list[str], start: float, end: float) -> list[LyricSegment]:
    count = len(lines)
    segment_duration = (end - start) / float(count)
    return [
        LyricSegment(
            index=index,
            text=line,
            start=start + (segment_duration * index),
            end=start + (segment_duration * (index + 1)),
        )
        for index, line in enumerate(lines)
    ]


def build_grid_segments(lines: list[str], start: float, end: float, grid_step: float) -> list[LyricSegment]:
    if grid_step <= 0:
        raise ValueError("Grid step must be > 0")
    count = len(lines)
    available = max(0.0, end - start)
    total_steps = int(math.floor(available / grid_step))
    if total_steps < count:
        raise ValueError(
            f"Tempo grid too coarse for {count} lines in {available:.2f}s span; "
            f"need at least {count} grid steps, got {total_steps}"
        )

    boundaries = [0]
    for i in range(1, count):
        step_index = int(round(i * total_steps / count))
        step_index = max(boundaries[-1] + 1, min(step_index, total_steps - (count - i)))
        boundaries.append(step_index)
    boundaries.append(total_steps)

    segments: list[LyricSegment] = []
    for index, line in enumerate(lines):
        seg_start = start + (boundaries[index] * grid_step)
        seg_end = start + (boundaries[index + 1] * grid_step)
        segments.append(LyricSegment(index=index, text=line, start=seg_start, end=seg_end))
    return segments


def plan_segments(lines: list[str], video_duration: float, args: argparse.Namespace) -> list[LyricSegment]:
    start = max(0.0, float(args.start_margin))
    end = video_duration - max(0.0, float(args.end_margin))
    if end <= start:
        raise ValueError("Margins leave no available time span")

    mode = args.mode
    if mode == "even":
        return build_even_segments(lines, start, end)
    if mode == "cps":
        if args.cps <= 0:
            raise ValueError("--cps must be > 0 in cps mode")
        return build_grid_segments(lines, start, end, 1.0 / float(args.cps))
    if mode == "bpm":
        if args.bpm <= 0:
            raise ValueError("--bpm must be > 0 in bpm mode")
        beats_per_line = float(args.beats_per_line)
        if beats_per_line <= 0:
            raise ValueError("--beats-per-line must be > 0 in bpm mode")
        beat_step = 60.0 / float(args.bpm)
        return build_grid_segments(lines, start, end, beat_step * beats_per_line)
    raise ValueError(f"Unsupported mode: {mode}")


def compute_text_position(args: argparse.Namespace, width: int, height: int) -> tuple[int, int]:
    if args.text_position == "top":
        return width // 2, int(height * 0.16)
    if args.text_position == "bottom":
        return width // 2, int(height * 0.84)
    if args.text_position == "custom":
        return int(width * args.text_x_ratio), int(height * args.text_y_ratio)
    return width // 2, height // 2


def build_ass_file(
    ass_path: Path,
    segments: list[LyricSegment],
    video_info: VideoInfo,
    args: argparse.Namespace,
) -> None:
    primary = rgb_to_ass_bgr_hex(parse_color(args.font_color))
    outline = rgb_to_ass_bgr_hex(parse_color(args.outline_color))
    shadow = rgb_to_ass_bgr_hex(parse_color(args.shadow_color))
    x, y = compute_text_position(args, video_info.width, video_info.height)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {video_info.width}
PlayResY: {video_info.height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Default,{args.font_name},{args.font_size},{primary},{primary},{outline},{shadow},0,0,0,0,100,100,0,0,1,{args.outline_width},{args.shadow_depth},5,40,40,40,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""

    events: list[str] = []
    for segment in segments:
        start = seconds_to_ass_time(segment.start)
        end = seconds_to_ass_time(segment.end)
        text = ass_escape(segment.text)
        events.append(
            f"Dialogue: 0,{start},{end},Default,,0,0,0,,{{\\pos({x},{y})}}{text}"
        )

    ass_path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")


async def synthesize_tts(text: str, output_path: Path, args: argparse.Namespace) -> None:
    voice = edge_tts.Communicate(
        text,
        voice=args.voice,
        rate=args.voice_rate,
        pitch=args.voice_pitch,
    )
    await voice.save(str(output_path))


def atempo_chain(speedup: float) -> str:
    if speedup <= 0:
        raise ValueError("speedup must be > 0")
    factors: list[float] = []
    remaining = speedup
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    return ",".join(f"atempo={factor:.6f}" for factor in factors)


async def generate_tts_tracks(
    segments: list[LyricSegment],
    work_dir: Path,
    args: argparse.Namespace,
) -> list[Path]:
    outputs: list[Path] = []
    for segment in segments:
        raw_path = work_dir / f"tts_{segment.index:03d}.mp3"
        await synthesize_tts(segment.text, raw_path, args)
        current_path = raw_path

        if args.tts_fit_mode != "none":
            raw_duration = get_media_duration(raw_path)
            target_duration = max(segment.duration, 0.05)
            if raw_duration > target_duration:
                if args.tts_fit_mode == "truncate":
                    fitted_path = work_dir / f"tts_{segment.index:03d}_fit.wav"
                    run_cmd(
                        [
                            "ffmpeg",
                            "-y",
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-i",
                            str(raw_path),
                            "-t",
                            f"{target_duration:.6f}",
                            "-c:a",
                            "pcm_s16le",
                            str(fitted_path),
                        ]
                    )
                    current_path = fitted_path
                elif args.tts_fit_mode == "speed":
                    speedup = raw_duration / target_duration
                    fitted_path = work_dir / f"tts_{segment.index:03d}_fit.wav"
                    run_cmd(
                        [
                            "ffmpeg",
                            "-y",
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-i",
                            str(raw_path),
                            "-filter:a",
                            atempo_chain(speedup),
                            "-c:a",
                            "pcm_s16le",
                            str(fitted_path),
                        ]
                    )
                    current_path = fitted_path
        outputs.append(current_path)
    return outputs


def build_plan_json(plan_path: Path, segments: list[LyricSegment], args: argparse.Namespace, video_info: VideoInfo) -> None:
    payload = {
        "input_video": str(args.input_video),
        "lyrics_file": str(args.lyrics_file),
        "mode": args.mode,
        "video_duration": video_info.duration,
        "video_width": video_info.width,
        "video_height": video_info.height,
        "segments": [asdict(segment) | {"duration": segment.duration} for segment in segments],
    }
    plan_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def render_output(
    input_video: Path,
    output_video: Path,
    ass_path: Path | None,
    tts_tracks: list[Path],
    segments: list[LyricSegment],
    video_info: VideoInfo,
    args: argparse.Namespace,
) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_video),
    ]
    for track in tts_tracks:
        cmd.extend(["-i", str(track)])

    filter_parts: list[str] = []
    base_audio_label = None
    if video_info.has_audio:
        base_audio_label = "[0:a]"

    delayed_labels: list[str] = []
    for i, segment in enumerate(segments, start=1):
        if i - 1 >= len(tts_tracks):
            break
        delay_ms = max(0, int(round(segment.start * 1000.0)))
        label = f"[tts{i}]"
        filter_parts.append(
            f"[{i}:a]adelay={delay_ms}|{delay_ms},volume={args.voice_volume:.6f}{label}"
        )
        delayed_labels.append(label)

    audio_output_label = None
    if delayed_labels:
        inputs = ([] if base_audio_label is None else [base_audio_label]) + delayed_labels
        audio_output_label = "[aout]"
        filter_parts.append("".join(inputs) + f"amix=inputs={len(inputs)}:normalize=0{audio_output_label}")
    elif base_audio_label is not None:
        audio_output_label = base_audio_label

    if filter_parts:
        cmd.extend(["-filter_complex", ";".join(filter_parts)])

    if ass_path is not None:
        cmd.extend(["-vf", f"ass={ffmpeg_filter_path(ass_path)}"])

    cmd.extend(["-map", "0:v:0"])
    if audio_output_label is not None:
        cmd.extend(["-map", audio_output_label])
    elif video_info.has_audio:
        cmd.extend(["-map", "0:a:0"])

    cmd.extend(
        [
            "-c:v",
            args.video_codec,
            "-preset",
            args.video_preset,
            "-crf",
            str(args.video_crf),
        ]
    )
    if audio_output_label is not None or video_info.has_audio:
        cmd.extend(["-c:a", "aac", "-b:a", args.audio_bitrate])
    cmd.extend(["-shortest", str(output_video)])
    run_cmd(cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay timed lyric lines on a video and synthesize aligned TTS speech."
    )
    parser.add_argument("input_video", type=Path, help="Input video path")
    parser.add_argument("lyrics_file", type=Path, help="UTF-8 text file with one lyric line per line")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output video path")
    parser.add_argument("--mode", choices=["even", "cps", "bpm"], default="even", help="Timing mode for lyric segments")
    parser.add_argument("--start-margin", type=float, default=0.0, help="Seconds to leave before the first lyric line")
    parser.add_argument("--end-margin", type=float, default=0.0, help="Seconds to leave after the last lyric line")
    parser.add_argument("--cps", type=float, default=0.0, help="Cycles per second grid when --mode cps")
    parser.add_argument("--bpm", type=float, default=0.0, help="Tempo grid when --mode bpm")
    parser.add_argument("--beats-per-line", type=float, default=1.0, help="Grid size per lyric line in BPM mode")
    parser.add_argument("--voice", default="en-GB-RyanNeural", help="Edge TTS voice name")
    parser.add_argument("--voice-rate", default="-8%", help="Edge TTS speaking rate, e.g. -10%% or +5%%")
    parser.add_argument("--voice-pitch", default="-6Hz", help="Edge TTS pitch, e.g. -4Hz or +0Hz")
    parser.add_argument("--voice-volume", type=float, default=0.9, help="Final voice mix gain multiplier")
    parser.add_argument("--tts-fit-mode", choices=["speed", "truncate", "none"], default="speed", help="How to handle spoken lines longer than their segment")
    parser.add_argument("--no-tts", action="store_true", help="Disable synthesized speech generation")
    parser.add_argument("--no-text", action="store_true", help="Disable on-screen lyric text")
    parser.add_argument("--font-name", default="DejaVu Sans", help="Subtitle font name")
    parser.add_argument("--font-size", type=int, default=64, help="Subtitle font size")
    parser.add_argument("--font-color", default="FFFFFF", help="Primary text color in RRGGBB")
    parser.add_argument("--outline-color", default="101010", help="Outline color in RRGGBB")
    parser.add_argument("--shadow-color", default="000000", help="Shadow color in RRGGBB")
    parser.add_argument("--outline-width", type=float, default=3.0, help="Subtitle outline thickness")
    parser.add_argument("--shadow-depth", type=float, default=1.5, help="Subtitle shadow depth")
    parser.add_argument("--text-position", choices=["center", "top", "bottom", "custom"], default="center", help="Subtitle anchor position")
    parser.add_argument("--text-x-ratio", type=float, default=0.5, help="Horizontal position ratio for custom text placement")
    parser.add_argument("--text-y-ratio", type=float, default=0.5, help="Vertical position ratio for custom text placement")
    parser.add_argument("--video-codec", default="libx264", help="FFmpeg video codec for output")
    parser.add_argument("--video-preset", default="medium", help="FFmpeg preset for output video")
    parser.add_argument("--video-crf", type=int, default=18, help="CRF for output video")
    parser.add_argument("--audio-bitrate", default="192k", help="Output AAC bitrate")
    parser.add_argument("--write-plan-json", type=Path, default=None, help="Optional path to write the computed lyric timing plan as JSON")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary subtitle and TTS files")
    parser.add_argument("--dry-run", action="store_true", help="Compute and print the timing plan without rendering output")
    return parser.parse_args()


def print_plan(segments: list[LyricSegment]) -> None:
    for segment in segments:
        print(
            f"[{segment.index:03d}] {segment.start:8.3f}s -> {segment.end:8.3f}s "
            f"({segment.duration:6.3f}s) | {segment.text}"
        )


def main() -> int:
    args = parse_args()
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe must be installed and on PATH")
    if not args.input_video.is_file():
        raise FileNotFoundError(f"Input video not found: {args.input_video}")
    if not args.lyrics_file.is_file():
        raise FileNotFoundError(f"Lyrics file not found: {args.lyrics_file}")

    video_info = get_video_info(args.input_video)
    lines = read_lyrics_lines(args.lyrics_file)
    segments = plan_segments(lines, video_info.duration, args)

    if args.write_plan_json is not None:
        build_plan_json(args.write_plan_json, segments, args, video_info)

    print_plan(segments)
    if args.dry_run:
        return 0

    output_path = args.output
    if output_path is None:
        output_path = args.input_video.with_name(f"{args.input_video.stem}_lyrics_tts.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_dir_obj = tempfile.TemporaryDirectory(prefix="voidstar_lyrics_tts_")
    work_dir = Path(temp_dir_obj.name)

    try:
        ass_path = None
        if not args.no_text:
            ass_path = work_dir / "lyrics.ass"
            build_ass_file(ass_path, segments, video_info, args)

        tts_tracks: list[Path] = []
        if not args.no_tts:
            tts_tracks = asyncio.run(generate_tts_tracks(segments, work_dir, args))

        render_output(args.input_video, output_path, ass_path, tts_tracks, segments, video_info, args)
        print(f"[lyrics-tts] wrote {output_path}")
        if args.write_plan_json is not None:
            print(f"[lyrics-tts] plan json {args.write_plan_json}")
        return 0
    finally:
        if args.keep_temp:
            print(f"[lyrics-tts] kept temp files in {work_dir}")
        else:
            temp_dir_obj.cleanup()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)