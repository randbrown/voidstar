#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from shutil import which


def log(message: str) -> None:
    print(f"[voidstar] {message}", flush=True)


def die(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(1)


def require_cmd(name: str) -> None:
    if not which(name):
        die(f"Missing required command: {name}")


def bool_flag(value: str) -> bool:
    v = value.strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {value}")


def run_cmd(cmd: list[str]) -> None:
    log("▶ " + " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}")


def ffprobe_info(input_path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(input_path),
    ]
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)


def ffprobe_duration_seconds(input_path: Path) -> float:
    info = ffprobe_info(input_path)
    raw = (info.get("format") or {}).get("duration")
    if raw is None:
        die(f"Could not determine duration via ffprobe: {input_path}")
    value = float(raw)
    if value <= 0:
        die(f"Invalid non-positive duration from ffprobe: {value}")
    return value


def has_audio_stream(input_path: Path) -> bool:
    info = ffprobe_info(input_path)
    streams = info.get("streams") or []
    return any((s or {}).get("codec_type") == "audio" for s in streams)


def slug_float(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def build_default_output_path(input_path: Path, out_dir: Path, args: argparse.Namespace) -> Path:
    oa = "1" if args.overwrite_audio else "0"
    name = (
        f"{input_path.stem}_hook"
        f"_mode-{args.mode}"
        f"_hs-{slug_float(args.hook_start)}"
        f"_hd-{slug_float(args.hook_duration)}"
        f"_oa-{oa}{input_path.suffix}"
    )
    return out_dir / name


def extract_segment_copy(
    input_path: Path,
    output_path: Path,
    *,
    start: float,
    duration: float,
    include_video: bool,
    include_audio: bool,
) -> None:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{max(0.0, start):.6f}",
        "-i",
        str(input_path),
        "-t",
        f"{max(0.0, duration):.6f}",
    ]

    if include_video:
        cmd += ["-map", "0:v:0"]
    else:
        cmd += ["-vn"]

    if include_audio:
        cmd += ["-map", "0:a:0"]
    else:
        cmd += ["-an"]

    cmd += ["-c", "copy", "-avoid_negative_ts", "make_zero", str(output_path)]
    run_cmd(cmd)


def concat_copy(parts: list[Path], output_path: Path) -> None:
    if not parts:
        die("concat requested with no parts")

    with tempfile.NamedTemporaryFile("w", suffix=".txt", prefix="voidstar_concat_", delete=False) as tmp:
        list_path = Path(tmp.name)
        for part in parts:
            escaped = str(part).replace("'", "'\\''")
            tmp.write(f"file '{escaped}'\n")

    try:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        run_cmd(cmd)
    finally:
        list_path.unlink(missing_ok=True)


def mux_video_with_original_audio(video_path: Path, source_path: Path, output_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(source_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    run_cmd(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Voidstar hook tool: prepend, overwrite, or wrap around the best section "
            "to move engagement-heavy moments to the front."
        )
    )
    parser.add_argument("input", help="Input video path")
    parser.add_argument("--mode", required=True, choices=["prepend", "overwrite", "wrap"], help="Hook mode")
    parser.add_argument("--hook-start", type=float, required=True, help="Hook start time (seconds)")
    parser.add_argument("--hook-duration", type=float, default=5.0, help="Hook duration (seconds)")
    parser.add_argument(
        "--overwrite-audio",
        type=bool_flag,
        default=True,
        help="Overwrite mode only: if true, overwrite opening audio with hook audio (default: true)",
    )
    parser.add_argument("--output", default="", help="Explicit output file path")
    parser.add_argument("--out-dir", default="", help="Output directory when --output is omitted")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output file")
    args = parser.parse_args()

    require_cmd("ffmpeg")
    require_cmd("ffprobe")

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        die(f"Input file not found: {input_path}")

    duration = ffprobe_duration_seconds(input_path)
    audio_present = has_audio_stream(input_path)

    if args.hook_start < 0:
        die("--hook-start must be >= 0")
    if args.hook_start >= duration:
        die(f"--hook-start must be < input duration ({duration:.3f}s)")

    if args.mode in {"prepend", "overwrite"} and args.hook_duration <= 0:
        die("--hook-duration must be > 0 for prepend/overwrite modes")

    hook_duration = float(args.hook_duration)
    max_hook_duration = max(0.0, duration - args.hook_start)
    if args.mode in {"prepend", "overwrite"} and hook_duration > max_hook_duration:
        log(
            f"Requested hook duration {hook_duration:.3f}s exceeds available range; "
            f"clamping to {max_hook_duration:.3f}s"
        )
        hook_duration = max_hook_duration

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        out_dir = output_path.parent
    else:
        out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else input_path.parent
        output_path = build_default_output_path(input_path, out_dir, args)

    out_dir.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.force:
        die(f"Output already exists (use --force): {output_path}")

    log(f"Input:  {input_path}")
    log(f"Mode:   {args.mode}")
    log(f"Hook:   start={args.hook_start:.3f}s duration={hook_duration:.3f}s")
    log(f"Audio:  present={audio_present} overwrite_audio={args.overwrite_audio}")
    log(f"Output: {output_path}")

    suffix = input_path.suffix if input_path.suffix else ".mp4"

    with tempfile.TemporaryDirectory(prefix="voidstar_hook_") as td:
        tdir = Path(td)

        if args.mode == "prepend":
            hook_part = tdir / f"hook{suffix}"
            full_part = tdir / f"full{suffix}"

            extract_segment_copy(
                input_path,
                hook_part,
                start=args.hook_start,
                duration=hook_duration,
                include_video=True,
                include_audio=audio_present,
            )
            extract_segment_copy(
                input_path,
                full_part,
                start=0.0,
                duration=duration,
                include_video=True,
                include_audio=audio_present,
            )
            concat_copy([hook_part, full_part], output_path)

        elif args.mode == "overwrite":
            tail_duration = max(0.0, duration - hook_duration)
            if tail_duration <= 0:
                die("Hook duration leaves no tail in overwrite mode; reduce --hook-duration")

            if args.overwrite_audio:
                hook_part = tdir / f"hook{suffix}"
                tail_part = tdir / f"tail{suffix}"

                extract_segment_copy(
                    input_path,
                    hook_part,
                    start=args.hook_start,
                    duration=hook_duration,
                    include_video=True,
                    include_audio=audio_present,
                )
                extract_segment_copy(
                    input_path,
                    tail_part,
                    start=hook_duration,
                    duration=tail_duration,
                    include_video=True,
                    include_audio=audio_present,
                )
                concat_copy([hook_part, tail_part], output_path)
            else:
                hook_video = tdir / f"hook_video{suffix}"
                tail_video = tdir / f"tail_video{suffix}"
                video_composite = tdir / f"video_composite{suffix}"

                extract_segment_copy(
                    input_path,
                    hook_video,
                    start=args.hook_start,
                    duration=hook_duration,
                    include_video=True,
                    include_audio=False,
                )
                extract_segment_copy(
                    input_path,
                    tail_video,
                    start=hook_duration,
                    duration=tail_duration,
                    include_video=True,
                    include_audio=False,
                )
                concat_copy([hook_video, tail_video], video_composite)

                if audio_present:
                    mux_video_with_original_audio(video_composite, input_path, output_path)
                else:
                    log("Input has no audio stream; outputting video-only overwrite result")
                    video_composite.replace(output_path)

        else:  # wrap
            if args.hook_duration > 0:
                log("Wrap mode ignores --hook-duration and uses only --hook-start")

            first_duration = max(0.0, duration - args.hook_start)
            second_duration = args.hook_start
            first_part = tdir / f"wrap_a{suffix}"
            second_part = tdir / f"wrap_b{suffix}"

            extract_segment_copy(
                input_path,
                first_part,
                start=args.hook_start,
                duration=first_duration,
                include_video=True,
                include_audio=audio_present,
            )

            if second_duration > 0:
                extract_segment_copy(
                    input_path,
                    second_part,
                    start=0.0,
                    duration=second_duration,
                    include_video=True,
                    include_audio=audio_present,
                )
                concat_copy([first_part, second_part], output_path)
            else:
                first_part.replace(output_path)

    log("Done.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        die(str(exc))