#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from shutil import which


TRANSITION_TYPES = [
    "none",
    "fade",
    "dissolve",
    "smoothleft",
    "smoothright",
    "wipeleft",
    "wiperight",
    "circleopen",
    "circleclose",
    "pixelize",
]


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


def parse_ffmpeg_time_to_seconds(value: str) -> float:
    if not value:
        return 0.0
    txt = value.strip()
    if txt.upper() in {"N/A", "NA", "INF", "-INF", "NAN", "-NAN"}:
        return 0.0
    if txt.isdigit():
        return max(0.0, float(int(txt)) / 1_000_000.0)
    if ":" in txt:
        try:
            h, m, s = txt.split(":")
            return max(0.0, (float(h) * 3600.0) + (float(m) * 60.0) + float(s))
        except Exception:
            return 0.0
    try:
        return max(0.0, float(txt))
    except Exception:
        return 0.0


def run_cmd(cmd: list[str], *, progress_label: str = "", expected_duration_sec: float = 0.0) -> None:
    log("▶ " + " ".join(shlex.quote(c) for c in cmd))

    is_ffmpeg = bool(cmd) and Path(cmd[0]).name.startswith("ffmpeg")
    if not is_ffmpeg:
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {proc.returncode}")
        return

    ff_cmd = cmd[:]
    if "-progress" not in ff_cmd:
        ff_cmd[1:1] = ["-nostats", "-progress", "pipe:2"]

    proc = subprocess.Popen(
        ff_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    last_emit = 0.0
    start_time = time.time()
    current_out_seconds = 0.0
    current_speed = 0.0
    emit_interval_sec = 1.5

    try:
        assert proc.stderr is not None
        for raw_line in proc.stderr:
            line = raw_line.strip()
            if not line:
                continue

            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                if key in {"out_time", "out_time_ms", "out_time_us"}:
                    current_out_seconds = parse_ffmpeg_time_to_seconds(value)
                elif key == "speed":
                    speed_txt = value[:-1] if value.endswith("x") else value
                    try:
                        current_speed = max(0.0, float(speed_txt))
                    except Exception:
                        current_speed = 0.0
                elif key == "progress":
                    now = time.time()
                    emit = (now - last_emit) >= emit_interval_sec or value == "end"
                    if emit:
                        elapsed = now - start_time
                        label = progress_label or "ffmpeg"
                        if expected_duration_sec > 0:
                            pct = min(100.0, 100.0 * (current_out_seconds / expected_duration_sec))
                            remaining = max(0.0, expected_duration_sec - current_out_seconds)
                            eta = (remaining / current_speed) if current_speed > 1e-6 else -1.0
                            if eta >= 0:
                                log(
                                    f"{label} ... {pct:5.1f}% out={current_out_seconds:.1f}s/"
                                    f"{expected_duration_sec:.1f}s speed={current_speed:.2f}x eta={eta:.1f}s"
                                )
                            else:
                                log(
                                    f"{label} ... {pct:5.1f}% out={current_out_seconds:.1f}s/"
                                    f"{expected_duration_sec:.1f}s speed={current_speed:.2f}x elapsed={elapsed:.1f}s"
                                )
                        else:
                            log(
                                f"{label} ... out={current_out_seconds:.1f}s speed={current_speed:.2f}x "
                                f"elapsed={elapsed:.1f}s"
                            )
                        last_emit = now
                continue

            print(raw_line, end="", file=sys.stderr)
    finally:
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed with exit code {rc}")


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


def get_primary_stream(info: dict, codec_type: str) -> dict:
    for stream in info.get("streams") or []:
        if (stream or {}).get("codec_type") == codec_type:
            return stream or {}
    return {}


def choose_video_encoder(source_codec: str) -> str:
    mapping = {
        "h264": "libx264",
        "hevc": "libx265",
        "mpeg4": "mpeg4",
        "vp9": "libvpx-vp9",
        "av1": "libsvtav1",
    }
    return mapping.get(source_codec, "libx264")


def choose_audio_encoder(source_codec: str) -> str:
    mapping = {
        "aac": "aac",
        "mp3": "libmp3lame",
        "opus": "libopus",
        "vorbis": "libvorbis",
    }
    return mapping.get(source_codec, "aac")


def parse_int_or_zero(value: object) -> int:
    try:
        out = int(str(value or "0"))
        return max(0, out)
    except Exception:
        return 0


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
    run_cmd(cmd, progress_label="extract segment", expected_duration_sec=duration)


def concat_copy(parts: list[Path], output_path: Path) -> None:
    if not parts:
        die("concat requested with no parts")

    expected_duration = 0.0
    for part in parts:
        expected_duration += ffprobe_duration_seconds(part)

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
        run_cmd(cmd, progress_label="concat", expected_duration_sec=expected_duration)
    finally:
        list_path.unlink(missing_ok=True)


def mux_video_with_original_audio(video_path: Path, source_path: Path, output_path: Path) -> None:
    expected_duration = ffprobe_duration_seconds(video_path)
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
    run_cmd(cmd, progress_label="mux", expected_duration_sec=expected_duration)


def transition_join(
    part_a: Path,
    part_b: Path,
    output_path: Path,
    *,
    transition_type: str,
    transition_duration: float,
    video_encoder: str,
    video_bitrate: int,
    audio_encoder: str,
    audio_bitrate: int,
    include_audio_transition: bool,
) -> None:
    dur_a = ffprobe_duration_seconds(part_a)
    dur_b = ffprobe_duration_seconds(part_b)
    offset = max(0.0, dur_a - transition_duration)
    expected_duration = max(0.0, dur_a + dur_b - transition_duration)

    filters = [
        (
            "[0:v][1:v]"
            f"xfade=transition={transition_type}:duration={transition_duration:.6f}:offset={offset:.6f}"
            "[v]"
        )
    ]
    has_audio = include_audio_transition and has_audio_stream(part_a) and has_audio_stream(part_b)
    if has_audio:
        filters.append(f"[0:a][1:a]acrossfade=d={transition_duration:.6f}[a]")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(part_a),
        "-i",
        str(part_b),
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[v]",
    ]

    if has_audio:
        cmd += ["-map", "[a]"]
    else:
        cmd += ["-an"]

    cmd += ["-c:v", video_encoder]
    if video_encoder in {"libx264", "libx265"}:
        cmd += ["-preset", "slow", "-crf", "15"]
    if video_bitrate > 0:
        cmd += ["-b:v", str(video_bitrate), "-maxrate", str(video_bitrate), "-bufsize", str(video_bitrate * 2)]

    if has_audio:
        cmd += ["-c:a", audio_encoder]
        if audio_bitrate > 0:
            cmd += ["-b:a", str(audio_bitrate)]

    cmd += ["-movflags", "+faststart", str(output_path)]
    run_cmd(cmd, progress_label="transition", expected_duration_sec=expected_duration)


def add_start_transition_with_preroll(
    source_path: Path,
    main_path: Path,
    output_path: Path,
    *,
    source_duration: float,
    preroll_from_end: bool,
    transition_type: str,
    transition_duration: float,
    video_encoder: str,
    video_bitrate: int,
    audio_encoder: str,
    audio_bitrate: int,
    include_audio_transition: bool,
) -> None:
    main_duration = ffprobe_duration_seconds(main_path)
    effective = min(transition_duration, source_duration, main_duration)
    if effective <= 0:
        main_path.replace(output_path)
        return

    with tempfile.TemporaryDirectory(prefix="voidstar_startpreroll_") as td:
        tdir = Path(td)
        suffix = main_path.suffix if main_path.suffix else ".mp4"
        preroll_path = tdir / f"preroll{suffix}"
        preroll_start = max(0.0, source_duration - effective) if preroll_from_end else 0.0

        extract_segment_copy(
            source_path,
            preroll_path,
            start=preroll_start,
            duration=effective,
            include_video=True,
            include_audio=include_audio_transition,
        )

        transition_join(
            preroll_path,
            main_path,
            output_path,
            transition_type=transition_type,
            transition_duration=effective,
            video_encoder=video_encoder,
            video_bitrate=video_bitrate,
            audio_encoder=audio_encoder,
            audio_bitrate=audio_bitrate,
            include_audio_transition=include_audio_transition,
        )


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
        "--transition-type",
        default="none",
        choices=TRANSITION_TYPES,
        help="Transition type applied at splice points (default: none)",
    )
    parser.add_argument(
        "--transition-duration",
        type=float,
        default=0.20,
        help="Transition duration in seconds (default: 0.20)",
    )
    parser.add_argument(
        "--no-start-transition",
        action="store_true",
        help="Disable transition at very start (default: enabled when transition type is not none)",
    )
    parser.add_argument(
        "--overwrite-audio",
        type=bool_flag,
        default=True,
        help="Overwrite mode only: if true, overwrite opening audio with hook audio (default: true)",
    )
    parser.add_argument("--output", default="", help="Explicit output file path")
    parser.add_argument("--out-dir", default="", help="Output directory when --output is omitted")
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Fail if output already exists (default behavior is overwrite)",
    )
    args = parser.parse_args()

    require_cmd("ffmpeg")
    require_cmd("ffprobe")

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        die(f"Input file not found: {input_path}")

    duration = ffprobe_duration_seconds(input_path)
    source_info = ffprobe_info(input_path)
    audio_present = has_audio_stream(input_path)
    v_stream = get_primary_stream(source_info, "video")
    a_stream = get_primary_stream(source_info, "audio")

    source_video_codec = str(v_stream.get("codec_name") or "h264")
    source_audio_codec = str(a_stream.get("codec_name") or "aac")
    source_video_bitrate = parse_int_or_zero(v_stream.get("bit_rate") or (source_info.get("format") or {}).get("bit_rate"))
    source_audio_bitrate = parse_int_or_zero(a_stream.get("bit_rate"))

    transition_type = str(args.transition_type)
    transition_duration = float(args.transition_duration)
    transition_enabled = transition_type != "none"
    start_transition_enabled = transition_enabled and not args.no_start_transition

    if args.hook_start < 0:
        die("--hook-start must be >= 0")
    if args.hook_start >= duration:
        die(f"--hook-start must be < input duration ({duration:.3f}s)")

    if args.mode in {"prepend", "overwrite"} and args.hook_duration <= 0:
        die("--hook-duration must be > 0 for prepend/overwrite modes")
    if transition_type != "none" and transition_duration <= 0:
        die("--transition-duration must be > 0 when --transition-type is not 'none'")

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
    if output_path.exists() and args.no_overwrite:
        die(f"Output already exists (delete it or omit --no-overwrite): {output_path}")

    log(f"Input:  {input_path}")
    log(f"Mode:   {args.mode}")
    log(f"Hook:   start={args.hook_start:.3f}s duration={hook_duration:.3f}s")
    log(f"Audio:  present={audio_present} overwrite_audio={args.overwrite_audio}")
    log(f"Trans:  type={transition_type} duration={transition_duration:.3f}s")
    log(f"Start:  transition={'on' if start_transition_enabled else 'off'}")
    log(f"Write:  overwrite_default={'yes' if not args.no_overwrite else 'no'}")
    log(f"Output: {output_path}")

    suffix = input_path.suffix if input_path.suffix else ".mp4"

    with tempfile.TemporaryDirectory(prefix="voidstar_hook_") as td:
        tdir = Path(td)
        mode_output = tdir / f"mode_output{suffix}"

        if args.mode == "prepend":
            hook_part = tdir / f"hook{suffix}"
            full_part = tdir / f"full{suffix}"
            effective_transition = 0.0
            hook_extract_duration = hook_duration
            if transition_enabled:
                effective_transition = min(transition_duration, hook_duration, duration)
                hook_extract_duration = min(max_hook_duration, hook_duration + effective_transition)
                if hook_extract_duration < (hook_duration + effective_transition - 1e-6):
                    log("Warning: insufficient source range to fully compensate transition overlap in prepend mode")

            extract_segment_copy(
                input_path,
                hook_part,
                start=args.hook_start,
                duration=hook_extract_duration,
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
            if not transition_enabled or effective_transition <= 0:
                concat_copy([hook_part, full_part], mode_output)
            else:
                transition_join(
                    hook_part,
                    full_part,
                    mode_output,
                    transition_type=transition_type,
                    transition_duration=effective_transition,
                    video_encoder=choose_video_encoder(source_video_codec),
                    video_bitrate=source_video_bitrate,
                    audio_encoder=choose_audio_encoder(source_audio_codec),
                    audio_bitrate=source_audio_bitrate,
                    include_audio_transition=audio_present,
                )

        elif args.mode == "overwrite":
            effective_transition = transition_duration if transition_enabled else 0.0
            effective_transition = min(effective_transition, hook_duration)
            tail_start = max(0.0, hook_duration - effective_transition)
            tail_duration = max(0.0, duration - tail_start)
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
                    start=tail_start,
                    duration=tail_duration,
                    include_video=True,
                    include_audio=audio_present,
                )
                if not transition_enabled or effective_transition <= 0:
                    concat_copy([hook_part, tail_part], mode_output)
                else:
                    transition_join(
                        hook_part,
                        tail_part,
                        mode_output,
                        transition_type=transition_type,
                        transition_duration=effective_transition,
                        video_encoder=choose_video_encoder(source_video_codec),
                        video_bitrate=source_video_bitrate,
                        audio_encoder=choose_audio_encoder(source_audio_codec),
                        audio_bitrate=source_audio_bitrate,
                        include_audio_transition=audio_present,
                    )
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
                    start=tail_start,
                    duration=tail_duration,
                    include_video=True,
                    include_audio=False,
                )
                if not transition_enabled or effective_transition <= 0:
                    concat_copy([hook_video, tail_video], video_composite)
                else:
                    transition_join(
                        hook_video,
                        tail_video,
                        video_composite,
                        transition_type=transition_type,
                        transition_duration=effective_transition,
                        video_encoder=choose_video_encoder(source_video_codec),
                        video_bitrate=source_video_bitrate,
                        audio_encoder=choose_audio_encoder(source_audio_codec),
                        audio_bitrate=source_audio_bitrate,
                        include_audio_transition=False,
                    )

                if audio_present:
                    mux_video_with_original_audio(video_composite, input_path, mode_output)
                else:
                    log("Input has no audio stream; outputting video-only overwrite result")
                    video_composite.replace(mode_output)

        else:  # wrap
            if args.hook_duration > 0:
                log("Wrap mode ignores --hook-duration and uses only --hook-start")

            first_duration = max(0.0, duration - args.hook_start)
            second_duration = args.hook_start
            effective_transition = 0.0
            if transition_enabled:
                effective_transition = min(transition_duration, first_duration, max(0.0, duration))
                desired_second_duration = second_duration + effective_transition
                second_duration = min(duration, desired_second_duration)
                if second_duration < (desired_second_duration - 1e-6):
                    log("Warning: insufficient source range to fully compensate transition overlap in wrap mode")
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

                if not transition_enabled or effective_transition <= 0:
                    concat_copy([first_part, second_part], mode_output)
                else:
                    transition_join(
                        first_part,
                        second_part,
                        mode_output,
                        transition_type=transition_type,
                        transition_duration=effective_transition,
                        video_encoder=choose_video_encoder(source_video_codec),
                        video_bitrate=source_video_bitrate,
                        audio_encoder=choose_audio_encoder(source_audio_codec),
                        audio_bitrate=source_audio_bitrate,
                        include_audio_transition=audio_present,
                    )
            else:
                first_part.replace(mode_output)

        if start_transition_enabled:
            preroll_from_end = args.mode in {"prepend", "wrap"}
            add_start_transition_with_preroll(
                input_path,
                mode_output,
                output_path,
                source_duration=duration,
                preroll_from_end=preroll_from_end,
                transition_type=transition_type,
                transition_duration=transition_duration,
                video_encoder=choose_video_encoder(source_video_codec),
                video_bitrate=source_video_bitrate,
                audio_encoder=choose_audio_encoder(source_audio_codec),
                audio_bitrate=source_audio_bitrate,
                include_audio_transition=audio_present,
            )
        else:
            mode_output.replace(output_path)

    log("Done.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        die(str(exc))