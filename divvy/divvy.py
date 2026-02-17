#!/usr/bin/env python3
"""
VoidStar Divvy

Split an input video into equal-length parts derived from a target segment length.
Cuts are made so every part has the same duration, and concatenating parts in order
reconstructs the full source timeline.

Default mode uses stream copy (`-c copy`) to preserve encoded streams/quality.
"""

import argparse
import json
import math
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def format_eta(seconds: float) -> str:
	if not math.isfinite(seconds) or seconds < 0:
		return "--:--:--"
	s = int(seconds)
	hh = s // 3600
	mm = (s % 3600) // 60
	ss = s % 60
	return f"{hh:02d}:{mm:02d}:{ss:02d}"


def ffprobe_info(input_path: Path) -> dict:
	cmd = [
		"ffprobe",
		"-v",
		"error",
		"-print_format",
		"json",
		"-show_format",
		"-show_streams",
		str(input_path),
	]
	out = subprocess.check_output(cmd, text=True)
	return json.loads(out)


def choose_num_parts(total_duration: float, target_seconds: float, rounding_mode: str) -> int:
	ratio = total_duration / max(1e-9, target_seconds)
	if rounding_mode == "ceil":
		parts = int(math.ceil(ratio))
	elif rounding_mode == "floor":
		parts = int(math.floor(ratio))
	else:
		parts = int(math.floor(ratio + 0.5))
	return max(1, parts)


def nearest_int_distance(x: float) -> float:
	return abs(x - round(x))


def choose_num_parts_bpm_quantized(
	total_duration: float,
	target_seconds: float,
	rounding_mode: str,
	bpm: float,
	beats_per_bar: float,
	search_radius: int,
) -> tuple[int, float, float]:
	base_parts = choose_num_parts(total_duration, target_seconds, rounding_mode)
	if bpm <= 0:
		part_len = total_duration / max(1, base_parts)
		beats = part_len * bpm / 60.0 if bpm > 0 else 0.0
		bars = beats / max(1e-9, beats_per_bar) if bpm > 0 else 0.0
		return base_parts, beats, bars

	best_parts = base_parts
	best_score = float("inf")
	best_beats = 0.0
	best_bars = 0.0

	lo = max(1, base_parts - max(0, search_radius))
	hi = max(lo, base_parts + max(0, search_radius))

	for parts in range(lo, hi + 1):
		part_len = total_duration / parts
		beats = part_len * bpm / 60.0
		bars = beats / max(1e-9, beats_per_bar)

		beat_err = nearest_int_distance(beats)
		bar_err = nearest_int_distance(bars)
		len_err = abs(part_len - target_seconds) / max(1e-9, target_seconds)
		part_bias = abs(parts - base_parts) / max(1, base_parts)

		# Prioritize musical grid lock first, then target duration closeness.
		score = (beat_err * 1000.0) + (bar_err * 120.0) + (len_err * 10.0) + part_bias
		if score < best_score:
			best_score = score
			best_parts = parts
			best_beats = beats
			best_bars = bars

	return best_parts, best_beats, best_bars


def has_nvenc_encoder() -> bool:
	try:
		out = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"], text=True, stderr=subprocess.STDOUT)
	except Exception:
		return False
	return "h264_nvenc" in out


def choose_video_encoder(user_encoder: str) -> str:
	if user_encoder != "auto":
		return user_encoder
	if has_nvenc_encoder():
		return "h264_nvenc"
	return "libx264"


def offset_tag(seconds: float, int_width: int, ms_digits: int = 3) -> str:
	scale = 10 ** ms_digits
	total_scaled = int(round(max(0.0, seconds) * scale))
	sec_int = total_scaled // scale
	frac = total_scaled % scale
	return f"{sec_int:0{int_width}d}_{frac:0{ms_digits}d}"


def make_output_name(input_path: Path, start: float, end: float, int_width: int) -> str:
	s_tag = offset_tag(start, int_width)
	e_tag = offset_tag(end, int_width)
	return f"{input_path.stem}__{s_tag}-{e_tag}{input_path.suffix}"


def run_segment_with_progress(
	cmd: list[str],
	seg_i: int,
	seg_n: int,
	seg_duration: float,
	done_before: float,
	total_duration: float,
	started_at: float,
	log_interval: float,
) -> None:
	proc = subprocess.Popen(
		cmd,
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		text=True,
		bufsize=1,
	)

	if proc.stdout is None:
		raise RuntimeError("ffmpeg progress stream unavailable")

	last_log = 0.0
	seg_progress = 0.0

	for raw in proc.stdout:
		line = raw.strip()
		if not line or "=" not in line:
			continue
		key, value = line.split("=", 1)

		if key == "out_time_ms":
			try:
				# ffmpeg progress out_time_ms is in microseconds.
				seg_progress = max(0.0, float(value) / 1_000_000.0)
			except ValueError:
				continue
			now = time.time()
			if now - last_log >= max(0.1, log_interval):
				done_now = min(total_duration, done_before + min(seg_progress, seg_duration))
				pct = 100.0 * done_now / max(1e-9, total_duration)
				elapsed = now - started_at
				speed = done_now / max(1e-9, elapsed)
				eta = (total_duration - done_now) / max(1e-9, speed)
				print(
					f"[voidstar] part={seg_i}/{seg_n} "
					f"overall={pct:.1f}% speed={speed:.3f}x eta={format_eta(eta)}"
				)
				last_log = now

	stderr = ""
	if proc.stderr is not None:
		stderr = proc.stderr.read().strip()
	rc = proc.wait()
	if rc != 0:
		raise RuntimeError(f"ffmpeg failed for part {seg_i}/{seg_n}: {stderr}")


def run_ffmpeg_with_progress(
	cmd: list[str],
	label: str,
	total_duration: float,
	started_at: float,
	log_interval: float,
) -> None:
	proc = subprocess.Popen(
		cmd,
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		text=True,
		bufsize=1,
	)

	if proc.stdout is None:
		raise RuntimeError("ffmpeg progress stream unavailable")

	last_log = 0.0
	done_now = 0.0

	for raw in proc.stdout:
		line = raw.strip()
		if not line or "=" not in line:
			continue
		key, value = line.split("=", 1)

		if key == "out_time_ms":
			try:
				done_now = max(0.0, float(value) / 1_000_000.0)
			except ValueError:
				continue
			now = time.time()
			if now - last_log >= max(0.1, log_interval):
				done_clamped = min(total_duration, done_now)
				pct = 100.0 * done_clamped / max(1e-9, total_duration)
				elapsed = now - started_at
				speed = done_clamped / max(1e-9, elapsed)
				eta = (total_duration - done_clamped) / max(1e-9, speed)
				print(
					f"[voidstar] {label} overall={pct:.1f}% "
					f"speed={speed:.3f}x eta={format_eta(eta)}"
				)
				last_log = now

	stderr = ""
	if proc.stderr is not None:
		stderr = proc.stderr.read().strip()
	rc = proc.wait()
	if rc != 0:
		raise RuntimeError(f"ffmpeg failed ({label}): {stderr}")


def parse_offset_pair_from_name(path: Path) -> tuple[float, float] | None:
	m = re.search(r"__(\d+)_(\d+)-(\d+)_(\d+)$", path.stem)
	if not m:
		return None
	s_int, s_frac, e_int, e_frac = m.groups()
	scale_s = 10 ** len(s_frac)
	scale_e = 10 ** len(e_frac)
	start = int(s_int) + (int(s_frac) / scale_s)
	end = int(e_int) + (int(e_frac) / scale_e)
	if end < start:
		return None
	return start, end


def list_segment_files(segment_dir: Path) -> list[Path]:
	allowed = {".mp4", ".mov", ".mkv", ".m4v", ".webm"}
	files = [p for p in segment_dir.iterdir() if p.is_file() and p.suffix.lower() in allowed]

	def sort_key(p: Path) -> tuple[int, float, float, str]:
		off = parse_offset_pair_from_name(p)
		if off is None:
			return (1, 0.0, 0.0, p.name.lower())
		return (0, off[0], off[1], p.name.lower())

	files.sort(key=sort_key)
	return files


def probe_duration_and_audio(path: Path) -> tuple[float, bool]:
	info = ffprobe_info(path)
	fmt = info.get("format", {})
	dur = float(fmt.get("duration", 0.0) or 0.0)
	streams = info.get("streams", [])
	has_audio = any(s.get("codec_type") == "audio" for s in streams)
	return dur, has_audio


def sanitize_tag(text: str) -> str:
	t = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip())
	t = re.sub(r"-+", "-", t).strip("-._")
	return t or "x"


def float_tag(value: float, digits: int = 3) -> str:
	s = f"{value:.{digits}f}".rstrip("0").rstrip(".")
	if not s:
		s = "0"
	return s.replace(".", "p")


def parse_time_token(text: str) -> float:
	t = text.strip()
	if not t:
		raise ValueError("empty time token")
	if ":" not in t:
		return float(t)
	parts = t.split(":")
	if len(parts) not in {2, 3}:
		raise ValueError(f"invalid time token: {text}")
	try:
		vals = [float(p) for p in parts]
	except ValueError as e:
		raise ValueError(f"invalid time token: {text}") from e
	if len(vals) == 2:
		mm, ss = vals
		return (mm * 60.0) + ss
	hh, mm, ss = vals
	return (hh * 3600.0) + (mm * 60.0) + ss


def parse_time_range_token(text: str) -> tuple[float, float]:
	t = text.strip()
	if not t:
		raise ValueError("empty range token")
	if "-" not in t:
		raise ValueError(f"invalid range token: {text}")
	left, right = t.split("-", 1)
	start = parse_time_token(left)
	end = parse_time_token(right)
	if start < 0 or end < 0:
		raise ValueError(f"range values must be non-negative: {text}")
	if end <= start:
		raise ValueError(f"range end must be greater than start: {text}")
	return start, end


def overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
	if a_end <= a_start or b_end <= b_start:
		return 0.0
	return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def overlap_fraction(start: float, duration: float, ranges: list[tuple[float, float]]) -> float:
	if duration <= 0:
		return 0.0
	end = start + duration
	overlap = 0.0
	for r_start, r_end in ranges:
		overlap += overlap_seconds(start, end, r_start, r_end)
	return min(1.0, max(0.0, overlap / max(1e-9, duration)))


def ffconcat_quote(path: Path) -> str:
	return str(path).replace("'", r"'\\''")


def build_recombine_default_filename(segment_dir: Path, args: argparse.Namespace) -> str:
	order_tag = "rev" if args.reverse_order else "fwd"
	glitch_sec_tag = float_tag(max(0.0, args.glitch_seconds), digits=3)
	glitch_style_tag = sanitize_tag(args.glitch_style)
	loop_tag = "loop1" if args.loop_perfect else "loop0"
	return (
		f"{segment_dir.name}__recombined"
		f"__ord-{order_tag}"
		f"__glitch-{glitch_style_tag}-{glitch_sec_tag}s"
		f"__{loop_tag}.mp4"
	)


def build_highlights_default_filename(input_path: Path, args: argparse.Namespace) -> str:
	mode_tag = sanitize_tag(args.sampling_mode)
	target_tag = float_tag(max(0.0, args.target_length_seconds), digits=3)
	start_tag = float_tag(max(0.0, args.start_seconds), digits=3)
	full_tag = float_tag(max(0.0, args.youtube_full_seconds), digits=3)
	return (
		f"{input_path.stem}__highlights"
		f"__mode-{mode_tag}"
		f"__start-{start_tag}s"
		f"__full-{full_tag}s"
		f"__target-{target_tag}s.mp4"
	)


def add_split_args(ap: argparse.ArgumentParser) -> None:
	ap.add_argument("input", help="Input video path")
	ap.add_argument(
		"--target-seconds",
		type=float,
		required=True,
		help="Target part length in seconds (used to choose number of equal parts)",
	)
	ap.add_argument(
		"--rounding",
		choices=["nearest", "ceil", "floor"],
		default="nearest",
		help="How target/total ratio maps to number of parts",
	)
	ap.add_argument(
		"--out-dir",
		default=None,
		help="Output directory (default: <input_dir>/<input_stem>_divvy)",
	)
	ap.add_argument(
		"--bpm",
		type=float,
		default=0.0,
		help="Optional BPM for loop diagnostics (beats per part shown in logs)",
	)
	ap.add_argument(
		"--beats-per-bar",
		type=float,
		default=4.0,
		help="Time signature denominator for bar diagnostics when --bpm is set",
	)
	ap.add_argument(
		"--log-interval",
		type=float,
		default=1.0,
		help="Progress print interval in seconds",
	)
	ap.add_argument(
		"--ffmpeg-extra",
		default="",
		help="Optional extra ffmpeg args appended to each segment command",
	)
	ap.add_argument(
		"--cut-accuracy",
		choices=["accurate", "copy"],
		default="accurate",
		help="accurate=re-encode for exact boundaries (default), copy=stream copy faster but less exact",
	)
	ap.add_argument(
		"--bpm-quantize",
		choices=["auto", "off", "on"],
		default="auto",
		help="Adjust number of parts to better land on BPM beat/bar grid (default: auto=on when --bpm > 0)",
	)
	ap.add_argument(
		"--bpm-search-radius",
		type=int,
		default=12,
		help="How far around target part count to search when BPM quantization is enabled",
	)
	ap.add_argument(
		"--video-encoder",
		default="auto",
		help="Video encoder for accurate mode (auto/libx264/h264_nvenc/libx265/hevc_nvenc)",
	)
	ap.add_argument(
		"--preset",
		default="p5",
		help="Encoder preset (e.g. p5 for nvenc, medium for libx264)",
	)
	ap.add_argument(
		"--crf",
		type=int,
		default=18,
		help="CRF for libx264/libx265 accurate mode",
	)
	ap.add_argument(
		"--nvenc-cq",
		type=int,
		default=19,
		help="CQ value for NVENC accurate mode (lower=better quality)",
	)
	ap.add_argument(
		"--audio-codec",
		default="aac",
		help="Audio codec for accurate mode",
	)
	ap.add_argument(
		"--audio-bitrate",
		default="320k",
		help="Audio bitrate for accurate mode",
	)


def add_recombine_args(ap: argparse.ArgumentParser) -> None:
	ap.add_argument("segment_dir", help="Directory containing divvy segment files")
	ap.add_argument(
		"--out-dir",
		default=None,
		help="Output directory for recombined file (used when --output is not set)",
	)
	ap.add_argument(
		"--output",
		default=None,
		help="Output recombined video path (default: <segment_dir>/<segment_dir_name>__recombined.mp4)",
	)
	ap.add_argument(
		"--reverse-order",
		action="store_true",
		help="Recombine in reverse segment order",
	)
	ap.add_argument(
		"--glitch-seconds",
		type=float,
		default=0.20,
		help="Transition duration between parts in seconds",
	)
	ap.add_argument(
		"--glitch-style",
		default="pixelize",
		help="ffmpeg xfade transition style for glitch effect (e.g. pixelize, hblur, fadeblack)",
	)
	ap.add_argument(
		"--intro-glitch-seconds",
		type=float,
		default=0.06,
		help="Small start-only intro transition duration in seconds (0 to disable)",
	)
	ap.add_argument(
		"--loop-perfect",
		action="store_true",
		default=True,
		help="Enable loop-seam transition and split offset so output loops cleanly (default: on)",
	)
	ap.add_argument(
		"--no-loop-perfect",
		action="store_false",
		dest="loop_perfect",
		help="Disable loop-seam transition/split offset",
	)
	ap.add_argument(
		"--log-interval",
		type=float,
		default=1.0,
		help="Progress print interval in seconds",
	)
	ap.add_argument(
		"--ffmpeg-extra",
		default="",
		help="Optional extra ffmpeg args appended to recombine command",
	)
	ap.add_argument(
		"--video-encoder",
		default="auto",
		help="Video encoder (auto/libx264/h264_nvenc/libx265/hevc_nvenc)",
	)
	ap.add_argument(
		"--preset",
		default="p5",
		help="Encoder preset (e.g. p5 for nvenc, medium for libx264)",
	)
	ap.add_argument(
		"--crf",
		type=int,
		default=18,
		help="CRF for libx264/libx265",
	)
	ap.add_argument(
		"--nvenc-cq",
		type=int,
		default=19,
		help="CQ value for NVENC (lower=better quality)",
	)
	ap.add_argument(
		"--audio-codec",
		default="aac",
		help="Audio codec",
	)
	ap.add_argument(
		"--audio-bitrate",
		default="320k",
		help="Audio bitrate",
	)
	ap.add_argument(
		"--target-total-seconds",
		type=float,
		default=0.0,
		help="Target max recombined duration in seconds; uniformly trims the start of each source segment (0 disables)",
	)


def add_highlights_args(ap: argparse.ArgumentParser) -> None:
	ap.add_argument("input", help="Input video path for automated highlights")
	ap.add_argument(
		"--out-dir",
		default=None,
		help="Output directory for highlights (used when --output is not set)",
	)
	ap.add_argument(
		"--output",
		default=None,
		help="Output highlights video path (default: auto name in input dir)",
	)
	ap.add_argument(
		"--start-seconds",
		type=float,
		default=0.0,
		help="Start time in source video for highlight sampling window",
	)
	ap.add_argument(
		"--youtube-full-seconds",
		type=float,
		required=True,
		help="Total source window length from start-seconds to sample from",
	)
	ap.add_argument(
		"--target-length-seconds",
		type=float,
		required=True,
		help="Desired output highlights length target",
	)
	ap.add_argument(
		"--sampling-mode",
		choices=["minute-averages", "n-averages"],
		default="minute-averages",
		help="Sampling strategy for selecting source snippets",
	)
	ap.add_argument(
		"--sample-seconds",
		type=float,
		default=0.0,
		help="Seconds to sample from each segment (0 = auto fill to target)",
	)
	ap.add_argument(
		"--n-segments",
		type=int,
		default=12,
		help="Number of equal segments for n-averages mode",
	)
	ap.add_argument(
		"--log-interval",
		type=float,
		default=1.0,
		help="Progress print interval in seconds",
	)
	ap.add_argument(
		"--ffmpeg-extra",
		default="",
		help="Optional extra ffmpeg args appended to highlights command",
	)
	ap.add_argument(
		"--video-encoder",
		default="auto",
		help="Video encoder (auto/libx264/h264_nvenc/libx265/hevc_nvenc)",
	)
	ap.add_argument(
		"--preset",
		default="p5",
		help="Encoder preset (e.g. p5 for nvenc, medium for libx264)",
	)
	ap.add_argument(
		"--crf",
		type=int,
		default=18,
		help="CRF for libx264/libx265",
	)
	ap.add_argument(
		"--nvenc-cq",
		type=int,
		default=19,
		help="CQ value for NVENC (lower=better quality)",
	)
	ap.add_argument(
		"--audio-codec",
		default="aac",
		help="Audio codec",
	)
	ap.add_argument(
		"--audio-bitrate",
		default="320k",
		help="Audio bitrate",
	)
	ap.add_argument(
		"--ignore-range",
		action="append",
		default=[],
		help="Exclude sampling in time ranges. Repeatable. Format: START-END (seconds or MM:SS or HH:MM:SS)",
	)
	ap.add_argument(
		"--deemphasize-range",
		action="append",
		default=[],
		help="Downweight sampling in time ranges. Repeatable. Format: START-END (seconds or MM:SS or HH:MM:SS)",
	)
	ap.add_argument(
		"--deemphasize-factor",
		type=float,
		default=0.35,
		help="Relative sampling weight inside deemphasized ranges (0..1, default 0.35)",
	)


def run_split(args: argparse.Namespace) -> None:
	input_path = Path(args.input).expanduser().resolve()
	if not input_path.exists():
		raise FileNotFoundError(f"Input not found: {input_path}")
	if args.target_seconds <= 0:
		raise ValueError("--target-seconds must be > 0")

	out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else input_path.parent / f"{input_path.stem}_divvy"
	out_dir.mkdir(parents=True, exist_ok=True)

	probe = ffprobe_info(input_path)
	fmt = probe.get("format", {})
	total_duration = float(fmt.get("duration", 0.0) or 0.0)
	if total_duration <= 0:
		raise RuntimeError("Could not determine input duration from ffprobe")

	quant_enabled = (args.bpm_quantize == "on") or (args.bpm_quantize == "auto" and args.bpm > 0)
	if quant_enabled and args.bpm > 0:
		parts, quant_beats_per_part, quant_bars_per_part = choose_num_parts_bpm_quantized(
			total_duration=total_duration,
			target_seconds=args.target_seconds,
			rounding_mode=args.rounding,
			bpm=args.bpm,
			beats_per_bar=args.beats_per_bar,
			search_radius=args.bpm_search_radius,
		)
	else:
		parts = choose_num_parts(total_duration, args.target_seconds, args.rounding)
		quant_beats_per_part = 0.0
		quant_bars_per_part = 0.0
	actual_part_len = total_duration / parts

	width_int = max(2, len(str(int(math.ceil(total_duration)))))

	beat_info = ""
	if args.bpm > 0:
		beat_sec = 60.0 / args.bpm
		beats_per_part = actual_part_len / beat_sec
		bars_per_part = beats_per_part / max(1e-9, args.beats_per_bar)
		beat_info = (
			f" beats/part={beats_per_part:.4f} "
			f"bars/part={bars_per_part:.4f} (at {args.bpm:g} BPM)"
		)

	print(f"[voidstar] input={input_path}")
	print(f"[voidstar] duration={total_duration:.6f}s target={args.target_seconds:.6f}s")
	print(f"[voidstar] parts={parts} actual_part_len={actual_part_len:.6f}s{beat_info}")
	if quant_enabled and args.bpm > 0:
		print(
			f"[voidstar] bpm_quantized=on beats/part={quant_beats_per_part:.6f} "
			f"bars/part={quant_bars_per_part:.6f}"
		)
	else:
		print("[voidstar] bpm_quantized=off")

	print(f"[voidstar] cut_accuracy={args.cut_accuracy}")

	enc = choose_video_encoder(args.video_encoder)
	if args.cut_accuracy == "accurate":
		print(f"[voidstar] accurate_encoder={enc} audio={args.audio_codec}@{args.audio_bitrate}")
	else:
		print("[voidstar] mode=stream-copy (faster, but boundaries may drift from requested times)")

	extra_args = shlex.split(args.ffmpeg_extra) if args.ffmpeg_extra.strip() else []

	started = time.time()
	created_files: list[Path] = []
	with tempfile.TemporaryDirectory(prefix="divvy_split_", dir="/tmp") as tmp_dir_str:
		tmp_dir = Path(tmp_dir_str)

		for i in range(parts):
			idx = i + 1
			start = (total_duration * i) / parts
			end = (total_duration * (i + 1)) / parts
			seg_dur = max(0.0, end - start)

			out_name = make_output_name(input_path, start, end, width_int)
			out_path = out_dir / out_name
			staged_out_path = tmp_dir / out_name

			if args.cut_accuracy == "copy":
				cmd = [
					"ffmpeg",
					"-hide_banner",
					"-loglevel",
					"error",
					"-nostats",
					"-progress",
					"pipe:1",
					"-y",
					"-ss",
					f"{start:.6f}",
					"-t",
					f"{seg_dur:.6f}",
					"-i",
					str(input_path),
					"-map",
					"0",
					"-c",
					"copy",
					"-avoid_negative_ts",
					"make_zero",
					*extra_args,
					str(staged_out_path),
				]
			else:
				cmd = [
					"ffmpeg",
					"-hide_banner",
					"-loglevel",
					"error",
					"-nostats",
					"-progress",
					"pipe:1",
					"-y",
					"-ss",
					f"{start:.6f}",
					"-i",
					str(input_path),
					"-t",
					f"{seg_dur:.6f}",
					"-map",
					"0:v:0",
					"-map",
					"0:a:0?",
					"-c:v",
					enc,
				]

				if enc in {"h264_nvenc", "hevc_nvenc"}:
					cmd += ["-preset", args.preset, "-rc", "vbr", "-cq", str(args.nvenc_cq), "-b:v", "0"]
				elif enc in {"libx264", "libx265"}:
					cmd += ["-preset", args.preset, "-crf", str(args.crf)]

				cmd += [
					"-c:a",
					args.audio_codec,
					"-b:a",
					args.audio_bitrate,
					"-movflags",
					"+faststart",
					*extra_args,
					str(staged_out_path),
				]

			print(
				f"[voidstar] start part={idx}/{parts} "
				f"start={start:.6f}s end={end:.6f}s out={out_path.name}"
			)

			run_segment_with_progress(
				cmd=cmd,
				seg_i=idx,
				seg_n=parts,
				seg_duration=seg_dur,
				done_before=start,
				total_duration=total_duration,
				started_at=started,
				log_interval=args.log_interval,
			)

			if not staged_out_path.exists():
				raise RuntimeError(f"Staged split output missing: {staged_out_path}")

			print(f"[voidstar] copy_to_final src={staged_out_path} dst={out_path}")
			shutil.copy2(staged_out_path, out_path)
			created_files.append(out_path)

			elapsed = time.time() - started
			done = end
			speed = done / max(1e-9, elapsed)
			eta = (total_duration - done) / max(1e-9, speed)
			pct = 100.0 * done / max(1e-9, total_duration)
			print(
				f"[voidstar] done part={idx}/{parts} overall={pct:.1f}% "
				f"speed={speed:.3f}x eta={format_eta(eta)}"
			)

	elapsed_total = time.time() - started
	throughput = total_duration / max(1e-9, elapsed_total)

	print("[voidstar] ===== summary =====")
	print(f"[voidstar] output_dir={out_dir}")
	print(f"[voidstar] input_duration={total_duration:.6f}s")
	print(f"[voidstar] parts={parts}")
	print(f"[voidstar] part_duration={actual_part_len:.6f}s")
	if quant_enabled and args.bpm > 0:
		print(
			f"[voidstar] quant_beats_per_part={quant_beats_per_part:.6f} "
			f"quant_bars_per_part={quant_bars_per_part:.6f}"
		)
	if args.bpm > 0:
		beat_sec = 60.0 / args.bpm
		beats_per_part = actual_part_len / beat_sec
		bars_per_part = beats_per_part / max(1e-9, args.beats_per_bar)
		print(f"[voidstar] bpm={args.bpm:g} beats_per_part={beats_per_part:.4f} bars_per_part={bars_per_part:.4f}")
	print(f"[voidstar] elapsed={elapsed_total:.2f}s throughput={throughput:.3f}x")
	print(f"[voidstar] files_written={len(created_files)}")


def run_recombine(args: argparse.Namespace) -> None:
	segment_dir = Path(args.segment_dir).expanduser().resolve()
	if not segment_dir.exists() or not segment_dir.is_dir():
		raise FileNotFoundError(f"Segment directory not found: {segment_dir}")

	if args.output:
		out_path = Path(args.output).expanduser().resolve()
	elif args.out_dir:
		out_dir = Path(args.out_dir).expanduser().resolve()
		out_path = out_dir / build_recombine_default_filename(segment_dir, args)
	else:
		out_path = segment_dir / build_recombine_default_filename(segment_dir, args)
	out_path.parent.mkdir(parents=True, exist_ok=True)

	segment_files = list_segment_files(segment_dir)
	# Ignore previous recombined renders and never allow the output file to be reused as an input.
	segment_files = [
		p for p in segment_files
		if p.resolve() != out_path and not p.stem.endswith("__recombined")
	]
	if len(segment_files) < 1:
		raise RuntimeError(f"No segment video files found in: {segment_dir}")
	if len(segment_files) < 2 and args.glitch_seconds > 0:
		print("[voidstar] only one segment found; disabling transitions")

	if args.reverse_order:
		segment_files = list(reversed(segment_files))

	durations: list[float] = []
	has_audio_all = True
	for p in segment_files:
		dur, has_audio = probe_duration_and_audio(p)
		if dur <= 0:
			raise RuntimeError(f"Could not determine duration for: {p}")
		durations.append(dur)
		has_audio_all = has_audio_all and has_audio

	glitch_dur = max(0.0, args.glitch_seconds)
	intro_glitch_dur = max(0.0, min(args.intro_glitch_seconds, 0.50))
	min_dur = min(durations)
	if glitch_dur > 0 and glitch_dur >= (min_dur * 0.95):
		raise ValueError(f"--glitch-seconds too large for shortest segment ({min_dur:.6f}s)")

	loop_perfect = bool(args.loop_perfect and len(segment_files) >= 2 and glitch_dur > 0)
	segment_count = len(segment_files)
	transition_count = segment_count if (glitch_dur > 0 and segment_count >= 2 and loop_perfect) else (segment_count - 1 if glitch_dur > 0 and segment_count >= 2 else 0)

	target_total = max(0.0, float(args.target_total_seconds))
	trim_each = 0.0
	if target_total > 0 and segment_count > 0:
		base_total = sum(durations)
		base_est_output = max(0.0, base_total - (transition_count * glitch_dur))
		if base_est_output > target_total:
			min_keep = 0.010
			if glitch_dur > 0 and segment_count >= 2:
				min_keep = max(min_keep, (glitch_dur / 0.95) + 0.001)
			max_trim_each = max(0.0, min_dur - min_keep)

			trim_each = max(0.0, (base_est_output - target_total) / max(1, segment_count))
			if trim_each > max_trim_each:
				min_possible_output = max(0.0, (base_total - (max_trim_each * segment_count)) - (transition_count * glitch_dur))
				raise ValueError(
					"--target-total-seconds cannot be reached with uniform per-segment start trim "
					f"while preserving transition constraints. min_possible={min_possible_output:.6f}s"
				)

	used_durations = [max(0.001, d - trim_each) for d in durations]
	min_used_dur = min(used_durations)
	if glitch_dur > 0 and glitch_dur >= (min_used_dur * 0.95):
		raise ValueError(f"--target-total-seconds trim leaves segments too short for glitch ({min_used_dur:.6f}s)")

	input_files = list(segment_files)
	input_durations = list(used_durations)
	if loop_perfect:
		input_files.append(segment_files[0])
		input_durations.append(used_durations[0])

	total_input = sum(used_durations)
	if glitch_dur > 0 and len(segment_files) >= 2:
		est_output_duration = max(0.0, total_input - (transition_count * glitch_dur))
	else:
		transition_count = 0
		est_output_duration = total_input

	if est_output_duration > 0:
		intro_glitch_dur = min(intro_glitch_dur, est_output_duration * 0.25)
	if intro_glitch_dur < 0.001:
		intro_glitch_dur = 0.0

	enc = choose_video_encoder(args.video_encoder)
	extra_args = shlex.split(args.ffmpeg_extra) if args.ffmpeg_extra.strip() else []

	filter_parts: list[str] = []
	video_inputs: list[str] = []
	audio_inputs: list[str] = []
	for i in range(len(input_files)):
		v_lab = f"[v{i}]"
		if trim_each > 0:
			filter_parts.append(
				f"[{i}:v]trim=start={trim_each:.6f},setpts=PTS-STARTPTS{v_lab}"
			)
		else:
			filter_parts.append(f"[{i}:v]setpts=PTS-STARTPTS{v_lab}")
		video_inputs.append(v_lab)

		if has_audio_all:
			a_lab = f"[a{i}]"
			if trim_each > 0:
				filter_parts.append(
					f"[{i}:a]atrim=start={trim_each:.6f},asetpts=PTS-STARTPTS{a_lab}"
				)
			else:
				filter_parts.append(f"[{i}:a]asetpts=PTS-STARTPTS{a_lab}")
			audio_inputs.append(a_lab)

	if glitch_dur > 0 and len(input_files) >= 2:
		v_cur = video_inputs[0]
		a_cur = audio_inputs[0] if has_audio_all else ""
		offset = max(0.0, input_durations[0] - glitch_dur)
		for i in range(1, len(input_files)):
			v_out = f"[vx{i}]"
			filter_parts.append(
				f"{v_cur}{video_inputs[i]}xfade=transition={args.glitch_style}:duration={glitch_dur:.6f}:offset={offset:.6f}{v_out}"
			)
			v_cur = v_out

			if has_audio_all:
				a_out = f"[ax{i}]"
				filter_parts.append(
					f"{a_cur}{audio_inputs[i]}acrossfade=d={glitch_dur:.6f}:c1=tri:c2=tri{a_out}"
				)
				a_cur = a_out

			offset += max(0.0, input_durations[i] - glitch_dur)
	else:
		v_inputs = "".join(video_inputs)
		filter_parts.append(f"{v_inputs}concat=n={len(input_files)}:v=1:a=0[vx0]")
		v_cur = "[vx0]"
		if has_audio_all:
			a_inputs = "".join(audio_inputs)
			filter_parts.append(f"{a_inputs}concat=n={len(input_files)}:v=0:a=1[ax0]")
			a_cur = "[ax0]"
		else:
			a_cur = ""

	if loop_perfect:
		split_offset = glitch_dur * 0.5
		cycle_duration = max(1e-6, est_output_duration)
		filter_parts.append(
			f"{v_cur}trim=start={split_offset:.6f}:duration={cycle_duration:.6f},setpts=PTS-STARTPTS[vbase]"
		)
		if has_audio_all:
			filter_parts.append(
				f"{a_cur}atrim=start={split_offset:.6f}:duration={cycle_duration:.6f},asetpts=PTS-STARTPTS[abase]"
			)
	else:
		filter_parts.append(f"{v_cur}setpts=PTS-STARTPTS[vbase]")
		if has_audio_all:
			filter_parts.append(f"{a_cur}asetpts=PTS-STARTPTS[abase]")

	if intro_glitch_dur > 0:
		filter_parts.append("[vbase]split=2[vintro_src][vintro_main]")
		filter_parts.append(
			f"[vintro_src]trim=duration=0.001,tpad=stop_mode=clone:stop_duration={intro_glitch_dur:.6f},setpts=PTS-STARTPTS[vintro_hold]"
		)
		filter_parts.append(
			f"[vintro_hold][vintro_main]xfade=transition={args.glitch_style}:duration={intro_glitch_dur:.6f}:offset=0[vout]"
		)
	else:
		filter_parts.append("[vbase]setpts=PTS-STARTPTS[vout]")

	if has_audio_all:
		filter_parts.append(
			"[abase]"
			"aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
			"aresample=48000:resampler=soxr:precision=28,"
			"alimiter=limit=0.89:attack=5:release=50,"
			"aresample=async=1:first_pts=0,"
			"asetpts=PTS-STARTPTS[aout]"
		)

	filter_complex = ";".join(filter_parts)

	cmd: list[str] = [
		"ffmpeg",
		"-hide_banner",
		"-loglevel",
		"error",
		"-nostats",
		"-progress",
		"pipe:1",
		"-y",
	]

	for p in input_files:
		cmd += ["-i", str(p)]

	cmd += ["-filter_complex", filter_complex, "-map", "[vout]"]
	if has_audio_all:
		cmd += ["-map", "[aout]"]

	cmd += ["-c:v", enc]
	if enc in {"h264_nvenc", "hevc_nvenc"}:
		cmd += ["-preset", args.preset, "-rc", "vbr", "-cq", str(args.nvenc_cq), "-b:v", "0"]
	elif enc in {"libx264", "libx265"}:
		cmd += ["-preset", args.preset, "-crf", str(args.crf)]

	if enc in {"h264_nvenc", "libx264"}:
		cmd += ["-pix_fmt", "yuv420p", "-profile:v", "high", "-level:v", "4.1"]

	if has_audio_all:
		cmd += ["-c:a", args.audio_codec, "-b:a", args.audio_bitrate]

	cmd += ["-movflags", "+faststart", *extra_args, str(out_path)]

	print(f"[voidstar] recombine_input_dir={segment_dir}")
	print(f"[voidstar] output={out_path}")
	print(f"[voidstar] segment_count={len(segment_files)} reverse_order={'on' if args.reverse_order else 'off'}")
	print(f"[voidstar] glitch_style={args.glitch_style} glitch_seconds={glitch_dur:.6f}")
	print(f"[voidstar] intro_glitch_seconds={intro_glitch_dur:.6f}")
	print(f"[voidstar] target_total_seconds={target_total:.6f} trim_each_start={trim_each:.6f}")
	print(f"[voidstar] loop_perfect={'on' if loop_perfect else 'off'}")
	print(f"[voidstar] encoder={enc} audio={'on' if has_audio_all else 'off'}")
	if has_audio_all:
		print("[voidstar] audio_stabilize=soxr+limiter+aresample_async1")
	if enc in {"h264_nvenc", "libx264"}:
		print("[voidstar] video_compat=h264_yuv420p_high_4.1")
	print(f"[voidstar] est_output_duration={est_output_duration:.6f}s")

	started = time.time()
	with tempfile.TemporaryDirectory(prefix="divvy_recombine_", dir="/tmp") as tmp_dir_str:
		tmp_dir = Path(tmp_dir_str)
		staged_out_path = tmp_dir / out_path.name
		cmd[-1] = str(staged_out_path)

		run_ffmpeg_with_progress(
			cmd=cmd,
			label="recombine",
			total_duration=max(1e-6, est_output_duration),
			started_at=started,
			log_interval=args.log_interval,
		)

		if not staged_out_path.exists():
			raise RuntimeError(f"Staged output missing after recombine: {staged_out_path}")

		print(f"[voidstar] copy_to_final src={staged_out_path} dst={out_path}")
		shutil.copy2(staged_out_path, out_path)
	elapsed_total = time.time() - started
	throughput = est_output_duration / max(1e-9, elapsed_total)

	print("[voidstar] ===== summary =====")
	print(f"[voidstar] output_file={out_path}")
	print(f"[voidstar] segment_count={len(segment_files)}")
	print(f"[voidstar] reverse_order={'on' if args.reverse_order else 'off'}")
	print(f"[voidstar] transitions={transition_count} style={args.glitch_style} duration={glitch_dur:.6f}s")
	print(f"[voidstar] loop_perfect={'on' if loop_perfect else 'off'}")
	print(f"[voidstar] elapsed={elapsed_total:.2f}s throughput={throughput:.3f}x")


def run_highlights(args: argparse.Namespace) -> None:
	input_path = Path(args.input).expanduser().resolve()
	if not input_path.exists():
		raise FileNotFoundError(f"Input not found: {input_path}")
	if not (0.0 <= float(args.deemphasize_factor) <= 1.0):
		raise ValueError("--deemphasize-factor must be between 0 and 1")

	if args.output:
		out_path = Path(args.output).expanduser().resolve()
	elif args.out_dir:
		out_path = Path(args.out_dir).expanduser().resolve() / build_highlights_default_filename(input_path, args)
	else:
		out_path = input_path.parent / build_highlights_default_filename(input_path, args)
	out_path.parent.mkdir(parents=True, exist_ok=True)

	if args.youtube_full_seconds <= 0:
		raise ValueError("--youtube-full-seconds must be > 0")
	if args.target_length_seconds <= 0:
		raise ValueError("--target-length-seconds must be > 0")

	video_duration, has_audio = probe_duration_and_audio(input_path)
	if video_duration <= 0:
		raise RuntimeError("Could not determine input duration")

	window_start = max(0.0, args.start_seconds)
	if window_start >= video_duration:
		raise ValueError("--start-seconds is beyond input duration")

	window_len = min(max(0.0, args.youtube_full_seconds), max(0.0, video_duration - window_start))
	if window_len <= 0:
		raise ValueError("Sampling window length is zero after clamping to input duration")
	window_end = window_start + window_len

	ignore_ranges_abs: list[tuple[float, float]] = []
	for raw in args.ignore_range:
		rs, re_ = parse_time_range_token(raw)
		rs = min(max(0.0, rs), video_duration)
		re_ = min(max(0.0, re_), video_duration)
		if re_ > rs:
			ignore_ranges_abs.append((rs, re_))

	deemph_ranges_abs: list[tuple[float, float]] = []
	for raw in args.deemphasize_range:
		rs, re_ = parse_time_range_token(raw)
		rs = min(max(0.0, rs), video_duration)
		re_ = min(max(0.0, re_), video_duration)
		if re_ > rs:
			deemph_ranges_abs.append((rs, re_))

	if args.sampling_mode == "minute-averages":
		segment_len = 60.0
		segment_count = max(1, int(math.ceil(window_len / segment_len)))
	else:
		segment_count = max(1, int(args.n_segments))
		segment_len = window_len / max(1, segment_count)

	if args.sample_seconds > 0:
		sample_len = args.sample_seconds
	else:
		sample_len = args.target_length_seconds / max(1, segment_count)

	if sample_len <= 0:
		raise ValueError("Computed sample length is zero; increase target length or set --sample-seconds")

	segments: list[tuple[float, float]] = []
	for i in range(segment_count):
		seg_start = window_start + (i * segment_len)
		if seg_start >= window_end:
			break
		max_for_segment = min(segment_len, window_end - seg_start)
		dur = min(sample_len, max_for_segment)

		ignore_frac = overlap_fraction(seg_start, max_for_segment, ignore_ranges_abs)
		if ignore_frac >= 0.999:
			continue

		deemph_frac = overlap_fraction(seg_start, max_for_segment, deemph_ranges_abs)
		weight = (1.0 - ignore_frac) * (1.0 - (deemph_frac * (1.0 - args.deemphasize_factor)))
		dur = min(dur, max_for_segment * max(0.0, weight))

		if dur > 0.001:
			segments.append((seg_start, dur))

	if not segments:
		raise RuntimeError("No highlight segments were generated from provided settings")

	total_target = max(0.0, args.target_length_seconds)
	selected: list[tuple[float, float]] = []
	total_selected = 0.0
	for start, dur in segments:
		if total_selected >= total_target:
			break
		remaining = total_target - total_selected
		use_dur = min(dur, remaining)
		if use_dur > 0.001:
			selected.append((start, use_dur))
			total_selected += use_dur

	if not selected:
		raise RuntimeError("No highlight clips selected; try increasing --target-length-seconds")

	enc = choose_video_encoder(args.video_encoder)
	extra_args = shlex.split(args.ffmpeg_extra) if args.ffmpeg_extra.strip() else []

	print(f"[voidstar] highlights_input={input_path}")
	print(f"[voidstar] output={out_path}")
	print(f"[voidstar] mode={args.sampling_mode} start={window_start:.3f}s window={window_len:.3f}s")
	print(f"[voidstar] target={args.target_length_seconds:.3f}s segments={len(selected)} sample_seconds={sample_len:.3f}")
	print(f"[voidstar] selected_total={total_selected:.3f}s encoder={enc} audio={'on' if has_audio else 'off'}")
	if ignore_ranges_abs:
		print(f"[voidstar] ignore_ranges={len(ignore_ranges_abs)}")
	if deemph_ranges_abs:
		print(f"[voidstar] deemphasize_ranges={len(deemph_ranges_abs)} factor={args.deemphasize_factor:.3f}")
	print("[voidstar] pipeline=segment-then-concat (low-memory)")

	for i, (start, dur) in enumerate(selected, start=1):
		print(f"[voidstar] sample={i}/{len(selected)} start={start:.3f}s dur={dur:.3f}s")

	started = time.time()
	with tempfile.TemporaryDirectory(prefix="divvy_highlights_", dir="/tmp") as tmp_dir_str:
		tmp_dir = Path(tmp_dir_str)
		clip_paths: list[Path] = []
		done_before = 0.0
		staged_out_path = tmp_dir / out_path.name

		for i, (start, dur) in enumerate(selected, start=1):
			clip_path = tmp_dir / f"clip_{i:03d}.mp4"
			clip_cmd: list[str] = [
				"ffmpeg",
				"-hide_banner",
				"-loglevel",
				"error",
				"-nostats",
				"-progress",
				"pipe:1",
				"-y",
				"-ss",
				f"{start:.6f}",
				"-i",
				str(input_path),
				"-t",
				f"{dur:.6f}",
				"-map",
				"0:v:0",
				"-map",
				"0:a:0?",
				"-c:v",
				enc,
			]

			if enc in {"h264_nvenc", "hevc_nvenc"}:
				clip_cmd += ["-preset", args.preset, "-rc", "vbr", "-cq", str(args.nvenc_cq), "-b:v", "0"]
			elif enc in {"libx264", "libx265"}:
				clip_cmd += ["-preset", args.preset, "-crf", str(args.crf)]

			if enc in {"h264_nvenc", "libx264"}:
				clip_cmd += ["-pix_fmt", "yuv420p", "-profile:v", "high", "-level:v", "4.1"]

			if has_audio:
				clip_cmd += ["-c:a", args.audio_codec, "-b:a", args.audio_bitrate]
			else:
				clip_cmd += ["-an"]

			clip_cmd += [*extra_args, str(clip_path)]

			run_segment_with_progress(
				cmd=clip_cmd,
				seg_i=i,
				seg_n=len(selected),
				seg_duration=dur,
				done_before=done_before,
				total_duration=max(1e-6, total_selected),
				started_at=started,
				log_interval=args.log_interval,
			)
			done_before += dur
			clip_paths.append(clip_path)

		concat_list = tmp_dir / "concat.txt"
		concat_lines = [f"file '{ffconcat_quote(p)}'" for p in clip_paths]
		concat_list.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")

		concat_cmd: list[str] = [
			"ffmpeg",
			"-hide_banner",
			"-loglevel",
			"error",
			"-nostats",
			"-progress",
			"pipe:1",
			"-y",
			"-f",
			"concat",
			"-safe",
			"0",
			"-i",
			str(concat_list),
			"-c",
			"copy",
			"-movflags",
			"+faststart",
			str(staged_out_path),
		]

		run_ffmpeg_with_progress(
			cmd=concat_cmd,
			label="highlights-concat",
			total_duration=max(1e-6, total_selected),
			started_at=time.time(),
			log_interval=args.log_interval,
		)

		if not staged_out_path.exists():
			raise RuntimeError(f"Staged output missing after concat: {staged_out_path}")

		print(f"[voidstar] copy_to_final src={staged_out_path} dst={out_path}")
		shutil.copy2(staged_out_path, out_path)
	elapsed_total = time.time() - started
	throughput = total_selected / max(1e-9, elapsed_total)

	print("[voidstar] ===== summary =====")
	print(f"[voidstar] output_file={out_path}")
	print(f"[voidstar] mode={args.sampling_mode} segments={len(selected)}")
	print(f"[voidstar] target={args.target_length_seconds:.3f}s actual={total_selected:.3f}s")
	print(f"[voidstar] elapsed={elapsed_total:.2f}s throughput={throughput:.3f}x")


def main() -> None:
	ap = argparse.ArgumentParser(description="VoidStar Divvy: split and recombine musical loop parts.")
	sub = ap.add_subparsers(dest="command")

	ap_split = sub.add_parser("split", help="Split source video into equal loop parts")
	add_split_args(ap_split)

	ap_recombine = sub.add_parser("recombine", help="Recombine divvy parts with glitch transitions")
	add_recombine_args(ap_recombine)

	ap_highlights = sub.add_parser("highlights", help="Auto-sample and assemble highlight reels")
	add_highlights_args(ap_highlights)

	argv = sys.argv[1:]
	if argv and argv[0] not in {"split", "recombine", "highlights", "-h", "--help"}:
		# Back-compat: legacy invocation defaults to split command.
		argv = ["split", *argv]

	args = ap.parse_args(argv)

	if args.command == "recombine":
		run_recombine(args)
	elif args.command == "highlights":
		run_highlights(args)
	else:
		run_split(args)


if __name__ == "__main__":
	main()
