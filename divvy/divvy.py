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
import shlex
import subprocess
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


def main() -> None:
	ap = argparse.ArgumentParser(description="Split video into equal-length musical loop parts.")
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
	args = ap.parse_args()

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

	for i in range(parts):
		idx = i + 1
		start = (total_duration * i) / parts
		end = (total_duration * (i + 1)) / parts
		seg_dur = max(0.0, end - start)

		out_name = make_output_name(input_path, start, end, width_int)
		out_path = out_dir / out_name

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
				str(out_path),
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
				str(out_path),
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


if __name__ == "__main__":
	main()
