#!/usr/bin/env python3
"""
Bridge script for calling existing Voidstar offline processors from TouchDesigner.

Usage examples:
python3 touchdesigner/voidstar_td_starter/voidstar_post_bridge.py \
  --effect dvdlogo \
  --input /path/in.mp4 \
  --output /path/out.mp4 \
  --logo dvd_logo/voidstar_logo_0.png

python3 touchdesigner/voidstar_td_starter/voidstar_post_bridge.py \
  --effect glitchfield \
  --input /path/in.mp4 \
  --output /path/out.mp4
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str]) -> None:
    print("[voidstar-post] " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def _require_path(path: str, flag_name: str) -> str:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    if not p.exists():
        raise SystemExit(f"Missing {flag_name}: {p}")
    return str(p)


def build_command(args: argparse.Namespace) -> list[str]:
    inp = _require_path(args.input, "--input")
    outp = str(Path(args.output).expanduser().resolve())
    py = sys.executable or "python3"

    if args.effect == "dvdlogo":
        logo = args.logo or "dvd_logo/voidstar_logo_0.png"
        logo_abs = _require_path(logo, "--logo")
        cmd = [
            py,
            str(ROOT / "dvd_logo" / "voidstar_dvd_logo.py"),
            inp,
            logo_abs,
            "--output",
            outp,
            "--audio-reactive-glow",
            str(args.audio_reactive_glow),
            "--audio-reactive-scale",
            str(args.audio_reactive_scale),
        ]
    elif args.effect == "glitchfield":
        cmd = [
            py,
            str(ROOT / "glitchfield" / "glitchfield.py"),
            inp,
            "--effect",
            args.glitch_effect,
            "--output",
            outp,
        ]
    elif args.effect == "reels_overlay":
        cmd = [
            py,
            str(ROOT / "reels_cv_overlay" / "reels_cv_overlay.py"),
            inp,
            "--output",
            outp,
        ]
    elif args.effect == "title_hook":
        cmd = [
            py,
            str(ROOT / "title_hook" / "voidstar_title_hook.py"),
            inp,
            "--output",
            outp,
            "--title",
            args.title,
            "--secondary-text",
            args.secondary_text,
        ]
        if args.logo:
            cmd.extend(["--logo", _require_path(args.logo, "--logo")])
    else:
        raise SystemExit(f"Unsupported --effect: {args.effect}")

    if args.extra:
        cmd.extend(shlex.split(args.extra))

    return cmd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--effect", required=True, choices=["dvdlogo", "glitchfield", "reels_overlay", "title_hook"])
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)

    ap.add_argument("--logo", default="")
    ap.add_argument("--title", default="VOIDSTAR")
    ap.add_argument("--secondary-text", default="AUDIO MOTION GLITCH")

    ap.add_argument("--audio-reactive-glow", type=float, default=0.7)
    ap.add_argument("--audio-reactive-scale", type=float, default=0.08)
    ap.add_argument("--glitch-effect", default="combo", choices=["glyph", "stutter", "combo"])

    ap.add_argument(
        "--extra",
        default="",
        help="Extra raw args appended to target script. Example: --extra '--duration 8 --start 12'",
    )

    args = ap.parse_args()
    cmd = build_command(args)
    _run(cmd)


if __name__ == "__main__":
    main()
