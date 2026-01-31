# reels-cv-overlay

Pose-based video overlay tool for vertical reels.

This script processes phone video using MediaPipe Pose and OpenCV, producing
clean, deterministic overlays suitable for Instagram / Reels / Shorts.

## Features

- CFR-first ffmpeg pipeline (no drift)
- Start / duration trimming
- Vertical (9:16) framing
- Pose skeleton with optional trails
- Optional landmark IDs
- Optional velocity-based coloring
- Scanline aesthetic pass
- Optional code overlay scrolling
- Optional external WAV audio mux

Baseline version: **v1**

## Requirements

- Python 3.10+
- ffmpeg (in PATH)

Install Python deps:

```bash
pip install -r requirements.txt
```

## Basic usage
python reels_cv_overlay.py input.mp4

## With landmark IDs
python reels_cv_overlay.py input.mp4 --draw-ids true

## Velocity coloring
python reels_cv_overlay.py input.mp4 --draw-ids true --velocity-color true

## Trim clip
python reels_cv_overlay.py input.mp4 --start 10 --duration 30

#Examples
```bash
#!/usr/bin/env bash
python reels_cv_overlay.py input.mp4 \
  --draw-ids true \
  --velocity-color true \
  --trail true
```

